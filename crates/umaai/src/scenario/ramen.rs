//! 拉面（`scenarioId=14`）回合处理：切局检测、链式决策计算与 luck 挂载 emit。

use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;
use rand::rngs::StdRng;
use umasim::{
    game::{
        Game,
        Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    output::{
        DecisionInfo,
        DecisionSink,
        GameView,
        reason::render_reason_lines
    }
};

use crate::decision::{emit_with_luck_decision, LastReasonSink, LuckScoreTracker};

/// 处理拉面剧本的一个回合快照（`thisTurn.json` → `ParsedGame::Ramen`）
///
/// 承接原 main watch loop 的 ramen 分支：`Begin`（未 dispatch）早退、切局检测、
/// human 屏幕打印、链式决策 emit 与 luck 挂载。
///
/// 注意：`single_mode_chara_id` 为切局键（`None` 时退化用 uma_id）。
pub fn process_ramen<T: Trainer<RamenGame>>(
    mut game: RamenGame,
    single_mode_chara_id: Option<u64>,
    trainer: &T,
    reason_slot: &LastReasonSink,
    sink: &Arc<dyn DecisionSink>,
    luck_tracker: &mut LuckScoreTracker,
    rng: &mut StdRng,
    json_mode: bool,
    emit_info: &dyn Fn(&str)
) -> Result<()> {
    if game.stage == RamenStage::Begin {
        // into_game 没 dispatch（事件 / 结算 / 数据不全），等下一条 JSON。
        // 本轮无决策，但仍要收尾 compute_done——保证每轮 JSON 流
        // `compute_start → compute_done` 成对，C# 端"计算中"状态不会悬挂。
        emit_info("compute_done");
        return Ok(());
    }

    // 切局检测：`single_mode_chara_id` 变化 → 新一局开始。
    // C# 端 single_mode_chara_id 单调递增，同 uma_id 重复训练也能识别新局。
    // 协议字段缺失（None）时退化到 uma_id 兜底（旧 json / 测试 fixture）。
    let chara_id = single_mode_chara_id
        .unwrap_or_else(|| game.uma().uma_id as u64);
    if luck_tracker.last_single_mode_id() != Some(chara_id) {
        // 检测到新一局：通知 AIRed 重置 UI 状态
        emit_info("new_game");
        eprintln!("{}", "---- 拉面: 育成开始 ----".bright_yellow());
        *luck_tracker = LuckScoreTracker::new();
    }

    // 屏幕侧（human mode）按需求在收到并解析回合数据后**立即**显示：
    // 马娘状态 / 剧本信息 / 训练分布——后续才进入推理（calc_ramen_training）。
    if !json_mode {
        if let Ok(status) = game.explain() {
            println!("{status}");
        }
        let script_info = game.explain_ramen_info();
        if !script_info.is_empty() {
            println!("{script_info}");
        }
        if let Ok(dist_info) = game.explain_distribution() {
            println!("{dist_info}");
        }
    }
    eprintln!("AI计算中...");
    // 连续决策：链式决策（一个快照对应多个决策）。
    // 中间决策（除最后一个）已在 calc_ramen_training 内部、决策#2 计算前
    // 立即 emit（不触 luck）——保证 JSON 流顺序为 decision#1 →
    // compute_next_step → decision#2；此处 `chain` 只剩末项，由下方走
    // 完整 luck 挂载（baseline 每回合只更新一次）。
    let chain = calc_ramen_training(trainer, &mut game, rng, json_mode, reason_slot, emit_info, sink)?;
    if !chain.is_empty() {
        // 拉面 MCTS 路径：从 LastReasonSink 缓存取 DecisionReasonData 挂到
        // scenario_extra.reason，让 AIRedirector 拿到 human mode reason
        // 所需信息（metric / chosen_desc / chosen_mean / rivals[]）。
        // 链式决策**最后一步**才消费 reason_slot，避免中间决策漏挂。
        //
        // decision_kind 用最后一步决策的 stage——主链通常是 train（拉面
        // 决策落地后的训练阶段）。ramen_action 同样用最后一步决策的
        // candidate_descriptions[action_index]（to_string 形式，含
        // 训练名 + 之前已经 ground 的吃面效果）。
        let last_info = chain.last().expect("non-empty chain").0.clone();
        let last_kind = last_info.decision_kind.clone();
        let ramen_action_text = last_info
            .candidate_descriptions
            .get(last_info.action_index)
            .cloned();

        // 无搜索评分的决策（`candidate_scores` 为空，如 region 未开搜索时的地区选择、
        // **比赛回合单候选**、RamenSelect 单候选短路、网络做出的地区选择）：
        // 没有真正的搜索评分，走 luck 挂载只会以 baseline=0 污染 luck tracker
        // （后续回合运气全被算错），且 sink 打印的「期望评分」只是回合加成换算、
        // 运气恒 0 会误导。故直接 emit（不触 luck）；
        // HumanReadableSink 会为该决策打印「选择...（手写逻辑）」，网络做出的地区选择
        // 按来源标签另行标注。搜索决策（常见 train/ramen_select）仍走完整 luck 挂载。
        if last_info.candidate_scores.is_empty() {
            sink.emit(&last_info, &game.view());
        } else {
            emit_with_luck_decision(
                Some(last_info),
                &game,
                sink,
                luck_tracker,
                chara_id,
                reason_slot.take().as_ref(),
                &last_kind,
                ramen_action_text.as_deref(),
            );
        }
    }

    // 计算完成：通知下游 watcher 进入阻塞状态
    emit_info("compute_done");
    eprintln!("计算完成，等待新数据...");
    Ok(())
}

/// 拉面训练：当前阶段出推荐，并在**两个特定场景**连续出下一个决策
///
///**设计原则**：
///- watch 收到一次 `thisTurn.json` 只代表"当前回合、当前阶段"的快照，AI 基于本次
///  快照出推荐（select_action）。**仅解决"一个快照对应两个决策"的场景**，其余
///  情况下**不**改 game（下次 watch 收到新 JSON → 主循环重建 game 从零计算）。
///- 定向连续决策（类似 onsen 的"选完温泉券后继续给训练推荐"）：
///  1. `RamenSelect` 选**不吃面**：不吃面没有真实操作产生新 JSON，手动
///     `apply_action` + `next()` 推进到 `Train`，再给训练决策。
///  2. `Train` 且 turn == 1（仅剧本机制启动前的第 1 回合）：训练决策后下一屏是
///     回合 2 的地区选择（同样无新 JSON），跨过 `NextTurn` 推进到 `RegionSelect`，
///     再给地区决策；到达 RegionSelect 后**立即停**，不继续向下级联。
///- 其它所有阶段维持单决策：AI 不推进游戏状态，玩家执行后由 C# 发新 JSON。
///
/// 返回链式决策 `Vec<(DecisionInfo, GameView)>`。
///
/// **emit 时机**：链式决策的**中间项**（决策#1）在函数内部、决策#2 真正执行前
/// （`compute_next_step` 通知前）经 `sink` 立即 emit（不触 luck）——保证 JSON 流
/// 顺序为 `decision#1 → compute_next_step → decision#2`，下游不会先收到
/// "还在计算"通知而以为本回合没有结果。**末项**（决策#2，或非链式场景的决策#1）
/// 留在返回值中，由 call 方走完整 luck 挂载；每个决策附带其**作出时**的
/// `GameView`，保证决策行的 `turn`/`scenario` 正确。
pub fn calc_ramen_training<T: Trainer<RamenGame>>(
    trainer: &T, game: &mut RamenGame, rng: &mut StdRng, json_mode: bool, reason_slot: &LastReasonSink,
    emit_info: &dyn Fn(&str), sink: &Arc<dyn DecisionSink>
) -> Result<Vec<(DecisionInfo, GameView)>> {
    // 链式决策收集：每次 select_action 捕获 DecisionInfo + 该阶段 view
    let mut out: Vec<(DecisionInfo, GameView)> = Vec::new();
    let mut any_decision = false;

    {
        // 对当前阶段做一次决策：捕获决策与其阶段 view，返回选中的动作
        // （g / out / rng 走参数，避免闭包长期独占借用与下方直接使用冲突；仅捕获共享 trainer）
        //
        // 2026-09 扩展：snapshot select_action 前的 stage 填到 info.decision_kind——
        // 让 AIRedirector 端按 partial decision 类型分发。trainer 不感知 stage，
        // 由"发起决策的 umaai"统一管理。
        let decide =
            |g: &mut RamenGame, out: &mut Vec<(DecisionInfo, GameView)>, rng: &mut StdRng| -> Result<Option<RamenAction>> {
                let before_stage = g.stage.clone();
                let actions = match g.stage {
                    RamenStage::NextTurn | RamenStage::Settlement | RamenStage::SuperRamenSelect => {
                        // 回合边界 / RMJ 结算 / 超级拉面选择 —— 等下一条 JSON，AI 不出推荐
                        Vec::new()
                    }
                    _ => g.list_actions()?
                };
                if actions.is_empty() {
                    return Ok(None);
                }
                let idx = trainer.select_action(g, &actions, rng)?;
                let chosen = actions[idx].clone();
                let view = g.view();
                // `last_decision()` 仅对真正走过 MCTS 搜索的阶段返回 `Some`；其它（门控
                // 关闭的 `region`、单候选等）返回 `None`。
                // 仅以下场景需合成一条输出（手写 fallback）——否则该决策没有结果可 emit：
                // 1) 地区选择（门控关闭，手写策略）——最初"无结果"的问题；
                // 2) **比赛回合**：`is_race_turn()` 下落 `Train`，list_actions 只有"比赛"
                //    一个固定动作，trainer 因单候选直接落 fallback、不搜索，`last_decision()`
                //    为 `None`，不合成的话 calc_ramen_training 返回空、屏幕上无策略输出。
                // 3) **RamenSelect 决策（兜底）**：合并搜索路径 2026-09 起按面聚合后暴露
                //    `last_decision()`；仅当回落三阶段逻辑（合并候选 ≤ 1 / 单候选短路）
                //    仍为 `None` 时才在此合成（吃面不链式时同样需要决策行）。
                //    其余 None 阶段保持旧行为（决策仍返回但**不**合成、不 emit）。
                let mut info = match trainer.last_decision() {
                    Some(info) => Some(info),
                    None if synthesizes_fallback(g) => Some(fallback_decision(&actions, idx, &before_stage)),
                    None => None,
                };
                if let Some(mut info) = info.take() {
                    info.decision_kind = ramen_stage_kind(before_stage).to_string();
                    out.push((info, view));
                }
                Ok(Some(chosen))
            };

        if let Some(chosen) = decide(game, &mut out, rng)? {
            any_decision = true;
            let before_stage = game.stage.clone();
            let before_turn = game.turn();
            // 定向连续决策判定：仅两个场景在决策#1 后继续给下一个决策
            let need_continue = (before_stage == RamenStage::RamenSelect && !chosen.is_eating_ramen())
                || (before_stage == RamenStage::Train && before_turn == 1);

            if need_continue {
                // 决策#1 是链式决策的**中间项**：先把它的结果 emit 出去（下游拿到
                // 即时反馈），再从 `out` 移除——否则它要等本函数返回后才由 call 方
                // emit，而 `compute_next_step` 已在下面决策#2 前发出，JSON 顺序变成
                // `compute_next_step` 先于任何决策结果到达，下游会误以为本回合没算。
                // 末项（决策#2）仍留在 `out` 返回，由 call 方走完整 luck 挂载。
                // 注：RamenSelect 决策已在上方 decide 合成（见合成条件 3），
                // `out` 通常有内容；其它 None 阶段不合成时这里无输出，与旧行为一致。
                if let Some((info, view)) = out.first() {
                    sink.emit(info, view);
                }
                out.clear();
                // 应用决策#1 并推进一个阶段（RamenSelect 不吃 → Train；Train(turn==1) → AfterTrain）
                game.apply_action(&chosen, rng)?;
                if game.next() {
                    // 逐阶段推进直到下一决策点（或真正需要等新 JSON 的结算 / 超级拉面阶段）
                    const MAX_STAGE_LOOP: usize = 32;
                    for _ in 0..MAX_STAGE_LOOP {
                        match game.stage {
                            // RMJ 结算 / 超级拉面选择：等新 JSON，不再续
                            RamenStage::Settlement | RamenStage::SuperRamenSelect => break,
                            // 到达决策点：给出决策#2，随后停止（定向，不再向下级联）
                            RamenStage::RamenSelect
                            | RamenStage::SpecialSelect
                            | RamenStage::Train
                            | RamenStage::RegionSelect => {
                                // 连续决策的**第 2 个决策**前先通知下游 "AI 还在算这一回合"：
                                // 必须在真正执行决策#2（select_action，MCTS 可能耗时数秒）
                                // **之前**打出屏幕并 emit——首决策前已在 watch loop 入口发射过
                                // compute_start，不需要重复。
                                eprintln!("计算后续动作...");
                                emit_info("compute_next_step");
                                let _ = decide(game, &mut out, rng)?;
                                break;
                            }
                            // 自动阶段（Begin / BeginAfterRegionSelect / Distribute / AfterTrain / NextTurn）：
                            // 交给 umasim 的 run_stage 执行载荷，再用 next() 推进到下一阶段
                            _ => {
                                game.run_stage(trainer, rng)?;
                                if !game.next() {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // 屏幕侧（human mode）输出推理结果（回合头部打印由 call 方在调用前完成）
    if !json_mode {
        if any_decision {
            if let Some(data) = reason_slot.take() {
                for line in render_reason_lines(&data) {
                    println!("{line}");
                }
            }
        }
        println!("{}", "[按 F2 保存当前回合状态]".bright_black());
    }
    Ok(out)
}

/// `last_decision()` 为 `None` 时，本决策点是否仍要合成一条 [`fallback_decision`]
///
/// `game` 是做决策时（应用动作之前）的局面。三种场景见 [`calc_ramen_training`] 内的注释：
/// 地区选择、比赛回合、`RamenSelect` 兜底。
pub(crate) fn synthesizes_fallback(game: &RamenGame) -> bool {
    game.stage == RamenStage::RegionSelect
        || (game.stage == RamenStage::Train && game.is_race_turn())
        || game.stage == RamenStage::RamenSelect
}

/// RamenStage → decision_kind 字符串映射
///
/// 拉面模块在 calc_ramen_training 内部 snapshot stage 填这个字段——trainer 不关心，
/// 由"发起决策的 umaai"统一管理（按用户拍板）。
fn ramen_stage_kind(stage: RamenStage) -> &'static str {
    match stage {
        RamenStage::Begin => "begin",
        RamenStage::Distribute => "distribute",
        RamenStage::RamenSelect => "ramen_select",
        RamenStage::SpecialSelect => "special_select",
        RamenStage::Train => "train",
        RamenStage::AfterTrain => "after_train",
        RamenStage::NextTurn => "next_turn",
        RamenStage::RegionSelect => "region_select",
        RamenStage::SuperRamenSelect => "super_ramen_select",
        RamenStage::Settlement => "settlement",
        RamenStage::BeginAfterRegionSelect => "begin_after_region_select"
    }
}

/// 为 `last_decision()` 为 `None` 的阶段合成一条最小 `DecisionInfo`
///
/// [`Trainer::last_decision`] 只在**真正走过 MCTS 搜索**时返回 `Some`；门控关闭的阶段
/// （如默认配置 `ramen_search_stages="train,ramen"` 下未开启的 `region`）落入手写
/// fallback，合并搜索的 `RamenSelect` 路径（吃面 / 不吃面）也会清掉 last_summary——
/// 这些情况 `last_decision()` 均为 `None`，但策略确实作出了选择，导致该阶段没有任何
/// 决策结果输出。这里按本次候选列表与选中下标合成一条无搜索评分的决策信息，保证
/// region_select / ramen_select 等阶段也有结果可 emit（candidate_scores 为空，
/// luck baseline 退化按等权）。
pub(crate) fn fallback_decision(
    actions: &[RamenAction], chosen_idx: usize, before_stage: &RamenStage
) -> DecisionInfo {
    DecisionInfo {
        action_index: chosen_idx,
        score: 0.0,
        decision_kind: ramen_stage_kind(before_stage.clone()).to_string(),
        candidate_scores: Vec::new(),
        candidate_descriptions: actions.iter().map(|a| a.to_string()).collect(),
        candidate_n: Vec::new(),
        scenario_extra: None
    }
}

#[cfg(test)]
mod tests {
    use std::{env, sync::Mutex};

    use rand::SeedableRng;
    use umasim::{
        game::InheritInfo,
        gamedata::init_global,
        search::SearchConfig,
        trainer::{RamenMctsTrainer, RamenSearchStages},
        utils::get_workspace_root
    };

    use super::*;
    #[cfg(feature = "onnx")]
    use crate::utils::unique_test_dir;
    use crate::{decision::LuckScoreTracker, utils::Checks};

    /// 最小 ONNX 模型生成器（workspace 根的 `testsupport/onnx_fixture.rs`，与 `umasim` 侧共用）
    #[cfg(feature = "onnx")]
    #[path = "../../../../../../testsupport/onnx_fixture.rs"]
    mod onnx_fixture;

    /// 临时 fixture 目录的 RAII 清理（测试中途早退也会删）
    #[cfg(feature = "onnx")]
    struct FixtureDir(std::path::PathBuf);

    #[cfg(feature = "onnx")]
    impl Drop for FixtureDir {
        fn drop(&mut self) {
            if let Err(e) = crate::utils::cleanup_test_dir(&self.0) {
                println!("fixture 目录清理失败: {e:#}");
            }
        }
    }

    /// 让 fixture 模型偏好的地区组合
    ///
    /// `RegionSelect` 的候选打分是三格之和，抬高某个合法组合的三个地区，网络的推荐
    /// 就被钉在该组合上（其他组合最多共享 2 个地区）。
    #[cfg(feature = "onnx")]
    #[derive(Debug, Clone, Copy)]
    enum FixturePref {
        /// 把这组地区 ID 对应的 policy 格位抬高
        Regions([usize; 3])
    }

    /// 当场生成一个可加载的 fixture 模型（含旁车），返回 `(清理守卫, 模型路径)`
    ///
    /// # 错误
    ///
    /// 地区 ID 越界、临时目录创建或文件写入失败时返回。
    #[cfg(feature = "onnx")]
    fn fixture_model(tag: &str, pref: FixturePref) -> Result<(FixtureDir, std::path::PathBuf)> {
        use umasim::{
            game::ramen::{
                features::INPUT_DIM,
                policy_schema::{POLICY_DIM, region_index}
            },
            training_sample::{CHOICE_DIM, VALUE_DIM}
        };

        let dir = unique_test_dir(tag)?;
        let model = dir.join("fixture.onnx");
        let out_dim = POLICY_DIM + CHOICE_DIM + VALUE_DIM;
        let FixturePref::Regions(ids) = pref;
        let mut logits = vec![0.0f32; out_dim];
        for rid in ids {
            logits[region_index(rid)?] = 10.0;
        }
        fs_err::write(&model, onnx_fixture::const_logits_model(INPUT_DIM, &logits)?)?;
        fs_err::write(
            dir.join("fixture.onnx.json"),
            format!(
                r#"{{"input_dim":{INPUT_DIM},"output_dim":{out_dim},"value_normalization":{{"center":[0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}}}}"#
            )
        )?;
        Ok((FixtureDir(dir), model))
    }

    /// 把 `decision` / `info` 两路输出按发生顺序记进一个流，用来核对 JSON 流次序
    ///
    /// 只记 `type:kind` 标签，不比任何指纹——次序与完整性看的就是这串标签本身。
    #[derive(Default)]
    struct EventLog {
        events: Mutex<Vec<String>>,
        /// 真正 emit 出去的决策原件（按顺序；经过了 luck 挂载，内容以它为准）
        infos: Mutex<Vec<DecisionInfo>>
    }

    impl EventLog {
        /// 追加一条事件
        fn push(&self, s: String) {
            self.events.lock().expect("事件流锁").push(s);
        }

        /// 追加一条 emit 出去的决策原件
        fn push_info(&self, info: DecisionInfo) {
            self.infos.lock().expect("决策原件锁").push(info);
        }

        /// 按发生顺序取出事件标签
        fn take(&self) -> Vec<String> {
            self.events.lock().expect("事件流锁").clone()
        }

        /// 按发生顺序取出 emit 出去的决策原件
        #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
        fn decisions(&self) -> Vec<DecisionInfo> {
            self.infos.lock().expect("决策原件锁").clone()
        }
    }

    /// 记录 `DecisionSink::emit` 的 sink（决策行进同一个事件流）
    struct RecordingSink(Arc<EventLog>);

    impl DecisionSink for RecordingSink {
        fn emit(&self, info: &DecisionInfo, _view: &GameView) {
            self.0.push_info(info.clone());
            // `src` 是决策来源标签（未标注记 `-`）；`reason` 记 scenario_extra 里
            // 有没有搜索理由——地区决策挂上它就说明把上一步搜索的理由带过来了。
            let has_reason = info
                .scenario_extra
                .as_ref()
                .is_some_and(|v| v.get("reason").is_some());
            self.0.push(format!(
                "decision:{}(scores={},cands={},idx={},src={},reason={has_reason})",
                info.decision_kind,
                info.candidate_scores.len(),
                info.candidate_descriptions.len(),
                info.action_index,
                info.source_label().unwrap_or("-")
            ));
        }
    }

    /// 把搜索压到最小的训练员（本测试只看输出次序，不看棋力）
    fn small_trainer() -> Result<RamenMctsTrainer> {
        staged_trainer("train,ramen")
    }

    /// 建一局标准拉面（卡组含新友人卡，`newgame` 会校验）
    fn new_game() -> Result<RamenGame> {
        RamenGame::newgame(
            101901,
            &[303124, 303114, 303084, 303094, 303064, 303054],
            InheritInfo {
                blue_count: [15, 0, 3, 0, 0],
                extra_count: [0, 40, 40, 20, 20, 40]
            }
        )
    }

    /// 把新局自然推进到 turn 1 的 `Train` 阶段（链式决策的触发点之一）
    ///
    /// # 错误
    ///
    /// 推进途中报错、提前终局，或超过步数上限仍没到达目标阶段时报错。
    fn advance_to_turn1_train(
        mut game: RamenGame, trainer: &RamenMctsTrainer, rng: &mut StdRng
    ) -> Result<RamenGame> {
        const MAX_STEPS: usize = 400;
        for _ in 0..MAX_STEPS {
            if game.turn() == 1 && game.stage == RamenStage::Train {
                return Ok(game);
            }
            if !game.next() {
                anyhow::bail!("推进到 turn 1 Train 之前本局就结束了");
            }
            if game.turn() == 1 && game.stage == RamenStage::Train {
                return Ok(game);
            }
            game.run_stage(trainer, rng)?;
        }
        anyhow::bail!("{MAX_STEPS} 步内未到达 turn 1 的 Train 阶段")
    }

    /// 地区回合（turn 2）：恰好一条 `region_select` 决策，不触发链式，`compute_done` 收尾
    ///
    /// 覆盖上游 e5cdd64 之后的输出契约：每轮 `compute_start → … → compute_done` 成对，
    /// 地区决策**不遗漏、不重复**；它走 `fallback_decision`，`candidate_scores` 为空
    /// —— 即没有把上一步搜索的理由挂到这次地区决策上。
    #[test]
    fn test_region_turn_emits_one_decision_and_compute_done() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let mut game = new_game()?;
        game.base.turn = 2;
        game.stage = RamenStage::RegionSelect;
        let trainer = small_trainer()?;
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260914);

        process_ramen(game, Some(1), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        let decisions: Vec<_> = ev.iter().filter(|e| e.starts_with("decision:")).collect();
        c.check(decisions.len() == 1, &format!("恰好 1 条决策（实际 {}）", decisions.len()));
        c.check(
            decisions.first().is_some_and(|d| d.starts_with("decision:region_select")),
            "该决策的 decision_kind 是 region_select"
        );
        c.check(
            decisions.first().is_some_and(|d| d.contains("scores=0")),
            "地区决策不带搜索评分（没有把上一步搜索的理由挂过来）"
        );
        c.check(
            decisions.first().is_some_and(|d| d.contains("cands=10")),
            "第 1 年 10 个候选全部进 candidate_descriptions"
        );
        c.check(
            decisions.first().is_some_and(|d| d.contains("reason=false")),
            "地区决策的 scenario_extra 里没有搜索理由"
        );
        c.check(
            !ev.iter().any(|e| e == "info:compute_next_step"),
            "地区回合不触发链式决策，无 compute_next_step"
        );
        c.check(
            ev.iter().filter(|e| *e == "info:compute_done").count() == 1,
            "恰好一条 compute_done 收尾"
        );
        c.check(ev.last().is_some_and(|e| e == "info:compute_done"), "compute_done 是最后一条");
        c.finish()
    }

    /// 按指定搜索阶段构造一个最小搜索训练员
    fn staged_trainer(stages: &str) -> Result<RamenMctsTrainer> {
        Ok(
            RamenMctsTrainer::new(SearchConfig::default().with_search_n(2).with_ucb(false))
                .with_stages(RamenSearchStages::parse(stages)?)
                .verbose(false)
        )
    }

    /// 加载 fixture 地区网络
    #[cfg(feature = "onnx")]
    fn load_fixture_nn(path: &std::path::Path) -> Result<umasim::trainer::RamenNnTrainer> {
        use umasim::trainer::{RamenNnTrainer, SpecialSelectMode};

        Ok(RamenNnTrainer::load(path)?
            .with_race_shield(true)
            .with_special_mode(SpecialSelectMode::Canonical))
    }

    /// 在第 1 年地区回合跑一次 `process_ramen`，返回 (事件标签流, emit 出去的决策, 之后的随机流)
    ///
    /// # 错误
    ///
    /// 建局或 `process_ramen` 报错时返回。
    #[cfg(feature = "onnx")]
    fn run_region_turn<T: Trainer<RamenGame>>(
        trainer: &T, seed: u64
    ) -> Result<(Vec<String>, Vec<DecisionInfo>, StdRng)> {
        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));
        let mut game = new_game()?;
        game.base.turn = 2;
        game.stage = RamenStage::RegionSelect;
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(seed);
        process_ramen(game, Some(3), trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;
        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        Ok((ev, log.decisions(), rng))
    }

    /// 取出候选 `i` 的三个地区下标（候选不是地区组合时为 `None`）
    #[cfg(feature = "onnx")]
    fn region_of(actions: &[RamenAction], i: usize) -> Option<[usize; 3]> {
        match actions.get(i)?.operation {
            umasim::game::ramen::Operation::RegionSelect(r) => Some(r),
            _ => None
        }
    }

    /// `nn`：地区回合由网络选，来源标为 ramen_nn，没有搜索评分
    ///
    /// 走真实 `process_ramen`，核对 emit 出去的决策：恰好一条 `region_select`、
    /// 来源标签是 `ramen_nn`（不会被渲染成手写）、没有搜索评分与理由、选中候选
    /// 就是 fixture 钉住的地区组合。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_whole_nn_executes_network_pick() -> Result<()> {
        use umasim::output::decision::SOURCE_RAMEN_NN;

        use crate::ramen_nn::WholeNnTrainer;

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        let pinned = [0, 2, 4];
        let (_fixture_dir, model_path) = fixture_model("whole_nn_exec", FixturePref::Regions(pinned))?;
        let mut probe = new_game()?;
        probe.base.turn = 2;
        probe.stage = RamenStage::RegionSelect;
        let actions = probe.list_actions()?;

        let trainer = WholeNnTrainer::new(load_fixture_nn(&model_path)?);
        let (ev, infos, _) = run_region_turn(&trainer, 20260914)?;
        let decisions: Vec<_> = ev.iter().filter(|e| e.starts_with("decision:")).collect();
        c.check(decisions.len() == 1, &format!("恰好 1 条决策（实际 {}）", decisions.len()));
        let first = decisions.first().map(|s| s.as_str()).unwrap_or_default();
        c.check(first.starts_with("decision:region_select"), "decision_kind 是 region_select");
        c.check(first.contains(&format!("src={SOURCE_RAMEN_NN}")), "来源标签是 ramen_nn");
        c.check(first.contains("scores=0"), "没有搜索评分");
        c.check(first.contains("reason=false"), "没有挂搜索理由");
        c.check(ev.last().is_some_and(|e| e == "info:compute_done"), "compute_done 收尾");
        let picked = infos.first().and_then(|i| region_of(&actions, i.action_index));
        println!("选中地区组合 {picked:?}");
        c.check(picked == Some(pinned), &format!("执行的是网络钉住的 {pinned:?}"));
        c.finish()
    }

    /// `mcts_nn_hint`：地区回合的执行与 `mcts` 完全一致，只多挂一条网络参考
    ///
    /// 同一种子下分别跑纯搜索训练员与 hint 装配，地区走手写与走搜索两种阶段集各一遍，
    /// 核对：事件流、执行的候选、候选评分、来源标签、之后的随机流都一致；hint 装配的
    /// 决策上挂着 `nn_hint`，且与 luck 快照同时存在。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_nn_hint_matches_mcts_on_region_turn() -> Result<()> {
        use rand::RngCore;
        use umasim::output::decision::NN_HINT_KEY;

        use crate::ramen_nn::NnHintTrainer;

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        let pinned = [0, 1, 2];
        let (_fixture_dir, model_path) = fixture_model("nn_hint_region", FixturePref::Regions(pinned))?;
        let mut probe = new_game()?;
        probe.base.turn = 2;
        probe.stage = RamenStage::RegionSelect;
        let actions = probe.list_actions()?;
        let pinned_text = (0..actions.len())
            .find(|&i| region_of(&actions, i) == Some(pinned))
            .map(|i| actions[i].to_string());

        for stages in ["train,ramen", "train,ramen,region"] {
            println!("-- mcts / stages={stages} --");
            let (ev_base, base, mut rng_base) = run_region_turn(&staged_trainer(stages)?, 20260915)?;
            println!("-- mcts_nn_hint / stages={stages} --");
            let hint_trainer = NnHintTrainer::new(staged_trainer(stages)?, load_fixture_nn(&model_path)?);
            let (ev_hint, hint, mut rng_hint) = run_region_turn(&hint_trainer, 20260915)?;

            c.check(ev_base == ev_hint, &format!("[{stages}] 事件流与 mcts 完全相同"));
            let (Some(b), Some(h)) = (base.first(), hint.first()) else {
                c.check(false, &format!("[{stages}] 两边都应 emit 一条决策"));
                continue;
            };
            c.check(b.action_index == h.action_index, &format!("[{stages}] 执行的候选相同"));
            c.check(b.candidate_scores == h.candidate_scores, &format!("[{stages}] 候选评分相同"));
            c.check(h.source_label().is_none(), &format!("[{stages}] 不带网络来源标签"));
            c.check(rng_base.next_u64() == rng_hint.next_u64(), &format!("[{stages}] 之后的随机流相同"));

            let hint_val = h.scenario_extra.as_ref().and_then(|v| v.get(NN_HINT_KEY));
            println!("nn_hint = {hint_val:?}");
            let choice = hint_val.and_then(|v| v.get("choice")).and_then(|v| v.as_str());
            let same = hint_val.and_then(|v| v.get("same_as_executed")).and_then(|v| v.as_bool());
            let executed = h.candidate_descriptions.get(h.action_index).map(String::as_str);
            c.check(choice == pinned_text.as_deref(), &format!("[{stages}] 参考推荐是网络钉住的组合"));
            c.check(same == Some(choice == executed), &format!("[{stages}] same_as_executed 与实际一致"));
            if stages.contains("region") {
                c.check(!h.candidate_scores.is_empty(), "[region] 执行侧确实走了地区搜索");
                c.check(
                    h.scenario_extra.as_ref().and_then(|v| v.get("total_luck_score")).is_some(),
                    "[region] luck 快照与参考推荐同时存在（合并而非覆盖）"
                );
            }
        }
        c.finish()
    }

    /// 用 [`umasim::bench::run_seeded`] 跑一整局（关掉逐步日志）
    ///
    /// # 错误
    ///
    /// 建局或对局中任一步报错时返回。
    #[cfg(feature = "onnx")]
    fn play_full<T: Trainer<RamenGame>>(trainer: T, run_idx: u64) -> Result<umasim::bench::GameOutcome> {
        let mut lt = umasim::trainer::LoggingTrainer::new(trainer, run_idx);
        lt.set_logging(false);
        umasim::bench::run_seeded(
            101901,
            &[303124, 303114, 303084, 303094, 303064, 303054],
            &InheritInfo {
                blue_count: [15, 0, 3, 0, 0],
                extra_count: [0, 40, 40, 20, 20, 40]
            },
            20260923,
            run_idx,
            &lt
        )
    }

    /// 整局：`mcts_nn_hint` 与 `mcts` 同种子终局逐项相同；`nn` 能完整跑完一局
    ///
    /// 覆盖地区回合之外的全部阶段（训练、吃面、SpecialSelect、比赛回合）：hint 装配
    /// 只要在任一阶段动了随机流或执行结果，终局就会分叉。`nn` 用钉住地区的 fixture，
    /// 核对整局不报错、第 1 年地区是网络钉住的那一组（钉住的是第 1 年的地区，后两年候选不含它们）。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_full_game_hint_identical_and_nn_completes() -> Result<()> {
        use crate::ramen_nn::{NnHintTrainer, WholeNnTrainer};

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        let pinned = [0, 2, 4];
        let (_fixture_dir, model_path) = fixture_model("full_game", FixturePref::Regions(pinned))?;
        for idx in [0u64, 1] {
            let mcts = play_full(staged_trainer("train,ramen,region")?, idx)?;
            let hint = play_full(
                NnHintTrainer::new(staged_trainer("train,ramen,region")?, load_fixture_nn(&model_path)?),
                idx
            )?;
            println!(
                "[{idx}] mcts {} {:?} / hint {} {:?}",
                mcts.score, mcts.yearly_selected_regions, hint.score, hint.yearly_selected_regions
            );
            c.check(mcts.score == hint.score, &format!("[{idx}] 终局分相同"));
            c.check(mcts.five_status == hint.five_status, &format!("[{idx}] 五维相同"));
            c.check(
                mcts.yearly_selected_regions == hint.yearly_selected_regions,
                &format!("[{idx}] 三年地区相同")
            );

            let nn = play_full(WholeNnTrainer::new(load_fixture_nn(&model_path)?), idx);
            println!(
                "[{idx}] nn → {:?}",
                nn.as_ref()
                    .map(|o| (o.score, o.yearly_selected_regions))
                    .map_err(|e| format!("{e:#}"))
            );
            c.check(nn.is_ok(), &format!("[{idx}] nn 整局跑完不报错"));
            if let Ok(o) = &nn {
                c.check(
                    o.yearly_selected_regions[0] == pinned,
                    &format!("[{idx}] nn 第 1 年地区是网络钉住的 {pinned:?}")
                );
            }
        }
        c.finish()
    }

    /// 链式决策（turn 1 的 Train）：`decision#1 → compute_next_step → … → compute_done`
    ///
    /// 这是上游 e5cdd64 修正的次序：中间决策必须**先于** `compute_next_step` 到达，
    /// 下游才不会以为本回合没算。
    #[test]
    fn test_chained_decision_order() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let trainer = small_trainer()?;
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260914);
        // ❗必须**自然推进**到 turn 1 的 Train：直接摆 stage 会留下空的训练分布，
        // 搜索一展开就越界。这里按 umasim 的阶段机推进，等价于客户端收到那一帧快照。
        let game = advance_to_turn1_train(new_game()?, &trainer, &mut rng)?;

        process_ramen(game, Some(2), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        let pos = |needle: &str| ev.iter().position(|e| e.starts_with(needle));
        let first_decision = pos("decision:");
        let next_step = pos("info:compute_next_step");
        let done = pos("info:compute_done");
        let decisions = ev.iter().filter(|e| e.starts_with("decision:")).count();
        c.check(decisions == 2, &format!("链式共 2 条决策（实际 {decisions}）"));
        c.check(next_step.is_some(), "发出了 compute_next_step");
        c.check(
            matches!((first_decision, next_step), (Some(a), Some(b)) if a < b),
            "决策#1 先于 compute_next_step"
        );
        c.check(
            matches!((next_step, done), (Some(b), Some(d)) if b < d),
            "compute_next_step 先于 compute_done"
        );
        c.check(
            ev.iter().filter(|e| *e == "info:compute_done").count() == 1,
            "compute_done 只发一次"
        );
        c.check(ev.last().is_some_and(|e| e == "info:compute_done"), "compute_done 是最后一条");
        c.finish()
    }

    /// `Begin` 早退回合也补 `compute_done`（上游 e5cdd64 的成对保证）
    #[test]
    fn test_begin_stage_still_emits_compute_done() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let mut game = new_game()?;
        game.stage = RamenStage::Begin;
        let trainer = small_trainer()?;
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260914);

        process_ramen(game, Some(3), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        println!("{ev:?}");
        c.check(ev == vec!["info:compute_done".to_string()], "Begin 早退只发 compute_done");
        c.finish()
    }
}
