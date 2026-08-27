//! 拉面杯 MCTS 训练员
//!
//! 用扁平蒙特卡洛搜索（[`FlatSearch<RamenGame>`]）替换手写策略的部分决策点，
//! 其余决策点仍走 [`RecommendedRamenTrainer`]。
//!
//! # 为什么不复用 `MctsTrainer`
//!
//! `MctsTrainer` 只 `impl Trainer<OnsenGame>`，且字段与温泉强耦合
//! （`HandwrittenEvaluator` 只实现了 `Evaluator<OnsenGame>`、`OnsenAction::Dig`
//! 特判）。把它泛型化要连带掀开 `umaai` 的调用签名，代价远大于另写一个薄壳。
//! 搜索核心 [`FlatSearch`] 本身已泛型化，拉面侧缺的只是最外层这一层。
//!
//! # 阶段门控
//!
//! 一局约 171 个决策点（实测单局：Train 69 / RamenSelect 61 / SpecialSelect 25 /
//! Event 15 / RegionSelect 3 / SuperRamenSelect 1），全搜代价高。[`RamenSearchStages`] 允许只搜指定阶段，
//! 未选中的阶段直接转发给手写策略。这样既能压预算，也能单独测量
//! 「只搜 Train」/「只搜 RamenSelect」各自的边际收益。
//!
//! # 事件选项不走搜索
//!
//! [`Trainer::select_choice`] / [`Trainer::select_event_choice`] 的候选不来自
//! [`Game::list_actions`]，通用 rollout 入口 `apply_action` 吃不下，一律转发手写策略。
//!
//! # 合并动作搜索（`use_combined_ramen_select`）
//!
//! 打开时 `RamenSelect` 用 `list_combined_ramen_select_actions` 一次搜
//! `(ramen, targets)`，再把最优 `ramen` 映射回三阶段候选下标；紧随其后的
//! `SpecialSelect` 直接返回缓存的 targets，不再搜索。
//!
//! 这会改变对外层 rng 的消耗：`FlatSearch::search` 每次恰好消耗一次
//! `next_u64`。三阶段路径在 RamenSelect + SpecialSelect 各搜一次（2 次），
//! 合并路径只在 RamenSelect 搜一次（1 次）。随机序列整体位移，拉面基线作废。
//! 这是预期行为，不是 bug。关闭本开关即退回改动前的三阶段分别搜。

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering}
    }
};

use anyhow::{Result, anyhow, bail};
use log::info;
use rand::prelude::StdRng;

use super::RecommendedRamenTrainer;
use crate::{
    game::{
        Game, Trainer,
        ramen::{Operation, RamenGame, RamenStage, policy::FIXED_SUPER_RAMEN_INDEX}
    },
    gamedata::{EventChoice, EventData},
    search::{ActionResult, FlatSearch, RamenSearchOutput, SearchConfig, TerminalStats}
};

/// 搜索哪些阶段的门控开关
///
/// 字段对应 [`RamenStage`] 中会产生多候选的阶段。未列出的阶段
/// （`Begin` / `BeginAfterRegionSelect` / `Distribute` / `AfterTrain` /
/// `NextTurn` / `Settlement`）不产生真正的选择空间，无需门控。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamenSearchStages {
    /// 训练/比赛选择（决策点最多，约占一局的 45%）
    pub train: bool,
    /// 吃哪碗面（约占 27%）
    pub ramen_select: bool,
    /// 隐藏风味用法
    pub special_select: bool,
    /// 年度地区选择（turn 2 / 23 / 47 的 `RegionSelect` 阶段，含第 1 年）
    pub region_select: bool,
    /// 超级拉面选择（回合 71 后、一局一次，3 个候选）
    ///
    /// 默认配置 `ramen_search_stages = "train,ramen"` 不打开本开关。
    /// [`Self::all`] 现在会真正搜这一步；历史 `all` 基线因此作废。
    pub super_ramen_select: bool
}

impl RamenSearchStages {
    /// 全部阶段都搜
    ///
    /// **历史 `all` 基线已作废**：超级拉面接入 trainer 后 `all` 会真的搜
    /// `SuperRamenSelect`；第 1 年地区抬到 `RegionSelect` 阶段边界后，`all`
    /// 也会真的搜 turn 2 的地区选择（以前卡在 `Begin` 内部搜不到）。
    pub fn all() -> Self {
        Self {
            train: true,
            ramen_select: true,
            special_select: true,
            region_select: true,
            super_ramen_select: true
        }
    }

    /// 一个阶段都不搜（等价于纯手写策略，用于对照组）
    pub fn none() -> Self {
        Self {
            train: false,
            ramen_select: false,
            special_select: false,
            region_select: false,
            super_ramen_select: false
        }
    }

    /// 只搜训练阶段
    pub fn train_only() -> Self {
        Self {
            train: true,
            ..Self::none()
        }
    }

    /// 只搜吃面阶段
    pub fn ramen_only() -> Self {
        Self {
            ramen_select: true,
            ..Self::none()
        }
    }

    /// 解析逗号分隔的阶段名（CLI 用）
    ///
    /// 可用名：`all` / `none` / `train` / `ramen` / `special` / `region` / `super`。
    /// 例：`"train,ramen"`。
    ///
    /// # 三条严格性约定
    ///
    /// 这些输入直接决定实验分组，静默接受歧义输入会让对照组悄悄退化成纯手写策略，
    /// 是最难发现的一类错，故一律 `Err`：
    ///
    /// - 未知阶段名
    /// - 空串 / 只有逗号（否则静默得到 `none`）
    /// - `all` / `none` 与其他名混用（否则 `train,none` 与 `none,train` 结果不同）
    pub fn parse(spec: &str) -> Result<Self> {
        let names: Vec<&str> = spec.split(',').map(str::trim).filter(|n| !n.is_empty()).collect();
        if names.is_empty() {
            anyhow::bail!("搜索阶段为空（要表达「不搜索」请显式写 none）");
        }
        if names.iter().any(|n| matches!(*n, "all" | "none")) {
            if names.len() > 1 {
                anyhow::bail!("all / none 必须单独使用，不能与其他阶段名混用: {spec}");
            }
            return Ok(if names[0] == "all" { Self::all() } else { Self::none() });
        }
        let mut stages = Self::none();
        for name in names {
            match name {
                "train" => stages.train = true,
                "ramen" => stages.ramen_select = true,
                "special" => stages.special_select = true,
                "region" => stages.region_select = true,
                "super" => stages.super_ramen_select = true,
                other => {
                    anyhow::bail!("未知搜索阶段: {other}（可用 all/none/train/ramen/special/region/super）")
                }
            }
        }
        Ok(stages)
    }

    /// 该阶段是否应走搜索
    ///
    /// 取引用而非按值：`RamenStage` 未实现 `Copy`（上游类型，不在本次改动范围内）。
    pub fn contains(&self, stage: &RamenStage) -> bool {
        match stage {
            RamenStage::Train => self.train,
            RamenStage::RamenSelect => self.ramen_select,
            RamenStage::SpecialSelect => self.special_select,
            RamenStage::RegionSelect => self.region_select,
            RamenStage::SuperRamenSelect => self.super_ramen_select,
            _ => false
        }
    }
}

impl Default for RamenSearchStages {
    fn default() -> Self {
        Self::all()
    }
}

/// 最优动作的取分口径
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamenSelection {
    /// 结算评分（`calc_score`）
    Score,
    /// 计入 PT 偏好的评分（`calc_score_with_pt_favor`）
    Pt
}

/// 拉面杯 MCTS 训练员
///
/// 被门控选中的阶段走 [`FlatSearch`]，其余转发给内置的
/// [`RecommendedRamenTrainer`]。搜索的 rollout 基策同样是推荐策略
/// （由 `FlatSearchGame::default_rollout_trainer` 提供），因此本训练员
/// 是「手写策略 + 搜索」的严格叠加：门控全关时行为与纯推荐策略一致。
pub struct RamenMctsTrainer {
    /// 扁平搜索器
    pub search: FlatSearch<RamenGame>,
    /// 未搜索阶段与事件选项的回退策略
    pub fallback: RecommendedRamenTrainer,
    /// 搜索哪些阶段
    pub stages: RamenSearchStages,
    /// 取分口径
    pub selection: RamenSelection,
    /// 是否输出每步决策日志
    pub verbose: bool,
    /// `RamenSelect` 是否用合并动作（ramen + targets 一次决策）搜索
    ///
    /// 打开时 `SpecialSelect` 不再是独立决策点：`RamenSelect` 的搜索结果里
    /// 已经含 targets，`SpecialSelect` 直接返回缓存值。
    /// 关闭时退回三阶段分别搜（改动前行为）。
    ///
    /// **RNG 消耗会变**：`FlatSearch::search` 对外层 rng 恰好消耗一次 `next_u64`。
    /// 打开后 SpecialSelect 零消耗，随机序列整体位移，拉面基线作废。这是预期的。
    pub use_combined_ramen_select: bool,
    /// 最近一次搜索决策的候选统计文本（供 `LoggingTrainer` 写入决策日志）
    ///
    /// 用 `Mutex` 而非 `RefCell`：`Trainer` 在搜索/并行场景要求 `Sync`。
    last_breakdown: Mutex<Option<String>>,
    /// 本训练员真正走过搜索的决策次数（转发给手写策略的不计）
    ///
    /// `Trainer::select_action` 只有 `&self`，故用原子量。用途是让「门控是否生效」
    /// 可观测：只看分数无法区分「搜索没提分」与「门控写错、根本没搜」。
    searched: AtomicUsize,
    /// `SpecialSelect` 直接命中合并搜索缓存的次数
    ///
    /// 与 [`Self::searched`] 同理，用原子量是因为 `select_action` 只有 `&self`。
    /// 用途是钉住「缓存检查必须在门控早退之前」：若它被挪到早退之后，
    /// `special_select` 门控关闭时合并搜索选出的 targets 会被**静默丢弃**、
    /// 改由手写策略另选，而分数上看不出来——本计数器归零才看得见。
    combined_cache_hits: AtomicUsize,
    /// `RamenSelect` 合并搜索选出的 targets，供紧随其后的 `SpecialSelect` 复用
    ///
    /// 用 `Mutex` 而非 `RefCell`：`Trainer` 在搜索/并行场景要求 `Sync`
    /// （与既有 `last_breakdown` 同理）。
    pending_combined_targets: Mutex<Option<[i32; 3]>>
}

impl RamenMctsTrainer {
    /// 用指定搜索配置创建（默认搜全部阶段、按 `score` 口径取最优、打开合并动作搜索）
    pub fn new(config: SearchConfig) -> Self {
        Self {
            search: FlatSearch::<RamenGame>::new(config),
            fallback: RecommendedRamenTrainer::new(),
            stages: RamenSearchStages::all(),
            selection: RamenSelection::Score,
            verbose: false,
            use_combined_ramen_select: true,
            last_breakdown: Mutex::new(None),
            searched: AtomicUsize::new(0),
            combined_cache_hits: AtomicUsize::new(0),
            pending_combined_targets: Mutex::new(None)
        }
    }

    /// 本训练员真正走过搜索的决策次数
    pub fn searched_count(&self) -> usize {
        self.searched.load(Ordering::Relaxed)
    }

    /// 设置搜索阶段门控
    pub fn with_stages(mut self, stages: RamenSearchStages) -> Self {
        self.stages = stages;
        self
    }

    /// 设置取分口径
    pub fn with_selection(mut self, selection: RamenSelection) -> Self {
        self.selection = selection;
        self
    }

    /// 设置是否输出每步决策日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// `SpecialSelect` 直接命中合并搜索缓存的次数
    pub fn combined_cache_hits(&self) -> usize {
        self.combined_cache_hits.load(Ordering::Relaxed)
    }

    /// 设置 `RamenSelect` 是否走合并动作搜索
    pub fn with_combined_ramen_select(mut self, on: bool) -> Self {
        self.use_combined_ramen_select = on;
        self
    }

    /// 获取搜索配置
    pub fn config(&self) -> &SearchConfig {
        self.search.config()
    }

    /// 缓存本次搜索的候选统计（次数 / 均分 / 标准差 / PT 均分）
    fn stash_search_breakdown(&self, output: &RamenSearchOutput) {
        let text = output
            .actions
            .iter()
            .zip(output.action_results.iter())
            .enumerate()
            .map(|(i, (action, (res, res_pt)))| {
                format!(
                    "#{i} {action} n={} mean={:.0} sd={:.0} pt={:.0}",
                    res.count(),
                    res.mean(),
                    res.stdev(),
                    res_pt.mean()
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        if let Ok(mut slot) = self.last_breakdown.lock() {
            *slot = Some(text);
        }
    }

    /// 输出终局多维记录：其余候选相对**实际选中动作**的差值
    ///
    /// 只打差值而非绝对值：各候选的绝对面板高度相似，人眼分辨不出；
    /// 「选这个动作，最终智力会多 300」才是可读的因果陈述。
    ///
    /// 锚点取 `chosen`（即 `select_action` 真正返回的下标）而非
    /// `best_action_idx`：`RamenSelection::Pt` 下两者可能不同，拿后者当锚点会
    /// 对着一个没被选中的动作报差值。
    ///
    /// 差值只在**均值**层面成立。阈值类维度（`rmj_ok_*`）本身已是每次 rollout
    /// 内部归约出的 0/1，其均值是达成率，差值即达成率之差——不要再拿它与 PT
    /// 均值互推，那正是这套观测要避免的错误。
    fn log_terminal_breakdown(&self, turn: i32, chosen: usize, output: &RamenSearchOutput) {
        if !self.verbose || output.terminal_results.len() != output.actions.len() {
            return;
        }
        let Some(base) = output.terminal_results.get(chosen) else {
            return;
        };

        // 基准按 key 建表：两次 visit 靠键名配对，而不是靠下标。
        // 宏保证同类型的遍历顺序一致，但下标对齐正是 `NamedMetricRef` 要消灭的
        // 那种耦合——增删维度时不该出现静默错位。
        let mut base_dims: HashMap<&'static str, f64> = HashMap::new();
        base.visit(&mut |m| {
            base_dims.insert(m.key, m.result.mean());
        });

        for (i, action) in output.actions.iter().enumerate() {
            if i == chosen {
                continue;
            }
            let Some(stats) = output.terminal_results.get(i) else {
                continue;
            };
            let mut parts: Vec<String> = Vec::new();
            stats.visit(&mut |m| {
                let Some(base_mean) = base_dims.get(m.key) else {
                    return;
                };
                let delta = m.result.mean() - base_mean;
                // 只报可见差异，否则每行都被 20 余维刷屏
                match m.unit {
                    "flag" if delta.abs() >= 0.02 => {
                        parts.push(format!("{}{:+.0}%", m.key, delta * 100.0));
                    }
                    "flag" => {}
                    _ if delta.abs() >= 1.0 => parts.push(format!("{}{delta:+.0}", m.key)),
                    _ => {}
                }
            });
            if !parts.is_empty() {
                info!("[回合 {}][终局差异] {action} vs 选中: {}", turn + 1, parts.join(" "));
            }
        }
    }

    /// 清空本次缓存（转发给手写策略时用，避免读到上一条搜索的陈旧文本）
    fn clear_breakdown(&self) {
        if let Ok(mut slot) = self.last_breakdown.lock() {
            *slot = None;
        }
    }

    /// 超级拉面平局回退：分数与选项二完全相同时改选选项二
    ///
    /// `deck_can_split == false`（卡组训练类型数 < 5）时
    /// [`RamenAction::distribute_super_ramen_clones`](crate::game::ramen::RamenAction)
    /// 直接早返回，三个选项对结局**完全等价**：CRN 下各候选逐位同分，
    /// `best_action_idx` 的 `max_by` 取到的是候选 0，于是打开 `super` 门控会把
    /// 手写一直固定的选项二静默换成选项一。**分数不变，变的是状态与日志**——
    /// 属于最难排查的一类差异。
    ///
    /// 因此只在**确实平局**时向手写回退对齐；一旦搜索真的分出高下，
    /// 就完全按搜索结果走，不干预。
    ///
    /// 非 `SuperRamenSelect` 阶段原样返回。
    ///
    /// 平局判定**必须与 `selection` 用同一口径**：`Score` 比 `.0.mean()`，
    /// `Pt` 比 `.1.weighted_mean(radical_factor)`。两边错位会把「Pt 口径下并非
    /// 平局」误判成平局，反而覆盖掉正确选择。
    fn break_super_ramen_tie(
        game: &RamenGame, actions: &[<RamenGame as Game>::Action], output: &RamenSearchOutput,
        selection: RamenSelection, idx: usize
    ) -> usize {
        if game.stage != RamenStage::SuperRamenSelect {
            return idx;
        }
        let Some(fallback_idx) = actions.iter().position(|a| {
            matches!(a.operation, Operation::SuperRamenSelect(i) if i == FIXED_SUPER_RAMEN_INDEX)
        }) else {
            return idx;
        };
        if fallback_idx == idx {
            return idx;
        }
        // 取不到统计就不干预
        let (Some(chosen), Some(fallback)) =
            (output.action_results.get(idx), output.action_results.get(fallback_idx))
        else {
            return idx;
        };
        let metric = |r: &(ActionResult, ActionResult)| match selection {
            RamenSelection::Score => r.0.mean(),
            RamenSelection::Pt => r.1.weighted_mean(output.radical_factor)
        };
        if metric(chosen) == metric(fallback) { fallback_idx } else { idx }
    }

    /// 取出并清空合并搜索缓存的 targets
    fn take_pending_combined_targets(&self) -> Option<[i32; 3]> {
        match self.pending_combined_targets.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take()
        }
    }

    /// 写入合并搜索缓存的 targets（`None` 表示不吃面或不缓存）
    fn store_pending_combined_targets(&self, targets: Option<[i32; 3]>) {
        let mut slot = match self.pending_combined_targets.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner()
        };
        *slot = targets;
    }
}

impl Default for RamenMctsTrainer {
    fn default() -> Self {
        Self::new(SearchConfig::default())
    }
}

impl Trainer<RamenGame> for RamenMctsTrainer {
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        // (A) SpecialSelect 命中缓存 —— 必须放在早退判断之前。
        // 候选可能只有 1 个，或 stages.special_select 关着，这两种情况都要消费缓存，
        // 否则会污染下一回合的 SpecialSelect。
        if game.stage == RamenStage::SpecialSelect {
            if let Some(t) = self.take_pending_combined_targets() {
                match actions.iter().position(|a| a.special_targets == Some(t)) {
                    Some(idx) => {
                        self.combined_cache_hits.fetch_add(1, Ordering::Relaxed);
                        self.clear_breakdown();
                        return Ok(idx);
                    }
                    None => {
                        bail!(
                            "SpecialSelect 缓存未命中: 缓存 targets={t:?}，实际候选=[{}]",
                            actions
                                .iter()
                                .map(|a| format!("{:?}", a.special_targets))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
        }

        // (C) RamenSelect 每次做决策时先无条件清一次，防止上一回合遗留
        if game.stage == RamenStage::RamenSelect {
            self.store_pending_combined_targets(None);
        }

        // 单候选无选择空间，跑搜索纯属浪费预算
        // 门控**必须**用未经纠正的 `game.stage`（第 1 年地区已是正规 `RegionSelect`）
        if actions.len() <= 1 || !self.stages.contains(&game.stage) {
            self.clear_breakdown();
            return self.fallback.select_action(game, actions, rng);
        }

        // (B) RamenSelect 走合并搜索（排除 race_turn：那边 list_actions 是比赛动作）
        if self.use_combined_ramen_select && game.stage == RamenStage::RamenSelect && !game.is_race_turn()
        {
            let combined = game.list_combined_ramen_select_actions();
            if combined.len() > 1 {
                self.searched.fetch_add(1, Ordering::Relaxed);
                let output = self.search.search(game, &combined, rng)?;
                let idx = match self.selection {
                    RamenSelection::Score => output.best_action_idx,
                    RamenSelection::Pt => output.best_action_pt_idx()
                };
                let best = combined
                    .get(idx)
                    .ok_or_else(|| anyhow!("合并搜索最优下标 {idx} 超出候选数 {}", combined.len()))?;
                // 不吃面时 next() 会直接推到 Train，不会有 SpecialSelect；留缓存会污染下一回合
                if best.ramen.is_none() {
                    self.store_pending_combined_targets(None);
                } else {
                    self.store_pending_combined_targets(best.special_targets);
                }
                self.stash_search_breakdown(&output);
                self.log_terminal_breakdown(game.turn() as i32, idx, &output);
                if self.verbose {
                    let (res, _) = &output.action_results[idx];
                    info!(
                        "[MCTS][回合 {}] 阶段 {:?} 合并 {} 候选 -> combined#{idx} {} (mean={:.0} n={})",
                        game.turn(),
                        game.stage,
                        combined.len(),
                        best,
                        res.mean(),
                        res.count()
                    );
                }
                match actions.iter().position(|a| a.ramen == best.ramen) {
                    Some(three_idx) => return Ok(three_idx),
                    None => {
                        bail!(
                            "RamenSelect 合并搜索结果在三阶段候选中找不到: best.ramen={:?}，实际候选=[{}]",
                            best.ramen,
                            actions
                                .iter()
                                .map(|a| format!("{:?}", a.ramen))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
            // combined.len() <= 1：不走合并，落回原逻辑
        }

        self.searched.fetch_add(1, Ordering::Relaxed);
        let output = self.search.search(game, actions, rng)?;
        let idx = match self.selection {
            RamenSelection::Score => output.best_action_idx,
            RamenSelection::Pt => output.best_action_pt_idx()
        };
        let idx = Self::break_super_ramen_tie(game, actions, &output, self.selection, idx);
        self.stash_search_breakdown(&output);
        self.log_terminal_breakdown(game.turn() as i32, idx, &output);
        if self.verbose {
            let (res, _) = &output.action_results[idx];
            info!(
                "[MCTS][回合 {}] 阶段 {:?} {} 候选 -> #{idx} {} (mean={:.0} n={})",
                game.turn(),
                game.stage,
                actions.len(),
                actions[idx],
                res.mean(),
                res.count()
            );
        }
        Ok(idx)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_breakdown();
        self.fallback.select_choice(game, choices, rng)
    }


    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_breakdown();
        self.fallback.select_event_choice(game, event, choices, rng)
    }

    /// 搜索决策返回候选统计；转发决策返回手写策略自己的分解
    fn last_breakdown(&self) -> Option<String> {
        match self.last_breakdown.lock().ok().and_then(|slot| slot.clone()) {
            Some(text) => Some(text),
            None => self.fallback.last_breakdown()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gamedata::{GAMECONSTANTS, init_global},
        global,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 准备一局固定种子的拉面局面
    fn setup(seed: u64) -> Result<(RamenGame, StdRng)> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let (decision_rng, rule_master) = crate::bench::seeded_rngs(seed, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        Ok((game, decision_rng))
    }

    /// 阶段门控字符串解析
    #[test]
    fn test_search_stages_parse() -> Result<()> {
        let mut c = Checks::new();
        let s = RamenSearchStages::parse("train,ramen")?;
        println!("parse(train,ramen) = {s:?}");
        c.check(
            s.train && s.ramen_select && !s.special_select && !s.region_select && !s.super_ramen_select,
            "只有 train / ramen_select 为真"
        );
        c.check(s.contains(&RamenStage::Train), "Train 命中");
        c.check(!s.contains(&RamenStage::SpecialSelect), "SpecialSelect 不命中");
        c.check(!RamenSearchStages::all().contains(&RamenStage::Begin), "Begin 永不命中");
        c.check(
            !RamenSearchStages::all().contains(&RamenStage::BeginAfterRegionSelect),
            "BeginAfterRegionSelect 永不命中"
        );

        // 四类必须报错的输入：静默接受会让实验对照组悄悄退化成纯手写策略
        c.check(RamenSearchStages::parse("train,bogus").is_err(), "未知阶段名报错");
        c.check(RamenSearchStages::parse("").is_err(), "空串报错（不静默当 none）");
        c.check(RamenSearchStages::parse(" , ").is_err(), "只有逗号报错");
        c.check(RamenSearchStages::parse("train,none").is_err(), "none 与其他名混用报错");
        c.check(RamenSearchStages::parse("all,train").is_err(), "all 与其他名混用报错");
        c.check(RamenSearchStages::parse("all")?.train, "单独 all 有效");
        c.check(!RamenSearchStages::parse("none")?.train, "单独 none 有效");
        c.finish()
    }

    /// 第 1 年地区选择已是阶段边界上的正规决策点，门控 `region` 必须进搜索
    #[test]
    fn test_year1_region_is_searched() -> Result<()> {
        use crate::{
            game::ramen::Operation,
            trainer::ramen_handwritten_trainer::ramen_effective_stage
        };

        let mut c = Checks::new();
        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        game.run_stage(&hw, &mut rng)?;
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::RegionSelect && game.turn() == 2 {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        c.check(reached, "真实推进到 turn 2 RegionSelect");
        c.check(game.turn() == 2, "回合仍为 2");

        let actions = game.list_actions()?;
        println!(
            "根: turn={} stage={:?} 候选={}",
            game.turn(),
            game.stage,
            actions.len()
        );
        c.check(actions.len() > 1, "第 1 年有多个地区候选");
        c.check(
            actions
                .iter()
                .all(|a| matches!(a.operation, Operation::RegionSelect(_))),
            "候选全是 RegionSelect（不是训练+吃面回退）"
        );

        let eff = ramen_effective_stage(&game, &actions);
        println!("ramen_effective_stage = {eff:?} raw = {:?}", game.stage);
        c.check(eff == RamenStage::RegionSelect, "有效阶段是 RegionSelect");
        c.check(game.stage == RamenStage::RegionSelect, "raw game.stage 已是 RegionSelect");

        let gate = RamenSearchStages {
            region_select: true,
            ..RamenSearchStages::none()
        };
        c.check(gate.contains(&game.stage), "门控 region 命中第 1 年");

        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(2).with_ucb(false))
            .with_stages(gate);
        let _idx = trainer.select_action(&game, &actions, &mut rng)?;
        c.check(trainer.searched_count() == 1, "第 1 年 RegionSelect 走过搜索");
        c.finish()
    }

    /// 门控全关时必须与正式推荐策略 [`RecommendedRamenTrainer`] **逐位一致**
    ///
    /// 这是实验的对照组正确性前提：若两者不一致，说明 MCTS 壳自己额外消耗了
    /// 随机流或改了决策，后续「搜索提分多少」的差值就无从归因。
    /// 2026-08-27 切换：原对照 `RamenHandwrittenTrainer`（纯 RamenPolicy，缺平衡/联动等
    /// 机制）已不再是生产路径；现在对照正式推荐策略，等同于把搜索壳的"无操作"边界钉死。
    #[test]
    fn test_stages_none_matches_recommended() -> Result<()> {
        let seed = 42;

        let (mut game_rec, mut rng_rec) = setup(seed)?;
        game_rec.run_full_game(&RecommendedRamenTrainer::new(), &mut rng_rec)?;
        let score_rec = game_rec.uma.calc_score();

        let (mut game_mcts, mut rng_mcts) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(8))
            .with_stages(RamenSearchStages::none());
        game_mcts.run_full_game(&trainer, &mut rng_mcts)?;
        let score_mcts = game_mcts.uma.calc_score();

        let mut c = Checks::new();
        println!("推荐={score_rec} / MCTS(stages=none)={score_mcts}");
        println!(
            "  五维 {:?} vs {:?}  PT {} vs {}  super_ramen {:?} vs {:?}",
            game_rec.uma.five_status,
            game_mcts.uma.five_status,
            game_rec.ramen.scenario_pt,
            game_mcts.ramen.scenario_pt,
            game_rec.ramen.super_ramen,
            game_mcts.ramen.super_ramen
        );
        c.check(score_rec == score_mcts, "门控全关 == 推荐策略");
        c.check(game_rec.uma.five_status == game_mcts.uma.five_status, "五维一致");
        c.check(game_rec.uma.skill_pt == game_mcts.uma.skill_pt, "技能点一致");
        c.check(game_rec.ramen.scenario_pt == game_mcts.ramen.scenario_pt, "剧本 PT 一致");
        c.check(game_rec.ramen.super_ramen == game_mcts.ramen.super_ramen, "super_ramen 一致");
        c.check(game_rec.ramen.super_ramen == Some(1), "门控关时仍是选项二");
        c.check(trainer.searched_count() == 0, "门控全关时一次搜索都没发生");
        c.finish()
    }

    /// 只搜训练阶段跑通整局（小预算冒烟）
    #[test]
    fn test_mcts_train_only_full_game() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::train_only());
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let score = game.uma.calc_score();
        println!(
            "MCTS(train, search_n=4) 整局: 回合={} 评分={} ({}) 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            global!(GAMECONSTANTS).get_rank_name(score)
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(score > 0, "评分为正");
        // 「末次决策有 breakdown」几乎恒真（末步必是转发、回落到手写分解），
        // 改为统计整局真正走过搜索的次数——这才是门控生效的证据
        println!("  整局走搜索的决策数={}", trainer.searched_count());
        c.check(trainer.searched_count() > 0, "确实走过搜索");
        c.check(trainer.searched_count() <= 80, "只搜 Train（约 69 个点），没有蔓延到其他阶段");
        c.finish()
    }

    /// rollout 的根动作必须走策略流，不能走通用 `apply_action`
    ///
    /// 真实对局中 `run_train` 用 `apply_action_with_strategy`（优先用局面内策略流），
    /// 而旧 `simulate_common` 直接 `apply_action(action, rng)`。本测试扫过整局所有
    /// 多候选 Train 决策点，统计两条路径跑到终局的分数有多少个点不同——
    /// 若一个都不同不了，说明该修复是空操作，需要重新评估。
    #[test]
    fn test_root_action_uses_strategy_stream() -> Result<()> {
        use rand::SeedableRng;

        use crate::search::FlatSearchGame;

        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        let seed = 12345u64;
        let (mut checked, mut differ) = (0usize, 0usize);
        let mut first_diff = None;

        while game.next() {
            if matches!(game.stage, RamenStage::Train) {
                let actions = game.list_actions()?;
                if actions.len() > 1 {
                    // 同一个动作、同一个种子，两条 apply 路径各自跑到终局
                    let mut scores = [0i32; 2];
                    for (k, score) in scores.iter_mut().enumerate() {
                        let mut g = game.fork_for_rollout(seed);
                        let mut r = StdRng::seed_from_u64(seed);
                        if k == 0 {
                            g.apply_action(&actions[0], &mut r)?;
                        } else {
                            g.apply_root_action(&actions[0], &mut r)?;
                        }
                        while g.next() {
                            g.run_stage(&hw, &mut r)?;
                        }
                        *score = g.uma.calc_score();
                    }
                    checked += 1;
                    if scores[0] != scores[1] {
                        differ += 1;
                        first_diff.get_or_insert((game.turn(), scores[0], scores[1]));
                    }
                }
            }
            game.run_stage(&hw, &mut rng)?;
        }

        println!("扫过 {checked} 个多候选 Train 决策点，其中 {differ} 个两条路径终局分数不同");
        if let Some((turn, a, b)) = first_diff {
            println!("  首个差异: 回合 {turn} 通用={a} 策略流={b}");
        }
        let mut c = Checks::new();
        c.check(differ > 0, "修复非空操作（至少一个 Train 决策点两条路径结果不同）");
        c.finish()
    }

    /// 同种子两次整局结果一致（搜索层的 CRN 种子由传入 rng 派生）
    #[test]
    fn test_mcts_reproducible() -> Result<()> {
        let seed = 7;
        let mut scores = Vec::new();
        for _ in 0..2 {
            let (mut game, mut rng) = setup(seed)?;
            let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
                .with_stages(RamenSearchStages::train_only());
            game.run_full_game(&trainer, &mut rng)?;
            scores.push(game.uma.calc_score());
        }
        let mut c = Checks::new();
        println!("两次评分: {scores:?}");
        c.check(scores[0] == scores[1], "可复现");
        c.finish()
    }

    /// 吃面 + 隐藏风味两阶段都搜（P1.2 / P1.3 对照与测量用）
    fn ramen_and_special_stages() -> RamenSearchStages {
        RamenSearchStages {
            ramen_select: true,
            special_select: true,
            ..RamenSearchStages::none()
        }
    }

    /// 按需运行：整局输出终局多维诊断，人工看可读性
    ///
    /// 不是断言测试，是**给合作伙伴看仪表长什么样**的观察壳，故 `#[ignore]`。
    /// 手动跑：
    /// `cargo test -p umasim --lib -- test_terminal_breakdown_demo --ignored --nocapture`
    #[test]
    #[ignore = "整局诊断输出演示，按需手动运行"]
    fn test_terminal_breakdown_demo() -> Result<()> {
        // 必须早于 setup：全局 logger 只初始化一次，setup 里设的是 error 级，
        // 会把诊断用的 info! 整个吞掉
        let _ = init_test_logger("info");
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        // search_n 取小值：本壳看的是输出形态，不是分数
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(16).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .verbose(true);
        game.run_full_game(&trainer, &mut rng)?;

        println!(
            "整局结束: 评分={} 五维={:?} 上限={:?} skill_pt={} 逐年PT={:?} RMJ={:?}",
            game.uma.calc_score(),
            game.uma.five_status,
            game.uma.five_status_limit,
            game.uma.skill_pt,
            game.ramen.yearly_scenario_pt,
            game.ramen.rmj_results
        );
        Ok(())
    }

    /// 硬性验收 1 的对照尺子：`use_combined_ramen_select = false` 必须与改动前逐位相同
    ///
    /// 改动前（字段尚不存在、等价于三阶段分别搜）实测：
    /// 评分=55153 五维=[2958, 1742, 2200, 866, 1112] skill_pt=7390 scenario_pt=0 searched_count=46
    #[test]
    fn test_combined_gate_off_full_game() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(false);
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let score = game.uma.calc_score();
        let searched = trainer.searched_count();
        println!(
            "gate-off 整局: 回合={} 评分={} 五维={:?} skill_pt={} scenario_pt={} searched_count={} 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            game.uma.five_status,
            game.uma.skill_pt,
            game.ramen.scenario_pt,
            searched
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        // 2026-08-25 更新：不在判定与得意率解耦 + 地区分身缺席优先，模拟数值变化，基准重抓
        // 2026-08-27 更新（两次叠加）：
        // (1) 五维上限剧本化，速度上限 2958→3337，整局数值变化；
        // (2) fallback 与 rollout 均切到 RecommendedRamenTrainer。
        //     ⚠ gate-off **不是**纯推荐策略跑局——本测试用 ramen_and_special_stages()，
        //     ramen/special 两阶段仍在搜（searched_count=66），只是不合并成单动作。
        //     纯推荐策略的对照在 test_stages_none_matches_recommended（stages=none，
        //     searched_count=0），同卡组 seed=42 的纯推荐快照见 bench.rs 的 64336。
        //     别拿这里的 62698 当 REC 基线，会误判搜索掉分幅度。
        // 上游 (2) 抓的 66705 / [3258,...] 是在 (1) 之前测的，两者叠加后已在本分支重抓。
        c.check(score == 62698, "评分与改动前逐位相同");
        c.check(
            game.uma.five_status == [3337, 1983, 2200, 1005, 1065],
            "五维与改动前逐位相同"
        );
        c.check(game.uma.skill_pt == 8441, "技能点与改动前逐位相同");
        c.check(game.ramen.scenario_pt == 0, "剧本 PT 与改动前逐位相同");
        c.check(searched == 66, "searched_count 与改动前逐位相同");
        c.finish()
    }

    /// 默认打开合并搜索；链式 setter 能关掉
    #[test]
    fn test_combined_default_on() -> Result<()> {
        // `RamenMctsTrainer::default()` 会构造 `HandwrittenEvaluator`，后者
        // `load_onsen_order().expect(..)` 依赖工作目录与全局数据；不初始化则本测试
        // 只在别的测试先跑过时才碰巧通过（顺序依赖）。
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let t = RamenMctsTrainer::default();
        println!("default use_combined_ramen_select = {}", t.use_combined_ramen_select);
        let mut c = Checks::new();
        c.check(t.use_combined_ramen_select, "new()/default 默认打开");
        let t2 = t.with_combined_ramen_select(false);
        println!(
            "after with_combined_ramen_select(false) = {}",
            t2.use_combined_ramen_select
        );
        c.check(!t2.use_combined_ramen_select, "setter 关闭");
        c.finish()
    }

    /// 只统计不干预的包装训练员：记录各阶段调用次数，以及其中真正走过搜索的次数
    struct CountingTrainer {
        /// 被包装的 MCTS 训练员
        inner: RamenMctsTrainer,
        /// `RamenSelect` 的 `select_action` 调用次数
        ramen_select_calls: AtomicUsize,
        /// `SpecialSelect` 的 `select_action` 调用次数
        special_select_calls: AtomicUsize,
        /// `RamenSelect` 中真正走过搜索的次数
        ramen_select_searches: AtomicUsize,
        /// `SpecialSelect` 中真正走过搜索的次数
        special_select_searches: AtomicUsize
    }

    impl CountingTrainer {
        /// 包装一个已构造好的 `RamenMctsTrainer`
        fn wrap(inner: RamenMctsTrainer) -> Self {
            Self {
                inner,
                ramen_select_calls: AtomicUsize::new(0),
                special_select_calls: AtomicUsize::new(0),
                ramen_select_searches: AtomicUsize::new(0),
                special_select_searches: AtomicUsize::new(0)
            }
        }
    }

    impl Trainer<RamenGame> for CountingTrainer {
        fn select_action(
            &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
        ) -> Result<usize> {
            let before = self.inner.searched_count();
            let idx = self.inner.select_action(game, actions, rng)?;
            let did_search = self.inner.searched_count() > before;
            match game.stage {
                RamenStage::RamenSelect => {
                    self.ramen_select_calls.fetch_add(1, Ordering::Relaxed);
                    if did_search {
                        self.ramen_select_searches.fetch_add(1, Ordering::Relaxed);
                    }
                }
                RamenStage::SpecialSelect => {
                    self.special_select_calls.fetch_add(1, Ordering::Relaxed);
                    if did_search {
                        self.special_select_searches.fetch_add(1, Ordering::Relaxed);
                    }
                }
                _ => {}
            }
            Ok(idx)
        }

        fn select_choice(
            &self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            self.inner.select_choice(game, choices, rng)
        }

        fn select_event_choice(
            &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            self.inner.select_event_choice(game, event, choices, rng)
        }

        fn last_breakdown(&self) -> Option<String> {
            self.inner.last_breakdown()
        }
    }

    /// 硬性验收 2：合并开启时 SpecialSelect 大多数走缓存命中，少数走搜索
    ///
    /// 2026-08-27 修订：原断言 `special_searches == 0` 在 fallback 切到 `RecommendedRamenTrainer`
    /// 后偶发失败——race_turn 时 `RamenSelect` 走非合并搜索路径（缓存写不进去），若 trainer
    /// 在该回合选了某个 ramen，下一阶段 SpecialSelect 出现时缓存 miss 必须重搜一次。这是
    /// REC 决策倾向带来的合法新行为，不是缓存检查逻辑问题。
    ///
    /// 验收口径收紧为「SpecialSelect 命中数 >> 搜索数」，保留"合并路径生效"的本意。
    #[test]
    fn test_combined_on_skips_special_search() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let inner = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(true);
        let trainer = CountingTrainer::wrap(inner);
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let score = game.uma.calc_score();
        let ramen_calls = trainer.ramen_select_calls.load(Ordering::Relaxed);
        let special_calls = trainer.special_select_calls.load(Ordering::Relaxed);
        let ramen_searches = trainer.ramen_select_searches.load(Ordering::Relaxed);
        let special_searches = trainer.special_select_searches.load(Ordering::Relaxed);
        let searched = trainer.inner.searched_count();
        println!(
            "gate-on 整局: 回合={} 评分={} searched_count={} 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            searched
        );
        println!(
            "  RamenSelect 调用={ramen_calls} 搜索={ramen_searches} / SpecialSelect 调用={special_calls} 搜索={special_searches}"
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(score > 0, "评分为正");
        c.check(ramen_calls > 0, "RamenSelect 被调用过");
        c.check(ramen_searches > 0, "RamenSelect 走过搜索");
        c.check(special_calls > 0, "SpecialSelect 被调用过（缓存命中路径）");
        // 2026-08-28 收紧：原断言 `special_calls > special_searches` 在 29 次调用里
        // 搜 28 次也绿，等于没有守门。合并路径整个失效都抓不住。
        // 改回本文件通行的逐位快照：29 次调用只有 1 次重搜（第 3 年 race_turn 选面，
        // `select_action` 的合并短路 `!game.is_race_turn()` 不成立，见本文件 495-547）。
        c.check(special_calls == 29, "SpecialSelect 调用数与改动前逐位相同");
        c.check(special_searches == 1, "SpecialSelect 重搜数与改动前逐位相同");
        // 再留一条与具体数字解耦的语义上界，防止将来重抓快照时把比例抬上去
        c.check(
            special_searches * 5 < special_calls,
            "SpecialSelect 绝大多数走缓存命中（重搜占比 < 20%）"
        );
        c.check(
            searched == ramen_searches + special_searches,
            "整局 searched_count 等于两阶段搜索合计"
        );
        c.finish()
    }

    /// 只搜 `ramen`、不搜 `special` 时，合并搜索选出的 targets 仍必须被采用
    ///
    /// 钉「缓存检查必须在门控早退之前」：挪到早退之后，targets 会被静默丢弃、
    /// 改由手写策略另选，分数上看不出来，只有 `combined_cache_hits()` 归零才暴露。
    #[test]
    fn test_combined_cache_used_when_special_gate_off() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let stages = RamenSearchStages {
            ramen_select: true,
            ..RamenSearchStages::none()
        };
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(stages)
            .with_combined_ramen_select(true);
        game.run_full_game(&trainer, &mut rng)?;
        let hits = trainer.combined_cache_hits();
        println!(
            "special 门控关: 回合={} 评分={} searched_count={} combined_cache_hits={hits}",
            game.turn(),
            game.uma.calc_score(),
            trainer.searched_count()
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(trainer.searched_count() > 0, "RamenSelect 走过合并搜索");
        c.check(hits > 0, "SpecialSelect 必须命中合并缓存（门控关也要用）");
        c.finish()
    }

    /// 门控 `super`：整局恰好搜索一次；门控关时为 0
    #[test]
    fn test_super_ramen_gate_searches_once() -> Result<()> {
        let seed = 42;

        let (mut game_on, mut rng_on) = setup(seed)?;
        let stages_on = RamenSearchStages {
            super_ramen_select: true,
            ..RamenSearchStages::none()
        };
        let trainer_on = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(stages_on);
        game_on.run_full_game(&trainer_on, &mut rng_on)?;
        let searched_on = trainer_on.searched_count();
        println!(
            "gate=super: 回合={} 评分={} searched={} super_ramen={:?}",
            game_on.turn(),
            game_on.uma.calc_score(),
            searched_on,
            game_on.ramen.super_ramen
        );

        let (mut game_off, mut rng_off) = setup(seed)?;
        let trainer_off = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::none());
        game_off.run_full_game(&trainer_off, &mut rng_off)?;
        let searched_off = trainer_off.searched_count();
        println!(
            "gate=none: 回合={} searched={} super_ramen={:?}",
            game_off.turn(),
            searched_off,
            game_off.ramen.super_ramen
        );

        let mut c = Checks::new();
        c.check(game_on.turn() == 77, "门控开跑满 77 回合");
        c.check(searched_on == 1, "门控 super 整局恰好搜索一次");
        c.check(searched_off == 0, "门控关时一次搜索都没有");
        c.check(game_off.ramen.super_ramen == Some(1), "门控关仍选选项二");
        c.finish()
    }

    /// 根节点冒烟：真实推进到 SuperRamenSelect，小 search_n 跑通 3 候选，
    /// apply_root_action 后下一阶段是 turn 72 的 Begin
    #[test]
    fn test_super_ramen_search_root_smoke() -> Result<()> {
        use crate::search::{FlatSearch, FlatSearchGame};

        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        game.run_stage(&hw, &mut rng)?;
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::SuperRamenSelect {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        let mut c = Checks::new();
        c.check(reached, "真实推进到 SuperRamenSelect");
        c.check(game.turn() == 71, "超级拉面选择发生在回合 71");

        let actions = game.list_actions()?;
        println!(
            "根: turn={} stage={:?} 候选={} {:?}",
            game.turn(),
            game.stage,
            actions.len(),
            actions.iter().map(|a| a.to_string()).collect::<Vec<_>>()
        );
        c.check(actions.len() == 3, "根上恰好 3 个候选");

        let search = FlatSearch::<RamenGame>::new(SearchConfig::default().with_search_n(4).with_ucb(false));
        let output = search.search(&game, &actions, &mut rng)?;
        println!(
            "search 最优 #{} {} 各候选 n={:?}",
            output.best_action_idx,
            actions[output.best_action_idx],
            output.action_results.iter().map(|(r, _)| r.count()).collect::<Vec<_>>()
        );
        c.check(output.action_results.len() == 3, "搜索覆盖 3 个候选");
        c.check(
            output.action_results.iter().all(|(r, _)| r.count() > 0),
            "每个候选都有样本"
        );

        let best = &actions[output.best_action_idx];
        game.apply_root_action(best, &mut rng)?;
        c.check(game.stage == RamenStage::SuperRamenSelect, "apply_root_action 不切阶段");
        c.check(game.turn() == 71, "apply_root_action 不推进回合");
        c.check(game.ramen.super_ramen.is_some(), "根动作已写入 super_ramen");

        let advanced = game.next();
        println!("next()={} turn={} stage={:?}", advanced, game.turn(), game.stage);
        c.check(advanced, "next() 能推进");
        c.check(game.turn() == 72, "下一回合是 72");
        c.check(game.stage == RamenStage::Begin, "下一阶段是 Begin");
        c.finish()
    }

    /// 门控 `region`：三年 RegionSelect 都是多候选且门控命中（测试 init 为 All）
    ///
    /// 第 3 年 `ramen_region_strategy=fixed` 时 `list_actions` 只有 1 个候选，
    /// 不会进搜索（少一次）。本测试不跑搜索，只数会触发搜索的决策点。
    #[test]
    fn test_region_gate_three_years() -> Result<()> {
        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        let gate = RamenSearchStages {
            region_select: true,
            ..RamenSearchStages::none()
        };
        let mut visits = 0usize;
        let mut searchable = 0usize;
        game.run_stage(&hw, &mut rng)?;
        while game.next() {
            if game.stage == RamenStage::RegionSelect {
                visits += 1;
                let actions = game.list_actions()?;
                let would = actions.len() > 1 && gate.contains(&game.stage);
                println!(
                    "RegionSelect turn={} 候选={} would_search={would}",
                    game.turn(),
                    actions.len()
                );
                if would {
                    searchable += 1;
                }
            }
            game.run_stage(&hw, &mut rng)?;
        }
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(visits == 3, "三年各到一次 RegionSelect");
        c.check(searchable == 3, "All 策略下三年都是多候选，门控各搜一次");
        c.finish()
    }

    /// 第 1 年根交给 FlatSearch：每个候选都能跑到终局且有样本；同根同种子两次逐位一致
    #[test]
    fn test_year1_region_search_root_smoke() -> Result<()> {
        use crate::search::{FlatSearch, FlatSearchGame};

        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        game.run_stage(&hw, &mut rng)?;
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::RegionSelect && game.turn() == 2 {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        let mut c = Checks::new();
        c.check(reached, "真实推进到 turn 2 RegionSelect");
        let actions = game.list_actions()?;
        println!(
            "根: turn={} stage={:?} 候选={}",
            game.turn(),
            game.stage,
            actions.len()
        );
        c.check(actions.len() > 1, "第 1 年多个地区候选");

        let search = FlatSearch::<RamenGame>::new(SearchConfig::default().with_search_n(2).with_ucb(false));
        let output = search.search(&game, &actions, &mut rng)?;
        println!(
            "search 最优 #{} 各候选 n={:?}",
            output.best_action_idx,
            output.action_results.iter().map(|(r, _)| r.count()).collect::<Vec<_>>()
        );
        c.check(output.action_results.len() == actions.len(), "搜索覆盖全部候选");
        c.check(
            output.action_results.iter().all(|(r, _)| r.count() > 0),
            "每个候选都有样本"
        );

        use rand::SeedableRng;
        let search2 = FlatSearch::<RamenGame>::new(SearchConfig::default().with_search_n(2).with_ucb(false));
        let mut rng_a = StdRng::seed_from_u64(99);
        let mut rng_b = StdRng::seed_from_u64(99);
        let a = search.search(&game, &actions, &mut rng_a)?;
        let b = search2.search(&game, &actions, &mut rng_b)?;
        let same = a.action_results.iter().zip(b.action_results.iter()).all(|((ra, _), (rb, _))| {
            ra.count() == rb.count() && (ra.mean() - rb.mean()).abs() < f64::EPSILON
        });
        println!(
            "同根同种子两次: best {} vs {} same={same}",
            a.best_action_idx, b.best_action_idx
        );
        c.check(same, "同根同种子两次逐位一致");
        c.check(a.best_action_idx == b.best_action_idx, "最优下标一致");

        let best = &actions[output.best_action_idx];
        game.apply_root_action(best, &mut rng)?;
        c.check(game.stage == RamenStage::RegionSelect, "apply_root_action 不切阶段");
        c.check(game.turn() == 2, "apply_root_action 不推进回合");
        let advanced = game.next();
        println!("next()={} turn={} stage={:?}", advanced, game.turn(), game.stage);
        c.check(advanced, "next() 能推进");
        c.check(game.turn() == 2, "地区选择后回合仍为 2");
        c.check(
            game.stage == RamenStage::BeginAfterRegionSelect,
            "turn 2 RegionSelect 下一阶段是 BeginAfterRegionSelect"
        );
        c.finish()
    }
}
