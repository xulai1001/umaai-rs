//! digest.json 组装与落盘（文档 §3.5 schema）
//!
//! 强类型 struct 直接 `Serialize`（schema 与文档对齐）；`inherit` 块在 M4
//! 里程碑补齐（缺失字段直接不出现在 JSON 中）。

use std::{collections::BTreeMap, fs, path::{Path, PathBuf}};

use anyhow::{Context, Result};
use serde::Serialize;
use umasim::{gamedata::UmaData, global, gamedata::ramen::RAMENDATA, game::SupportCard};

use crate::{decisions::DecisionsResult, pack::Pack, schedule::Schedule, timeline::TimelineResult};
use umaai::protocol::{FinalScorePayload, GameStatusBase};
use umasim::gamedata::GAMEDATA;

/// digest 顶层
#[derive(Debug, Serialize)]
pub struct Digest {
    pub meta: Meta,
    pub timeline: Vec<crate::timeline::TimelineRow>,
    pub decisions: Vec<crate::decisions::DecRow>,
    pub execution: Vec<crate::execution::ExecRow>,
    pub luck: crate::decisions::LuckBlock,
    pub schedule: Schedule,
    /// 继承质量（§7；配置不可用 / 回合缺快照时缺席）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inherit: Option<crate::inherit::InheritBlock>,
    /// 分身彩圈观测（§6.5；gamedata 缺失时缺席）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clones: Option<crate::clones::ClonesBlock>,
    /// 训练画像（逐年训练次数 × 五维净增 × 运气归属；增长顺序分析用）
    pub training: crate::profile::TrainingProfile,
    pub coverage: crate::decisions::Coverage,
    pub findings: Vec<crate::execution::Finding>,
    pub context: DigestContext,
}

/// 局元信息（§3.5 meta；meta.json 缺失时从文件名与 CSV 降级推导）
#[derive(Debug, Serialize)]
pub struct Meta {
    pub game: u64,
    pub uma_id: u32,
    pub uma_name: String,
    pub deck: Vec<DeckCard>,
    pub start_turn: u32,
    pub mid_entry: bool,
    pub end_reason: String,
    pub snapshots: u64,
    pub decision_rows: u64,
    pub total_luck_end: Option<f64>,
    pub final_score: Option<i32>,
    pub rank: Option<String>,
    /// 终局评分数据来源：`final_frame`（真机终局帧）/ `last_snapshot`（末快照估算，
    /// 缺结局事件 ≈ -2700）/ `unavailable`
    pub final_source: String,
    /// 末快照携带的切者（能人，状态 7）标志：终局评分 PT 项 ×1.1 已计入
    pub is_qiezhe: bool,
    /// 末快照携带的小切（勤勉好学，状态 40）标志：PT 项 ×1.04 已计入
    pub is_xiao_qie: bool,
}

/// 卡组条目（card_id = 协议 idrank = cardId×10 + 突破等级）
#[derive(Debug, Serialize)]
pub struct DeckCard {
    pub card_id: u32,
    pub name: String,
    /// 0速 1耐 2力 3根 4智 5友人 6团队
    pub card_type: i32,
    /// 突破等级（idrank % 10）
    pub limit_break: u32,
}

/// 归因环境（§3.5 context：换环境也能归因）
#[derive(Debug, Serialize)]
pub struct DigestContext {
    /// 地区 id → 名称（RAMENDATA.ramen_region_effect）
    pub region_names: BTreeMap<String, String>,
    /// 运气分口径说明
    pub luck_formula: String,
    /// 判据摘要 + 降级注记
    pub criteria: Vec<String>,
}

/// digest 组装入参
pub struct Inputs<'a> {
    pub pack: &'a Pack,
    pub timeline: &'a TimelineResult,
    pub decisions: &'a DecisionsResult,
    /// 实际执行推断（§11 步骤 4）
    pub execution: crate::execution::ExecutionResult,
    pub schedule: Schedule,
    /// gamedata 是否可用（names / score / region 表的降级开关）
    pub gamedata_ok: bool,
    /// 伪波动标记（§6.7；`checks::flagged_turns` 产出）
    pub flags: Vec<crate::decisions::FlaggedTurn>,
    /// 继承质量块（§7）
    pub inherit: Option<crate::inherit::InheritBlock>,
    /// 分身彩圈观测块（§6.5）
    pub clones: Option<crate::clones::ClonesBlock>,
    /// 训练画像（逐年训练次数 × 五维净增 × 运气归属；2026-10-07 用户拍板）
    pub training: crate::profile::TrainingProfile,
    /// 检查项 findings（§6.1 / §6.6）
    pub extra_findings: Vec<crate::execution::Finding>,
    /// 自带 gamedata 的版本注记（`Some` = 用的是 skill 携带的旧版数据）
    pub gamedata_bundled: Option<String>,
}

/// 组装 digest（meta 降级推导 + deck / uma_name / final_score / context）
pub fn build(inputs: &Inputs) -> Digest {
    let pack = inputs.pack;
    let tl = inputs.timeline;
    let dec = inputs.decisions;

    // —— meta（meta.json 缺失时降级推导）——
    let meta_game = pack.meta.as_ref().map(|m| m.game).unwrap_or(pack.game);
    let uma_id = pack
        .meta
        .as_ref()
        .map(|m| m.uma_id)
        .or_else(|| tl.first_status.as_ref().map(|s| s.base_game.uma_id))
        .unwrap_or(0);
    let start_turn = pack
        .meta
        .as_ref()
        .map(|m| m.start_turn)
        .or_else(|| tl.rows.first().map(|r| r.turn))
        .unwrap_or(0);
    let end_reason = pack
        .meta
        .as_ref()
        .map(|m| m.end_reason.clone())
        .unwrap_or_else(|| "unknown(meta缺失)".to_string());
    let total_luck_end = pack
        .meta
        .as_ref()
        .and_then(|m| m.total_luck_end)
        .or_else(|| dec.luck.series.last().map(|p| p.total_luck));

    // deck（首份快照的 cardId = idrank；gamedata 缺失 → 纯 ID）
    let deck = deck_of(tl.first_status.as_ref().map(|s| &s.base_game), inputs.gamedata_ok);

    // uma_name（gamedata 缺失 → 纯 ID 展示，§9.2 第 7 条降级）
    let uma_name = if inputs.gamedata_ok {
        uma_data(uma_id)
            .map(|d| d.short_name().to_string())
            .unwrap_or_else(|| format!("unknown({uma_id})"))
    } else {
        format!("unknown({uma_id})")
    };

    // 终局评分 + 等级
    //
    // 优先用**真机终局帧**（`game{id}_final.json`，育成结束·点技能前，含全部结局
    // 事件）——它是本局终局数据的唯一真机来源；缺失时回落到末份快照估算（缺结局
    // 事件，约 -2700，口径见 context.criteria）。gamedata 缺失 → None + 注记。
    //
    // 切者（能人，状态 7）×1.1 / 小切（勤勉好学，状态 40）×1.04 的 PT 项加成
    // （2026-10-01 起 `calc_score` 生效）：终局帧不带该标志，从末份快照的
    // `isQieZhe` / `isXiaoQie` 代入（快照缺失时按无状态计）。
    let final_frame = pack
        .final_raw
        .as_deref()
        .and_then(|raw| serde_json::from_str::<FinalScorePayload>(raw).ok());
    if pack.final_raw.is_some() && final_frame.is_none() {
        println!("终局帧解析失败（{} 字节），评分回落到末快照", pack.final_raw.as_deref().map(str::len).unwrap_or(0));
    }
    // 字段在场但全 0（插件字段漂移）→ 同样不可用：放行会让五维分整体清零
    let final_frame = final_frame.filter(|f| {
        let ok = f.is_usable();
        if !ok {
            println!("终局帧五维/上限为空，判为不可用，评分回落到末快照");
        }
        ok
    });
    let (is_qiezhe, is_xiao_qie) = tl
        .last_status
        .as_ref()
        .map_or((false, false), |s| (s.base_game.is_qiezhe, s.base_game.is_xiao_qie));
    let (final_score, rank, final_source) = match (&final_frame, inputs.gamedata_ok) {
        (Some(frame), true) => {
            let score = crate::score::final_score_from_frame(frame, is_qiezhe, is_xiao_qie);
            (Some(score), Some(crate::score::rank_name(score)), "final_frame")
        }
        (None, true) => match tl.last_status.as_ref().map(|s| &s.base_game) {
            Some(base) => {
                let score = crate::score::final_score(base);
                (Some(score), Some(crate::score::rank_name(score)), "last_snapshot")
            }
            None => (None, None, "unavailable")
        },
        _ => (None, None, "unavailable")
    };

    let meta = Meta {
        game: meta_game,
        uma_id,
        uma_name,
        deck,
        start_turn,
        mid_entry: pack.meta.as_ref().map(|m| m.mid_entry).unwrap_or(false),
        end_reason,
        snapshots: pack
            .meta
            .as_ref()
            .map(|m| m.snapshots)
            .unwrap_or((pack.snaps.len() + pack.unparsed.len()) as u64),
        decision_rows: dec.rows.len() as u64,
        total_luck_end,
        final_score,
        rank,
        final_source: final_source.to_string(),
        is_qiezhe,
        is_xiao_qie,
    };

    // —— context ——
    let mut region_names = BTreeMap::new();
    if inputs.gamedata_ok {
        for re in &global!(RAMENDATA).ramen_region_effect {
            region_names.insert(re.id.to_string(), re.name.clone());
        }
    }
    let mut criteria = vec![
        "YEAR_BOUNDARIES=[24,48,72]（剧本年份边界，代码常量）".to_string(),
        "INHERIT_TURNS=[30,54]（两次继承回合，代码常量）".to_string(),
        "SUPER_RAMEN_TURNS=turn>=72（超级拉面期；实测分身自 72 起）".to_string(),
        "运气分读法：① 残余已知偏差——RMJ 结算结果按『每年成功』假设补齐（真机若\
         实际失败，该年起期望整体偏高）；② 位置标记（检测保留）——年界 / 继承 /\
         RMJ / 开局 2-3 回合第 1 年地区选择：跨年假跳 2026-10 修复后已基本消除，\
         这些位置如今的跳变基本是真实结算 / 继承落地 / 选择本身带来的期望变化，\
         标记用于识别性质，归因时说明性质、不当连续损益读，仅再出现一增一减配对\
         式大幅账面波动时才降级；③ 获得切者（能人）的回合 T(n) 有 PT 项 ×1.1 量级\
         的期望跳升（2026-10-01 起计入评分，timeline.is_qiezhe 由 false→true 的回合\
         即获得回合）——属真实好运，不当程序性波动读；④「这局运气差」判读阈值\
         < -2000（≈方差量级，保守线）".to_string(),
        format!(
            "final_score 口径 = AI 端按**实际终局五维与技能点**的估算（Uma::calc_score：\
             五维查表 + PT 折算 + Hint 折算 + 切者/小切 PT 项加成），**略高于小黑板与\
             实际分数**（PT 折算高于实际买技能的得分）；数据来源 = {}。final_frame：\
             育成结束·点技能前的真机终局帧（含全部结局事件与末回合比赛奖励，即 AI 评估\
             轴的真机终局状态；帧内无切者/小切标志，按末快照 isQieZhe/isXiaoQie 代入）；\
             last_snapshot：末份快照估算，**缺结局事件**（育成结束 401407 / 通用 5011 /\
             友人结束 ≈ -2700），相对再偏低。两种来源都不含已学技能分（skillScore 恒 0）",
            match final_source {
                "final_frame" => "final_frame（真机终局帧）",
                "last_snapshot" => "last_snapshot（末快照估算，偏低）",
                _ => "unavailable（gamedata 缺失）"
            }
        ),
        "flagged_turns = 位置标记（年界前2至后1回合 / 继承回合 / RMJ 结算 / 开局\
         2-3 回合第 1 年地区选择；检测保留——跨年假跳已基本消除，按性质归因）；\
         turn 72 双属性：既标记为年界、也算进超级拉面期统计（该回合份量实打实，\
         正跳不是纯程序性回吐）"
            .to_string(),
        "检查项覆盖：已验证判据（目标赛未跑赢 / 关键资源过早耗尽 / 心情掉落未恢复）直接产\
         findings；训练失败出候选清单（info，人工复核）；其余判据（体力健康 / 吃面节奏 / 友人\
         完成度 / free_race / 状态健康 / 属性溢出）待实测，由 SKILL 层用 digest 数据判读"
            .to_string(),
        "彩圈判定 = 分身新增落位 == cardType（本体占的得意位不算）；有效增加彩圈来源二分：\
         luck=随机有效增加（吃面前本体在场 → 好运气）/ strategy=规则有效增加（吃面前本体\
         缺席，地区/超拉规则带入得意位 → 好策略）；A 类地区分身（turn<72）评判判据待定\
         义，clones 块为观测数据（game6234 实测分身真实落得意位仅 2/31，分身主要价值是\
         加人头）；B 类只统计训练卡，没吃到彩圈不算亏，只能用该期运气分判盈亏".to_string(),
        "运气极值归因（top_gain/top_loss 的叙事分类）由 SKILL 层结合 timeline + flagged_turns + \
         decisions 完成——注意 turn_delta 是「局面期望终局分」变化，不等于本回合属性增量"
            .to_string()
    ];
    // —— 录制年代注记（按局包内容自动判定；旧口径 bug 只影响旧局包）——
    // ① keyEvents：2026-10-01 起协议导入把小黑板 keyEvents 归一到模拟事件历史，
    //    支援卡连续事件进度未统计的系统性压低自此修复——旧局包仍带该偏差。
    //    判定用「任一快照非空」（tl.key_events_seen）：末快照可能是收尾空帧
    //    （game6263 实测末帧空而前 142 份非空），turn 0 也天然为空。
    if !tl.key_events_seen {
        criteria.push(
            "本局快照未携带 keyEvents（2026-10-01 前录制 / 旧插件 / 中途接入）：支援卡连续\
             事件进度未对齐，整条运气曲线系统性略偏低（该偏差 2026-10-01 已修复，仅影响\
             旧局包），total_luck_end 按偏低读"
                .to_string()
        );
    }
    // ② turn 2 假跳：2026-10-03 修复链式推进漏加 NPC（选区前期望偏低约 1400 →
    //    turn 2 出现 +1300 量级假跳）
    let turn_delta_sum = |turn: u32| {
        dec.rows
            .iter()
            .filter(|r| r.turn == turn)
            .filter_map(|r| r.turn_delta)
            .sum::<f64>()
    };
    let t2 = turn_delta_sum(2);
    if t2 >= 1000.0 {
        criteria.push(format!(
            "turn 2 回合合计 Δ = {t2:+.0}，疑似 2026-10-03 前录制：开局链式推进漏加 NPC 使\
             「选区前」期望偏低约 1400，turn 2 出现 +1300 量级假跳（已修复，仅影响旧局包），\
             该跳变按程序性波动降级"
        ));
    }
    // ③ turn 72 假跳：2026-10-07 修复 URA 段比赛收益低估（turn 72 一次性赛后加成
    //    race_bonus +100 在重放路径丢失 → 71→72 有 −1000 量级程序性假跳，修复后
    //    收敛至 −388 量级）
    let t72 = turn_delta_sum(72);
    if t72 <= -800.0 {
        criteria.push(format!(
            "turn 72 回合合计 Δ = {t72:+.0}，疑似 2026-10-07 前录制：URA 段一次性赛后加成\
             在重放路径丢失，turn 71→72 有 −1000 量级程序性假跳（已修复，仅影响旧局包，\
             修复后收敛至 −388 量级），该跳变按程序性波动降级"
        ));
    }
    if !inputs.gamedata_ok {
        criteria.push("gamedata 缺失：uma/卡名、地区名、赛程、终局评分与等级均已降级（纯 ID）".to_string());
    }
    if let Some(v) = &inputs.gamedata_bundled {
        criteria.push(format!(
            "gamedata 为 skill 自带旧版（{v}）：卡名 / 赛程 / 地区名可能与当前游戏版本不一致，\
             结论按旧版口径读"
        ));
    }
    if pack.meta.is_none() {
        criteria.push("meta.json 缺失：局元信息从文件名与 CSV 降级推导".to_string());
    }
    if pack.decisions_csv.is_none() {
        criteria.push("decisions.csv 缺失：决策明细 / 运气分 / coverage 不可用".to_string());
    }
    if !tl.parse_errors.is_empty() {
        criteria.push(format!("快照解析失败 {} 份（coverage.parse_error）", tl.parse_errors.len()));
    }

    // coverage（unparsed / parse_error 来自 pack 与 timeline）
    let mut coverage = dec.coverage.clone();
    coverage.unparsed = pack.unparsed.len() as u64;
    coverage.parse_error = tl.parse_errors.len() as u64;

    let context = DigestContext {
        region_names,
        luck_formula: "total_luck = T(n) − T(1)（T = 期望终局分；t_n_display = raw + (78 − turn) × \
                       mcts_turn_bonus，t_n_raw 为反推 raw）；turn_delta = T(n+1) − T(n) 是「局面期望\
                       终局分」变化，不等于本回合属性增量；同回合多段 Δ 以回合合计为基本观察单位"
            .to_string(),
        criteria,
    };

    // luck 块：填充伪波动标记（decisions 层无 timeline 信息，由 checks 产出）
    let mut luck = dec.luck.clone();
    luck.flagged_turns = inputs.flags.clone();

    // findings：execution 偏离 + 检查项命中
    let mut findings = inputs.execution.findings.clone();
    findings.extend(inputs.extra_findings.clone());

    Digest {
        meta,
        timeline: tl.rows.clone(),
        decisions: dec.rows.clone(),
        execution: inputs.execution.rows.clone(),
        luck,
        schedule: inputs.schedule.clone(),
        inherit: inputs.inherit.clone(),
        clones: inputs.clones.clone(),
        training: inputs.training.clone(),
        coverage,
        findings,
        context,
    }
}

/// deck 组装（idrank → SupportCard；gamedata 缺失 → 纯 ID 展示）
fn deck_of(first_base: Option<&GameStatusBase>, gamedata_ok: bool) -> Vec<DeckCard> {
    let Some(base) = first_base else {
        return Vec::new();
    };
    base.card_id
        .iter()
        .map(|&idrank| {
            if gamedata_ok {
                match SupportCard::new(idrank) {
                    Ok(card) => DeckCard {
                        card_id: idrank,
                        name: card.data.card_name.clone(),
                        card_type: card.card_type,
                        limit_break: card.rank,
                    },
                    Err(_) => DeckCard {
                        card_id: idrank,
                        name: format!("unknown({idrank})"),
                        card_type: -1,
                        limit_break: idrank % 10,
                    },
                }
            } else {
                DeckCard {
                    card_id: idrank,
                    name: format!("unknown({idrank})"),
                    card_type: -1,
                    limit_break: idrank % 10,
                }
            }
        })
        .collect()
}

/// 取马娘数据（gamedata 可用但 id 查不到时返回 None）
fn uma_data(uma_id: u32) -> Option<&'static UmaData> {
    GAMEDATA.get().and_then(|g| g.get_uma(uma_id).ok())
}

/// digest 落盘（紧凑 JSON，不换行缩进）→ 返回文件路径
pub fn write_json(digest: &Digest, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("创建输出目录失败: {}", out_dir.display()))?;
    let path = out_dir.join("digest.json");
    let f = fs::File::create(&path).with_context(|| format!("创建 digest.json 失败: {}", path.display()))?;
    // 紧凑序列化（不换行不缩进）：timeline/decisions 行数多，省空间优先于可读性
    serde_json::to_writer(f, digest).with_context(|| "序列化 digest 失败")?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decisions, timeline};

    /// 最小可用 digest 组装（不依赖 gamedata，走降级路径）+ 落盘回读
    #[test]
    fn test_build_and_write_digest() -> Result<()> {
        let json = r#"{
            "baseGame": {
                "scenarioId": 14, "umaId": 112402, "umaStar": 5, "turn": 3,
                "vital": 80, "maxVital": 100, "motivation": 4,
                "fiveStatus": [100, 200, 300, 400, 500],
                "fiveStatusLimit": [1200, 1200, 1200, 1200, 1200],
                "skillPt": 10, "skillScore": 0, "totalHints": 5,
                "trainLevelCount": [1, 2, 3, 4, 5],
                "ptScoreRate": 2.0, "failureRateBias": 0,
                "isIll": false, "isQieZhe": false, "isAiJiao": false,
                "isXiaoQie": false, "PositiveThinkingCount": 0, "isRefreshMind": false, "LuckyCount": 0,
                "zhongMaBlueCount": [0, 0, 0, 0, 0], "isRacing": false,
                "cardId": [302424, 302894],
                "persons": [], "personDistribution": [[], [], [], [], []],
                "lockedTrainingId": -1,
                "friendship_noncard_yayoi": 0, "friendship_noncard_reporter": 0,
                "friend_stage": 0, "friend_outgoingUsed": 0,
                "playing_state": 1, "raceHistory": [], "story": null,
                "source": "command"
            },
            "ramen": {}
        }"#;
        let snaps = vec![crate::pack::SnapEntry {
            file: "g1_turn3.json".to_string(),
            game: 1,
            turn: 3,
            seq: 0,
            bytes: json.as_bytes().to_vec(),
        }];
        let tl = timeline::build(&snaps);
        let dec = decisions::parse(None).unwrap();
        let exec = crate::execution::build(&tl.rows, &dec.rows, &[]);
        let inputs = Inputs {
            pack: &Pack {
                game: 1,
                snaps: snaps.clone(),
                unparsed: vec![],
                decisions_csv: None,
                meta: None,
                luck_trend_svg: None,
                final_raw: None,
                ignored: vec![],
            },
            timeline: &tl,
            decisions: &dec,
            execution: exec,
            schedule: Schedule::default(),
            gamedata_ok: false,
            flags: vec![],
            inherit: None,
            clones: None,
            training: Default::default(),
            extra_findings: vec![],
            gamedata_bundled: None,
        };
        let digest = build(&inputs);
        let out = std::env::temp_dir().join(format!("digest_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&out);
        let path = write_json(&digest, &out).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        println!("digest.json 回读:\n{}", serde_json::to_string_pretty(&v["meta"]).unwrap());
        println!("context.criteria: {:#?}", v["context"]["criteria"]);
        assert_eq!(v["meta"]["game"], serde_json::json!(1));
        assert_eq!(v["meta"]["uma_id"], serde_json::json!(112402));
        assert_eq!(v["meta"]["uma_name"], serde_json::json!("unknown(112402)"));
        assert_eq!(v["meta"]["deck"].as_array().unwrap().len(), 2, "deck 来自首快照 cardId");
        assert_eq!(v["meta"]["final_score"], serde_json::Value::Null, "gamedata 缺失 → 评分降级");
        assert_eq!(v["timeline"].as_array().unwrap().len(), 1);
        assert!(
            v.get("inherit").is_none(),
            "inherit 块 M4 才实现，应缺席"
        );
        assert_eq!(
            v["execution"].as_array().unwrap().len(),
            0,
            "无决策行 → execution 空数组"
        );
        assert_eq!(v["findings"].as_array().unwrap().len(), 0);
        let _ = fs::remove_dir_all(&out);
        Ok(())
    }

    /// 录制年代注记（按局包内容自动判定）：旧局包出注记、新局包不出
    #[test]
    fn test_recording_vintage_notes() {
        // 快照：A 局不带 keyEvents（旧录制）；B 局带 keyEvents（2026-10-01 后）
        let snap_json = |turn: u32, key_events: &str| {
            format!(
                r#"{{
            "baseGame": {{
                "scenarioId": 14, "umaId": 112402, "umaStar": 5, "turn": {turn},
                "vital": 80, "maxVital": 100, "motivation": 4,
                "fiveStatus": [100, 200, 300, 400, 500],
                "fiveStatusLimit": [1200, 1200, 1200, 1200, 1200],
                "skillPt": 10, "skillScore": 0, "totalHints": 5,
                "trainLevelCount": [1, 2, 3, 4, 5],
                "ptScoreRate": 2.0, "failureRateBias": 0,
                "isIll": false, "isQieZhe": false, "isAiJiao": false,
                "isXiaoQie": false, "PositiveThinkingCount": 0, "isRefreshMind": false, "LuckyCount": 0,
                "zhongMaBlueCount": [0, 0, 0, 0, 0], "isRacing": false,
                "cardId": [302424],
                "persons": [], "personDistribution": [[], [], [], [], []],
                "lockedTrainingId": -1,
                "friendship_noncard_yayoi": 0, "friendship_noncard_reporter": 0,
                "friend_stage": 0, "friend_outgoingUsed": 0,
                "playing_state": 1, "raceHistory": [], "story": null,
                "keyEvents": {key_events},
                "source": "command"
            }},
            "ramen": {{}}
        }}"#
            )
        };
        let row = |turn: u32, delta: Option<f64>| crate::decisions::DecRow {
            file: format!("f{turn}.json"),
            turn,
            seq: 0,
            stage: "Train".to_string(),
            decision_kind: "train".to_string(),
            candidates: vec![],
            chosen: Default::default(),
            t_n_raw: None,
            t_n_display: None,
            total_luck: None,
            turn_delta: delta,
            chain_len: 1,
            outcome: "calc".to_string(),
            reason: String::new(),
            step: 0,
        };
        let build_digest = |snaps_spec: Vec<(u32, &str)>, rows: Vec<crate::decisions::DecRow>| {
            let snaps: Vec<crate::pack::SnapEntry> = snaps_spec
                .iter()
                .map(|&(turn, key_events)| crate::pack::SnapEntry {
                    file: format!("game1_turn{turn}.json"),
                    game: 1,
                    turn,
                    seq: 0,
                    bytes: snap_json(turn, key_events).into_bytes(),
                })
                .collect();
            let tl = timeline::build(&snaps);
            let dec = decisions::DecisionsResult { rows, ..Default::default() };
            let exec = crate::execution::build(&tl.rows, &dec.rows, &[]);
            build(&Inputs {
                pack: &Pack {
                    game: 1,
                    snaps,
                    unparsed: vec![],
                    decisions_csv: None,
                    meta: None,
                    luck_trend_svg: None,
                    final_raw: None,
                    ignored: vec![],
                },
                timeline: &tl,
                decisions: &dec,
                execution: exec,
                schedule: Schedule::default(),
                gamedata_ok: false,
                flags: vec![],
                inherit: None,
                clones: None,
                training: Default::default(),
                extra_findings: vec![],
                gamedata_bundled: None,
            })
        };

        // A：旧局包——无 keyEvents + turn2 假跳 + turn72 假跳 → 3 条注记齐出
        let a = build_digest(vec![(77, "[]")], vec![row(2, Some(1500.0)), row(72, Some(-1083.0))]);
        println!("旧局包 criteria:\n{}", a.context.criteria.join("\n"));
        let has_a = |kw: &str| a.context.criteria.iter().any(|c| c.contains(kw));
        assert!(has_a("keyEvents"), "无 keyEvents → 连续事件偏差注记");
        assert!(has_a("turn 2 回合合计"), "turn2 Δ≥+1000 → 假跳注记");
        assert!(has_a("turn 72 回合合计"), "turn72 Δ≤-800 → 假跳注记");

        // B：新局包——keyEvents 在场、跳变幅度正常 → 3 条注记都不出
        let b = build_digest(vec![(77, "[830297001]")], vec![row(2, Some(300.0)), row(72, Some(-388.0))]);
        println!("新局包 criteria 数 = {}", b.context.criteria.len());
        let has_b = |kw: &str| b.context.criteria.iter().any(|c| c.contains(kw));
        assert!(!has_b("keyEvents"), "keyEvents 在场 → 不出连续事件偏差注记");
        assert!(!has_b("turn 2 回合合计") && !has_b("turn 72 回合合计"), "无大跳 → 不出假跳注记");

        // C：末快照 keyEvents 为空、前份非空（game6263 实测形态：末帧收尾空帧）
        //    → 判定用「任一非空」，不出注记
        let c = build_digest(
            vec![(5, "[830297001]"), (77, "[]")],
            vec![row(2, Some(300.0))]
        );
        let has_c = |kw: &str| c.context.criteria.iter().any(|c| c.contains(kw));
        println!("末帧空 keyEvents 局 criteria 数 = {}", c.context.criteria.len());
        assert!(!has_c("keyEvents"), "任一快照有 keyEvents → 不出注记（末帧空是收尾特例）");
    }

    /// 终局评分来源分流：真机终局帧优先（含结局事件），缺失回落末快照
    ///
    /// 需要真实 gamedata（评分查表 + uma 名），与项目其它测试同法：cwd 切
    /// workspace 根 + init_global。
    #[test]
    fn test_final_score_source_priority() -> Result<()> {
        let root = umasim::utils::get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        umasim::gamedata::init_global()?;

        // 末快照 = 末回合比赛前（game6243 实测值，缺全部结局事件）
        let last_snap = r#"{
            "baseGame": {
                "scenarioId": 14, "umaId": 109701, "umaStar": 5, "turn": 77,
                "vital": 80, "maxVital": 100, "motivation": 4,
                "fiveStatus": [3226, 2162, 1678, 1089, 2338],
                "fiveStatusLimit": [3242, 2444, 2206, 2200, 2506],
                "skillPt": 7717, "skillScore": 0, "totalHints": 0,
                "trainLevelCount": [1, 1, 1, 1, 1],
                "ptScoreRate": 2.0, "failureRateBias": 0,
                "isIll": false, "isQieZhe": false, "isAiJiao": false,
                "isXiaoQie": false, "PositiveThinkingCount": 0, "isRefreshMind": false, "LuckyCount": 0,
                "zhongMaBlueCount": [0, 0, 0, 0, 0], "isRacing": false,
                "cardId": [302424], "persons": [], "personDistribution": [[], [], [], [], []],
                "lockedTrainingId": -1,
                "friendship_noncard_yayoi": 0, "friendship_noncard_reporter": 0,
                "friend_stage": 0, "friend_outgoingUsed": 0,
                "playing_state": 1, "raceHistory": [], "story": null,
                "source": "command"
            },
            "ramen": {}
        }"#;
        // 终局帧 = 育成结束·点技能前（结局事件已结算：五维 +45、SP +270）
        let final_frame = r#"{
            "scenarioId": 14, "single_mode_chara_id": 6243, "umaId": 109701,
            "turn": 77, "state": 2,
            "fiveStatus": [3242, 2222, 1723, 1134, 2398],
            "fiveStatusLimit": [3242, 2444, 2206, 2200, 2506],
            "skillPt": 7987, "inheritGains": [11, 22]
        }"#;
        let snaps = vec![crate::pack::SnapEntry {
            file: "game6243_turn77_2.json".to_string(),
            game: 6243,
            turn: 77,
            seq: 2,
            bytes: last_snap.as_bytes().to_vec(),
        }];
        let tl = timeline::build(&snaps);
        let dec = decisions::parse(None)?;
        let exec = crate::execution::build(&tl.rows, &dec.rows, &[]);
        let build_with = |final_raw: Option<&str>| {
            build(&Inputs {
                pack: &Pack {
                    game: 6243,
                    snaps: snaps.clone(),
                    unparsed: vec![],
                    decisions_csv: None,
                    meta: None,
                    luck_trend_svg: None,
                    final_raw: final_raw.map(str::to_string),
                    ignored: vec![],
                },
                timeline: &tl,
                decisions: &dec,
                execution: exec.clone(),
                schedule: Schedule::default(),
                gamedata_ok: true,
                flags: vec![],
                inherit: None,
                clones: None,
                training: Default::default(),
                extra_findings: vec![],
                gamedata_bundled: None,
            })
        };

        let with_frame = build_with(Some(final_frame));
        let without = build_with(None);
        println!(
            "真机终局帧评分={:?}（来源 {}）/ 末快照评分={:?}（来源 {}）",
            with_frame.meta.final_score, with_frame.meta.final_source,
            without.meta.final_score, without.meta.final_source
        );
        assert_eq!(with_frame.meta.final_source, "final_frame");
        assert_eq!(without.meta.final_source, "last_snapshot");
        let a = with_frame.meta.final_score.ok_or_else(|| anyhow::anyhow!("终局帧评分应可用"))?;
        let b = without.meta.final_score.ok_or_else(|| anyhow::anyhow!("末快照评分应可用"))?;
        println!("缺口 Δ={}", a - b);
        assert!(a > b, "真机终局帧评分应高于缺结局事件的末快照评分");
        assert!(
            (1500..3500).contains(&(a - b)),
            "缺口量级应为结局事件贡献（实测 ≈2700），实际 {}",
            a - b
        );
        // criteria 应注明数据来源
        let note = with_frame
            .context
            .criteria
            .iter()
            .find(|c| c.contains("final_score 口径"))
            .cloned()
            .unwrap_or_default();
        println!("口径注记: {note}");
        assert!(note.contains("final_frame"), "口径注记应标明真机终局帧来源");
        Ok(())
    }
}
