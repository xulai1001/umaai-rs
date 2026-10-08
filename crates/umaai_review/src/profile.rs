//! 训练画像（2026-10-07 用户拍板）：逐年「训练次数 × 五维净增长 × 运气归属」
//!
//! 供报告「属性成长曲线」处分析各维增长的**先后顺序**（如先耐后速：第 2 年带
//! 1 张卡的耐增长反而高于带 2 张卡的速），以及「增加运气的训练主要类型」。
//! 判断力留给 SKILL 层，本模块只做数据预处理。
//!
//! 口径：
//! - **训练次数**：`execution` 推断的实际动作「X训练」按年段计数（推断口径；
//!   「X训练·继承混合」该回合确实做了训练，也计入）
//! - **净增长**：timeline 回合末五维的**年段差分**（段末 − 段前；含事件 / 继承 /
//!   比赛奖励，**不纯是训练贡献**——读「增长顺序」够用，读训练收益要结合次数）
//! - **运气归属**：回合合计 Δ（与 `checks::super_ramen_stats` 同口径）按**当回合
//!   实际训练维**归属；「X训练·继承混合」与无训练回合（休息 / 出行 / 比赛 /
//!   剧本 / 未知）归入 `luck_other`——继承落地与非训练动作的 Δ 不算到训练头上；
//!   **程序性大波动回合**（自选比赛期限 swing / 学技能标记，见 flags）归入
//!   `luck_program`——game6263 实测 t25 的 +11683 回升若不剔除会把「涨运气的
//!   训练类型」整体带偏

use std::collections::BTreeMap;

use serde::Serialize;

use crate::checks::SUPER_RAMEN_START;
use crate::decisions::{DecRow, FlaggedTurn};
use crate::execution::ExecRow;
use crate::timeline::TimelineRow;

/// 年段（与 brief 运气分段同界：三个自然年 + 超级拉面期）
const SEGMENTS: [(&str, u32, u32); 4] = [
    ("第1年", 0, 23),
    ("第2年", 24, 47),
    ("第3年", 48, 71),
    ("超拉期", SUPER_RAMEN_START, u32::MAX)
];

/// 五维属性名（训练位索引同序）
const ATTR_NAMES: [&str; 5] = ["速", "耐", "力", "根", "智"];

/// 单年段画像
#[derive(Debug, Clone, Default, Serialize)]
pub struct YearProfile {
    pub label: String,
    /// 本段实际「X训练」次数（速/耐/力/根/智；推断口径）
    pub train_counts: [u32; 5],
    /// 本段五维净增长（段末 − 段前；含事件 / 继承 / 比赛奖励）
    pub gains: [i32; 5]
}

/// 训练画像块（digest.training）
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrainingProfile {
    /// 三个自然年 + 超拉期（年段无数据时该段仍占位、全零）
    pub years: Vec<YearProfile>,
    /// 全局：回合合计 Δ 按当回合实际训练维归属（速/耐/力/根/智）
    pub luck_by_attr: [f64; 5],
    /// 非训练回合（休息 / 出行 / 比赛 / 剧本 / 继承混合 / 未知）的回合合计 Δ 总和
    pub luck_other: f64,
    /// 程序性大波动回合（自选比赛期限 swing / 学技能）的回合合计 Δ 总和——
    /// 不计入训练归属（带偏「涨运气的训练类型」）
    pub luck_program: f64
}

/// 训练画像主入口（`tl` 升序；`exec` 为推断行；`dec` 为 calc 决策行；
/// `flags` 为伪波动标记——swing / 学技能回合的 Δ 归 `luck_program`）
pub fn build(
    tl: &[TimelineRow],
    exec: &[ExecRow],
    dec: &[DecRow],
    flags: &[FlaggedTurn]
) -> TrainingProfile {
    // 回合末五维（timeline 升序 → 后写覆盖）
    let mut state: BTreeMap<u32, [i32; 5]> = BTreeMap::new();
    for r in tl {
        state.insert(r.turn, r.five_status);
    }
    // 回合合计 Δ（同 super_ramen_stats / digest 年代注记口径）
    let mut delta_by_turn: BTreeMap<u32, f64> = BTreeMap::new();
    for r in dec {
        if let Some(d) = r.turn_delta {
            *delta_by_turn.entry(r.turn).or_default() += d;
        }
    }
    // 每回合的实际训练维（宽松：含「·继承混合」——计数用）
    let train_lenient: BTreeMap<u32, usize> = exec
        .iter()
        .filter_map(|e| attr_lenient(&e.actual_action).map(|i| (e.turn, i)))
        .collect();
    // 严格：纯「X训练」——运气归属用（继承回合的 Δ 含继承落地，不归训练）
    let train_strict: BTreeMap<u32, usize> = exec
        .iter()
        .filter_map(|e| attr_strict(&e.actual_action).map(|i| (e.turn, i)))
        .collect();

    // —— 逐年：训练次数 + 净增长 ——
    // 段前基准：首段取最小回合（正常为 turn 0 的开局五维），其后各段沿用上段末
    let mut years = Vec::with_capacity(SEGMENTS.len());
    let mut prev_state: Option<[i32; 5]> = None;
    for &(label, lo, hi) in SEGMENTS.iter() {
        let end = state.range(lo..=hi).next_back().map(|(_, v)| *v);
        let mut yp = YearProfile { label: label.to_string(), ..Default::default() };
        for (&t, &i) in train_lenient.range(lo..=hi) {
            yp.train_counts[i] += 1;
        }
        if let Some(cur) = end {
            let base = prev_state.or_else(|| state.values().next().copied()).unwrap_or([0; 5]);
            for i in 0..5 {
                yp.gains[i] = cur[i] - base[i];
            }
            prev_state = Some(cur);
        }
        years.push(yp);
    }

    // —— 运气归属：回合 Δ 计入当回合训练维；程序性大波动回合与无训练回合分桶 ——
    // program 回合 = flags 里带 free_race_deadline_swing 或 skill_learned(-Npt) 的回合
    let program_turns: std::collections::BTreeSet<u32> = flags
        .iter()
        .filter(|f| {
            f.reason == "free_race_deadline_swing" || f.reason.starts_with("skill_learned")
        })
        .map(|f| f.turn)
        .collect();
    let mut luck_by_attr = [0f64; 5];
    let mut luck_other = 0f64;
    let mut luck_program = 0f64;
    for (&t, &d) in &delta_by_turn {
        if program_turns.contains(&t) {
            luck_program += d;
        } else {
            match train_strict.get(&t) {
                Some(&i) => luck_by_attr[i] += d,
                None => luck_other += d
            }
        }
    }

    TrainingProfile { years, luck_by_attr, luck_other, luck_program }
}

/// 实际动作 → 训练维（计数用，宽松：「X训练·继承混合」也算训练了该维）
fn attr_lenient(actual: &str) -> Option<usize> {
    ATTR_NAMES.iter().position(|n| actual.starts_with(&format!("{n}训练")))
}

/// 实际动作 → 训练维（运气归属用，严格：纯「X训练」）
fn attr_strict(actual: &str) -> Option<usize> {
    if actual.contains("继承混合") {
        return None;
    }
    attr_lenient(actual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::Evidence;

    /// 构造 timeline 行（只填五维）
    fn tl_row(turn: u32, five: [i32; 5]) -> TimelineRow {
        TimelineRow {
            turn,
            seq: 0,
            stage: "Train".to_string(),
            reason: None,
            source: "command".to_string(),
            playing_state: 1,
            vital: 80,
            max_vital: 100,
            motivation: 4,
            five_status: five,
            five_status_display: five,
            five_status_limit: [3000; 5],
            skill_pt: 0,
            train_level_count: [1; 5],
            friend_outgoing_used: 0,
            selected_regions: vec![],
            scenario_pt: 0,
            feeling_stock: vec![],
            super_ramen: -1,
            is_ill: false,
            is_qiezhe: false,
            is_xiao_qie: false,
            race_count: 0,
            absent_persons: vec![]
        }
    }

    /// 构造推断行
    fn exec_row(turn: u32, actual: &str) -> ExecRow {
        ExecRow {
            turn,
            stage: "Train".to_string(),
            ai_choice: "速训练".to_string(),
            actual_action: actual.to_string(),
            matches: Some(true),
            evidence: Evidence::default(),
            alt_candidate: None
        }
    }

    /// 构造决策行（只填 turn / turn_delta）
    fn dec_row(turn: u32, delta: Option<f64>) -> DecRow {
        DecRow {
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
            step: 0
        }
    }

    /// 逐年次数 / 净增（先耐后速形态）与运气归属（含继承混合与 other）
    #[test]
    fn test_training_profile() {
        // 五维：t0 开局 [100,100,100,100,100]；t23 末 [200,300,150,120,130]；
        // t47 末 [400,700,300,200,220]；t71 末 [900,900,400,250,260]；t77 末 [1000,950,420,260,280]
        let tl = vec![
            tl_row(0, [100, 100, 100, 100, 100]),
            tl_row(23, [200, 300, 150, 120, 130]),
            tl_row(47, [400, 700, 300, 200, 220]),
            tl_row(71, [900, 900, 400, 250, 260]),
            tl_row(77, [1000, 950, 420, 260, 280])
        ];
        // 训练：Y1 速×2、耐×1；Y2 耐×3（其中 t30 继承混合）、速×1、休息 1；
        // Y3 速×2；超拉期 速×1
        let exec = vec![
            exec_row(5, "速训练"),
            exec_row(10, "速训练"),
            exec_row(20, "耐训练"),
            exec_row(30, "耐训练·继承混合"),
            exec_row(35, "耐训练"),
            exec_row(40, "耐训练"),
            exec_row(44, "速训练"),
            exec_row(50, "休息"),
            exec_row(55, "速训练"),
            exec_row(60, "速训练"),
            exec_row(74, "速训练")
        ];
        // 回合 Δ：速训练回合 +100、耐 +80、继承混合 t30 +500（归 other）、休息 t50 -30（other）
        let dec = vec![
            dec_row(5, Some(100.0)),
            dec_row(10, Some(100.0)),
            dec_row(20, Some(80.0)),
            dec_row(30, Some(500.0)),
            dec_row(35, Some(80.0)),
            dec_row(40, Some(80.0)),
            dec_row(44, Some(100.0)),
            dec_row(50, Some(-30.0)),
            dec_row(55, Some(100.0)),
            dec_row(60, Some(100.0)),
            dec_row(74, Some(100.0))
        ];
        // flags：t44 自选期限 swing、t50 学技能 → 两回合 Δ 归 luck_program
        let flags = vec![
            crate::decisions::FlaggedTurn { turn: 44, reason: "free_race_deadline_swing".to_string() },
            crate::decisions::FlaggedTurn { turn: 50, reason: "skill_learned(-30pt)".to_string() },
            crate::decisions::FlaggedTurn { turn: 23, reason: "year_boundary(24)".to_string() },
        ];
        let p = build(&tl, &exec, &dec, &flags);
        println!("{p:#?}");

        // 年段：训练次数
        assert_eq!(p.years[0].train_counts, [2, 1, 0, 0, 0], "第1年 速×2 耐×1");
        assert_eq!(p.years[1].train_counts, [1, 3, 0, 0, 0], "第2年 耐×3（含继承混合）速×1");
        assert_eq!(p.years[2].train_counts, [2, 0, 0, 0, 0], "第3年 速×2（休息不计）");
        assert_eq!(p.years[3].train_counts, [1, 0, 0, 0, 0], "超拉期 速×1");

        // 净增：段末 − 段前（Y2 耐 +400 > 速 +200 → 先耐后速形态）
        assert_eq!(p.years[0].gains, [100, 200, 50, 20, 30]);
        assert_eq!(p.years[1].gains, [200, 400, 150, 80, 90]);
        assert_eq!(p.years[2].gains, [500, 200, 100, 50, 40]);
        assert_eq!(p.years[3].gains, [100, 50, 20, 10, 20]);

        // 运气归属：速 5×100（t5/t10/t55/t60/t74；t44 归 swing）、耐 3×80（t20/t35/t40）；
        // t30 继承混合 +500 归 other；t44 +100 与 t50 -30 归 program（年界标记不剔除）
        assert_eq!(p.luck_by_attr[0], 500.0, "速训练回合合计 Δ（t44 swing 剔除）");
        assert_eq!(p.luck_by_attr[1], 240.0, "耐训练回合（不含继承混合 t30）");
        assert_eq!(p.luck_by_attr[2..], [0.0, 0.0, 0.0]);
        assert_eq!(p.luck_other, 500.0, "继承混合 +500（t50 学技能移入 program）");
        assert_eq!(p.luck_program, 70.0, "swing +100 与学技能 -30");
    }
}
