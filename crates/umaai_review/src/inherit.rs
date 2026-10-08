//! 继承质量分析（文档 §7）
//!
//! 继承是玩家**局外选择**（选哪个继承、养哪匹马娘）的核心输入，不属于单局
//! 操作质量，单独成块（§7 开头）。
//!
//! - 继承回合 `[30, 54]`（代码常量，与年份边界错开 6 回合的独立未记录事件）
//! - 参考值 = `Σ(extra_count[0..5]) + 126`（`game_config.toml [config_override]`
//!   的种马额外属性，只取五维不含技能点）
//! - 贡献 = 继承回合前后**五维差** − 该回合行动本身的贡献（训练目标属性增量）
//! - 比较口径：**只比五维，不含 PT**

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{checks::INHERIT_TURNS, execution::ExecutionResult, timeline::TimelineRow};

/// 继承质量块（digest inherit）
#[derive(Debug, Clone, Serialize)]
pub struct InheritBlock {
    /// 两次继承回合
    pub turns: [u32; 2],
    /// 参考值（`None` = 配置不可用，如 gamedata 缺失）
    pub reference_value: Option<i32>,
    pub contributions: Vec<InheritContrib>,
    /// 口径说明
    pub dev_note: String
}

/// 单次继承的贡献与偏差
#[derive(Debug, Clone, Serialize)]
pub struct InheritContrib {
    pub turn: u32,
    /// 剥离训练贡献后的五维增量合计（继承贡献）
    pub five_status_sum: i32,
    /// 贡献 − 参考值（`None` = 参考值不可用）
    pub deviation: Option<i32>
}

/// 组装继承质量块
///
/// - `reference_value`：`Σ(extra_count[0..5]) + 126`（调用方从 `load_game_config`
///   解算；不可用时传 `None` → 只出贡献不出偏差）
/// - 实测校准（§7.4 game6234）：turn 30 贡献 +241 vs 236 → 偏差 +5 ✓；
///   turn 54 偏差 −83 是**确实的损失**（继承质量偏弱，不是剥离方法误差）
pub fn build(tl: &[TimelineRow], exec: &ExecutionResult, reference_value: Option<i32>) -> InheritBlock {
    // 回合末（行动前）与回合首快照：窗口 (t-1 末 → t 首) = turn t-1 的行动结果
    // + turn t 的继承落地（§5.3「行动结果体现在下一回合快照」）
    let last_by_turn: BTreeMap<u32, &TimelineRow> = {
        let mut m: BTreeMap<u32, &TimelineRow> = BTreeMap::new();
        for r in tl {
            m.insert(r.turn, r); // 升序 → 后写覆盖 = 回合末
        }
        m
    };
    let first_by_turn: BTreeMap<u32, &TimelineRow> = {
        let mut m: BTreeMap<u32, &TimelineRow> = BTreeMap::new();
        for r in tl {
            m.entry(r.turn).or_insert(r); // 先写保留 = 回合首
        }
        m
    };
    // 窗口内含的是 **turn t-1 的行动**（剥离其训练目标属性增量）
    // 兼容「·继承混合」后缀（execution 对 29/53 锚点的标注）——用前缀匹配
    let trained_attr: BTreeMap<u32, usize> = {
        let mut m: BTreeMap<u32, usize> = BTreeMap::new();
        for r in &exec.rows {
            for (i, name) in ["速", "耐", "力", "根", "智"].iter().enumerate() {
                if r.actual_action.starts_with(&format!("{name}训练")) {
                    m.insert(r.turn, i);
                }
            }
        }
        m
    };

    let mut contributions = Vec::new();
    for &t in INHERIT_TURNS.iter() {
        let (Some(prev), Some(cur)) = (last_by_turn.get(&(t - 1)), first_by_turn.get(&t)) else {
            continue; // 回合缺快照（中途接管等）
        };
        let delta: [i32; 5] = std::array::from_fn(|i| cur.five_status[i] - prev.five_status[i]);
        // 剥离 turn t-1 训练目标属性的增量（§7.3 算法；非训练回合不剥离）
        let stripped = match trained_attr.get(&(t - 1)) {
            Some(&a) => delta[a],
            None => 0
        };
        let sum: i32 = delta.iter().sum::<i32>() - stripped;
        let deviation = reference_value.map(|r| sum - r);
        contributions.push(InheritContrib { turn: t, five_status_sum: sum, deviation });
    }

    InheritBlock {
        turns: INHERIT_TURNS,
        reference_value,
        contributions,
        dev_note: "不含 PT，只比五维（§7.3）；贡献 = 继承回合前后五维差 − 训练目标属性增量"
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{Evidence, ExecRow};

    /// 构造 timeline 行（只填五维）
    fn row(turn: u32, five: [i32; 5]) -> TimelineRow {
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
            selected_regions: vec![1, 2, 3],
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

    /// 测试用 execution 行（指定回合的实际动作）
    fn exec_row_at(turn: u32, actual: &str) -> ExecRow {
        ExecRow {
            turn,
            stage: "Train".to_string(),
            ai_choice: String::new(),
            actual_action: actual.to_string(),
            matches: None,
            evidence: Evidence::default(),
            alt_candidate: None
        }
    }

    /// 文档 §7.4 实测形态：窗口 (29 末 → 30 首) Δ=[85,138,26,81,49]，
    /// 剥离 turn 29 耐训练（138）→ +241 vs 236 偏差 +5；
    /// turn 54 同理剥离 turn 53 速训练（316）→ +153 偏差 −83
    #[test]
    fn test_inherit_doc_calibration() {
        let tl = vec![
            row(29, [1000, 1000, 1000, 1000, 1000]),
            row(30, [1085, 1138, 1026, 1081, 1049]),
            row(53, [2000, 2000, 2000, 2000, 2000]),
            row(54, [2316, 2010, 2104, 2001, 2038]),
        ];
        let exec = ExecutionResult {
            rows: vec![exec_row_at(29, "耐训练"), exec_row_at(53, "速训练")],
            ..Default::default()
        };
        let block = build(&tl, &exec, Some(236));
        println!("inherit: {block:#?}");
        assert_eq!(block.reference_value, Some(236));
        assert_eq!(block.contributions.len(), 2);
        assert_eq!(block.contributions[0].turn, 30);
        assert_eq!(block.contributions[0].five_status_sum, 241, "379 − 138（耐训练）");
        assert_eq!(block.contributions[0].deviation, Some(5));
        assert_eq!(block.contributions[1].five_status_sum, 153, "469 − 316（速训练）");
        assert_eq!(block.contributions[1].deviation, Some(-83));
    }

    /// 参考值不可用 → deviation None；回合缺快照 → 跳过
    #[test]
    fn test_inherit_degraded() {
        let tl = vec![row(29, [1000; 5]), row(30, [1100; 5])]; // 缺 turn 53/54
        let exec = ExecutionResult {
            rows: vec![exec_row_at(29, "休息")],
            ..Default::default()
        };
        let block = build(&tl, &exec, None);
        println!("降级 inherit: {block:#?}");
        assert_eq!(block.contributions.len(), 1, "缺 turn 54 快照 → 只出 turn 30");
        assert_eq!(block.contributions[0].deviation, None);
        assert_eq!(block.contributions[0].five_status_sum, 500, "休息不剥离");
    }
}
