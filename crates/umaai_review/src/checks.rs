//! 检查项引擎（文档 §6.1 坏手法 / §6.6 超级拉面期 / §6.7 伪波动标记）
//!
//! 原则（§6.1）：**已验证判据**直接产 findings；**判据待实测**的项（训练与
//! 体力健康 / 吃面节奏 / 友人完成度 / free_race 次数 / 状态健康 / 属性溢出）
//! 不纳入自动 findings，留待 SKILL 层用 digest 数据人工判读。
//!
//! 代码常量（§10）：`YEAR_BOUNDARIES` / `INHERIT_TURNS` / `SUPER_RAMEN_START`。

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    decisions::{DecRow, FlaggedTurn},
    execution::{ExecutionResult, Finding},
    schedule::Schedule,
    timeline::TimelineRow
};

/// 剧本年份边界（代码常量，§10）
pub const YEAR_BOUNDARIES: [u32; 3] = [24, 48, 72];
/// 两次继承回合（代码常量，§10）
pub const INHERIT_TURNS: [u32; 2] = [30, 54];
/// 第 1 年地区选择回合（开局 2-3 回合；地区选择本身带来的期望跳变也是
/// 程序性波动——与年界前的 RegionSelect 正跳同源，用户拍板）
pub const Y1_REGION_SELECT_TURNS: [u32; 2] = [2, 3];
/// 超级拉面期起点（turn ≥ 72；实测分身自 72 起，用户拍板）
pub const SUPER_RAMEN_START: u32 = 72;

/// 友人出行次数上限（完成要求）
const FRIEND_OUTING_CAP: i32 = 5;
/// 友人次数用完距结束的「仍远」阈值（初判，待标定；game6234：turn 58 用完剩 19）
const FRIEND_EARLY_REMAINING: u32 = 6;
/// 体力低点阈值（与 §6.1「体力长期低」同档；实测 turn 69 跌至 27）
pub const LOW_VITAL: i32 = 35;
/// 心情掉落后未及时恢复的观察窗口 N（§6.1 建议 N=3）
pub const MOTIVATION_WINDOW: u32 = 3;

/// 伪波动标记（§6.7）：年界 / 继承 / RMJ 结算 / 开局第 1 年地区选择回合
///
/// - 年界窗口 = 边界前 2 回合至后 1 回合；**turn 72 双属性**：既标记为年界
///   波动回合，也仍算进超级拉面期统计（§6.6 例外：该回合份量实打实，
///   归因时注意其正跳不是纯程序性回吐）
/// - 继承回合（[30,54]，与年份边界错开 6 回合的独立事件）
/// - RMJ 结算（从 `playing_state` 46/48 自动检测，不硬编码位置）
/// - **开局 2-3 回合 = 第 1 年地区选择**：选择带来的期望跳变（小赚或小亏）
///   也是程序性波动，与年界前的 RegionSelect 正跳同源（用户拍板）
pub fn flagged_turns(tl: &[TimelineRow]) -> Vec<FlaggedTurn> {
    let mut flags: Vec<FlaggedTurn> = Vec::new();
    for &t in Y1_REGION_SELECT_TURNS.iter() {
        flags.push(FlaggedTurn { turn: t, reason: "region_select(y1)".to_string() });
    }
    for &b in YEAR_BOUNDARIES.iter() {
        for t in b.saturating_sub(2)..=b + 1 {
            flags.push(FlaggedTurn { turn: t, reason: format!("year_boundary({b})") });
        }
    }
    for &t in INHERIT_TURNS.iter() {
        flags.push(FlaggedTurn { turn: t, reason: "inherit".to_string() });
    }
    for r in tl {
        if let Some(reason) = &r.reason {
            if reason.starts_with("rmj_settle") || reason.starts_with("rmj_final") {
                flags.push(FlaggedTurn { turn: r.turn, reason: reason.clone() });
            }
        }
    }
    flags.sort_by_key(|f| f.turn);
    flags.dedup_by(|a, b| a.turn == b.turn && a.reason == b.reason);
    flags
}

/// 超级拉面期（turn ≥ 72）运气统计（§6.6）
///
/// B 类分身按机制保证 100% 落得意位、每回合只能选一个训练 →「没吃到彩圈」
/// 不算亏，**只能用该期运气分判盈亏**（§6.5.3）。
#[derive(Debug, Default, Clone, Serialize)]
pub struct SuperRamenStats {
    /// 逐回合合计 Δ（回合合计口径，§6.2）
    pub by_turn: BTreeMap<u32, f64>,
    /// 该期回合合计 Δ 总和（正 = 赚 / 负 = 亏）
    pub total: f64
}

/// 超级拉面期统计（窗口内无任何 Δ 数据时返回 `None`）
pub fn super_ramen_stats(dec: &[DecRow]) -> Option<SuperRamenStats> {
    let mut by_turn: BTreeMap<u32, f64> = BTreeMap::new();
    for r in dec {
        if r.turn >= SUPER_RAMEN_START {
            if let Some(d) = r.turn_delta {
                *by_turn.entry(r.turn).or_default() += d;
            }
        }
    }
    if by_turn.is_empty() {
        return None;
    }
    let total = by_turn.values().sum();
    Some(SuperRamenStats { by_turn, total })
}

/// 超级拉面期盈亏 finding（§6.6：实测 game6234 该期 −1482 判亏）
pub fn super_ramen_finding(stats: &SuperRamenStats) -> Finding {
    let detail = stats
        .by_turn
        .iter()
        .map(|(t, d)| format!("{t}:{d:.1}"))
        .collect::<Vec<_>>()
        .join(" ");
    let verdict = if stats.total < 0.0 { "判亏" } else { "判赚" };
    Finding {
        kind: "super_ramen_luck".to_string(),
        turn: SUPER_RAMEN_START,
        file: None,
        evidence: format!(
            "turn>=72 回合合计 Δ = {:.2}（{detail}）{verdict}；B 类分身机制保证落得意位，\
             没吃到彩圈不算亏，只能用该期运气分判盈亏",
            stats.total
        ),
        severity: if stats.total < 0.0 { "warn".to_string() } else { "info".to_string() }
    }
}

/// 自选比赛期限波动的幅度阈值（用户口径：>8000 的运气波动通常与自选比赛期限有关）
const FREE_RACE_SWING_MIN: f64 = 8000.0;
/// 骤降后多少回合内出现的回升视为配对
const FREE_RACE_SWING_WINDOW: u32 = 3;
/// 配对容差：净变 ≤ 幅度 × 此比例（对应「补赛达标后运气分恢复到以前水平」）
const FREE_RACE_SWING_NET_TOL: f64 = 0.25;

/// 自选比赛期限波动（骤降 → 补赛达标后回升，净变≈0）
#[derive(Debug, Clone, Serialize)]
pub struct FreeRaceSwing {
    pub drop_turn: u32,
    pub drop_delta: f64,
    pub recover_turn: u32,
    pub recover_delta: f64,
    /// 净变（≈0 = 回到原水平）
    pub net: f64,
    /// 命中的自选窗口（未匹配到则为 `None`）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<FreeRaceWindow>,
}

/// 命中的自选比赛窗口
#[derive(Debug, Clone, Serialize)]
pub struct FreeRaceWindow {
    pub start_turn: u32,
    pub end_turn: u32,
    pub required: u32,
    pub picked_turns: Vec<i32>,
    /// 是否在截止回合（或前 1 回合）才补赛 → 「自选比赛极限达标」
    pub last_minute: bool,
}

/// 自选比赛期限波动（用户口径）
///
/// **机制**：自选比赛窗口截止前若仍未达标，MCTS 对「将育成失败」的状态给出大幅
/// 低估 → 运气分骤降；随后补赛达标 → 回升。净变≈0，故**不影响最终运气**，
/// 但「自选比赛极限达标」是值得关注的操作（叙事要点出，不当坏运气）。
///
/// **判据**：回合合计 Δ 出现 ≤ −8000 的骤降，且其后 3 回合内出现 ≥ +8000 的回升、
/// 净变 ≤ 幅度的 25%（= 回到原水平）。命中后骤降/回升两回合都进伪波动标记
/// （原有年界标记可能同时命中且归因有误，见 `free_race_swing_note`）。
pub fn free_race_swings(dec: &[DecRow], sched: &Schedule) -> Vec<FreeRaceSwing> {
    // 逐回合合计 Δ（与 super_ramen_stats 同口径）
    let mut by_turn: BTreeMap<u32, f64> = BTreeMap::new();
    for r in dec {
        if let Some(d) = r.turn_delta {
            *by_turn.entry(r.turn).or_default() += d;
        }
    }
    let turns: Vec<(u32, f64)> = by_turn.into_iter().collect();

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < turns.len() {
        let (drop_turn, drop_delta) = turns[i];
        if drop_delta > -FREE_RACE_SWING_MIN {
            i += 1;
            continue;
        }
        // 往后 SWING_WINDOW 回合内找配对回升
        let mut paired: Option<(usize, u32, f64)> = None;
        for (j, &(rt, rd)) in turns.iter().enumerate().skip(i + 1) {
            if rt > drop_turn + FREE_RACE_SWING_WINDOW {
                break;
            }
            if rd >= FREE_RACE_SWING_MIN {
                let mag = drop_delta.abs().max(rd.abs());
                if (drop_delta + rd).abs() <= mag * FREE_RACE_SWING_NET_TOL {
                    paired = Some((j, rt, rd));
                }
                break;
            }
        }
        match paired {
            Some((j, recover_turn, recover_delta)) => {
                out.push(FreeRaceSwing {
                    drop_turn,
                    drop_delta,
                    recover_turn,
                    recover_delta,
                    net: drop_delta + recover_delta,
                    window: match_free_race_window(sched, drop_turn),
                });
                i = j + 1; // 跳过已配对的回升回合
            }
            None => i += 1,
        }
    }
    out
}

/// 匹配该骤降回合所属的自选窗口（截止回合后留 SWING_WINDOW 容差，容纳迟到的回升）
fn match_free_race_window(sched: &Schedule, drop_turn: u32) -> Option<FreeRaceWindow> {
    sched.free_races.iter().find_map(|f| {
        if drop_turn < f.start_turn || drop_turn > f.end_turn + FREE_RACE_SWING_WINDOW {
            return None;
        }
        let last_minute = f.picked_turns.iter().any(|t| {
            let t = *t as u32;
            t == f.end_turn || t + 1 == f.end_turn
        });
        Some(FreeRaceWindow {
            start_turn: f.start_turn,
            end_turn: f.end_turn,
            required: f.required,
            picked_turns: f.picked_turns.clone(),
            last_minute,
        })
    })
}

/// 自选比赛期限波动 → 伪波动标记（骤降与回升两回合都标）
pub fn free_race_swing_flags(swings: &[FreeRaceSwing]) -> Vec<FlaggedTurn> {
    let mut out = Vec::new();
    for s in swings {
        for turn in [s.drop_turn, s.recover_turn] {
            out.push(FlaggedTurn { turn, reason: "free_race_deadline_swing".to_string() });
        }
    }
    out
}

/// 自选比赛期限波动 → 赛程注记（供叙事引用「极限达标」，并纠正年界标记的误归因）
pub fn free_race_swing_note(s: &FreeRaceSwing) -> String {
    let where_ = match &s.window {
        Some(w) if w.last_minute => format!(
            "t{} 是自选窗口 {}..={}（要求 {} 次，实跑 {:?}）的截止回合，属「自选比赛极限达标」",
            s.drop_turn, w.start_turn, w.end_turn, w.required, w.picked_turns
        ),
        Some(w) => format!(
            "t{} 落在自选窗口 {}..={}（要求 {} 次，实跑 {:?}）",
            s.drop_turn, w.start_turn, w.end_turn, w.required, w.picked_turns
        ),
        None => format!("t{} 未匹配到自选窗口（按同形态波动处理）", s.drop_turn),
    };
    format!(
        "自选比赛期限波动：{where_}；运气分 t{} 骤降 {:+.0} → t{} 补赛达标后回升 {:+.0}\
         （净变 {:+.0}，属程序性波动、不影响最终运气）",
        s.drop_turn, s.drop_delta, s.recover_turn, s.recover_delta, s.net
    )
}

/// 学技能标记（用户口径）
///
/// **技能点只减不增——减少就一定是玩家学了技能**（没有别的减少途径），故不设运气分门槛，
/// 逐回合检查技能点（回合末口径）、减少即标。
///
/// **为什么重要**：花掉技能点后 AI 的期望终局分把「未花掉的技能点」计入价值 → 运气分下降，
/// 但这**不是真实运气下降**。标记 reason = `skill_learned(-Npt)`，供叙事识别为程序性波动。
pub fn skill_learned_flags(tl: &[TimelineRow]) -> Vec<FlaggedTurn> {
    let mut pt: BTreeMap<u32, i32> = BTreeMap::new();
    for r in tl {
        pt.insert(r.turn, r.skill_pt); // 升序 → 后写覆盖 = 回合末
    }
    let turns: Vec<u32> = pt.keys().copied().collect();
    let mut out = Vec::new();
    for pair in turns.windows(2) {
        let (t0, t1) = (pair[0], pair[1]);
        let spent = pt[&t0] - pt[&t1]; // >0 = 技能点变少（学了技能）
        if spent > 0 {
            out.push(FlaggedTurn { turn: t1, reason: format!("skill_learned(-{spent}pt)") });
        }
    }
    out
}

/// 坏手法检查项（§6.1 已验证判据 + 训练失败候选清单）
///
/// 覆盖：
/// 1. **目标赛未跑赢**（已验证）：必赛回合不在 `raceHistory`（raceHistory 只记
///    跑赢——实测 game6234 命中 turn 45）
/// 2. **关键资源过早耗尽**（已验证）：友人次数达上限距结束仍远 + 此后出现
///    体力低点（实测：turn 58 用完 → turn 69 体力 27）
/// 3. **心情掉落后未及时恢复**（已验证，§6.1 首个已验证项）：干劲下降后
///    N=3 回合内既无恢复动作（出行/友人出行/休息）也未回升
/// 4. **训练失败候选**（先出候选人工复核）：AI 建议训练 + 跨回合五维零增长
///    + 体力有训练量级消耗
pub fn bad_habits(
    tl: &[TimelineRow],
    exec: &ExecutionResult,
    sched: &Schedule,
    race_history: &[i32]
) -> Vec<Finding> {
    let mut out = Vec::new();
    let max_turn = tl.last().map(|r| r.turn).unwrap_or(77);

    // ① 目标赛未跑赢
    for &t in &sched.mandatory_turns {
        if !race_history.contains(&t) {
            out.push(Finding {
                kind: "mandatory_race_not_won".to_string(),
                turn: t as u32,
                file: None,
                evidence: "必赛回合未跑赢（raceHistory 只记跑赢的比赛）".to_string(),
                severity: "warn".to_string()
            });
        }
    }

    // ② 关键资源过早耗尽（友人次数）
    // 「用完」的判定：friend_outgoing_used 达 5 的首份快照在 t5 回合 —— 但第 5 次
    // 出行发生在 t5-1 回合（行动结果体现在下一回合快照，§5.3），对齐文档口径
    // （实测 game6234：turn 58 用完）
    if let Some(t5) = tl.iter().find(|r| r.friend_outgoing_used >= FRIEND_OUTING_CAP).map(|r| r.turn) {
        let t = t5.saturating_sub(1);
        let remaining = max_turn.saturating_sub(t);
        let min_vital = tl
            .iter()
            .filter(|r| r.turn > t)
            .map(|r| r.vital)
            .min();
        if remaining >= FRIEND_EARLY_REMAINING && min_vital.is_some_and(|v| v < LOW_VITAL) {
            out.push(Finding {
                kind: "friend_quota_exhausted_early".to_string(),
                turn: t,
                file: None,
                evidence: format!(
                    "turn {t} 友人出行次数用完（距结束还剩 {remaining} 回合），此后体力最低 \
                     跌至 {}（< {LOW_VITAL}）——一次性资源过早耗尽",
                    min_vital.unwrap_or(0)
                ),
                severity: "warn".to_string()
            });
        }
    }

    // ③ 心情掉落后未及时恢复
    let mot_by_turn: BTreeMap<u32, i32> = per_turn_last(tl, |r| r.motivation);
    let recovery_turns: Vec<u32> = exec
        .rows
        .iter()
        .filter(|r| matches!(r.actual_action.as_str(), "出行" | "友人出行" | "休息"))
        .map(|r| r.turn)
        .collect();
    let turns: Vec<u32> = mot_by_turn.keys().copied().collect();
    for (i, &t) in turns.iter().enumerate() {
        let Some(&prev) = i.checked_sub(1).and_then(|p| mot_by_turn.get(&turns[p])) else {
            continue;
        };
        let Some(&cur) = mot_by_turn.get(&t) else { continue };
        if cur >= prev {
            continue; // 未下降
        }
        // 窗口 t..=t+3：既无恢复动作、干劲也未回升（高于掉落后的值）
        let recovered = (1..=MOTIVATION_WINDOW).any(|k| {
            mot_by_turn
                .get(&(t + k))
                .is_some_and(|&m| m > cur)
        });
        let acted = (0..=MOTIVATION_WINDOW).any(|k| recovery_turns.contains(&(t + k)));
        if !recovered && !acted {
            out.push(Finding {
                kind: "motivation_drop_unrecovered".to_string(),
                turn: t,
                file: None,
                evidence: format!(
                    "干劲 {prev}→{cur} 后 {MOTIVATION_WINDOW} 回合内既无出行/休息恢复动作、\
                     干劲也未回升（AI 认为后续支援卡事件会把心情补回来，不主动恢复）"
                ),
                severity: "warn".to_string()
            });
        }
    }

    // ④ 训练失败候选（人工复核：事件导致的属性变化可能误判）
    for row in &exec.rows {
        let five_total: i32 = row.evidence.five_status_delta.iter().sum();
        if row.ai_choice.contains("训练")
            && five_total == 0
            && row.evidence.vital_delta <= -10
        {
            out.push(Finding {
                kind: "train_failure_candidate".to_string(),
                turn: row.turn,
                file: None,
                evidence: format!(
                    "AI 建议 {}，体力消耗 {} 但五维零增长（训练失败候选——\
                     正常训练对应属性 +13~+134；需人工复核事件影响）",
                    row.ai_choice, row.evidence.vital_delta
                ),
                severity: "info".to_string()
            });
        }
    }

    out
}

/// 每回合最后一份快照的字段值（回合末状态口径）
fn per_turn_last<T: Copy>(tl: &[TimelineRow], f: impl Fn(&TimelineRow) -> T) -> BTreeMap<u32, T> {
    let mut m: BTreeMap<u32, T> = BTreeMap::new();
    for r in tl {
        m.insert(r.turn, f(r)); // timeline 升序 → 后写覆盖 = 回合末
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::TimelineRow;

    /// 构造测试 timeline 行（只填检查项相关字段）
    fn row(turn: u32, vital: i32, motivation: i32, friend: i32, reason: Option<&str>) -> TimelineRow {
        TimelineRow {
            turn,
            seq: 0,
            stage: "Train".to_string(),
            reason: reason.map(String::from),
            source: "command".to_string(),
            playing_state: 1,
            vital,
            max_vital: 100,
            motivation,
            five_status: [0; 5],
            five_status_display: [0; 5],
            five_status_limit: [3000; 5],
            skill_pt: 0,
            train_level_count: [1; 5],
            friend_outgoing_used: friend,
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

    /// 年界 / 继承 / RMJ / turn72 例外
    #[test]
    fn test_flagged_turns() {
        let tl = vec![
            row(23, 80, 4, 0, None),
            row(46, 80, 4, 0, Some("rmj_settle(46)")),
            row(48, 80, 4, 0, Some("rmj_final(48)")),
            row(50, 80, 4, 0, None),
        ];
        let flags = flagged_turns(&tl);
        println!("flagged: {flags:?}");
        let has = |t: u32, reason: &str| {
            flags.iter().any(|f| f.turn == t && f.reason.contains(reason))
        };
        assert!(has(22, "year_boundary(24)") && has(25, "year_boundary(24)"));
        assert!(has(46, "year_boundary(48)") && has(49, "year_boundary(48)"));
        assert!(has(70, "year_boundary(72)") && has(71, "year_boundary(72)"));
        assert!(
            has(72, "year_boundary(72)"),
            "turn 72 双属性：既标记年界波动，也算进超级拉面统计"
        );
        assert!(has(73, "year_boundary(72)"));
        assert!(has(30, "inherit") && has(54, "inherit"));
        assert!(has(2, "region_select(y1)") && has(3, "region_select(y1)"), "开局第 1 年地区选择");
        assert!(has(46, "rmj_settle") && has(48, "rmj_final"), "RMJ 自动检测");
    }

    /// 超级拉面期统计（turn ≥ 72 回合合计）
    #[test]
    fn test_super_ramen_stats() {
        let dec = vec![
            dec_row(70, Some(500.0)),
            dec_row(72, Some(-800.0)),
            dec_row(72, Some(-200.0)),
            dec_row(74, Some(-482.13)),
        ];
        let stats = super_ramen_stats(&dec);
        println!("super_ramen: {:?}", stats);
        let s = stats.expect("turn>=72 有数据");
        assert_eq!(s.by_turn.len(), 2);
        assert!((s.total - (-1482.13)).abs() < 1e-6);
        let f = super_ramen_finding(&s);
        println!("finding: {f:?}");
        assert_eq!(f.severity, "warn", "合计为负 → 判亏");
        assert!(f.evidence.contains("判亏"));

        let none = super_ramen_stats(&[dec_row(70, Some(1.0))]);
        assert!(none.is_none(), "窗口内无数据 → None");
    }

    /// 坏手法四项（含 game6234 实测形态：友人 58 耗尽 + 体力 27 / 心情 45 掉落）
    #[test]
    fn test_bad_habits() {
        // 友人次数：turn 58 达 5，此后 turn 69 体力 27
        let mut tl = vec![];
        for t in 0..=77 {
            let friend = if t >= 58 { 5 } else { 0 };
            let vital = if t == 69 { 27 } else { 80 };
            // 心情：turn 44=5 → 45 掉到 4，46-49 停在 4，50 回 5（事件恢复）
            let motivation = match t {
                t if t < 45 => 5,
                t if (45..50).contains(&t) => 4,
                _ => 5
            };
            tl.push(row(t, vital, motivation, friend, None));
        }
        let exec = ExecutionResult {
            rows: vec![crate::execution::ExecRow {
                turn: 45,
                stage: "Train".to_string(),
                ai_choice: "速训练".to_string(),
                actual_action: "剧本".to_string(),
                matches: Some(false),
                evidence: crate::execution::Evidence {
                    five_status_delta: [0; 5],
                    vital_delta: -20,
                    motivation_delta: 0,
                    race_count_delta: 0,
                    friend_outgoing_delta: 0,
                    is_ill_cured: false,
                    scenario_pt_delta: 0,
                    feeling_stock_len_delta: 1,
                    super_ramen_delta: 0
                },
                alt_candidate: None
            }],
            findings: vec![],
            comparable: 1,
            matched: 0
        };
        let sched = Schedule {
            mandatory_turns: vec![45],
            free_races: vec![],
            notes: vec![]
        };
        let race_history: Vec<i32> = vec![]; // turn 45 未跑赢
        let findings = bad_habits(&tl, &exec, &sched, &race_history);
        for f in &findings {
            println!("{}: turn={} sev={} ev={}", f.kind, f.turn, f.severity, f.evidence);
        }
        let kind = |k: &str| findings.iter().filter(|f| f.kind == k).count();
        assert_eq!(kind("mandatory_race_not_won"), 1, "turn 45 必赛未跑赢");
        assert_eq!(kind("friend_quota_exhausted_early"), 1, "58 耗尽 + 69 体力 27");
        assert_eq!(kind("motivation_drop_unrecovered"), 1, "45 掉心情 3 回合未恢复");
        assert_eq!(kind("train_failure_candidate"), 1, "速训练五维零增长体力 -20");
    }

    /// 学技能标记：技能点减少即标（不设运气分门槛——技能点只减不增）
    #[test]
    fn test_skill_learned_flags() {
        let mut tl = vec![];
        // 技能点：t10=100 → t11=60（学了）→ t12=60（未学）→ t13=55（又学）
        for (t, pt) in [(10u32, 100), (11, 60), (12, 60), (13, 55)] {
            let mut r = row(t, 80, 4, 0, None);
            r.skill_pt = pt;
            tl.push(r);
        }
        let flags = skill_learned_flags(&tl);
        println!("{flags:?}");
        assert_eq!(flags.len(), 2, "两次技能点减少 → 两个标记");
        assert_eq!(flags[0].turn, 11);
        assert_eq!(flags[0].reason, "skill_learned(-40pt)");
        assert_eq!(flags[1].turn, 13);
        assert_eq!(flags[1].reason, "skill_learned(-5pt)");
        // 技能点不减少 → 无标记
        let flat = vec![row(1, 80, 4, 0, None), row(2, 80, 4, 0, None)];
        assert!(skill_learned_flags(&flat).is_empty());
    }

    /// 测试用自由比赛窗口
    fn free_window(start: u32, end: u32, required: u32, picked: Vec<i32>) -> crate::schedule::FreeRaceInfo {
        crate::schedule::FreeRaceInfo {
            start_turn: start,
            end_turn: end,
            required,
            grade: None,
            picked_turns: picked,
        }
    }

    /// 自选比赛期限波动（game3099 实测形态）：
    /// 窗口 12..=22 要求 1 次、实跑 [22]（截止回合才补赛）→ t22 −12003 / t23 +13383
    #[test]
    fn test_free_race_swing_game3099() {
        let dec = vec![
            dec_row(21, Some(-178.0)),
            dec_row(22, Some(-12003.0)),
            dec_row(23, Some(13383.0)),
            dec_row(24, Some(-773.0)),
        ];
        let sched = Schedule {
            mandatory_turns: vec![11, 41],
            free_races: vec![free_window(12, 22, 1, vec![22])],
            notes: vec![],
        };
        let sw = free_race_swings(&dec, &sched);
        println!("swings: {sw:#?}");
        assert_eq!(sw.len(), 1, "应检出 1 条自选期限波动");
        let s = &sw[0];
        assert_eq!((s.drop_turn, s.recover_turn), (22, 23));
        assert!((s.net - 1380.0).abs() < 1e-6, "净变 ≈ +1380");
        let w = s.window.as_ref().expect("应匹配到自选窗口");
        assert_eq!((w.start_turn, w.end_turn, w.required), (12, 22, 1));
        assert!(w.last_minute, "实跑 [22] == 截止回合 → 极限达标");
        // 注记要点出「极限达标」且说明不影响最终运气
        let note = free_race_swing_note(s);
        println!("note: {note}");
        assert!(note.contains("自选比赛极限达标"));
        assert!(note.contains("不影响最终运气"));
        // 两回合都进伪波动标记
        let flags = free_race_swing_flags(&sw);
        assert_eq!(flags.len(), 2);
        assert!(flags.iter().all(|f| f.reason == "free_race_deadline_swing"));
        assert!(flags.iter().any(|f| f.turn == 22) && flags.iter().any(|f| f.turn == 23));
    }

    /// 非截止回合补赛 → 命中窗口但不标「极限达标」
    #[test]
    fn test_free_race_swing_not_last_minute() {
        let dec = vec![dec_row(20, Some(-9000.0)), dec_row(21, Some(9100.0))];
        let sched = Schedule {
            mandatory_turns: vec![],
            free_races: vec![free_window(12, 22, 1, vec![20])],
            notes: vec![],
        };
        let sw = free_race_swings(&dec, &sched);
        assert_eq!(sw.len(), 1);
        let w = sw[0].window.as_ref().unwrap();
        assert!(!w.last_minute, "实跑 [20] 距截止 22 还有余量 → 非极限达标");
        assert!(free_race_swing_note(&sw[0]).contains("落在自选窗口"));
    }

    /// 未配对的骤降 / 净变过大 / 幅度不足 → 均不检出
    #[test]
    fn test_free_race_swing_negative_cases() {
        let sched = Schedule {
            mandatory_turns: vec![],
            free_races: vec![free_window(12, 22, 1, vec![22])],
            notes: vec![],
        };
        // ① 骤降后无回升
        let no_recover = vec![dec_row(22, Some(-12003.0)), dec_row(23, Some(-500.0))];
        assert!(free_race_swings(&no_recover, &sched).is_empty(), "无回升不配对");
        // ② 回升远不足以回到原水平（净变 -9000，超 25% 容差）
        let not_recovered = vec![dec_row(22, Some(-12003.0)), dec_row(23, Some(3003.0))];
        assert!(free_race_swings(&not_recovered, &sched).is_empty(), "未回到原水平不配对");
        // ③ 幅度不足阈值
        let small = vec![dec_row(22, Some(-5000.0)), dec_row(23, Some(5100.0))];
        assert!(free_race_swings(&small, &sched).is_empty(), "幅度 < 8000 不判");
        // ④ 回升超出配对窗口（>3 回合）
        let late = vec![dec_row(22, Some(-12003.0)), dec_row(27, Some(12000.0))];
        assert!(free_race_swings(&late, &sched).is_empty(), "回升超出窗口不配对");
    }

    /// 测试用决策行（只填 turn / turn_delta）
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
}
