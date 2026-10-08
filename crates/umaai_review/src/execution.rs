//! 实际执行动作推断（文档 §5.4、§6.9）与偏离清单（§11 步骤 4）
//!
//! 方法（§5.4）：取「回合末 train 决策点 → 下一回合首份快照」的**状态差**推断
//! 实际动作，回答「玩家有没有听 AI、听的是哪一选」。
//!
//! - **锚点**：每回合**最后一条 `decision_kind=train`** 的 calc 行。链序保证
//!   Train 阶段行在 RamenSelect 行之后（吃面路径），不吃面路径的
//!   RamenSelect 行即该回合最终项；地区选择是回合内子决策（kind≠train），
//!   自动排除（§5.3）
//! - **分类口径**（§12.5 修正方向 + 实测校准）：
//!   1. 比赛 = `raceHistory` +1（跑赢）**或锚点回合 ∈ 必赛回合**——输掉的比赛
//!      `raceHistory` 不记（§6.1），用赛程表兜底（URA 决赛 73/75/77 剧本固定赛）
//!   2. **训练**（先判训练、再用体力形态否证）：目标维增量 ≥ 当年阈值。
//!      **目标维的选取**（用户拍板，替换原「纯取最大增量」——多属性齐涨时会误判）：
//!      - **推荐维优先**：它是最大增量，**或**自身增量达 `TRAIN_ATTR_MIN`（+20）
//!        ——事件给属性的量级通常低于此，故「推荐维 +20 以上」是练了该维的强证据
//!      - 否则取增量最大的维
//!      **体力形态只作否证**：体力 ≥ +25 且目标维未达 `TRAIN_ATTR_MIN` → 判休息
//!      （休息 + 事件小额属性）；反过来，某维达 `TRAIN_ATTR_MIN` 时即使体力也回升
//!      仍判训练（智训练本身不耗体力，且事件可同时回体）
//!   3. 休息 = 体力 ≥ +25（容忍事件小额属性）；事件小额回体（未达 +25）不判休息，
//!      落到出行 / 剧本 / 未知
//!   4. **继承窗口混合**：锚点 29/53 的窗口（t 末 → t+1 首）含 t+1 的继承落地
//!      全维加成，行动与继承无法从数据剥离 → 实际动作标注「继承混合」、
//!      `matches` 置 None（不参与一致率、不产偏离 finding），证据照留
//! - **粒度限制**：只能识别「动作类别 + 目标属性」；事件给属性会混入训练
//!   判读（证据全量输出供 LLM 复核）
//! - **AI 自动执行局**（AIRedirector）本块仅作校验、不产生结论（§6.9）——
//!   bin 层无法判定是否自动执行，由 SKILL 层结合 context 解读

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use umasim::utils::Array5;

use crate::{checks::INHERIT_TURNS, decisions::DecRow, timeline::TimelineRow};

/// 五维属性名（训练动作的目标属性）
const ATTR_NAMES: [&str; 5] = ["速", "耐", "力", "根", "智"];

/// 休息判定的体力下限（表值休息 +30/+50/+70；智训练 +5 叠事件实测最多 +20，
/// 分界取 25——「休息 + 事件小额属性」不被误判训练、「智训练大回复」不误判休息）
pub const REST_MIN_VITAL: i32 = 25;

/// 属性增量达此值 → 该维「确定被训练」（用户拍板）
///
/// 事件给属性的量级通常低于此，故用于两处：
/// - **推荐维优先**：推荐维增量达此值即推定练了该维（即使它不是最大增量）
/// - **体力形态否证**：达此值时即使体力也回升仍判训练（智训练不耗体力 + 事件回体）
const TRAIN_ATTR_MIN: i32 = 20;

/// 训练判定的主增量下限（分年分段，用户拍板）：
/// - 第 1 年（turn < 24）：训练等级低、加成少 → **训练基础值表最低主维**
///   （等级 1 裸值 + 0 加成空人头，运行时从 `scenario_ramen.json` 推导，
///   实测表值 7 为智位等级 1；开局低加成训练实测低至 +12，文档 +13~+134
///   覆盖不到）
/// - 第 2 年起（turn ≥ 24）：训练等级与加成上来了 → 固定 **12**（事件增益
///   通常 < 12，与其他动作区分）
/// gamedata 未初始化时第 1 年退回表实测值 7
fn min_train_delta(turn: u32) -> i32 {
    const AFTER_Y1: i32 = 12;
    if turn >= crate::checks::YEAR_BOUNDARIES[0] {
        return AFTER_Y1;
    }
    const FALLBACK: i32 = 7;
    let Some(data) = umasim::gamedata::ramen::RAMENDATA.get() else {
        return FALLBACK;
    };
    let mut min = i32::MAX;
    for (pos, levels) in data.training_basic_value.iter().enumerate() {
        for row in levels {
            if let Some(&v) = row.get(pos) {
                if v > 0 {
                    min = min.min(v);
                }
            }
        }
    }
    if min == i32::MAX { FALLBACK } else { min }
}

/// AI 建议 vs 实际执行对照行（digest execution 块）
#[derive(Debug, Clone, Serialize)]
pub struct ExecRow {
    pub turn: u32,
    /// 锚点决策的阶段（`Train`；不吃面路径为 `RamenSelect`）
    pub stage: String,
    /// AI 建议选中项描述（锚点 chosen_desc）
    pub ai_choice: String,
    /// 推断的实际动作（动作类别 + 目标属性）
    pub actual_action: String,
    /// 是否一致（`None` = ai_choice 无法映射到动作类别）
    pub matches: Option<bool>,
    /// 状态差证据（§5.4 签名表全量）
    pub evidence: Evidence,
    /// 偏离一选时实际动作对应的候选（如 `#2 力训练`）——用户口径：偏离主要是
    /// **选了 2-3 选**而非完全不听建议；实际动作命中候选表 #2 起的某项时标注，
    /// `None` = 未命中任何候选（推断失准或候选外操作）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alt_candidate: Option<String>,
}

/// 状态差证据（锚点快照 → 下一回合首份快照）
#[derive(Debug, Default, Clone, Serialize)]
pub struct Evidence {
    /// 五维增量
    pub five_status_delta: Array5,
    pub vital_delta: i32,
    pub motivation_delta: i32,
    /// `raceHistory` 长度增量（只记跑赢）
    pub race_count_delta: i64,
    /// 友人出行次数增量
    pub friend_outgoing_delta: i32,
    /// `isIll` true → false
    pub is_ill_cured: bool,
    /// 剧本 PT 增量
    pub scenario_pt_delta: i32,
    /// 诀窍队列长度增量
    pub feeling_stock_len_delta: i64,
    /// 超级拉面档位增量
    pub super_ramen_delta: i32,
}

/// 偏离/检查项命中（digest findings 块；execution 偏离 + M4 检查项共用）
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    /// 类型（`execution_mismatch` / `mandatory_race_not_won` / 检查项类型等）
    #[serde(rename = "type")]
    pub kind: String,
    pub turn: u32,
    /// 关联快照文件（检查项无关联文件时 `None`）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// 证据摘要（人可读）
    pub evidence: String,
    pub severity: String,
}

/// 推断结果 + 一致率统计
#[derive(Debug, Default, Clone)]
pub struct ExecutionResult {
    pub rows: Vec<ExecRow>,
    pub findings: Vec<Finding>,
    /// 可比行数（ai_choice 可映射）
    pub comparable: usize,
    /// 一致行数
    pub matched: usize,
}

/// 推断主入口
///
/// - `tl`：timeline 行（须按 (turn, seq) 升序，与 `timeline::build` 产出同序）
/// - `dec`：calc 决策行（CSV 原序 = 回合链序）
/// - `race_turns`：必赛回合（+ URA 决赛固定赛）——输掉的比赛 `raceHistory`
///   不记，用赛程表兜底判「比赛」
pub fn build(tl: &[TimelineRow], dec: &[DecRow], race_turns: &[i32]) -> ExecutionResult {
    // (turn, seq) → timeline 下标
    let idx: HashMap<(u32, u32), usize> =
        tl.iter().enumerate().map(|(i, r)| ((r.turn, r.seq), i)).collect();

    // 每回合最后一条 train 决策（BTreeMap 同 key 后写覆盖 → 保留链序最后一条）
    let mut anchor_by_turn: BTreeMap<u32, &DecRow> = BTreeMap::new();
    for r in dec {
        if r.decision_kind == "train" {
            anchor_by_turn.insert(r.turn, r);
        }
    }

    let mut rows = Vec::new();
    let mut findings = Vec::new();
    let mut comparable = 0usize;
    let mut matched = 0usize;

    for (turn, anchor) in &anchor_by_turn {
        let Some(&ai) = idx.get(&(*turn, anchor.seq)) else { continue };
        // 下一回合首份快照（timeline 升序 → 从锚点起首个 turn 更大的行）
        let Some(next) = tl.iter().skip(ai).find(|r| r.turn > *turn) else { continue };
        let a = &tl[ai];

        let ev = evidence(a, next);
        let min_train = min_train_delta(*turn);
        let actual = classify(&ev, *turn, race_turns, min_train, choice_attr(&anchor.chosen.desc));
        // 继承窗口混合：锚点 29/53 的窗口含 t+1 继承落地，行动与继承无法剥离
        // → 标注「继承混合」、不参与一致率（数据与证据照留）
        let inherit_mixed = INHERIT_TURNS.contains(&(*turn + 1));
        let (actual, matches) = if inherit_mixed {
            (format!("{actual}·继承混合"), None)
        } else {
            let ai_mapped = map_choice(&anchor.chosen.desc);
            let m = ai_mapped.map(|m| m == actual.as_str());
            (actual, m)
        };
        if matches.is_some() {
            comparable += 1;
        }
        if matches == Some(true) {
            matched += 1;
        }
        // 偏离一选 → 找实际动作对应的候选（#2 起；用户口径：偏离主要是选了 2-3 选）
        let alt_candidate = if matches == Some(false) {
            anchor
                .candidates
                .iter()
                .filter(|c| c.rank >= 2)
                .find(|c| map_choice(&c.desc).is_some_and(|m| m == actual.as_str()))
                .map(|c| format!("#{} {}", c.rank, c.desc))
        } else {
            None
        };
        if matches == Some(false) {
            findings.push(Finding {
                kind: "execution_mismatch".to_string(),
                turn: *turn,
                file: Some(anchor.file.clone()),
                evidence: format!(
                    "ai_choice={} actual={}{} five_delta={:?} vital={} friend={} race={}",
                    anchor.chosen.desc,
                    actual,
                    alt_candidate
                        .as_ref()
                        .map(|s| format!("（对应候选 {s}）"))
                        .unwrap_or_default(),
                    ev.five_status_delta,
                    ev.vital_delta,
                    ev.friend_outgoing_delta,
                    ev.race_count_delta
                ),
                severity: "info".to_string(),
            });
        }
        rows.push(ExecRow {
            turn: *turn,
            stage: anchor.stage.clone(),
            ai_choice: anchor.chosen.desc.clone(),
            actual_action: actual,
            matches,
            evidence: ev,
            alt_candidate,
        });
    }

    ExecutionResult { rows, findings, comparable, matched }
}

/// 状态差证据（锚点快照 → 下一回合首份快照）
fn evidence(a: &TimelineRow, n: &TimelineRow) -> Evidence {
    let mut five = [0i32; 5];
    for i in 0..5 {
        five[i] = n.five_status[i] - a.five_status[i];
    }
    Evidence {
        five_status_delta: five,
        vital_delta: n.vital - a.vital,
        motivation_delta: n.motivation - a.motivation,
        race_count_delta: n.race_count as i64 - a.race_count as i64,
        friend_outgoing_delta: n.friend_outgoing_used - a.friend_outgoing_used,
        is_ill_cured: a.is_ill && !n.is_ill,
        scenario_pt_delta: n.scenario_pt - a.scenario_pt,
        feeling_stock_len_delta: n.feeling_stock.len() as i64 - a.feeling_stock.len() as i64,
        super_ramen_delta: n.super_ramen - a.super_ramen,
    }
}

/// 实际动作分类（优先级见模块头「分类口径」）
///
/// `ai_attr`：AI 建议的训练目标维（非训练建议 / 无法映射时为 `None`）——
/// 只用于**在候选维中挑选目标维**，不直接决定 actual_action，
/// 故「建议训练却练了别的维」仍会被判成偏离。
fn classify(ev: &Evidence, turn: u32, race_turns: &[i32], min_train: i32, ai_attr: Option<usize>) -> String {
    let five_total: i32 = ev.five_status_delta.iter().sum();
    // ① 比赛：跑赢（raceHistory +1）或必赛回合（输掉的比赛赛程表兜底）
    if ev.race_count_delta > 0 || race_turns.contains(&(turn as i32)) {
        return "比赛".to_string();
    }
    if ev.friend_outgoing_delta > 0 {
        return "友人出行".to_string();
    }
    if ev.is_ill_cured {
        return "治病".to_string();
    }
    // ② 先判是否训练了，再用体力形态否证
    // 目标维：推荐维优先（它是最大增量，或自身增量达 TRAIN_ATTR_MIN），否则取最大增量
    let dom = ev
        .five_status_delta
        .iter()
        .enumerate()
        .fold(0usize, |best, (i, &d)| if d > ev.five_status_delta[best] { i } else { best });
    let attr = match ai_attr {
        Some(a) if a == dom || ev.five_status_delta[a] >= TRAIN_ATTR_MIN => a,
        _ => dom,
    };
    let attr_v = ev.five_status_delta[attr];
    // 休息形态：体力大幅回升且目标维未达强证据 → 属性来自事件而非训练
    let rest_like = ev.vital_delta >= REST_MIN_VITAL && attr_v < TRAIN_ATTR_MIN;
    if attr_v >= min_train && !rest_like {
        return format!("{}训练", ATTR_NAMES[attr]);
    }
    // ③ 休息：体力大幅回复（容忍事件小额属性）；未达阈值的回体不判休息
    if ev.vital_delta >= REST_MIN_VITAL {
        return "休息".to_string();
    }
    if ev.motivation_delta > 0 && five_total == 0 {
        return "出行".to_string();
    }
    if ev.scenario_pt_delta != 0 || ev.feeling_stock_len_delta != 0 || ev.super_ramen_delta != 0
    {
        return "剧本".to_string();
    }
    "未知".to_string()
}

/// AI 建议描述 → 训练目标维下标（非训练建议 / 无具体维时返回 `None`）
fn choice_attr(desc: &str) -> Option<usize> {
    let mapped = map_choice(desc)?;
    ATTR_NAMES.iter().position(|n| mapped.starts_with(n))
}

/// AI 建议描述 → 动作类别（关键词匹配，覆盖实测 desc 词表：
/// `速/耐/力/根/智训练` / `休息` / `普通出行` / `友人出行` / `比赛` / `治病` /
/// `吃面`；无法映射返回 `None` → matches = null）
fn map_choice(desc: &str) -> Option<&'static str> {
    if desc.contains("训练") {
        Some(match () {
            _ if desc.contains("速") => "速训练",
            _ if desc.contains("耐") => "耐训练",
            _ if desc.contains("力") => "力训练",
            _ if desc.contains("根") => "根训练",
            _ if desc.contains("智") => "智训练",
            _ => "训练"
        })
    } else if desc.contains("休息") {
        Some("休息")
    } else if desc.contains("友人") {
        Some("友人出行")
    } else if desc.contains("出行") || desc.contains("外出") {
        Some("出行")
    } else if desc.contains("治病") || desc.contains("治疗") {
        Some("治病")
    } else if desc.contains("比赛") || desc.contains("参赛") {
        Some("比赛")
    } else if desc.contains("吃面") {
        Some("吃面")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试 timeline 行（只填推断相关字段）
    fn tl_row(
        turn: u32,
        seq: u32,
        five: [i32; 5],
        vital: i32,
        motivation: i32,
        friend: i32,
        race: usize,
        ill: bool,
    ) -> TimelineRow {
        TimelineRow {
            turn,
            seq,
            stage: "Train".to_string(),
            reason: None,
            source: "command".to_string(),
            playing_state: 1,
            vital,
            max_vital: 100,
            motivation,
            five_status: five,
            five_status_display: five,
            five_status_limit: [3000; 5],
            skill_pt: 0,
            train_level_count: [1; 5],
            friend_outgoing_used: friend,
            selected_regions: vec![1, 2, 3],
            scenario_pt: 0,
            feeling_stock: vec![],
            super_ramen: -1,
            is_ill: ill,
            is_qiezhe: false,
            is_xiao_qie: false,
            race_count: race,
            absent_persons: vec![],
        }
    }

    /// 构造测试决策行（train 决策）
    fn dec_row(turn: u32, seq: u32, chosen: &str) -> DecRow {
        DecRow {
            file: format!("f{turn}_{seq}.json"),
            turn,
            seq,
            stage: "Train".to_string(),
            decision_kind: "train".to_string(),
            candidates: vec![],
            chosen: crate::decisions::Chosen {
                idx: Some(0),
                desc: chosen.to_string(),
                action_luck: None,
            },
            t_n_raw: None,
            t_n_display: None,
            total_luck: None,
            turn_delta: None,
            chain_len: 1,
            outcome: "calc".to_string(),
            reason: String::new(),
            step: 0,
        }
    }

    /// 训练 / 休息 / 友人出行 / 比赛（跑赢 + 必赛兜底）/ 未映射 各分支 + findings
    #[test]
    fn test_execution_inference() {
        let tl = vec![
            // turn 5：行动前（速 100，体力 80）→ turn 6：速 +20、体力 -20 → 速训练
            tl_row(5, 0, [100, 100, 100, 100, 100], 80, 4, 1, 2, false),
            tl_row(6, 0, [120, 100, 100, 100, 100], 60, 4, 1, 2, false),
            // turn 7 → 8：休息（体力 +30、五维不变）
            tl_row(7, 0, [120, 100, 100, 100, 100], 40, 4, 1, 2, false),
            tl_row(8, 0, [120, 100, 100, 100, 100], 70, 4, 1, 2, false),
            // turn 9 → 10：友人出行（friend +1、无五维）
            tl_row(9, 0, [120, 100, 100, 100, 100], 70, 3, 1, 2, false),
            tl_row(10, 0, [120, 100, 100, 100, 100], 70, 5, 2, 2, false),
            // turn 11 → 12：比赛跑赢（race +1）
            tl_row(11, 0, [120, 100, 100, 100, 100], 70, 4, 2, 2, false),
            tl_row(12, 0, [120, 100, 100, 100, 100], 70, 4, 2, 3, false),
            // turn 13 → 14：必赛回合输了（race 不变，赛程表兜底判比赛）
            tl_row(13, 0, [120, 100, 100, 100, 100], 70, 4, 2, 3, false),
            tl_row(14, 0, [127, 100, 100, 100, 107], 70, 4, 2, 3, false),
            // turn 15 → 16：智训练（智 +32、体力 +10 —— 智训练不耗体力）
            tl_row(15, 0, [127, 100, 100, 100, 107], 70, 4, 2, 3, false),
            tl_row(16, 0, [135, 100, 100, 100, 139], 80, 4, 2, 3, false),
            // turn 17 → 18：休息 + 事件小额属性（速 +8 达第 1 年阈值但体力大增 → 休息）
            tl_row(17, 0, [135, 100, 100, 100, 139], 40, 4, 2, 3, false),
            tl_row(18, 0, [143, 100, 100, 100, 139], 90, 4, 2, 3, false),
        ];
        let dec = vec![
            dec_row(5, 0, "速训练"),   // 一致
            dec_row(7, 0, "休息"),     // 一致
            dec_row(9, 0, "普通出行"), // 实际友人出行 → 偏离
            dec_row(11, 0, "速训练"),  // 实际比赛 → 偏离
            dec_row(13, 0, "比赛"),    // 必赛兜底 → 一致
            dec_row(15, 0, "智训练"),  // 智训练不耗体力 → 一致
            dec_row(17, 0, "休息"),    // 休息+事件属性 → 休息（防误判训练）
        ];
        let r = build(&tl, &dec, &[13]);
        for row in &r.rows {
            println!(
                "turn={} ai={} actual={} match={:?} ev={:?}",
                row.turn, row.ai_choice, row.actual_action, row.matches, row.evidence
            );
        }
        println!("findings: {:#?}", r.findings);
        assert_eq!(r.rows.len(), 7);
        assert_eq!(r.rows[0].actual_action, "速训练");
        assert_eq!(r.rows[0].matches, Some(true));
        assert_eq!(r.rows[1].actual_action, "休息");
        assert_eq!(r.rows[1].matches, Some(true));
        assert_eq!(r.rows[2].actual_action, "友人出行");
        assert_eq!(r.rows[2].matches, Some(false), "普通出行 vs 友人出行应偏离");
        assert_eq!(r.rows[3].actual_action, "比赛");
        assert_eq!(r.rows[3].matches, Some(false), "速训练 vs 比赛应偏离");
        assert_eq!(r.rows[4].actual_action, "比赛", "必赛回合兜底判比赛（输掉不记 raceHistory）");
        assert_eq!(r.rows[4].matches, Some(true));
        assert_eq!(r.rows[5].actual_action, "智训练", "智训练不耗体力（vital +10）仍判训练");
        assert_eq!(r.rows[5].matches, Some(true));
        assert_eq!(
            r.rows[6].actual_action, "休息",
            "休息+事件属性（速 +8 达第 1 年阈值但体力 +50）应判休息"
        );
        assert_eq!(r.rows[6].matches, Some(true));
        assert_eq!(r.comparable, 7);
        assert_eq!(r.matched, 5);
        assert_eq!(r.findings.len(), 2);
        assert!(r.findings.iter().all(|f| f.kind == "execution_mismatch"));
    }

    /// 偏离一选 → 对应候选标注：实际动作命中 #2/#3 时给出编号（用户口径：偏离主要是选了 2-3 选）
    #[test]
    fn test_alt_candidate_annotation() {
        let tl = vec![
            // t5 速 100 → t6 力 +30（力训练），建议为速训练（一选）
            tl_row(5, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(6, 0, [100, 100, 130, 100, 100], 60, 4, 0, 0, false),
            // t7 智 100 → t8 智 +32（智训练），建议为速训练，候选表无智训练
            tl_row(7, 0, [100, 100, 130, 100, 100], 70, 4, 0, 0, false),
            tl_row(8, 0, [100, 100, 130, 100, 132], 60, 4, 0, 0, false),
        ];
        let mut with_alt = dec_row(5, 0, "速训练");
        with_alt.candidates = vec![
            crate::decisions::Cand { rank: 1, desc: "速训练".to_string(), score: None, n: None, gap_to_best: None },
            crate::decisions::Cand { rank: 2, desc: "力训练".to_string(), score: None, n: None, gap_to_best: None },
            crate::decisions::Cand { rank: 3, desc: "耐训练".to_string(), score: None, n: None, gap_to_best: None },
        ];
        let mut no_alt = dec_row(7, 0, "速训练");
        no_alt.candidates = vec![
            crate::decisions::Cand { rank: 1, desc: "速训练".to_string(), score: None, n: None, gap_to_best: None },
            crate::decisions::Cand { rank: 2, desc: "耐训练".to_string(), score: None, n: None, gap_to_best: None },
        ];
        let r = build(&tl, &[with_alt, no_alt], &[]);
        for row in &r.rows {
            println!("turn={} ai={} actual={} alt={:?}", row.turn, row.ai_choice, row.actual_action, row.alt_candidate);
        }
        assert_eq!(r.rows[0].matches, Some(false));
        assert_eq!(r.rows[0].alt_candidate.as_deref(), Some("#2 力训练"), "实际力训练命中候选 #2");
        assert_eq!(r.rows[1].matches, Some(false));
        assert_eq!(r.rows[1].alt_candidate, None, "候选表无智训练 → 不标注");
        // finding 证据带候选编号
        assert!(r.findings[0].evidence.contains("（对应候选 #2 力训练）"));
        assert!(!r.findings[1].evidence.contains("对应候选"));
    }

    /// 锚点规则：同回合多条 train 行取最后一条（RamenSelect 初判 → Train 最终）
    #[test]
    fn test_anchor_last_train_row() {
        let tl = vec![
            tl_row(5, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(5, 1, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(6, 0, [130, 100, 100, 100, 100], 55, 4, 0, 0, false),
        ];
        let mut first = dec_row(5, 0, "耐训练"); // RamenSelect 初判（seq 0）
        first.stage = "RamenSelect".to_string();
        let final_row = dec_row(5, 1, "速训练"); // Train 最终（seq 1）
        let r = build(&tl, &[first, final_row], &[]);
        println!("锚点: ai={} stage={}", r.rows[0].ai_choice, r.rows[0].stage);
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0].ai_choice, "速训练", "取链序最后一条 train");
        assert_eq!(r.rows[0].stage, "Train");
        assert_eq!(r.rows[0].matches, Some(true));
    }

    /// ai_choice 无法映射 → matches = None（不计入一致率）
    #[test]
    fn test_unmappable_choice() {
        let tl = vec![
            tl_row(5, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(6, 0, [120, 100, 100, 100, 100], 60, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(5, 0, "超级迷惑操作")];
        let r = build(&tl, &dec, &[]);
        println!("未映射: match={:?}", r.rows[0].matches);
        assert_eq!(r.rows[0].matches, None);
        assert_eq!(r.comparable, 0);
        assert_eq!(r.matched, 0);
        assert!(r.findings.is_empty(), "未映射不产生 findings");
    }

    /// 开局低加成训练（智 +12，低于文档 +13~+134 下限）仍判训练
    /// （game1444 turn 0 实测形态，阈值取 12 的依据）
    #[test]
    fn test_opening_small_train() {
        let tl = vec![
            tl_row(0, 0, [3, 0, 0, 0, 0], 30, 4, 0, 0, false),
            tl_row(1, 0, [6, 0, 0, 0, 12], 30, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(0, 0, "智训练")];
        let r = build(&tl, &dec, &[]);
        println!("开局小训练: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert_eq!(r.rows[0].actual_action, "智训练", "智+12 应判训练（阈值 12）");
        assert_eq!(r.rows[0].matches, Some(true));
    }

    /// 继承窗口混合：锚点 29/53 的窗口含继承落地 → 标注「继承混合」、
    /// 不参与一致率、不产偏离 finding
    #[test]
    fn test_inherit_mixed_window() {
        // turn 29 → 30：全维大涨（行动 + 继承落地混合形态）
        let tl = vec![
            tl_row(29, 0, [1000, 1000, 1000, 1000, 1000], 80, 4, 0, 0, false),
            tl_row(30, 0, [1108, 1010, 1126, 1081, 1049], 80, 4, 0, 0, false),
            tl_row(31, 0, [1108, 1010, 1126, 1081, 1049], 80, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(29, 0, "智训练")];
        let r = build(&tl, &dec, &[]);
        println!("继承窗口: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert!(
            r.rows[0].actual_action.ends_with("·继承混合"),
            "锚点 29 应标继承混合，实为 {}",
            r.rows[0].actual_action
        );
        assert_eq!(r.rows[0].matches, None, "继承混合不参与一致率");
        assert_eq!(r.comparable, 0);
        assert_eq!(r.matched, 0);
        assert!(r.findings.is_empty(), "继承混合不产偏离 finding");
    }

    /// 第 2 年起阈值提高到 12：事件级增益（速 +8）不再判训练
    #[test]
    fn test_after_y1_threshold() {
        let tl = vec![
            tl_row(25, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(26, 0, [108, 100, 100, 100, 100], 80, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(25, 0, "速训练")];
        let r = build(&tl, &dec, &[]);
        println!("第 2 年事件级: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert_ne!(r.rows[0].actual_action, "速训练", "速+8 < 12 不应判训练");
        assert_eq!(r.rows[0].matches, Some(false));
    }

    /// 推荐维优先（规则②）：推荐维不是最大增量，但达 TRAIN_ATTR_MIN → 判推荐维
    /// （多属性齐涨时纯 argmax 会误判成力训练，用户拍板修此）
    #[test]
    fn test_recommended_attr_above_threshold() {
        let tl = vec![
            tl_row(25, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            // 速 +25（达 20）、力 +40（最大）——事件给力多，但玩家练的是速
            tl_row(26, 0, [125, 100, 140, 100, 100], 60, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(25, 0, "速训练")];
        let r = build(&tl, &dec, &[]);
        println!("推荐维达阈值: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert_eq!(r.rows[0].actual_action, "速训练", "速+25 ≥ 20 应判速训练（不取最大的力）");
        assert_eq!(r.rows[0].matches, Some(true));
        assert!(r.findings.is_empty(), "不应产生偏离");
    }

    /// 推荐维未达阈值 → 回退取最大增量维，仍判偏离（不能因「推荐过」就判一致）
    #[test]
    fn test_recommended_attr_below_threshold() {
        let tl = vec![
            tl_row(25, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            // 速 +15（< 20）、力 +40 → 目标维取力
            tl_row(26, 0, [115, 100, 140, 100, 100], 60, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(25, 0, "速训练")];
        let r = build(&tl, &dec, &[]);
        println!("推荐维未达阈值: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert_eq!(r.rows[0].actual_action, "力训练", "速+15 < 20 不优先，取最大的力");
        assert_eq!(r.rows[0].matches, Some(false), "建议速却练力 → 偏离");
        assert_eq!(r.findings.len(), 1);
    }

    /// 智训练 vs 休息：智增量达 TRAIN_ATTR_MIN 时，即使体力也回升仍判智训练
    /// （智训练本身不耗体力，且事件可同时回体）
    #[test]
    fn test_int_train_vs_rest() {
        // 智 +32、体力 +30（事件回体）→ 智训练
        let tl = vec![
            tl_row(25, 0, [100, 100, 100, 100, 100], 40, 4, 0, 0, false),
            tl_row(26, 0, [100, 100, 100, 100, 132], 70, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(25, 0, "智训练")];
        let r = build(&tl, &dec, &[]);
        println!("智达阈值: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert_eq!(r.rows[0].actual_action, "智训练", "智+32 ≥ 20 应判智训练（体力回升不否证）");
        assert_eq!(r.rows[0].matches, Some(true));
    }

    /// 休息 vs 事件给智：智增量未达 TRAIN_ATTR_MIN 且体力大增 → 判休息
    #[test]
    fn test_rest_with_small_int_event() {
        // 智 +15（事件给的）、体力 +50 → 休息
        let tl = vec![
            tl_row(25, 0, [100, 100, 100, 100, 100], 40, 4, 0, 0, false),
            tl_row(26, 0, [100, 100, 100, 100, 115], 90, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(25, 0, "休息")];
        let r = build(&tl, &dec, &[]);
        println!("休息+事件给智: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert_eq!(r.rows[0].actual_action, "休息", "智+15 < 20 且体力 +50 → 休息");
        assert_eq!(r.rows[0].matches, Some(true));
    }

    /// 休息 vs 事件回体：体力回升未达 REST_MIN_VITAL 且无训练 → 不判休息
    /// （先判训练、再解释体力变化；小额回体落到出行 / 剧本 / 未知）
    #[test]
    fn test_small_vital_recovery_is_not_rest() {
        let tl = vec![
            tl_row(25, 0, [100, 100, 100, 100, 100], 40, 4, 0, 0, false),
            // 体力 +15（事件级）、五维不变、干劲不变 → 非休息
            tl_row(26, 0, [100, 100, 100, 100, 100], 55, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(25, 0, "休息")];
        let r = build(&tl, &dec, &[]);
        println!("小额回体: actual={} match={:?}", r.rows[0].actual_action, r.rows[0].matches);
        assert_ne!(r.rows[0].actual_action, "休息", "体力 +15 < 25 不判休息");
        assert_eq!(r.rows[0].matches, Some(false));
    }
}
