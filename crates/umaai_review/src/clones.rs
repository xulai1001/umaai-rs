//! 分身彩圈观测（文档 §6.5）
//!
//! 识别判据（§6.5.1，已验证 + game6234 实测修正）：
//! - persons 索引 0-5 = 6 张训练卡；训练位索引 0-4 同序：速/耐/力/根/智
//! - **分身** = 同一人物索引出现在不同训练位（同一训练位内不会重复人物；
//!   索引 8 是 NPC 占位符，多个 8 = 多个 NPC）
//! - **分身落位 = 末快照位置 − 本体位**（before ≤ 1 → 本体至多占一位；
//!   absent 卡的全部末位都是分身落位）
//! - **彩圈 = 分身的 cardType == 分身落位**（§6.5.1 ⑤「分身所在位」——
//!   **本体自己站的得意位不算**。game6234 实测：任意位口径 16/31 命中里
//!   14 个是本体占位假象，分身真实落得意位仅 2——「地区面多了很多彩圈」
//!   不成立，分身的主要价值是加人头）
//! - **有效增加彩圈的来源二分**（用户口径）：
//!   - 吃面前**本体在场** → 「随机有效增加」＝**好运气**（分身随机落位）
//!   - 吃面前**本体缺席** → 「规则有效增加」＝**好策略**（地区/超拉规则
//!     把缺席卡带入其得意位，归因于吃面/地区选择）
//! - **只统计「吃面后新增的分身」**：同一回合内比较吃面前（首个快照）与
//!   吃面后（末个快照）的 `personDistribution`，取「该卡位从 ≤1 次变为
//!   ≥2 次」的（§6.5.1 末段）
//!
//! A / B 两类**必须分开统计**（§6.5.2，机制与预期完全不同，混算双重失真）：
//! - **地区分身**（turn < 72）：真正的随机运气；评判判据**待精确定义**
//!   （§6.5.4），本块只出观测数据
//! - **超级拉面分身**（turn ≥ 72）：`finals_effect` 的 `clone_count`，
//!   机制保证落得意位命中率 100%——**只统计训练卡（cardType 0-4）**：
//!   友人卡 cardType=5 无得意位可落，该保证对它不成立（game6234 实测排除
//!   友人卡后 14/14 落得意位，对齐文档 §6.5.5）；「没吃到」不算亏，只能用
//!   该期运气分判盈亏（§6.5.3 / §6.6）

use serde::Serialize;

use umaai::protocol::ramen::GameStatusRamen;

use crate::{checks::SUPER_RAMEN_START, execution::ExecutionResult, pack::SnapEntry};

/// 五维属性名（训练位索引同序）
const ATTR_NAMES: [&str; 5] = ["速", "耐", "力", "根", "智"];

/// 分身观测块（digest clones）
#[derive(Debug, Default, Clone, Serialize)]
pub struct ClonesBlock {
    /// A 类（turn < 72，地区分身）——判据待定义，观测数据
    pub region: CloneClass,
    /// B 类（turn ≥ 72，超级拉面分身）——机制保证落得意位
    pub super_ramen_clones: CloneClass,
    /// 地区分身逐次彩圈明细（供判据标定与 LLM 复核）
    pub region_per_turn: Vec<CloneTurn>
}

/// 一类分身的汇总统计
#[derive(Debug, Default, Clone, Serialize)]
pub struct CloneClass {
    /// 新增分身数（卡 × 回合）
    pub new_clones: u32,
    /// 分身落得意位（新增落位 == cardType；本体位不算）
    pub rainbow_clones: u32,
    /// 分身落得意位且该位被当回合实际训练
    pub trained_clones: u32,
    /// 有效增加彩圈 · 随机（吃面前本体在场 → 好运气）
    pub rainbow_luck: u32,
    /// 有效增加彩圈 · 规则（吃面前本体缺席 → 好策略）
    pub rainbow_strategy: u32
}

/// 单回合的新增分身明细（地区分身）
#[derive(Debug, Default, Clone, Serialize)]
pub struct CloneTurn {
    pub turn: u32,
    pub cards: Vec<CloneCard>
}

/// 单张卡的新增分身明细
#[derive(Debug, Default, Clone, Serialize)]
pub struct CloneCard {
    /// 支援卡索引 0-5
    pub card: u32,
    /// 分身落位（末快照位置 − 本体位；absent 卡 = 全部末位）
    pub positions: Vec<u32>,
    /// 彩圈位（分身落位 == cardType；本体位不算）
    pub rainbow_positions: Vec<u32>,
    /// 有效增加彩圈的来源（仅彩圈命中时有值）：
    /// `luck` = 随机有效增加（吃面前本体在场 → 好运气）；
    /// `strategy` = 规则有效增加（吃面前本体缺席 → 好策略）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// 彩圈是否吃到（仅彩圈命中时有值）：新增回合的当回合实际训练命中彩圈位
    /// （速位彩圈 + 速训练 = 吃到；用户口径，未考虑分身存续期的后续训练）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used: Option<bool>
}

/// 地区分身「产生了彩圈」的逐次明细（**已预格式化**；report.html 与 brief.md 共用）
///
/// 只收彩圈事件（`rainbow_positions` 非空），逐 (回合, 卡) 一行 + 一句文字说明。
#[derive(Debug, Clone, Serialize)]
pub struct CloneDetailRow {
    pub turn: u32,
    /// 支援卡名（来自卡组；缺名时回退 `card{N}`）
    pub card: String,
    /// 彩圈所在训练位（速 / 耐 / 力 / 根 / 智；多命中用 `/` 连）
    pub train: String,
    /// 来源：运气好（随机有效增加）/ 规则性增加（好策略）
    pub origin: String,
    /// 玩家当回合是否选了该训练：是 / 否
    pub used: String,
    /// 一句文字说明：哪张卡去了哪个训练产生彩圈、来源、玩家选了没有
    pub text: String,
}

/// 五维训练位名（索引同序）
const TRAIN_POS: [&str; 5] = ["速", "耐", "力", "根", "智"];

impl ClonesBlock {
    /// 地区分身逐次彩圈明细（**只收产生了彩圈的**；逐 (回合, 卡) 一行 + 文字说明）
    ///
    /// 行数应等于 `region.rainbow_clones`，对不上说明取数漏了（见 pitfalls 第 9 条）。
    /// `deck_names`：卡组卡名（`meta.deck[].name`，按卡索引取；缺则回退 `card{N}`）。
    pub fn region_detail_rows(&self, deck_names: &[String]) -> Vec<CloneDetailRow> {
        let mut out = Vec::new();
        for ct in &self.region_per_turn {
            for c in &ct.cards {
                if c.rainbow_positions.is_empty() {
                    continue;
                }
                let name = deck_names
                    .get(c.card as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("card{}", c.card));
                let train = c
                    .rainbow_positions
                    .iter()
                    .map(|p| TRAIN_POS.get(*p as usize).copied().unwrap_or("?"))
                    .collect::<Vec<_>>()
                    .join("/");
                // origin 缺省按「运气好」处理（彩圈项才有 origin，非彩圈已被过滤）
                let by_luck = c.origin.as_deref() != Some("strategy");
                let (origin, cause) = if by_luck {
                    ("运气好", "分身随机落到了该卡的得意位")
                } else {
                    ("规则性增加", "地区/超拉规则把缺席的卡带入了得意位")
                };
                let (used, pick) = match c.used {
                    Some(true) => ("是", format!("玩家当回合选了{train}训练，吃到了这个彩圈")),
                    Some(false) => ("否", format!("玩家当回合没选{train}训练，没吃到")),
                    None => ("—", "当回合训练无法判定".to_string()),
                };
                out.push(CloneDetailRow {
                    turn: ct.turn,
                    card: name.clone(),
                    train: train.clone(),
                    origin: origin.to_string(),
                    used: used.to_string(),
                    text: format!(
                        "t{}：「{}」的分身落到{}位（得意位），{}，属{}；{}",
                        ct.turn, name, train, cause, origin, pick
                    ),
                });
            }
        }
        out
    }
}

/// 组装分身观测块
///
/// - `card_types`：卡组 6 张卡的 cardType（0速/1耐/2力/3根/4智/5友人；来自
///   `SupportCard::new(idrank)`，gamedata 缺失时传 `None` → 整块跳过）
/// - `exec`：实际执行推断（「被训练」判定用当回合实际动作）
pub fn build(snaps: &[SnapEntry], card_types: Option<&[i32]>, exec: &ExecutionResult) -> Option<ClonesBlock> {
    let types = card_types?;
    let mut block = ClonesBlock::default();

    // 每回合（首份 = 吃面前，末份 = 吃面后）的 personDistribution
    let mut groups: Vec<(u32, Vec<Vec<i32>>, Vec<Vec<i32>>)> = Vec::new();
    per_turn_first_last(snaps, |turn, first, last| {
        groups.push((turn, first, last));
    });

    for (turn, first, last) in groups {
        // 每张卡（0-5）：出现次数（所有训练位合计）
        for card in 0..6u32 {
            let before = count_person(&first, card);
            let after = count_person(&last, card);
            if before <= 1 && after >= 2 {
                // 本体位（before ≤ 1 → 至多一位；absent 卡为 None）
                let base_pos = positions_of(&first, card).first().copied();
                // 分身落位 = 末快照位置 − 本体位
                let positions: Vec<u32> = positions_of(&last, card)
                    .into_iter()
                    .filter(|&p| Some(p) != base_pos)
                    .collect();
                let ct = types.get(card as usize).copied().unwrap_or(-1);
                // 彩圈 = 分身落位 == cardType（本体位不算，§6.5.1 ⑤）
                let rainbow_positions: Vec<u32> = positions
                    .iter()
                    .copied()
                    .filter(|&p| (p as i32) == ct)
                    .collect();
                // 被训练：彩圈落位被当回合实际动作训练（前缀匹配——继承窗口
                // 的 actual_action 带「·继承混合」后缀，精确匹配会漏判）
                let trained = exec.rows.iter().any(|r| {
                    r.turn == turn
                        && rainbow_positions.iter().any(|&p| {
                            r.actual_action
                                .starts_with(&format!("{}训练", ATTR_NAMES[p as usize]))
                        })
                });
                // 有效增加彩圈的来源：本体在场 → 随机（好运气）；
                // 本体缺席 → 规则带入（好策略）
                let origin = if rainbow_positions.is_empty() {
                    None
                } else if before == 1 {
                    Some("luck".to_string())
                } else {
                    Some("strategy".to_string())
                };
                let entry = CloneClass {
                    new_clones: 1,
                    rainbow_clones: u32::from(!rainbow_positions.is_empty()),
                    trained_clones: u32::from(trained),
                    rainbow_luck: u32::from(origin.as_deref() == Some("luck")),
                    rainbow_strategy: u32::from(origin.as_deref() == Some("strategy"))
                };
                if turn < SUPER_RAMEN_START {
                    // A 类：判据待定义，全量观测（实测无友人卡分身，与文档 31 同口径）
                    let used = if rainbow_positions.is_empty() { None } else { Some(trained) };
                    block.region.add(&entry);
                    block.region_per_turn.push(CloneTurn {
                        turn,
                        cards: vec![CloneCard {
                            card,
                            positions,
                            rainbow_positions,
                            origin,
                            used
                        }]
                    });
                } else if (0..=4).contains(&ct) {
                    // B 类：只统计训练卡（cardType 0-4）——友人卡 cardType=5 无得意位
                    // 可落（game6234 实测：排除友人卡后新增 14，对齐文档 §6.5.5；
                    // 彩圈按分身落位口径实测 1/14——「机制保证 100% 落得意位」的
                    // 旧说法基于任意位口径，不成立）
                    block.super_ramen_clones.add(&entry);
                }
            }
        }
    }
    // 地区分身逐次彩圈明细按回合合并（同回合多张卡）
    merge_per_turn(&mut block);
    Some(block)
}

/// 组装辅助（分组回调：每回合首末 personDistribution）
fn per_turn_first_last(
    snaps: &[SnapEntry],
    mut f: impl FnMut(u32, Vec<Vec<i32>>, Vec<Vec<i32>>)
) {
    let mut cur_turn: Option<u32> = None;
    let mut first: Option<Vec<Vec<i32>>> = None;
    let mut last: Vec<Vec<i32>> = Vec::new();
    for s in snaps {
        let Ok(st) = serde_json::from_slice::<GameStatusRamen>(&s.bytes) else {
            continue;
        };
        let dist = st.base_game.person_distribution.clone();
        if cur_turn != Some(s.turn) {
            if let (Some(t), Some(fi)) = (cur_turn, first.take()) {
                f(t, fi, last.clone());
            }
            cur_turn = Some(s.turn);
            first = Some(dist.clone());
        }
        last = dist;
    }
    if let (Some(t), Some(fi)) = (cur_turn, first) {
        f(t, fi, last);
    }
}

/// 人物索引在分布中的总出现次数（所有训练位合计）
fn count_person(dist: &[Vec<i32>], person: u32) -> i32 {
    dist.iter().map(|slot| slot.iter().filter(|&&p| p == person as i32).count() as i32).sum()
}

/// 人物索引出现的训练位列表
fn positions_of(dist: &[Vec<i32>], person: u32) -> Vec<u32> {
    dist.iter()
        .enumerate()
        .filter(|(_, slot)| slot.contains(&(person as i32)))
        .map(|(i, _)| i as u32)
        .collect()
}

impl CloneClass {
    /// 累加单条统计
    fn add(&mut self, other: &CloneClass) {
        self.new_clones += other.new_clones;
        self.rainbow_clones += other.rainbow_clones;
        self.trained_clones += other.trained_clones;
        self.rainbow_luck += other.rainbow_luck;
        self.rainbow_strategy += other.rainbow_strategy;
    }
}

/// 同回合多张卡的明细合并为一行
fn merge_per_turn(block: &mut ClonesBlock) {
    let mut merged: Vec<CloneTurn> = Vec::new();
    for t in std::mem::take(&mut block.region_per_turn) {
        match merged.last_mut() {
            Some(last) if last.turn == t.turn => last.cards.extend(t.cards),
            _ => merged.push(t)
        }
    }
    block.region_per_turn = merged;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造带 personDistribution 的快照（GameStatusBase 必填字段全覆盖）
    fn snap_json(dist: &str) -> String {
        format!(
            r#"{{
                "baseGame": {{
                    "scenarioId": 14, "umaId": 1, "umaStar": 5, "turn": 1,
                    "vital": 80, "maxVital": 100, "motivation": 4,
                    "fiveStatus": [100, 100, 100, 100, 100],
                    "fiveStatusLimit": [1200, 1200, 1200, 1200, 1200],
                    "skillPt": 0, "skillScore": 0, "totalHints": 0,
                    "trainLevelCount": [1, 1, 1, 1, 1],
                    "ptScoreRate": 2.0, "failureRateBias": 0,
                    "isIll": false, "isQieZhe": false, "isAiJiao": false,
                    "isXiaoQie": false, "PositiveThinkingCount": 0, "isRefreshMind": false, "LuckyCount": 0,
                    "zhongMaBlueCount": [0, 0, 0, 0, 0], "isRacing": false,
                    "cardId": [], "persons": [],
                    "personDistribution": {dist},
                    "lockedTrainingId": -1,
                    "friendship_noncard_yayoi": 0, "friendship_noncard_reporter": 0,
                    "friend_stage": 0, "friend_outgoingUsed": 0,
                    "playing_state": 1, "raceHistory": [], "story": null,
                    "source": "command"
                }},
                "ramen": {{}}
            }}"#
        )
    }

    /// 构造 SnapEntry
    fn snap(file: &str, turn: u32, seq: u32, dist: &str) -> SnapEntry {
        SnapEntry {
            file: file.to_string(),
            game: 1,
            turn,
            seq,
            bytes: snap_json(dist).into_bytes()
        }
    }

    /// 地区分身（真实彩圈 + 本体占位假象反例）+ B 类分离统计（排除友人卡）
    #[test]
    fn test_clones_ab_split() {
        // turn 10（A 类）：卡1(t1 耐) 本体位0 → 分身落位1（耐位 = 彩圈）；
        // 本体在场 → luck（随机有效增加，好运气）；当回合实际耐训练 → 被训练
        // turn 12（A 类）：卡1(t1 耐) 本体位1（=得意位）→ 分身落位2；
        // **本体占的得意位不算彩圈**（game6234 实测 14/16 假象的反例钉死）
        // turn 14（A 类）：卡4(t0 速) 吃面前缺席 → 分身落位[0,2]，位0 = 彩圈；
        // 本体缺席 → strategy（规则有效增加，好策略）
        // turn 74（B 类）：卡2(t4 智) 本体位2 → 分身落位4（智位 = 彩圈）→ luck；
        // 卡5（友人）新增分身（位3+位4）但 cardType=5 → B 类不计
        let snaps = vec![
            snap("f1.json", 10, 0, "[[1],[8],[8],[3],[5]]"),
            snap("f2.json", 10, 1, "[[1],[1,8],[8],[3],[5]]"),
            snap("f3.json", 12, 0, "[[8],[1],[8],[8],[5]]"),
            snap("f4.json", 12, 1, "[[8],[1],[8,1],[8],[5]]"),
            snap("f5.json", 14, 0, "[[0],[8],[8],[8],[5]]"),
            snap("f6.json", 14, 1, "[[0,4],[8],[4,8],[8],[5]]"),
            snap("f7.json", 74, 0, "[[0],[1],[2],[8],[5]]"),
            snap("f8.json", 74, 1, "[[0],[1],[2,4],[8,5],[2,5]]"),
        ];
        let card_types: Vec<i32> = vec![0, 1, 4, 3, 0, 5];
        let exec = ExecutionResult {
            rows: vec![crate::execution::ExecRow {
                turn: 10,
                stage: "Train".to_string(),
                ai_choice: "耐训练".to_string(),
                // 继承窗口的 actual_action 带「·继承混合」后缀（matches 不参与一致率）
                actual_action: "耐训练·继承混合".to_string(),
                matches: None,
                evidence: Default::default(),
                alt_candidate: None
            }],
            ..Default::default()
        };
        let block = build(&snaps, Some(&card_types), &exec).expect("应产出观测块");
        println!("clones: {block:#?}");
        // A 类：3 个新增分身，彩圈命中 2（卡1@turn10 luck / 卡4@turn14 strategy），
        // 卡1@turn12 的命中位是本体位 → 不算彩圈
        assert_eq!(block.region.new_clones, 3);
        assert_eq!(block.region.rainbow_clones, 2, "本体占的得意位不算彩圈");
        assert_eq!(block.region.trained_clones, 1, "耐位彩圈被当回合耐训练");
        assert_eq!(block.region.rainbow_luck, 1, "turn10 本体在场 → 随机（好运气）");
        assert_eq!(block.region.rainbow_strategy, 1, "turn14 本体缺席 → 规则（好策略）");
        assert_eq!(block.region_per_turn.len(), 3);
        assert_eq!(block.region_per_turn[0].turn, 10);
        assert_eq!(block.region_per_turn[0].cards[0].positions, vec![1], "分身落位 = 末位 − 本体位");
        assert_eq!(block.region_per_turn[0].cards[0].rainbow_positions, vec![1]);
        assert_eq!(block.region_per_turn[0].cards[0].origin.as_deref(), Some("luck"));
        assert_eq!(block.region_per_turn[0].cards[0].used, Some(true), "turn10 耐位彩圈被当回合耐训练（含·继承混合后缀）→ 吃到");
        assert_eq!(block.region_per_turn[1].turn, 12);
        assert_eq!(block.region_per_turn[1].cards[0].positions, vec![2], "本体位1 被剔除");
        assert!(block.region_per_turn[1].cards[0].rainbow_positions.is_empty(), "分身落力位非彩圈");
        assert_eq!(block.region_per_turn[1].cards[0].origin, None, "非彩圈命中无来源");
        assert_eq!(block.region_per_turn[1].cards[0].used, None, "非彩圈无 used");
        assert_eq!(block.region_per_turn[2].turn, 14);
        assert_eq!(block.region_per_turn[2].cards[0].rainbow_positions, vec![0]);
        assert_eq!(block.region_per_turn[2].cards[0].origin.as_deref(), Some("strategy"));
        assert_eq!(block.region_per_turn[2].cards[0].used, Some(false), "turn14 无当回合速训练 → 没吃到");
        // B 类：卡2 落智位彩圈（本体在场 → luck）；友人卡不计；turn 74 无 exec 行
        assert_eq!(block.super_ramen_clones.new_clones, 1);
        assert_eq!(block.super_ramen_clones.rainbow_clones, 1);
        assert_eq!(block.super_ramen_clones.rainbow_luck, 1);
        assert_eq!(block.super_ramen_clones.rainbow_strategy, 0);
        assert_eq!(block.super_ramen_clones.trained_clones, 0);
    }

    /// 无变化（同回合首末相同）→ 无新增；gamedata 缺失 → None
    #[test]
    fn test_clones_no_change_and_degraded() {
        let snaps = vec![
            snap("f1.json", 10, 0, "[[1],[8],[8],[3],[5]]"),
            snap("f2.json", 10, 1, "[[1],[8],[8],[3],[5]]"),
        ];
        let block = build(&snaps, Some(&[0, 1, 2, 3, 4, 5]), &Default::default());
        let b = block.expect("应产出");
        println!("无变化: {b:?}");
        assert_eq!(b.region.new_clones, 0);
        let none = build(&snaps, None, &Default::default());
        assert!(none.is_none(), "card_types 缺失 → 整块跳过");
    }
}
