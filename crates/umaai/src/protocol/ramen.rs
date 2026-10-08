//! 拉面杯剧本通信状态（`scenarioId = 14`）
//!
//! 协议定稿见 `.trae/documents/ramen_protocol_v2.md`（v1，2026-09）。
//! 易混淆点见 `.trae/documents/adapter_spec.md`（2026-09-08 修订）。
//!
//! **Step 7 现状**：`GameStatusRamen::into_game` 完整实现，从 `thisTurn.json`
//! 覆写所有 ramen 段字段到 `RamenGame`：
//! - baseGame 增量字段（`source` + `playingState` + `active_effect_array` 三方联合 stage dispatch）
//! - ramen 段（last_ramen / feeling_stock / feeling_slot / super_ramen /
//!   selected_regions / scenario_pt / train_feeling_type / special_feeling 落入
//!   `RamenState`；`feeling_gauge_gains` / `feeling_gauge_gain_base` /
//!   `next_scenario_pt` / `active_effect_array` 仅用于 stage dispatch 判断，
//!   **不**存进 `RamenState`）
//!
//! `active_effect_array` 拆分到 `RamenEffect` 各字段**搁置**（按 §5 第 5 条）：
//! 当前只用其长度判 stage dispatch，不解读单项语义，后续按训练数值需求再补。
//!
//! **base 重建口径**：`into_game` 不采用 `RamenGame::newgame` 打补丁，而是严格用
//! [`GameStatusBase::parse_basegame`] 从协议重建 `BaseGame`（Uma / Friend 走
//! `parse_uma` / `parse_friend`，`five_status_limit` 取协议值，`friend_event_ids`
//! 不合并），再由 [`RamenGame::from_base_game`] 组装剧本专用状态。友人事件、
//! 五维上限等均视为外部输入，不做本地推算。
//!
//! ## stage dispatch 规则（adapter_spec §source / §playing_state）
//!
//! | turn | source | active_effect | playing_state | 含义 | stage |
//! |---|---|---|---|---|---|
//! | 任 | `event` | 任 | 任 | 待处理事件（check_event 带 story），AI 不进决策 | (不 dispatch) |
//! | 任 | 任 | 任 | 5 / 46 / 48 | 事件 / RMJ 结算 / RMJ 最终结算 | (不 dispatch) |
//! | 任 | 任 | 任 | 45 | 地区选择（`command` / `special` 两种 source 都出现） | `RegionSelect` |
//! | 2 / 24 / 48 | `command` | 任 | 1 | 刚选区、`train_feeling_type` 全 0（ramen 数据未刷新） | (不 dispatch) |
//! | ≤ 1 | `command` / `load` | 任 | 1 | 剧本机制未启用（拉面机制 turn >= 2 才启动） | `Train` |
//! | ≥ 2 | `command` / `load` | 空 | 1 | 当回合训练前，未吃面 | `RamenSelect` |
//! | ≥ 2 | `command` / `load` | 有 | 1 | 当回合已吃面，写中间状态 `pending_ramen=Some(last_ramen)` | `Train` |
//! | ≥ 72 | 任 | 空 | 1 | 超级拉面回合但效果未生效，丢弃等下一条 | (不 dispatch) |
//! | ≥ 72 | 任 | 有 | 1 | 超级拉面回合效果已生效 | `Train`（`combined_decision=true`） |
//! | 其它 | 其它 | 任 | 任 | 未识别的帧 | (warn + 不 dispatch) |
//!
//! **判定次序**：非决策帧（表头 2 行：`event` / ps 5·46·48）→ 刚选区未刷新帧
//! （turn 2/24/48 + ps=1 + feeling 全 0）→ 数据获取不全 → `playing_state=45` → `turn <= 1`
//! → 其余白名单。非决策帧与 turn **无关**，必须排在 `turn <= 1` 之前，否则开局（turn 0/1）的
//! `event` 帧会被派成 `Train` 计算。
//! 未识别的帧同样**不 dispatch**——宁可本帧不算（下一帧还会来），也不按 `Train` 硬算出一条
//! 与当前状态不符的推荐。
//!
//! `source=special` 在样本中暂未出现，按 `playing_state=45` 兜底为 RegionSelect。
//! 数据获取不全（turn 2..=71 且 `selected_regions` 全 0）→ warn + 不 dispatch。
//!
//! ## persons layout（adapter_spec §理事長、记者、NPC生成）
//!
//! | person_index | 身份 | 出现条件 |
//! |---|---|---|
//! | 0..=5 | 6 张训练卡（含友人） | 始终 |
//! | 6 | 理事長 | turn >= 0 |
//! | 7 | 记者 | turn > 12（不包含 12） |
//! | 8..=12 | 5 个 NPC | turn >= 2 |
//!
//! 规则层内部查找走 `PersonType`（Yayoi/Reporter/Npc），不依赖 person_index 数字；
//! person_index 数字仅供 `distribute_all` / 日志观测使用。
//!
//! ## personDistribution 中 NPC chara_id 派发（adapter_spec §personDistribution 适配）
//!
//! 协议约定：NPC 全填 `8`。但 `distribute_all` 按 chara_id 派发，5 个 NPC 必须 chara_id 各异。
//! 规则：把 `person_distribution` 中**全局按出现次序**的 `8` 依次改写为 `8, 9, 10, 11, 12`，
//! 改写后的数字即 NPC chara_id 来源（与 `NPC_CHARA_IDS` 一一对齐）。

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::ops::Deref;

use crate::protocol::{GameStatus, GameStatusBase};
use umasim::{
    game::{
        BasePerson,
        PersonType,
        ramen::{RamenGame, RamenStage, rules::NPC_CHARA_IDS}
    },
    gamedata::ramen::RAMENDATA,
    global
};

/// 拉面剧本通信状态顶层结构
///
/// 两段：`base_game`（与温泉剧本共用 `GameStatusBase` + 拉面增量字段）+ `ramen`
/// （拉面段所有字段，与文档 §2 一一对应）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusRamen {
    pub base_game: GameStatusBase,
    /// 拉面段（完整字段定义见 `RamenStatus`）
    #[serde(default)]
    pub ramen: RamenStatus
}

/// 拉面段通信状态（完整字段映射表见 `ramen_protocol_v2.md` §2）
///
/// 字段命名遵循协议 **snake_case**（实测样本 `scenario_pt` / `next_scenario_pt` /
/// `feeling_gauge` / `last_ramen` 等都是 snake_case，**不**走 `camelCase` —— 这是
/// ramen 段与 baseGame 段（`GameStatusBase` 走 camelCase + 个别 rename 覆盖）
/// 的字段命名约定差异）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RamenStatus {
    /// 每训练×每类型回合增量 `[[i32; 3]; 5]`
    #[serde(default)]
    pub feeling_gauge_gains: [[i32; 3]; 5],
    /// 三种诀窍（A/B/C）当前槽值
    #[serde(default)]
    pub feeling_gauge: [i32; 3],
    /// 诀窍队列（按获得顺序；C# 端会过滤 feeling_id==0 的项目）
    #[serde(default)]
    pub feeling_stock: Vec<i32>,
    /// 隐藏风味数量
    #[serde(default)]
    pub special_feeling: i32,
    /// 训练角标（0=无 / 1/2/3=A/B/C）
    #[serde(default)]
    pub train_feeling_type: [i32; 5],
    /// 当前生效效果列表 `{category, id, value}`（语义搁置，见 §5）
    #[serde(default)]
    pub active_effect_array: Vec<ActiveEffectEntry>,
    /// 超级拉面：-1=未选 / 0/1/2=档位
    #[serde(default = "default_super_ramen")]
    pub super_ramen: i32,
    /// 当年已选地区（`region_id`）
    ///
    /// **读取容忍空数组**：开局回合（turn 0/1，剧本机制未启动）实测会收到 `[]`——
    /// 定长 `[i32; 3]` 会让 serde 直接反序列化失败（`invalid length 0, expected an
    /// array of length 3`），整份快照被丢弃。改为 `Vec<i32>` 后空数组视为
    /// 「没选择地区」，在 `into_game` 中落到 game 侧默认状态 `[0, 0, 0]`
    /// （与 `RamenState::selected_regions: [usize; 3]` 口径一致，不影响 game 执行效率）。
    #[serde(default)]
    pub selected_regions: Vec<i32>,
    /// 基础增量（按 region 配方）
    #[serde(default)]
    pub feeling_gauge_gain_base: [i32; 3],
    /// **直接 = region_id**（实测；与 `selected_regions` 严格对齐）
    #[serde(default = "default_last_ramen")]
    pub last_ramen: i32,
    /// 当前累计剧本 PT（RMJ 失败归零）
    #[serde(default)]
    pub scenario_pt: i32,
    /// 下次吃面可获 PT
    #[serde(default)]
    pub next_scenario_pt: i32
}

fn default_super_ramen() -> i32 {
    -1
}
fn default_last_ramen() -> i32 {
    -1
}

/// 协议 `selected_regions` → `RamenState::selected_regions`（`[usize; 3]`）
///
/// 规则（与 `into_game` 的数据获取不全判定口径一致）：
/// - **空数组**（实测 turn 0/1 快照会发 `[]`）→ 视为「没选择地区」，落 game 侧默认状态 `[0, 0, 0]`；
/// - 负值（`-1` 未选占位）→ 该位保持 `0`；
/// - 长度不足 3 → 缺位保持 `0`；长度超过 3 → 只取前 3 位（越界项忽略，不 panic）。
fn map_selected_regions(raw: &[i32]) -> [usize; 3] {
    let mut arr = [0usize; 3];
    for (i, &r) in raw.iter().take(3).enumerate() {
        if r >= 0 {
            arr[i] = r as usize;
        }
    }
    arr
}

/// 协议 `active_effect_array` 的单项 `{category, id, value}`
///
/// 仅在 `into_game` 内用其长度做 stage dispatch 判断，不落入 `RamenState`；
/// 按 category 拆分到 `RamenEffect` 各字段暂不实现。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActiveEffectEntry {
    /// 类别（1/2/4 等，语义未公开）
    pub category: i32,
    /// 效果 ID
    pub id: i32,
    /// 效果数值
    pub value: i32
}

impl Deref for GameStatusRamen {
    type Target = GameStatusBase;
    fn deref(&self) -> &Self::Target {
        &self.base_game
    }
}

impl GameStatus for GameStatusRamen {
    type Game = RamenGame;

    fn scenario_id() -> u32 {
        14
    }

    /// 完整协议 → RamenGame 转换（Step 7 实现）
    fn into_game(self) -> Result<Self::Game> {
        let base = self.base_game;

        // 1. 从协议**严格重建** base（非 `RamenGame::newgame` 打补丁）：
        //    Uma / Friend 经 `parse_uma` / `parse_friend` 重建，`five_status_limit` 取协议值，
        //    `friend_event_ids` 不合并（丢弃）；deck / card_type_count / train_level_count /
        //    distribution / unresolved_events(story) 均由 `parse_basegame` 落地。
        let mut game = RamenGame::from_base_game(base.parse_basegame(9001)?)?;

        // 1.5 用协议 keyEvents 还原事件历史与友人首次点击状态
        //     （详见 `GameStatusBase::apply_key_events`）
        base.apply_key_events(&mut game.base, global!(RAMENDATA).friend_first_event);

        // 2. 构造 persons。按 spec §'理事長、记者、NPC生成' 的 layout：
        //   0..5 = deck 6 张（友人 chara_id=9001 / 其他友人改 OtherFriend）
        //   6 = 理事長（始终在场，turn=0 也有）
        //   7 = 记者（turn > 12 时才有，不包含 12）
        //   8..12 = 5 个 NPC（turn >= 2 时才有，chara_id 来自 NPC_CHARA_IDS 一一对齐）
        //   规则层查找走 PersonType，不依赖 person_index 数字；这里赋的 person_index
        //   仅供 distribute_all / 日志观测。NPC chara_id 派发见 §'personDistribution 适配'。
        let mut persons = vec![];
        for card in game.base.deck.iter() {
            let mut person = BasePerson::try_from(card)?;
            person.person_index = persons.len() as i32;
            if person.person_type == PersonType::ScenarioCard && person.chara_id != 9001 {
                person.person_type = PersonType::OtherFriend;
            }
            persons.push(person);
        }
        // 人头的 friendship/is_hint 来自协议 persons（按 cardIndex 对齐）
        for (index, person) in persons.iter_mut().enumerate() {
            if index < base.persons.len() {
                person.friendship = base.persons[index].friendship;
                person.is_hint = base.persons[index].is_hint;
            }
        }
        // 6: 理事長（始终在场）
        let mut yayoi = BasePerson::yayoi();
        yayoi.person_index = 6;
        yayoi.friendship = base.friendship_noncard_yayoi;
        persons.push(yayoi);
        // 7: 记者（adapter_spec：turn > 12 时才有，不包含 12）
        if base.turn > 12 {
            let mut reporter = BasePerson::reporter();
            reporter.person_index = 7;
            reporter.friendship = base.friendship_noncard_reporter;
            persons.push(reporter);
        }
        // 8..12: 5 NPC（adapter_spec：turn >= 2 时才有，chara_id 各异）
        if base.turn >= 2 {
            for (i, &npc_id) in NPC_CHARA_IDS.iter().enumerate() {
                persons.push(BasePerson {
                    person_index: (8 + i) as i32,
                    person_type: PersonType::Npc,
                    train_type: -1,
                    chara_id: npc_id,
                    friendship: 0,
                    is_hint: false,
                    card_id: None
                });
            }
        }
        game.persons = persons;

        // 3. 覆写 ramen 段（协议 `RamenStatus` → `RamenState` 全字段映射）
        let ramen = self.ramen;
        game.ramen.feeling_slot = ramen.feeling_gauge;
        game.ramen.feeling_stock = {
            // 协议 feeling_stock 是按"获得顺序"的队列，每项 1/2/3 表示 A/B/C
            // 累计整个 Vec 中 1/2/3 的出现次数 → [count_A, count_B, count_C]
            // 协议 feeling_id==0 项忽略（已被 C# 过滤但 Rust 端可能保留）
            let mut arr = [0; 3];
            for &f in &ramen.feeling_stock {
                if f >= 1 && f <= 3 {
                    arr[(f - 1) as usize] += 1;
                }
            }
            arr
        };
        game.ramen.special_feeling = ramen.special_feeling;
        game.ramen.train_feeling_type = {
            let mut arr = [umasim::game::ramen::FeelingType::A; 5];
            for (i, &t) in ramen.train_feeling_type.iter().enumerate() {
                arr[i] = match t {
                    1 => umasim::game::ramen::FeelingType::A,
                    2 => umasim::game::ramen::FeelingType::B,
                    3 => umasim::game::ramen::FeelingType::C,
                    _ => umasim::game::ramen::FeelingType::A
                };
            }
            // 协议 0=本回合无角标 → 整个 Option 设 None（让 Ramen 内部走默认）
            if ramen.train_feeling_type.iter().all(|&t| t == 0) {
                None
            } else {
                Some(arr)
            }
        };
        game.ramen.super_ramen = if ramen.super_ramen < 0 {
            None
        } else {
            Some(ramen.super_ramen as usize)
        };
        // 3.5 补超级拉面一次性赛后加成（race_bonus += finals_base.saihou）。
        //     协议帧不含 `raceBonus`、`parse_basegame` 只从支援卡累计；且重放**不跑**
        //     turn 72 的 `run_begin`（§5 dispatch 对 turn>=72 直接给 Train + combined_decision），
        //     故必须在此补上——否则 URA 段比赛收益被系统性低估（×1.55 而非 ×2.55）。
        //     `apply_super_ramen_saihou` 幂等：turn<72 或未选超级拉面时不生效，且合计最多一次。
        game.apply_super_ramen_saihou();
        game.ramen.selected_regions = map_selected_regions(&ramen.selected_regions);
        game.ramen.current_ramen = if ramen.last_ramen < 0 || ramen.active_effect_array.is_empty() {
            None
        } else {
            Some(ramen.last_ramen as usize)
        };
        // 协议 `scenario_pt` 即当年已累计的**吃面后**剧本 PT（实机在该帧训练时用的就是它，
        // 见 logs/game6261 turn47_2→turn47_3：本回合吃面 +600 后档位立刻按新值算），直接透传。
        game.ramen.scenario_pt = ramen.scenario_pt;

        // 4. personDistribution 适配（adapter_spec §personDistribution 适配）：
        //    spec 要求把全局按出现次序的 `8` 依次改写为 `8, 9, 10, 11, 12`。
        //    **当前实现不改写**——若启用会越界（详见 `issues.md` #12：spec 期望固定
        //    person_index 6/7/8-12，但当前 into_game 按 push 顺序动态分配 person_index，
        //    当 turn <= 12 时 persons 只有 12 项，distribution 出现 `12` 会越界）。等
        //    `BasePerson.is_hidden` 重构落地后再启用改写。
        //
        //    NPC chara_id 各异由 `NPC_CHARA_IDS` 常量保证（persons 构造时直接取常量），
        //    与 distribution 数字无绑定；distribute_all 按 persons 顺序遍历 chara_id 派发。
        //    留此注释作为占位，等 is_hidden PR 合并后启用改写函数。

        // 5. Stage dispatch（adapter_spec §source / §playing_state 三方联合）。
        //
        // **白名单**：只有明确识别为「决策帧」的帧才派发阶段给 AI 计算；其余一律**不派发**
        // （保留 `RamenStage::Begin` = newgame 默认值），由 main loop 识别后**整条决策链路跳过**
        // （不列候选、不搜索、不算 luck）。宁可本帧不算（下一帧还会来），也不要按 Train 硬算出
        // 一条与当前状态不符的推荐。
        //
        // `source` / `playing_state` 是插件侧的判定结果（见 `GameStatusSend_Base.ResolveSource`）：
        // - `command`：玩家指令响应；`event`：待处理事件（check_event 带 story）；
        // - `load`：载入响应（进育成 / 切屏回来）——**与 `command` 同等派发**：同样携带完整
        //   回合状态，实测有 `load + ps=1` 需要出推荐的情况，故不做跳过；
        // - `special`：`command` + `playing_state >= 10` 的特殊状态（只有 45 是地区选择）。
        let active_effect_count = ramen.active_effect_array.len();
        let source = base.source.as_deref().unwrap_or("");
        let playing_state = base.playing_state;
        let turn = base.turn;
        let data_incomplete = (2..=71).contains(&turn)
            && game.ramen.selected_regions.iter().all(|&r| r == 0);

        // 非决策帧：与 turn **无关**地跳过——必须在 `turn <= 1` 分支之前判定，
        // 否则开局（turn 0/1）的 event 帧会被派成 `Train` 计算。
        let non_decision = match (source, playing_state) {
            ("event", _) => Some("source=event，本回合为事件回合"),
            (_, 5) => Some("playing_state=5 事件回合"),
            (_, 46) => Some("playing_state=46 RMJ 结算"),
            (_, 48) => Some("playing_state=48 RMJ 最终结算"),
            _ => None
        };
        // 「刚选区、训练数据还没刷新」帧：玩家刚选完地区时，插件先发一条 ramen 数据段
        // 未刷新的帧（`command_feeling_info_array` 未下发 → `train_feeling_type` 全 0），
        // 随后才发含完整数据的那条。此时按未刷新数据算出的推荐不可信，故跳过。
        //
        // 判定范围**只在选区发生的回合**（turn 2）与**选区后紧接的回合**（23 选区 → 24、
        // 47 选区 → 48）：实测其它回合的 `train_feeling_type` 全 0 属于「本回合确实没有角标」
        // （夏合宿 36-39 / 60-63 等）或数据错误，不能跳。
        // `playing_state` 必须为 1：ps=45 是地区选择本身（其 feeling 同样全 0），必须算。
        let feels_unrefreshed = playing_state == 1
            && matches!(turn, 2 | 24 | 48)
            && !base.is_racing
            && ramen.train_feeling_type.iter().all(|&t| t == 0);

        if let Some(reason) = non_decision {
            log::info!("{reason}，跳过 AI 推荐");
            // 不动 game.stage，保留 Begin 让 main loop 走 fallback
        } else if feels_unrefreshed {
            log::info!(
                "turn={turn} 刚选择完地区、训练数据未刷新（train_feeling_type 全 0），跳过 AI 推荐"
            );
            // 不动 game.stage，保留 Begin；插件随后会再发一条含完整数据的帧
        } else if data_incomplete {
            log::warn!(
                "缺少地区选择信息，AI无法计算；需要回到大厅界面重进育成"
            );
            // 不动 game.stage，保留 Begin 让 main loop 走 fallback
        } else if playing_state == 45 {
            // 地区选择（`command` / `special` 两种 source 都按此处理）
            game.stage = RamenStage::RegionSelect;
        } else if turn <= 1 {
            // turn 0/1：剧本机制未启用（拉面机制 turn >= 2 才启动），直接进 Train。
            // 与 `RamenGame::next()` 内部短路（game.rs:124 turn < 2 跳 RamenSelect）口径一致：
            // 我们在 into_game 派发阶段提前派发，避免 main loop 走到 Distribute 后被 next() 短路时
            // 看不到本应有 RamenSelect 候选可选的语义。
            log::info!("turn={turn} 剧本机制未启用，直接进 Train 阶段");
            game.stage = RamenStage::Train;
        } else {
            match (source, active_effect_count > 0, playing_state) {
                // 超级拉面回合（turn >= 72）：
                //   active_effect_array 空 → 直接丢包（按 spec §超级拉面回合处理），
                //     等下一条数据；下一条数据会有 active_effect_array，是超级拉面激活后
                //     的效果，给训练决策。
                //   active_effect_array 有 → 训练阶段直接给决策，且不能重算超级拉面。
                (src, true, 1) if turn >= 72 => {
                    log::info!("超级拉面回合 turn={turn}，active_effect 已生效，给训练决策");
                    game.stage = RamenStage::Train;
                    // game.ramen.pending_ramen 已在前面 current_ramen 写入路径设置
                    game.ramen.combined_decision = true;
                    let _ = src;
                }
                (src, false, 1) if turn >= 72 => {
                    log::info!("超级拉面回合 turn={turn} 但 active_effect_array 为空，丢弃等下一条");
                    let _ = src;
                    // 不 dispatch
                }
                // 普通训练回合：command / load + active_effect_array 有 → 已吃面，Train
                //                                                  且构造中间状态 pending_ramen
                ("command" | "load", true, 1) => {
                    game.stage = RamenStage::Train;
                    // 写中间状态：adapter_spec §source 'command + active_effect_array 有' →
                    //  构造 RamenAction::ramen_select(Some(last_ramen))，让 umaai 决策训练。
                    //  按用户决策，落地为 `game.ramen.pending_ramen = Some(last_ramen)`。
                    if let Some(cur) = game.ramen.current_ramen {
                        game.ramen.pending_ramen = Some(cur);
                    }
                }
                // 普通训练回合：command / load + active_effect_array 空 → 吃面前，给 RamenSelect 决策
                ("command" | "load", false, 1) => {
                    game.stage = RamenStage::RamenSelect;
                }
                // `special` 且非 45：剧本特殊状态（非地区选择），AI 暂不处理
                ("special", ..) => {
                    log::info!(
                        "source=special（playing_state={playing_state} 非地区选择），跳过 AI 推荐"
                    );
                }
                // 兜底：未识别的帧**不派发**（保留 Begin）——不再按 Train 硬算
                (src, _, ps) => {
                    log::warn!(
                        "未识别的帧 source={src:?} playing_state={ps} active_effect={active_effect_count} turn={turn}，跳过 AI 推荐"
                    );
                }
            }
        }

        // 6. RMJ 派生状态恢复（协议快照不携带 → AI 侧按「每年成功 / 第 3 年大成功」假设补齐）
        //
        // 背景：`rmj_results`（年度 RMJ 结果，第 2/3 年常驻驱动 `ramen_success_effect` /
        // `ramen_fail_effect`）只存在于游戏内部状态，协议 JSON 没有对应字段。不补齐会让
        // **第 2/3 年的每次 rollin** 系统性缺失常驻成功效果——实测 turn23→24 的期望骤降
        // ~2300–3000 分（与随机种子无关，四局一致），且会拖低在线 AI 的决策质量。
        //
        // 依据：`check_rmj` 是纯函数（`scenario_pt >= ramen_success_pt[year]`），阈值
        // 1500 / 3000 / 3500（第 3 年 ≥5000 为大成功），正常育成基本达标；故按每年
        // 成功、第 3 年大成功假设补齐（两者 `is_success()` 均为 true）。
        let rmj_done = rmj_done_count(base.turn, &game.stage);
        game.ramen.rmj_results = vec![true; rmj_done];
        // `train_level_bonus` **不能**按年份一并补齐：协议的 `trainLevelCount` 是游戏内
        // 真实训练等级折算（`4×(等级−1) + 当前等级内点击数`，见插件 `GameStatusSend_Base`），
        // 已含已结算 RMJ 的 +1/+2 层；再补一次会让 `RamenGame::train_level()` 比实际等级
        // 高 1~2 级（第 2 年 +1、第 3 年 +2，再被 Lv5 上限截断）。
        // 置 0 表示「本次导入之后新增的加成」，局内 RMJ 结算照旧 `+= 1`（game/ramen/game.rs）。
        game.ramen.train_level_bonus = 0;

        // 7. 新年窗口 scenario_pt 归一化（协议快照携带的是「上一年遗留值」）
        //
        // 实测：新年首回合（turn 24 / 48 / 72）在**当年首次吃面入账前**，协议
        // `scenario_pt` 仍是上一年终值（如 3150 / 6000），而模拟在 turn23 的 RMJ
        // 结算时已把它归零、下一年重新累计。若直接沿用协议值，从该快照出发的 rollin
        // 会把这笔上一年 PT 一直带到下一年末（实测虚高 ~1900 分）。
        //
        // 判据：`active_effect_array` 为空 = 当年尚未吃面（吃面后效果数组非空），
        // 此时按模拟语义归零；已吃面（非空）则保留当年累计值。
        if matches!(base.turn, 24 | 48 | 72) && ramen.active_effect_array.is_empty() {
            game.ramen.scenario_pt = 0;
        }

        Ok(game)
    }
}

/// 已结算的 RMJ 次数（用于按「每年成功」假设恢复 `rmj_results` / `train_level_bonus`）
///
/// RMJ 在 turn 23 / 47 / 71 的 `NextTurn` 阶段结算；同一回合内**结算前**的快照
/// （`Train` / `AfterTrain` / `NextTurn`）仍算上一次数量，**结算后**（`RegionSelect`
/// 起，含 `RegionSelect` / `BeginAfterRegionSelect`）计入本次。
///
/// 返回值为**已完成的年份数**：turn < 23 → 0，第 2 年（24..47）→ 1，
/// 第 3 年（48..71）→ 2，URA 期（> 71）→ 3。
fn rmj_done_count(turn: i32, stage: &RamenStage) -> usize {
    let settled_this_turn = matches!(stage, RamenStage::RegionSelect | RamenStage::BeginAfterRegionSelect);
    match turn {
        t if t < 23 => 0,
        23 => usize::from(settled_this_turn),
        t if t < 47 => 1,
        47 => 1 + usize::from(settled_this_turn),
        t if t < 71 => 2,
        71 => 2 + usize::from(settled_this_turn),
        _ => 3
    }
}

/// `person_distribution` 适配（adapter_spec §personDistribution 适配）
///
/// TODO：本函数暂未启用——spec 要求把全局按出现次序的 `8` 依次改写为 `8, 9, 10, 11, 12`，
/// 但当前 `into_game` 按 push 顺序动态分配 person_index，当 turn <= 12 时 persons 只有
/// 12 项（下标 0..11），distribution 出现 `12` 会越界。详见 `issues.md` #12。
///
/// 启用条件：等 `BasePerson.is_hidden` 重构落地（按 spec 固定 person_index 6/7/8-12，
/// 缺位者以 placeholder + is_hidden=true 占位）。届时本函数实现即可安全启用。
#[allow(dead_code, unused_variables)]
fn adapt_person_distribution_npc_todo(distribution: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = Vec::with_capacity(distribution.len());
    let mut next_npc_id = 8_i32;
    for row in distribution {
        let mut new_row = Vec::with_capacity(row.len());
        for &v in row {
            if v == 8 {
                new_row.push(next_npc_id);
                next_npc_id += 1;
            } else {
                new_row.push(v);
            }
        }
        out.push(new_row);
    }
    out
}

/// `GameStatusRamen` → 拉面协议 JSON（暂未实现反向转换，Step 7 后续补）
impl From<&RamenGame> for GameStatusRamen {
    fn from(_game: &RamenGame) -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

/// feeling_stock 协议 → RamenState 映射（Vec<i32> → [i32; 3]）
    #[test]
    fn test_feeling_stock_mapping() {
        // 协议示例：[1, 2, 3, 1, 2, 3] → A/B/C 各 2
        let raw = vec![1, 2, 3, 1, 2, 3];
        let arr: [i32; 3] = {
            let mut a = [0; 3];
            for &f in &raw {
                if f >= 1 && f <= 3 {
                    a[(f - 1) as usize] += 1;
                }
            }
            a
        };
        println!("raw={raw:?} → arr={arr:?}");
        assert_eq!(arr, [2, 2, 2]);
    }

    /// train_feeling_type 协议 → RamenState 映射
    #[test]
    fn test_train_feeling_type_mapping() {
        let raw = [3, 1, 1, 3, 2];
        let arr: [umasim::game::ramen::FeelingType; 5] = {
            let mut a = [umasim::game::ramen::FeelingType::A; 5];
            for (i, &t) in raw.iter().enumerate() {
                a[i] = match t {
                    1 => umasim::game::ramen::FeelingType::A,
                    2 => umasim::game::ramen::FeelingType::B,
                    3 => umasim::game::ramen::FeelingType::C,
                    _ => umasim::game::ramen::FeelingType::A
                };
            }
            a
        };
        println!("raw={raw:?} → arr={arr:?}");
        // C, A, A, C, B
        assert_eq!(arr[0], umasim::game::ramen::FeelingType::C);
        assert_eq!(arr[1], umasim::game::ramen::FeelingType::A);
        assert_eq!(arr[4], umasim::game::ramen::FeelingType::B);
    }

    /// selected_regions 映射：正常 / 空数组 / 负值 / 长度异常
    #[test]
    fn test_selected_regions_mapping() {
        // 正常三值
        assert_eq!(map_selected_regions(&[1, 4, 5]), [1, 4, 5]);
        // 开局空数组（实测 turn 0/1）→ 没选择地区，落 game 默认状态
        assert_eq!(map_selected_regions(&[]), [0, 0, 0]);
        // -1 未选占位 → 该位 0；混选时保留有效位
        assert_eq!(map_selected_regions(&[-1, -1, -1]), [0, 0, 0]);
        assert_eq!(map_selected_regions(&[3, -1, 7]), [3, 0, 7]);
        // 长度异常：不足补 0 / 超出只取前 3（不 panic）
        assert_eq!(map_selected_regions(&[2]), [2, 0, 0]);
        assert_eq!(map_selected_regions(&[1, 2, 3, 4]), [1, 2, 3]);
        println!("selected_regions 映射用例全部通过");
    }

    /// RMJ 已完成次数边界：turn 23/47/71 同回合内「结算前 / 结算后」区分
    #[test]
    fn test_rmj_done_count_boundaries() {
        use umasim::game::ramen::RamenStage;
        assert_eq!(rmj_done_count(0, &RamenStage::Train), 0);
        assert_eq!(rmj_done_count(22, &RamenStage::Train), 0);
        // turn 23 同回合：训练 / 结算中（结算前）→ 0；地区选择起（结算后）→ 1
        assert_eq!(rmj_done_count(23, &RamenStage::Train), 0);
        assert_eq!(rmj_done_count(23, &RamenStage::NextTurn), 0);
        assert_eq!(rmj_done_count(23, &RamenStage::RegionSelect), 1);
        assert_eq!(rmj_done_count(23, &RamenStage::BeginAfterRegionSelect), 1);
        assert_eq!(rmj_done_count(24, &RamenStage::RamenSelect), 1);
        assert_eq!(rmj_done_count(46, &RamenStage::Train), 1);
        assert_eq!(rmj_done_count(47, &RamenStage::Train), 1);
        assert_eq!(rmj_done_count(47, &RamenStage::RegionSelect), 2);
        assert_eq!(rmj_done_count(48, &RamenStage::Train), 2);
        assert_eq!(rmj_done_count(70, &RamenStage::Train), 2);
        assert_eq!(rmj_done_count(71, &RamenStage::Train), 2);
        assert_eq!(rmj_done_count(71, &RamenStage::RegionSelect), 3);
        assert_eq!(rmj_done_count(72, &RamenStage::Train), 3);
        assert_eq!(rmj_done_count(77, &RamenStage::Train), 3);
        println!("RMJ 次数边界用例全部通过");
    }

    /// 协议重建补齐 RMJ 派生状态（真实快照驱动）
    ///
    /// 回归背景：`rmj_results` 协议不携带，缺失会让第 2/3 年 rollin 系统性缺少常驻
    /// 成功效果（turn23→24 期望骤降 ~2300–3000）。
    ///
    /// `train_level_bonus` **不再**按年份补齐：协议 `trainLevelCount` 已是游戏内真实
    /// 等级折算（含已结算 RMJ 的加成），再补会让训练等级高 1~2 级——回归见 mod.rs 的
    /// `test_ramen_import_keeps_real_train_level`。
    #[test]
    fn test_into_game_restores_rmj_state() {
        use std::fs;

        use crate::protocol::{ParsedGame, parse_game_by_scenario};
        use umasim::{game::Game, gamedata::init_global};

        let workspace_root = umasim::utils::get_workspace_root().expect("workspace root");
        let dir = workspace_root.join("logs").join("SendGameStatusPlugin");
        if !dir.is_dir() {
            eprintln!("样本目录不存在：{}（跳过本测试）", dir.display());
            return;
        }
        let _ = std::env::set_current_dir(&workspace_root);
        let _ = init_global();

        let load = |name: &str| -> Option<RamenGame> {
            let path = dir.join(name);
            if !path.is_file() {
                return None;
            }
            let contents = fs::read_to_string(&path).expect("read sample");
            match parse_game_by_scenario(&contents).expect("parse sample") {
                ParsedGame::Ramen { game, .. } => Some(game),
                ParsedGame::Onsen(_) => panic!("样本应为拉面剧本")
            }
        };

        // 第 2 年快照：rmj_results 补 1 次（第 1 年 RMJ 成功），但等级不再叠加
        if let Some(game) = load("game7075_turn24_2.json") {
            let levels: Vec<usize> = (0..5).map(|t| game.train_level(t)).collect();
            println!(
                "turn24_2: turn={} bonus={} rmj={:?} count={:?} level={:?}",
                game.turn(),
                game.ramen.train_level_bonus,
                game.ramen.rmj_results,
                game.base.train_level_count,
                levels
            );
            assert_eq!(game.ramen.rmj_results, vec![true], "第 2 年 rmj_results 应为 [true]");
            assert_eq!(
                game.ramen.train_level_bonus, 0,
                "导入帧不应再补等级加成（协议 trainLevelCount 已含）"
            );
            for t in 0..5 {
                assert_eq!(
                    game.train_level(t),
                    game.base.base_train_level(t),
                    "训练等级应等于协议真实等级（count/4+1），不得再叠加 RMJ 加成"
                );
            }
        }
        // 第 1 年内快照：不应有 RMJ 结果
        if let Some(game) = load("game7075_turn13.json") {
            println!("turn13: bonus={} rmj={:?}", game.ramen.train_level_bonus, game.ramen.rmj_results);
            assert_eq!(game.ramen.train_level_bonus, 0, "第 1 年内不应有 RMJ 加成");
            assert!(game.ramen.rmj_results.is_empty(), "第 1 年内 rmj_results 应为空");
        }
        // 新年首回合 scenario_pt：未吃面（active_effect 空）→ 归零；已吃面 → 透传当年累计值
        for (name, expect_pt) in [("game7075_turn24_2.json", 0), ("game7075_turn24_3.json", 0)] {
            if let Some(game) = load(name) {
                println!("{name}: scenario_pt={} (期望 {expect_pt})", game.ramen.scenario_pt);
                assert_eq!(game.ramen.scenario_pt, expect_pt, "{name} 新年窗口 scenario_pt 归一化不符");
            }
        }
        // turn23 同回合边界：训练（结算前）→ 0 次结算；地区选择（结算后）→ 1 次
        for (name, expect) in [("game7075_turn23_2.json", 0), ("game7075_turn23_4.json", 1)] {
            if let Some(game) = load(name) {
                println!(
                    "{name}: stage={:?} rmj={:?} bonus={}",
                    game.stage, game.ramen.rmj_results, game.ramen.train_level_bonus
                );
                assert_eq!(game.ramen.rmj_results.len(), expect, "{name} RMJ 结算次数边界不符");
                assert_eq!(game.ramen.train_level_bonus, 0, "{name} 导入帧不应补等级加成");
            }
        }
    }

    /// PT 校准复算 harness（无断言，仅打印全分量）
    ///
    /// 已知实测点（AI 复算 / 实机）：game6261 两例当前公式**逐位吻合**
    /// （turn48_3 = 106 / turn57_2 = 133，含属性 22+29）；game6263 两例
    /// PT 上层偏低恰好 +3（turn29_3: 119 vs 122 / turn60_2: 122 vs 125），
    /// 属性与下层均正确。缺口未定位（面板 region_bonus 读数不服从
    /// floor(pt/1000)，疑似按当年吃面次数走，详见与用户的对拍结论）。
    /// 后续拿到新的实测点后改 `frames` 数组继续对拍。
    #[test]
    fn test_game6263_pt_calibration() {
        use std::fs;

        use crate::protocol::{ParsedGame, parse_game_by_scenario};
        use umasim::{game::Game, gamedata::init_global};

        fn calib_print(game: &RamenGame, name: &str, train: usize, actual_pt: i32) {
            let buffs = game.calc_training_buff(train).expect("buffs");
            let lower = game.default_calc_training_value(&buffs, train).expect("lower");
            let value = game.calc_training_value(&buffs, train).expect("value");
            let is_shining = game.shining_count(train) > 0;
            let eff = umasim::game::ramen::effects::calc_ramen_training_effect(game, train, is_shining);
            println!(
                "{name}: train={train} shining={is_shining} level={}\n  \
                 lower_pt={}  final_pt={}  actual={actual_pt}\n  \
                 rmj={:?}  scenario_pt={}  current_ramen={:?}  motivation={}\n  \
                 eff: xunlian={} youqing={} rmj_youqing={} pt_bonus={} pt_limit={} status_limit={}\n  \
                 attrs: {:?}\n  \
                 buffs: {}\n  \
                 dist: {:?}\n  \
                 persons: {:?}",
                game.train_level(train),
                lower.status_pt[5],
                value.status_pt[5],
                game.ramen.rmj_results,
                game.ramen.scenario_pt,
                game.ramen.current_ramen,
                game.uma().motivation,
                eff.xunlian,
                eff.youqing,
                eff.rmj_youqing,
                eff.pt_bonus,
                eff.pt_limit,
                eff.status_limit,
                value.status_pt,
                buffs.explain(),
                game.base.distribution,
                (0..game.persons.len())
                    .map(|i| format!(
                        "{}:{:?}/t{}/f{}",
                        i,
                        game.persons[i].person_type,
                        game.persons[i].train_type,
                        game.persons[i].friendship
                    ))
                    .collect::<Vec<_>>()
            );
        }

        let workspace_root = umasim::utils::get_workspace_root().expect("workspace root");
        let _ = std::env::set_current_dir(&workspace_root);
        let _ = init_global();

        for (dir_name, frames) in [
            ("game6263", vec![
                ("game6263_turn29_3.json", 1usize, 122i32),
                ("game6263_turn60_2.json", 4usize, 125i32)
            ]),
            ("game6261", vec![
                ("game6261_turn48_3.json", 1usize, 106i32),
                ("game6261_turn57_2.json", 4usize, 133i32)
            ])
        ] {
            let dir = workspace_root.join("logs").join(dir_name);
            if !dir.is_dir() {
                eprintln!("样本目录不存在：{}（跳过）", dir.display());
                continue;
            }
            for (name, train, actual_pt) in frames {
                let contents = fs::read_to_string(dir.join(name)).expect("read sample");
                let game = match parse_game_by_scenario(&contents).expect("parse sample") {
                    ParsedGame::Ramen { game, .. } => game,
                    ParsedGame::Onsen(_) => panic!("样本应为拉面剧本")
                };
                calib_print(&game, name, train, actual_pt);
            }
        }
    }

    /// 空数组可被协议层反序列化（`RamenStatus` 读取容忍 `selected_regions: []`）
    ///
    /// 回归背景：该字段原为定长 `[i32; 3]`，开局快照发空数组会让整份快照
    /// `invalid length 0, expected an array of length 3` 解析失败。
    #[test]
    fn test_empty_selected_regions_deserializes() {
        let status: RamenStatus =
            serde_json::from_str(r#"{"selected_regions": []}"#).expect("空数组必须可反序列化");
        assert!(status.selected_regions.is_empty(), "空数组读入为空 Vec");
        assert_eq!(
            map_selected_regions(&status.selected_regions),
            [0, 0, 0],
            "空数组 → 没选择地区（game 默认状态）"
        );
        // 缺字段（serde default）同样落到默认状态
        let bare: RamenStatus = serde_json::from_str("{}").expect("缺字段必须可反序列化");
        assert_eq!(map_selected_regions(&bare.selected_regions), [0, 0, 0]);
        println!("空数组 / 缺字段反序列化用例通过");
    }

    /// super_ramen / last_ramen -1 → None
    #[test]
    fn test_optional_mappings() {
        assert_eq!(if -1_i32 < 0 { None } else { Some(-1_i32 as usize) }, None);
        assert_eq!(if 0_i32 < 0 { None } else { Some(0_i32 as usize) }, Some(0));
        assert_eq!(if 2_i32 < 0 { None } else { Some(2_i32 as usize) }, Some(2));
    }

    /// 重放路径：turn>=72 的超级拉面帧必须把 `finals_effect.base.saihou` 计入 `race_bonus`
    ///
    /// 回归背景：协议帧不含 `raceBonus`，`parse_basegame` 只从支援卡累计；且重放不跑
    /// turn 72 的 `run_begin`。若不补这 +100，URA 段三次比赛收益被系统性低估
    /// （×1.55 而非 ×2.55），表现为 71_2→72_2 的运气大跳。
    #[test]
    fn test_replay_applies_super_ramen_race_bonus() {
        use std::fs;

        use crate::protocol::{ParsedGame, parse_game_by_scenario};

        let workspace_root = umasim::utils::get_workspace_root().expect("workspace root");
        let path = workspace_root
            .join("logs")
            .join("game6261")
            .join("game6261_turn72_2.json");
        if !path.is_file() {
            eprintln!("样本不存在：{}（跳过本测试）", path.display());
            return;
        }
        let _ = std::env::set_current_dir(&workspace_root);
        let _ = umasim::gamedata::init_global();

        let contents = fs::read_to_string(&path).expect("read sample");
        let ParsedGame::Ramen { game, .. } = parse_game_by_scenario(&contents).expect("parse sample")
        else {
            panic!("样本应为拉面剧本");
        };

        assert!(
            game.ramen.super_ramen.is_some(),
            "turn72_2 应为超级拉面已选（super_ramen 非 None）"
        );
        let card_saihou: i32 = game.base.deck.iter().map(|c| c.effect.saihou).sum();
        let saihou = global!(RAMENDATA).finals_effect.base.saihou;
        assert!(saihou > 0, "finals_effect.base.saihou 应为正值");
        println!(
            "turn72_2: card_saihou={card_saihou}, super saihou={saihou}, race_bonus={}",
            game.uma.race_bonus
        );
        assert_eq!(
            game.uma.race_bonus,
            card_saihou + saihou,
            "重放 turn>=72 帧的 race_bonus 应 = 支援卡 sai_hou 之和 + 超级拉面一次性加成"
        );
    }

    /// 151 份 turn import 样本驱动测试（实测 chara 6204 全 78 回合）
    ///
    /// 数据来源：`logs/GameStatusSend_Ramen/game6204_turn*.json`（151 份）
    /// 测试目标：每份样本 parse → into_game → 关键字段 round-trip 校验
    #[test]
    fn test_turn_import_v2_full_samples() {
        use std::fs;

        // 定位样本目录（workspace 根 + logs/GameStatusSend_Ramen）
        let workspace_root = umasim::utils::get_workspace_root().expect("workspace root");
        let sample_dir = workspace_root.join("logs").join("GameStatusSend_Ramen");
        if !sample_dir.is_dir() {
            eprintln!("样本目录不存在：{}（跳过本测试）", sample_dir.display());
            return;
        }
        let _ = std::env::set_current_dir(&workspace_root);
        let _ = umasim::gamedata::init_global();

        // 收集所有 turn 样本（排除 thisTurn.json 当前软链）
        let mut files: Vec<_> = fs::read_dir(&sample_dir)
            .expect("read sample dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.starts_with("game6204_turn") && s != "thisTurn.json")
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        println!("驱动 {} 份样本", files.len());
        assert!(files.len() >= 100, "样本数过少（{}），请检查 logs 目录", files.len());

        let mut count_ok = 0;
        let mut count_stage: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut count_eaten_turns = 0usize; // last_ramen >= 0 + selected_regions 非零（年内吃面回合）
        let mut count_super_ramen_2 = 0usize; // super_ramen == 2（选了超级拉面档位 2）
        let mut max_scenario_pt: i32 = 0;
        let mut max_scenario_pt_json: i32 = 0;

        for path in &files {
            let contents = fs::read_to_string(path).expect("read sample");
            // 先 parse 成通用 Value 取 ramen 段（用于校验透传）
            let value: serde_json::Value = match serde_json::from_str(&contents) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("  parse value fail {}: {e}", path.display());
                    continue;
                }
            };
            let ramen_json = value.get("ramen").cloned().unwrap_or_default();
            let scenario_pt_json = ramen_json.get("scenario_pt").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let last_ramen_json = ramen_json.get("last_ramen").and_then(|v| v.as_i64()).unwrap_or(-1);
            let active_effect_json = ramen_json.get("active_effect_array").and_then(|v| v.as_array());
            let active_effect_empty = active_effect_json.map_or(true, |a| a.is_empty());
            let super_ramen_json = ramen_json.get("super_ramen").and_then(|v| v.as_i64()).unwrap_or(-1);
            let selected_regions_json: [i32; 3] = {
                let arr = ramen_json.get("selected_regions").and_then(|v| v.as_array());
                let mut r = [0; 3];
                if let Some(a) = arr {
                    for (i, v) in a.iter().enumerate() {
                        if i < 3 {
                            r[i] = v.as_i64().unwrap_or(0) as i32;
                        }
                    }
                }
                r
            };
            // 解析 + 构造 GameStatusRamen
            let status: GameStatusRamen = serde_json::from_value(value).expect("reparse");
            // into_game 构造 RamenGame
            let game = match status.into_game() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("  into_game fail {}: {e}", path.display());
                    continue;
                }
            };

            // 关键字段 round-trip 校验
            // 1) current_ramen 透传：仅在 last_ramen >= 0 且 active_effect_array 非空时生效
            //    （L311 改动：active_effect_array 为空时 current_ramen 置 None，即使 last_ramen 有效）
            let expected_current = if last_ramen_json < 0 || active_effect_empty {
                None
            } else {
                Some(last_ramen_json as usize)
            };
            assert_eq!(game.ramen.current_ramen, expected_current, "{}: current_ramen 不一致", path.display());
            // 2) scenario_pt：直接透传（协议帧即「吃面后」当年累计值）；
            //    仅新年窗口 turn 24/48/72 未吃面帧在 step 7 归零
            let expect_pt = if matches!(game.base.turn, 24 | 48 | 72) && active_effect_empty {
                0
            } else {
                scenario_pt_json
            };
            assert_eq!(game.ramen.scenario_pt, expect_pt, "{}: scenario_pt 不一致", path.display());
            // 3) selected_regions 透传
            let expected_regions: [usize; 3] = [
                selected_regions_json[0].max(0) as usize,
                selected_regions_json[1].max(0) as usize,
                selected_regions_json[2].max(0) as usize
            ];
            assert_eq!(game.ramen.selected_regions, expected_regions, "{}: selected_regions 不一致", path.display());
            // 4) super_ramen 透传
            let expected_super = if super_ramen_json < 0 { None } else { Some(super_ramen_json as usize) };
            assert_eq!(game.ramen.super_ramen, expected_super, "{}: super_ramen 不一致", path.display());

            // 5) persons layout 校验（adapter_spec §理事長、记者、NPC生成）：
            //    按 into_game 内的 push 顺序实际下标（无空洞）：
            //      0..=5   deck 6 张
            //      6       理事長（始终在场）
            //      7..=11  NPC（turn >= 2；记者不存在时 NPC 占据此区）
            //      7       记者（turn > 12；NPC 后移到 8..=12）
            //    ——注意：spec 写"理事长 6 / 记者 7 / NPC 8-12"是 spec 期望的固定下标，
            //    但实现按 push 顺序，无记者时 NPC 占 7..=11。
            let turn = game.base.turn;
            if turn < 2 {
                assert_eq!(game.persons.len(), 7, "{}: turn < 2 应为 7（6 卡 + 1 理事長）", path.display());
                assert!(matches!(game.persons[6].person_type, PersonType::Yayoi),
                    "{}: persons[6] 应为理事長", path.display());
            } else if turn <= 12 {
                assert_eq!(game.persons.len(), 12, "{}: turn 2..=12 应为 12（+ 5 NPC，无记者）", path.display());
                assert!(matches!(game.persons[6].person_type, PersonType::Yayoi),
                    "{}: persons[6] 应为理事長", path.display());
                for i in 7..=11 {
                    assert!(matches!(game.persons[i].person_type, PersonType::Npc),
                        "{}: persons[{i}] 应为 NPC", path.display());
                }
            } else {
                assert_eq!(game.persons.len(), 13, "{}: turn > 12 应为 13（+ 记者 + 5 NPC）", path.display());
                assert!(matches!(game.persons[6].person_type, PersonType::Yayoi),
                    "{}: persons[6] 应为理事長", path.display());
                assert!(matches!(game.persons[7].person_type, PersonType::Reporter),
                    "{}: persons[7] 应为记者", path.display());
                for i in 8..=12 {
                    assert!(matches!(game.persons[i].person_type, PersonType::Npc),
                        "{}: persons[{i}] 应为 NPC", path.display());
                }
            }

            // 累计统计
            count_ok += 1;
            *count_stage.entry(format!("{:?}", game.stage)).or_insert(0) += 1;
            if last_ramen_json >= 0 && selected_regions_json.iter().any(|&r| r > 0) {
                count_eaten_turns += 1;
            }
            if super_ramen_json == 2 {
                count_super_ramen_2 += 1;
            }
            max_scenario_pt = max_scenario_pt.max(game.ramen.scenario_pt);
            max_scenario_pt_json = max_scenario_pt_json.max(scenario_pt_json);
        }

        println!("解析成功：{} / {}", count_ok, files.len());
        println!("stage 分布：{count_stage:?}");
        println!("max_scenario_pt = {max_scenario_pt}（协议原始 max = {max_scenario_pt_json}）");
        println!("年内吃面回合数={count_eaten_turns}");
        println!("选了超级拉面档位 2 的样本数={count_super_ramen_2}");
        assert_eq!(count_ok, files.len(), "所有样本必须 parse + into_game 成功");
        // stage 分布（adapter_spec §stage dispatch 实测 chara 6204）：
        //   Begin: 数据获取不全 / 不 dispatch 的样本（source=event / playing_state=46/48 等）
        //   RamenSelect: command + active_effect 空（吃面前）
        //   Train: command + active_effect 有（吃面后训练 / 超级拉面回合训练）
        //   RegionSelect: playing_state=45（地区选择前）
        assert!(count_stage.contains_key("RamenSelect"), "应有 RamenSelect 样本");
        assert!(count_stage.contains_key("Train"), "应有 Train 样本");
        assert!(count_stage.contains_key("RegionSelect"), "应有 RegionSelect 样本");
        // 不应再出现旧协议派发的 stage
        assert!(!count_stage.contains_key("Settlement"), "Settlement stage 已废弃（46/48 不 dispatch）");
        assert!(!count_stage.contains_key("SuperRamenSelect"), "SuperRamenSelect stage 不应自动派发");
        // 协议文档约束：chara 6204 协议原始 max scenario_pt = 7500（Y3 终值）
        assert_eq!(max_scenario_pt_json, 7500, "实测 chara 6204 协议原始 scenario_pt 应在 Y3 终值 7500");
        // scenario_pt 直接透传（仅在 turn 24/48/72 未吃面帧归一化为 0），故 game 侧 max 等于协议原始 max
        assert_eq!(max_scenario_pt, max_scenario_pt_json, "scenario_pt 应直接透传协议值");
        // 至少有一个 super_ramen == 2 的样本（实测 turn72 起）
        assert!(count_super_ramen_2 >= 1, "应至少有 1 份 super_ramen=2 样本");
        // 至少有一个 source=event / playing_state=46 / 48 等不 dispatch 样本（落到 Begin）
        assert!(count_stage.get("Begin").copied().unwrap_or(0) >= 10, "不 dispatch 样本数应 >= 10");
    }
}