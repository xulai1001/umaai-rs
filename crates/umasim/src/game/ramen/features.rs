//! 拉面杯神经网络特征编码器（NN 管线 Phase 2 下半）
//!
//! 把一个 [`RamenGame`] 局面编码成定长 `f32` 向量，布局分三段：
//!
//! | 段 | 形状 | 说明 |
//! |---|---|---|
//! | global | `[GLOBAL_DIM]` | 回合 / 马娘面板 / 剧本状态 / 地区 / 比赛 |
//! | cards | `[CARD_NUM, CARD_DIM]` | 6 张支援卡，置换等变序列 |
//! | persons | `[PERSON_NUM, PERSON_DIM]` | 人头，第二个置换等变序列 |
//!
//! # 与温泉版 `extract_nn_features` 的区别
//!
//! 温泉版是上游的直接移植，**一次都没有编码 `five_status_bonus`**（成长率）。
//! 单马娘数据集下它是常数、无影响；第一代教师数据要用 7 个马娘，成长率直接进入
//! 训练收益公式的 `(1.0 + 0.01 * status_bonus[i])` 项，是区分马娘的最核心变量。
//! 本模块把 `five_status_bonus` 与 `five_status_limit` 一并编码。
//!
//! 温泉版还禁用了人头分支（上游 `Game_Input_C_Person = 0`）。拉面的人头分布
//! （彩圈、友人、得意率）是核心信息，故本模块开启 persons 段。
//!
//! # 设计约束
//!
//! - **分块记账**：每个块用 [`FeatureWriter::block`] 声明期望宽度，写完立即校验，
//!   末尾再断言总维度。改字段忘了改常量会在测试里立刻炸，而不是静默错位。
//! - **归一化用实际值域**：每个 `SCALE_*` 常量都注明取值依据，不用魔数。
//! - **失败返回 `Result`**：查表失败、下标越界一律报错，不静默填 0。
//!   「填 0」与「真值是 0」不可区分，会把数据错误伪装成合法样本。
//! - **只喂原始状态分量与纯计数派生量**：不喂任何带权重的估值。
//!   自选比赛缺口、剩余合格回合是原始状态的确定性函数、不含权重判断，故编码；
//!   而「这个训练值多少分」之类的打分结果一律不进特征。
//! - **人头下标 ≠ 卡组下标**：拉面的 `init_persons` 只放 5 张训练卡再追加理事长，
//!   友人卡到回合 2 才加入，于是理事长占人头 5、友人卡占人头 6，而 `deck[5]` 是友人卡。
//!   cards 段与 persons 段之间的互相引用一律按 `card_id` 反查，不假设两个序列同序。
//! - **组合类内容展开编码而非编 id**：地区段展开 `xunlian/youqing/at_trains`，
//!   超级拉面限制选项同理展开成合法训练位 multi-hot——泛化到没见过的组合，
//!   数据表调整时不改代码。
//!
//! # 超级拉面维度（schema v2）
//!
//! 原 v1 只编码「选了第几个选项」（one-hot）与「选没选」（flag），有两处盲区：
//! 选中的限制选项**允许哪些训练位**只能靠网络背 id→内容表，且「已选但尚未到
//! 72 回合生效」与「正在生效」无法区分。v2 在 `ramen` 块追加：
//!
//! - 1 维 `super_window`：`turn ∈ [72,77]`（[`RamenGame::is_super_ramen_turn`]）；
//! - [`TRAIN_NUM`] 维「已选选项的合法训练位」multi-hot（自
//!   `RAMENDATA.finals_effect.training_limit_options` 展开；未选时整组为 0，
//!   由既有的 `is_some` flag 区分「没选」与「选了但不覆盖该位」）。
//!
//! 决策时刻（`SuperRamenSelect` 三候选采样）不依赖新维度区分候选：由样本导出方
//! 把候选下标临时写入 `ramen.super_ramen` 后调用本函数，既有 one-hot 即可承载；
//! 该字段在数据里是纯位置下标、无 option ID 概念，写入-恢复对状态无副作用。
//!
//! **维度变更使已落盘的教师数据作废**：重新采数或给旧样本打 v1 标记后方可混用。
//!
//! # 已知未覆盖
//!
//! - `RamenGame::current_effect` 恒为全零（上游遗留的死字段），原先为它保留的 14 维
//!   已移除。若上游后续真正填充该字段，需要重新加回并给样本打新的 schema 版本号。

use anyhow::{Result, bail, ensure};

use super::{
    FeelingType, RamenStage,
    policy::remaining_race_slots,
    state::RamenGame
};
use crate::{
    game::{FriendCardState, FriendOutState, Game, PersonType},
    gamedata::ramen::RAMENDATA,
    global
};

// ========== 段维度 ==========

/// 支援卡槽位数（固定 6）
pub const CARD_NUM: usize = 6;

/// 人头序列长度
///
/// 开局 6 个（0-4 五张训练卡 + 5 理事长），回合 2 加剧本友人（人头 6）与 5 个 NPC
/// （7-11），回合 12 加记者（12），最终 13 个。**注意人头下标与卡组槽位不同序**。
/// 固定为最大值并用「已登场」掩码位标记，使维度恒定；未登场的整行为 0。
pub const PERSON_NUM: usize = 13;

/// global 段：回合与阶段（2 个 num + [`YEAR_NUM`] + [`STAGE_NUM`]）
const G_TURN: usize = 18;
/// global 段：马娘面板
const G_UMA: usize = 27;
/// global 段：马娘 buff 状态
const G_FLAGS: usize = 9;
/// global 段：训练设施
const G_FACILITY: usize = 6;
/// global 段：剧本状态
///
/// = 18（v1）+ 1 超级拉面生效窗口 flag + [`TRAIN_NUM`] 限制选项合法位 multi-hot。
const G_RAMEN: usize = 24;
/// global 段：诀窍角标
const G_MARK: usize = 16;
/// global 段：友人
const G_FRIEND: usize = 15;
/// global 段：地区
const G_REGION: usize = 35;
/// global 段：比赛与自选比赛
const G_RACE: usize = 10;

/// global 段总维度
pub const GLOBAL_DIM: usize = G_TURN + G_UMA + G_FLAGS + G_FACILITY + G_RAMEN + G_MARK + G_FRIEND + G_REGION + G_RACE;

/// 单张支援卡的特征维度
pub const CARD_DIM: usize = 35;

/// 单个人头的特征维度
pub const PERSON_DIM: usize = 30;

/// 编码后的总输入维度
pub const INPUT_DIM: usize = GLOBAL_DIM + CARD_NUM * CARD_DIM + PERSON_NUM * PERSON_DIM;

// ========== 归一化尺度 ==========

/// 总回合数 - 1（回合范围 0..=77）
const SCALE_TURN: f32 = 77.0;
/// 一年的回合数（RMJ 在回合 23/47/71 结算）
const SCALE_YEAR_TURN: f32 = 23.0;
/// 五维属性的典型量级（1200 以上不减半，实际可超）
const SCALE_STATUS: f32 = 1200.0;
/// 成长率百分点（`five_status_bonus` 进公式为 `1 + 0.01 * bonus`）
const SCALE_BONUS: f32 = 100.0;
/// 体力上限典型值
const SCALE_VITAL: f32 = 100.0;
/// 技能点典型量级
const SCALE_SKILL_PT: f32 = 1000.0;
/// 已学技能评分典型量级
const SCALE_SKILL_SCORE: f32 = 10000.0;
/// 打折级数典型量级
const SCALE_HINTS: f32 = 50.0;
/// 百分比类字段（训练加成 / 得意率 / 失败率等）
const SCALE_PCT: f32 = 100.0;
/// 设施等级计数（等级上限 5，计数为等级乘 4）
const SCALE_FACILITY: f32 = 20.0;
/// 诀窍库存上限
const SCALE_FEELING_STOCK: f32 = 10.0;
/// 诀窍槽满值（满 7 清零并加 1 诀窍）；小整数字段统一借用此尺度
const SCALE_SMALL: f32 = 7.0;
/// 隐藏风味库存上限
const SCALE_SPECIAL: f32 = 4.0;
/// 剧本 PT 典型量级
const SCALE_SCENARIO_PT: f32 = 1000.0;
/// 每年吃面次数上限
const SCALE_EAT: f32 = 5.0;
/// 训练等级剧本加成上限（RMJ 成功加 1，上限 5）
const SCALE_TRAIN_LEVEL_BONUS: f32 = 5.0;
/// 休息心得可持续回合的典型量级
const SCALE_REFRESH: f32 = 5.0;
/// 可自选比赛的回合数（回合 11-71 共 61 个 bit）
const SCALE_RACE_SLOTS: f32 = 61.0;
/// 单个自选比赛区间的要求场数典型量级
const SCALE_FREE_RACE_COUNT: f32 = 3.0;
/// 支援卡突破等级（0-4）
const SCALE_RANK: f32 = 4.0;
/// 团队卡 buff 可持续回合的典型量级
const SCALE_GROUP_BUFF: f32 = 5.0;

// ========== 枚举基数 ==========

/// 训练位数量（速耐力根智）
const TRAIN_NUM: usize = 5;
/// 得意训练类型数量（速耐力根智团）
const TRAIN_TYPE_NUM: usize = 6;
/// 支援卡类型数量（0-4 五种训练卡、5 友人、6 团队）
const CARD_TYPE_NUM: usize = 7;
/// [`PersonType`] 变体数
const PERSON_TYPE_NUM: usize = 7;
/// [`RamenStage`] 变体数 + **1 个预留空槽**（槽 0–9 给既有变体，槽 10 给
/// [`RamenStage::BeginAfterRegionSelect`]，槽 11 仍空）。
///
/// 预留是为了拆分阶段时 **只填空槽、不改 [`INPUT_DIM`]**，避免已落盘的教师数据作废。
/// 新增变体时只需在 [`stage_index`] 里给它分配一个 ≥ 10 的下标，本常量不动。
const STAGE_NUM: usize = 12;
/// 年份分档数（年 1 / 年 2 / 年 3 / URA）
const YEAR_NUM: usize = 4;
/// 干劲档位数（1-5）
const MOTIVATION_NUM: usize = 5;
/// 超级拉面选项数
const SUPER_RAMEN_NUM: usize = 3;
/// RMJ 结算次数
const RMJ_NUM: usize = 3;
/// 友人出行段数
const FRIEND_OUT_NUM: usize = 5;
/// [`FriendCardState`] 变体数
const FRIEND_CARD_NUM: usize = 3;
/// [`FriendOutState`] 变体数
const FRIEND_OUT_STATE_NUM: usize = 4;
/// 每年选中的地区数
const REGION_PER_YEAR: usize = 3;
/// 自选比赛等级档数（G1 / G2 / G3）
const GRADE_NUM: usize = 3;
/// 诀窍类型数
const FEELING_NUM: usize = 3;

/// 定长特征累加器
///
/// 只提供追加语义，不提供随机写——偏移量由写入顺序隐式决定，
/// 配合 [`Self::block`] 的宽度校验，避免手工维护 offset 常量。
struct FeatureWriter {
    /// 已写入的特征值
    buf: Vec<f32>
}

impl FeatureWriter {
    /// 建立一个预分配好容量的累加器
    fn new() -> Self {
        Self { buf: Vec::with_capacity(INPUT_DIM) }
    }

    /// 写入一个已归一化的原始值
    fn raw(&mut self, v: f32) {
        self.buf.push(v);
    }

    /// 按给定尺度归一化后写入
    fn num(&mut self, v: impl Into<f64>, scale: f32) {
        self.buf.push((v.into() / scale as f64) as f32);
    }

    /// 写入一个布尔位
    fn flag(&mut self, b: bool) {
        self.buf.push(if b { 1.0 } else { 0.0 });
    }

    /// 写入 `n` 维 one-hot；`idx` 为 `None` 或越界时全 0
    ///
    /// 越界不报错是有意的：`train_type = -1`（没有得意训练）、未登场人头等
    /// 都是合法的「没有」，用全 0 表示，与掩码位配合即可区分。
    fn onehot(&mut self, idx: Option<usize>, n: usize) {
        let start = self.buf.len();
        self.buf.resize(start + n, 0.0);
        if let Some(i) = idx {
            if i < n {
                self.buf[start + i] = 1.0;
            }
        }
    }

    /// 写入 `n` 维 multi-hot
    fn multihot(&mut self, idxs: impl IntoIterator<Item = usize>, n: usize) {
        let start = self.buf.len();
        self.buf.resize(start + n, 0.0);
        for i in idxs {
            if i < n {
                self.buf[start + i] = 1.0;
            }
        }
    }

    /// 补齐 `n` 个 0（未登场人头等整行留空的场合）
    fn zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0.0);
    }

    /// 写一个声明了期望宽度的特征块，写完立即校验
    fn block<F>(&mut self, name: &'static str, expect: usize, f: F) -> Result<()>
    where
        F: FnOnce(&mut Self) -> Result<()>
    {
        let start = self.buf.len();
        f(self)?;
        let got = self.buf.len() - start;
        ensure!(got == expect, "特征块 `{name}` 宽度不符: 期望 {expect}, 实际 {got}");
        Ok(())
    }
}

/// 把局面编码成定长特征向量
///
/// 返回长度恒为 [`INPUT_DIM`]；布局见模块文档。
pub fn encode(game: &RamenGame) -> Result<Vec<f32>> {
    let mut w = FeatureWriter::new();
    encode_global(game, &mut w)?;
    encode_cards(game, &mut w)?;
    encode_persons(game, &mut w)?;
    ensure!(w.buf.len() == INPUT_DIM, "特征总维度不符: 期望 {INPUT_DIM}, 实际 {}", w.buf.len());
    Ok(w.buf)
}

/// 人头下标到所在训练位的反查表（multi-hot 掩码）
///
/// **必须是 multi-hot 而非单个 `Option<usize>`**：拉面的分身不新建人头，而是把同一个
/// `person_index` 再 push 进另一个训练位（见 `RamenGame` 的地区分身与超级拉面分身），
/// 因此一个人头会同时出现在 `distribution` 的多行。用「最后写入的训练位」表示会让
/// 彩圈分身在特征里直接消失，只保留编号最大的那一位。
///
/// 返回长度为 `persons.len()`，每项是长度 [`TRAIN_NUM`] 的布尔掩码；负数下标
/// （空位占位）与越界下标一律忽略。
fn person_train_slots(game: &RamenGame) -> Vec<[bool; TRAIN_NUM]> {
    let mut slots = vec![[false; TRAIN_NUM]; game.persons.len()];
    for (t, row) in game.base.distribution.iter().enumerate().take(TRAIN_NUM) {
        for &p in row {
            if let Ok(idx) = usize::try_from(p) {
                if idx < slots.len() {
                    slots[idx][t] = true;
                }
            }
        }
    }
    slots
}

/// 编码 global 段
fn encode_global(game: &RamenGame, w: &mut FeatureWriter) -> Result<()> {
    let turn = game.turn();
    let uma = &game.base.uma;

    w.block("turn", G_TURN, |w| {
        w.num(turn, SCALE_TURN);
        w.num(turn % 24, SCALE_YEAR_TURN);
        // 年份分档：RMJ 在回合 23/47/71 结算，回合 72 起进入 URA
        let year = if turn <= 23 {
            0
        } else if turn <= 47 {
            1
        } else if turn <= 71 {
            2
        } else {
            3
        };
        w.onehot(Some(year), YEAR_NUM);
        w.onehot(Some(stage_index(&game.stage)), STAGE_NUM);
        Ok(())
    })?;

    w.block("uma", G_UMA, |w| {
        // 体力比例比绝对值更能跨马娘泛化（max_vital 因马娘而异），两者都给
        let max_vital = uma.max_vital.max(1) as f32;
        w.raw(uma.vital as f32 / max_vital);
        w.num(uma.vital, SCALE_VITAL);
        w.num(uma.max_vital, SCALE_VITAL);
        w.onehot(usize::try_from(uma.motivation - 1).ok(), MOTIVATION_NUM);
        for i in 0..5 {
            w.num(uma.five_status[i], SCALE_STATUS);
        }
        for i in 0..5 {
            w.num(uma.five_status_bonus[i], SCALE_BONUS);
        }
        for i in 0..5 {
            w.num(uma.five_status_limit[i], SCALE_STATUS);
        }
        w.num(uma.skill_pt, SCALE_SKILL_PT);
        w.num(uma.skill_score, SCALE_SKILL_SCORE);
        w.num(uma.total_hints, SCALE_HINTS);
        w.num(uma.race_bonus, SCALE_PCT);
        Ok(())
    })?;

    w.block("flags", G_FLAGS, |w| {
        let f = &uma.flags;
        w.flag(f.qiezhe);
        w.flag(f.aijiao);
        w.flag(f.good_trainer);
        w.flag(f.bad_trainer);
        w.flag(f.positive_thinking);
        w.flag(f.lucky);
        w.flag(f.doll);
        w.flag(f.ill);
        w.num(f.refresh_mind, SCALE_REFRESH);
        Ok(())
    })?;

    w.block("facility", G_FACILITY, |w| {
        for i in 0..TRAIN_NUM {
            w.num(game.base.train_level_count[i], SCALE_FACILITY);
        }
        w.num(game.base.absent_rate_drop, SCALE_PCT);
        Ok(())
    })?;

    w.block("ramen", G_RAMEN, |w| {
        let r = &game.ramen;
        for i in 0..FEELING_NUM {
            w.num(r.feeling_stock[i], SCALE_FEELING_STOCK);
        }
        for i in 0..FEELING_NUM {
            w.num(r.feeling_slot[i], SCALE_SMALL);
        }
        w.num(r.special_feeling, SCALE_SPECIAL);
        w.num(r.scenario_pt, SCALE_SCENARIO_PT);
        // rmj_results 长度随结算次数增长，固定 3 位并按「已结算且成功」编码
        for i in 0..RMJ_NUM {
            w.flag(r.rmj_results.get(i).copied().unwrap_or(false));
        }
        w.num(r.train_level_bonus, SCALE_TRAIN_LEVEL_BONUS);
        w.onehot(r.super_ramen, SUPER_RAMEN_NUM);
        w.flag(r.super_ramen.is_some());
        // —— schema v2 新增（处理上游合并时的 fixme）——
        // 1) 生效窗口：「已选但未生效」与「72-77 正在生效」由此位区分；
        w.flag(game.is_super_ramen_turn());
        // 2) 已选限制选项的合法训练位 multi-hot：网络直接读内容而不用背 id→内容表，
        //    数据表调整（training_limit_options）时无需改代码。未选时整组 0，
        //    由上面的 is_some 区分「没选」。下标查表失败属于数据异常，报错不静默补 0。
        match r.super_ramen {
            Some(opt) => {
                let data = global!(RAMENDATA);
                let slots = data
                    .finals_effect
                    .training_limit_options
                    .get(opt)
                    .ok_or_else(|| anyhow::anyhow!("超级拉面选项缺失: opt={opt}"))?;
                w.multihot(slots.iter().filter_map(|&t| usize::try_from(t).ok()), TRAIN_NUM);
            }
            None => w.zeros(TRAIN_NUM)
        }
        w.num(r.eat_count, SCALE_EAT);
        w.flag(game.deck_can_split);
        Ok(())
    })?;

    w.block("mark", G_MARK, |w| {
        // 诀窍角标：回合 2-71 每个训练位随机分配一种诀窍类型
        match game.ramen.train_feeling_type {
            Some(types) => {
                for t in types.iter().take(TRAIN_NUM) {
                    w.onehot(Some(feeling_index(*t)), FEELING_NUM);
                }
                w.flag(true);
            }
            None => {
                for _ in 0..TRAIN_NUM {
                    w.onehot(None, FEELING_NUM);
                }
                w.flag(false);
            }
        }
        Ok(())
    })?;

    w.block("friend", G_FRIEND, |w| {
        let f = &game.base.friend;
        w.onehot(Some(friend_card_index(f.card_state)), FRIEND_CARD_NUM);
        w.onehot(Some(friend_out_index(f.out_state)), FRIEND_OUT_STATE_NUM);
        for i in 0..FRIEND_OUT_NUM {
            w.flag(f.out_used.get(i).copied().unwrap_or(false));
        }
        w.num(f.group_buff_turn, SCALE_GROUP_BUFF);
        w.num(f.vital_bonus, SCALE_PCT);
        w.num(f.event_bonus, SCALE_PCT);
        Ok(())
    })?;

    w.block("region", G_REGION, |w| encode_regions(game, w))?;
    w.block("race", G_RACE, |w| encode_races(game, w))?;
    Ok(())
}

/// 编码地区段：当年选中的 3 个地区展开成效果数值，而非地区 id
///
/// 用 id one-hot 的话，网络必须从数据里自己学出每个 id 的效果，
/// 且第 3 年 C(10,3)=120 种组合下样本极稀疏；展开成
/// `xunlian / youqing / pt_bonus / hint_count / at_trains` 后
/// 可以泛化到没见过的地区组合。
///
/// 尚未选出时（live `selected_regions` 仍是默认 `[0,0,0]` 这类非法重复组合）
/// 效果块填零，避免编成三份「地区 0」。不新增维度。
fn encode_regions(game: &RamenGame, w: &mut FeatureWriter) -> Result<()> {
    // live 默认 [0,0,0] 不是合法组合。第 1 年 RegionSelect 根上若直接
    // `get(0)` 会编成三份「地区 0」的效果。当年尚未选出（id 有重复）时填零。
    let live = game.ramen.selected_regions;
    let unset = live[0] == live[1] || live[1] == live[2] || live[0] == live[2];
    if unset {
        // 每地区：xunlian / youqing / pt_bonus / hint_count + at_trains[TRAIN_NUM]
        w.zeros(REGION_PER_YEAR * (4 + TRAIN_NUM));
    } else {
        let data = global!(RAMENDATA);
        for &rid in live.iter().take(REGION_PER_YEAR) {
            let region = data
                .ramen_region_effect
                .get(rid)
                .ok_or_else(|| anyhow::anyhow!("地区效果缺失: region_id={rid}"))?;
            w.num(region.xunlian, SCALE_PCT);
            w.num(region.youqing, SCALE_PCT);
            w.num(region.pt_bonus, SCALE_PCT);
            w.num(region.hint_count, SCALE_SMALL);
            w.multihot(region.at_trains.iter().filter_map(|&t| usize::try_from(t).ok()), TRAIN_NUM);
        }
    }
    // 本回合正在吃 / 已选定要吃的面，指向 selected_regions 的哪一个
    for pick in [game.ramen.current_ramen, game.ramen.pending_ramen] {
        let slot = pick.and_then(|rid| game.ramen.selected_regions.iter().position(|&r| r == rid));
        w.onehot(slot, REGION_PER_YEAR);
        w.flag(pick.is_some());
    }
    Ok(())
}

/// 编码比赛段：位图聚合成计数，自选比赛按区间给「缺口 / 剩余合格回合」
///
/// 缺口与剩余合格回合是原始状态的确定性函数（不含任何权重判断），
/// 不给的话网络得从 `win_races` 位图自己学出比赛等级过滤规则，代价过高。
fn encode_races(game: &RamenGame, w: &mut FeatureWriter) -> Result<()> {
    let uma = &game.base.uma;
    let turn = game.turn();
    w.num(uma.career_races.count_ones(), SCALE_RACE_SLOTS);
    w.num(uma.win_races.count_ones(), SCALE_RACE_SLOTS);
    w.flag(game.base.can_self_race());

    match uma.find_free_race(turn) {
        Some(free) => {
            w.flag(true);
            let done = uma.count_free_race(free);
            w.num(free.count.saturating_sub(done), SCALE_FREE_RACE_COUNT);
            w.num(remaining_race_slots(turn, free), SCALE_RACE_SLOTS);
            // grade 取值 1/2/3 对应 G1/G2/G3；减 1 后落到 one-hot 下标
            w.onehot(free.grade.and_then(|g| usize::try_from(g).ok()).and_then(|g| g.checked_sub(1)), GRADE_NUM);
        }
        None => {
            w.flag(false);
            w.raw(0.0);
            w.raw(0.0);
            w.onehot(None, GRADE_NUM);
        }
    }
    // 后面是否还有未开始的自选比赛区间
    let next_start = u32::try_from(turn.max(0)).unwrap_or(0);
    let has_next = uma.get_data()?.free_races.iter().any(|f| f.start_turn > next_start);
    w.flag(has_next);
    Ok(())
}

/// 编码 cards 段（置换等变序列，槽位顺序即卡组顺序）
fn encode_cards(game: &RamenGame, w: &mut FeatureWriter) -> Result<()> {
    let slots = person_train_slots(game);
    for i in 0..CARD_NUM {
        w.block("card", CARD_DIM, |w| {
            let Some(card) = game.base.deck.get(i) else {
                // 卡组恒为 6 张；缺位只可能是构造异常，不静默补 0
                bail!("卡组槽位缺失: index={i}, 实际长度={}", game.base.deck.len());
            };
            w.onehot(usize::try_from(card.card_type).ok(), CARD_TYPE_NUM);
            w.num(card.rank, SCALE_RANK);
            w.num(card.friendship, SCALE_PCT);
            w.flag(card.is_link_card);
            w.flag(card.is_locked);
            w.num(card.total_hints, SCALE_SMALL);
            let e = &card.effect;
            w.raw(e.youqing / SCALE_PCT);
            w.num(e.ganjing, SCALE_PCT);
            w.num(e.xunlian, SCALE_PCT);
            w.num(e.saihou, SCALE_PCT);
            w.raw(e.deyilv / SCALE_PCT);
            for j in 0..6 {
                w.num(e.bonus[j], SCALE_PCT);
            }
            w.num(e.wiz_vital_bonus, SCALE_PCT);
            w.raw(e.fail_rate_drop / SCALE_PCT);
            w.raw(e.vital_cost_drop / SCALE_PCT);
            w.num(e.event_effect_up, SCALE_PCT);
            w.num(e.event_recovery_amount_up, SCALE_PCT);
            w.num(e.hint_count_bonus, SCALE_SMALL);
            // 该卡对应的人头当前在哪个训练位。
            // 人头下标 ≠ 卡组下标（拉面里理事长占人头 5、友人卡在人头 6），必须按 card_id 反查。
            let person_idx = game.persons.iter().position(|p| p.card_id == Some(card.card_id));
            let mask = person_idx.and_then(|pi| slots.get(pi)).copied().unwrap_or([false; TRAIN_NUM]);
            // multi-hot：同一张卡的分身会同时占多个训练位
            for in_train in mask {
                w.flag(in_train);
            }
            w.flag(mask.iter().any(|x| *x));
            Ok(())
        })?;
    }
    Ok(())
}

/// 编码 persons 段（第二个置换等变序列，未登场的人头整行为 0）
fn encode_persons(game: &RamenGame, w: &mut FeatureWriter) -> Result<()> {
    let slots = person_train_slots(game);
    for i in 0..PERSON_NUM {
        w.block("person", PERSON_DIM, |w| {
            let Some(p) = game.persons.get(i) else {
                // 未登场：整行 0（含「已登场」掩码位），与真实全 0 由掩码位区分
                w.zeros(PERSON_DIM);
                return Ok(());
            };
            w.onehot(Some(person_type_index(p.person_type)), PERSON_TYPE_NUM);
            w.onehot(usize::try_from(p.train_type).ok(), TRAIN_TYPE_NUM);
            w.num(p.friendship, SCALE_PCT);
            w.flag(p.is_hint);
            w.flag(true); // 已登场
            let mask = slots.get(i).copied().unwrap_or([false; TRAIN_NUM]);
            // multi-hot：分身会让同一个人头同时占多个训练位
            for in_train in mask {
                w.flag(in_train);
            }
            w.flag(mask.iter().any(|x| *x));
            // 剧本友人旗标。不能用 `friend.person_index`——它在回合 0-1 还是卡组下标，
            // 要到 `add_friend_and_npcs` 才改成真正的人头下标，期间会把理事长标成友人。
            w.flag(p.person_type == PersonType::ScenarioCard);
            // 对应的支援卡槽位。人头下标 ≠ 卡组下标，按 card_id 反查。
            let card_slot = p.card_id.and_then(|cid| game.base.deck.iter().position(|c| c.card_id == cid));
            w.onehot(card_slot, CARD_NUM);
            w.flag(card_slot.is_some());
            Ok(())
        })?;
    }
    Ok(())
}

/// [`RamenStage`] 到 one-hot 下标
///
/// 必须显式 `match`，**不得依赖枚举判别值**：变体顺序若调整，显式 match 会编译
/// 报错提醒同步，而判别值会静默改变所有已落盘样本的含义。
fn stage_index(stage: &RamenStage) -> usize {
    match stage {
        RamenStage::Begin => 0,
        RamenStage::Distribute => 1,
        RamenStage::RamenSelect => 2,
        RamenStage::SpecialSelect => 3,
        RamenStage::Train => 4,
        RamenStage::AfterTrain => 5,
        RamenStage::NextTurn => 6,
        RamenStage::RegionSelect => 7,
        RamenStage::SuperRamenSelect => 8,
        RamenStage::Settlement => 9,
        // 预留槽 10；槽 11 仍空。0–9 不得重排。
        RamenStage::BeginAfterRegionSelect => 10
    }
}

/// [`FeelingType`] 到 one-hot 下标（理由同 [`stage_index`]）
fn feeling_index(t: FeelingType) -> usize {
    match t {
        FeelingType::A => 0,
        FeelingType::B => 1,
        FeelingType::C => 2
    }
}

/// [`PersonType`] 到 one-hot 下标（理由同 [`stage_index`]）
fn person_type_index(t: PersonType) -> usize {
    match t {
        PersonType::Card => 0,
        PersonType::ScenarioCard => 1,
        PersonType::Npc => 2,
        PersonType::Yayoi => 3,
        PersonType::Reporter => 4,
        PersonType::OtherFriend => 5,
        PersonType::TeamCard => 6
    }
}

/// [`FriendCardState`] 到 one-hot 下标（理由同 [`stage_index`]）
fn friend_card_index(s: FriendCardState) -> usize {
    match s {
        FriendCardState::Empty => 0,
        FriendCardState::SSR => 1,
        FriendCardState::R => 2
    }
}

/// [`FriendOutState`] 到 one-hot 下标（理由同 [`stage_index`]）
fn friend_out_index(s: FriendOutState) -> usize {
    match s {
        FriendOutState::UnClicked => 0,
        FriendOutState::BeforeUnlock => 1,
        FriendOutState::AfterUnlock => 2,
        FriendOutState::Away => 3
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};

    use super::*;
    use crate::{
        gamedata::init_global,
        sampler::{SamplerConfig, SamplingSpace, sample_position},
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    /// 造一个开局局面（默认卡组 102601）
    fn make_game() -> Result<RamenGame> {
        let inherit = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        RamenGame::newgame(102601, &deck, inherit)
    }

    /// 回归：cards 段与 persons 段按 `card_id` 互相反查，不假设两个序列同序
    ///
    /// 拉面布局下理事长占人头 5、友人卡占人头 6，而友人卡在 `deck[5]`。
    /// 修复前编码器用「人头下标 == 卡组下标」的假设，会把卡槽 5 连到理事长的训练位，
    /// 并让人头 6 的卡链接整个丢失。
    #[test]
    fn test_card_person_cross_lookup() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut c = Checks::new();
        let mut game = make_game()?;
        game.add_friend_and_npcs()?;

        let friend_person = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::ScenarioCard)
            .ok_or_else(|| anyhow!("友人卡人头应存在"))?;
        let friend_deck =
            Game::deck_index_of(&game, friend_person).ok_or_else(|| anyhow!("友人卡应能反查到卡组槽位"))?;
        println!("友人卡: 人头 {friend_person} -> 卡组 {friend_deck}");
        c.check(friend_person != friend_deck, "本用例的前提是两个下标不相等");

        // 把友人卡放进 2 号训练位（力），理事长放进 0 号
        let yayoi_person = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Yayoi)
            .ok_or_else(|| anyhow!("理事长人头应存在"))?;
        game.base.distribution = vec![vec![]; TRAIN_NUM];
        game.base.distribution[0] = vec![yayoi_person as i32];
        game.base.distribution[2] = vec![friend_person as i32];

        let v = encode(&game)?;

        // card 块尾部布局：onehot(slot, TRAIN_NUM) + flag(slot.is_some())
        let card_base = GLOBAL_DIM + friend_deck * CARD_DIM;
        let card_slot_onehot = card_base + CARD_DIM - 1 - TRAIN_NUM;
        let card_in_train = v[card_base + CARD_DIM - 1];
        println!(
            "卡槽 {friend_deck} 的训练位 onehot = {:?}, 在训练标志 = {card_in_train}",
            &v[card_slot_onehot..card_slot_onehot + TRAIN_NUM]
        );
        c.check(card_in_train == 1.0, "友人卡所在的卡槽应标记为「在训练中」");
        c.check(v[card_slot_onehot + 2] == 1.0, "友人卡的卡槽应指向 2 号训练位");

        // person 块尾部布局：onehot(card_slot, CARD_NUM) + flag(card_slot.is_some())
        let person_base = GLOBAL_DIM + CARD_NUM * CARD_DIM + friend_person * PERSON_DIM;
        let card_slot_onehot = person_base + PERSON_DIM - 1 - CARD_NUM;
        println!(
            "人头 {friend_person} 的卡槽 onehot = {:?}",
            &v[card_slot_onehot..card_slot_onehot + CARD_NUM]
        );
        c.check(v[person_base + PERSON_DIM - 1] == 1.0, "友人卡人头应有卡链接");
        c.check(
            v[card_slot_onehot + friend_deck] == 1.0,
            &format!("友人卡人头应链接到卡组槽位 {friend_deck}")
        );

        // 理事长是无卡人头：不应有任何卡链接
        let yayoi_base = GLOBAL_DIM + CARD_NUM * CARD_DIM + yayoi_person * PERSON_DIM;
        c.check(
            v[yayoi_base + PERSON_DIM - 1] == 0.0,
            &format!("理事长(人头 {yayoi_person}) 不应有卡链接")
        );

        c.finish()
    }

    /// 回归：分身占据的多个训练位必须全部编进特征（multi-hot，不是最后写入的那一个）
    ///
    /// 拉面的分身不新建人头，而是把同一个 `person_index` 再 push 进另一个训练位。
    /// 修复前 `person_train_slots` 用 `slots[idx] = Some(t)` 后写覆盖，
    /// 彩圈分身在特征里直接消失，只保留编号最大的训练位。
    #[test]
    fn test_split_person_multi_hot() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = make_game()?;
        game.add_friend_and_npcs()?;

        // 人头 0（第一张训练卡）同时出现在 1 号与 3 号训练位，模拟分身
        game.base.distribution = vec![vec![]; TRAIN_NUM];
        game.base.distribution[1] = vec![0];
        game.base.distribution[3] = vec![0];

        let v = encode(&game)?;
        let deck_idx = Game::deck_index_of(&game, 0).ok_or_else(|| anyhow!("人头 0 应能反查到卡组槽位"))?;

        // card 块尾部：TRAIN_NUM 个训练位标志 + 1 个「在训练中」标志
        let card_base = GLOBAL_DIM + deck_idx * CARD_DIM;
        let card_mask = &v[card_base + CARD_DIM - 1 - TRAIN_NUM..card_base + CARD_DIM - 1];
        println!("卡槽 {deck_idx} 的训练位掩码 = {card_mask:?}");
        let mut c = Checks::new();
        c.check(card_mask[1] == 1.0 && card_mask[3] == 1.0, "分身应同时占 1 号与 3 号训练位");
        c.check(
            card_mask[0] == 0.0 && card_mask[2] == 0.0 && card_mask[4] == 0.0,
            "未占用的训练位应为 0"
        );

        // person 块的训练位掩码位于「已登场」之后
        let person_base = GLOBAL_DIM + CARD_NUM * CARD_DIM;
        let person_mask_start = person_base + PERSON_TYPE_NUM + TRAIN_TYPE_NUM + 3;
        let person_mask = &v[person_mask_start..person_mask_start + TRAIN_NUM];
        println!("人头 0 的训练位掩码 = {person_mask:?}");
        c.check(person_mask[1] == 1.0 && person_mask[3] == 1.0, "persons 段同样应是 multi-hot");

        c.finish()
    }

    /// 各分块宽度之和必须等于声明的总维度（纯常量校验，不跑局面）
    #[test]
    fn test_dim_constants_consistent() -> Result<()> {
        let global_parts = [
            ("turn", G_TURN),
            ("uma", G_UMA),
            ("flags", G_FLAGS),
            ("facility", G_FACILITY),
            ("ramen", G_RAMEN),
            ("mark", G_MARK),

            ("friend", G_FRIEND),
            ("region", G_REGION),
            ("race", G_RACE)
        ];
        let sum: usize = global_parts.iter().map(|(_, n)| n).sum();
        for (name, n) in global_parts {
            println!("  global/{name:<9} {n:>4}");
        }
        println!("global 合计 {sum} (GLOBAL_DIM={GLOBAL_DIM})");
        println!("cards   {CARD_NUM} x {CARD_DIM} = {}", CARD_NUM * CARD_DIM);
        println!("persons {PERSON_NUM} x {PERSON_DIM} = {}", PERSON_NUM * PERSON_DIM);
        println!("总维度 INPUT_DIM = {INPUT_DIM}");
        let mut c = Checks::new();
        c.check(sum == GLOBAL_DIM, "global 分块之和必须等于 GLOBAL_DIM");
        c.check(
            INPUT_DIM == GLOBAL_DIM + CARD_NUM * CARD_DIM + PERSON_NUM * PERSON_DIM,
            "INPUT_DIM 必须等于 global + cards + persons 三段之和"
        );
        c.finish()
    }

    /// P0.5：`STAGE_NUM` 预留空槽后仍大于 `RamenStage` 实际变体数
    ///
    /// 槽 10 已分给 `BeginAfterRegionSelect`，还剩 1 个空槽。
    /// schema v2：超级拉面补「生效窗口 + 限制位 multi-hot」各 1/5 维后
    /// `INPUT_DIM` 从 754 升到 760。
    #[test]
    fn test_stage_num_reserve_slots() -> Result<()> {
        let n_stage = enum_iterator::cardinality::<RamenStage>();
        println!("RamenStage 变体数 {n_stage}");
        println!("STAGE_NUM = {STAGE_NUM}");
        println!("G_TURN = {G_TURN}");
        println!("GLOBAL_DIM = {GLOBAL_DIM}");
        println!("INPUT_DIM = {INPUT_DIM}");
        let mut c = Checks::new();
        c.check(INPUT_DIM == 760, "schema v2（超级拉面 +6 维）后 INPUT_DIM 必须为 760");
        c.check(G_RAMEN == 24, "ramen 块应为 v1 的 18 + 1 窗口 + 5 multi-hot = 24");
        c.check(
            STAGE_NUM > n_stage,
            &format!("STAGE_NUM={STAGE_NUM} 必须大于 RamenStage 变体数 {n_stage}（还剩 1 个空槽）")
        );
        c.check(
            STAGE_NUM - n_stage == 1,
            "预留槽用掉一个后应还剩 1 个"
        );
        c.check(
            G_TURN == 2 + YEAR_NUM + STAGE_NUM,
            "G_TURN 必须等于 2 个 num + YEAR_NUM + STAGE_NUM"
        );
        c.finish()
    }

    /// 开局局面能编码，长度恒为 INPUT_DIM，且不含 NaN / Inf
    #[test]
    fn test_encode_newgame() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let game = make_game()?;
        let v = encode(&game)?;
        println!("开局编码长度 {} (期望 {INPUT_DIM})", v.len());
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        let nonzero = v.iter().filter(|x| **x != 0.0).count();
        println!("非有限值 {bad} 个, 非零 {nonzero} 个 ({:.1}%)", nonzero as f64 * 100.0 / v.len() as f64);
        let mut c = Checks::new();
        c.check(v.len() == INPUT_DIM, "开局编码长度等于 INPUT_DIM");
        c.check(bad == 0, "特征不得含 NaN / Inf");
        c.finish()
    }

    /// 采样器产出的各阶段根局面都能编码，且维度恒定
    ///
    /// 覆盖 `RamenSelect / SpecialSelect / Train / RegionSelect` 各阶段与全回合段，
    /// 是「各分块在真实局面上边界正确」的实证——分块宽度不符会由 `block` 直接报错。
    #[test]
    fn test_encode_sampled_positions() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();
        let mut ok = 0usize;
        let mut skipped = 0usize;
        let mut stages: std::collections::BTreeMap<String, usize> = Default::default();
        let mut nonzero_min = usize::MAX;
        let mut nonzero_max = 0usize;
        let mut c = Checks::new();
        for index in 0..300u64 {
            match sample_position(&space, &config, index)? {
                crate::sampler::SampleOutcome::Captured(pos) => {
                    let v = encode(&pos.game)?;
                    if v.len() != INPUT_DIM {
                        c.check(false, &format!("index={index} 维度不符（得到 {}）", v.len()));
                    }
                    if !v.iter().all(|x| x.is_finite()) {
                        c.check(false, &format!("index={index} 含 NaN / Inf"));
                    }
                    let nz = v.iter().filter(|x| **x != 0.0).count();
                    nonzero_min = nonzero_min.min(nz);
                    nonzero_max = nonzero_max.max(nz);
                    *stages.entry(format!("{:?}", pos.game.stage)).or_default() += 1;
                    ok += 1;
                }
                crate::sampler::SampleOutcome::Exhausted { .. } => skipped += 1
            }
        }
        println!("300 次采样：编码成功 {ok}，Exhausted {skipped}");
        println!("阶段分布: {stages:?}");
        println!("非零特征数区间 [{nonzero_min}, {nonzero_max}] / {INPUT_DIM}");
        c.check(ok > 0, "应至少编码成功一个局面");
        c.finish()
    }

    /// 同一局面编码两次必须逐位相同（编码器不得含随机性或读取可变全局态）
    #[test]
    fn test_encode_deterministic() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let game = make_game()?;
        let a = encode(&game)?;
        let b = encode(&game)?;
        let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        println!("两次编码不同位数: {diff}");
        let mut c = Checks::new();
        c.check(diff == 0, "同一局面编码两次必须逐位相同");
        c.finish()
    }

    /// 成长率必须真的进特征：改 `five_status_bonus` 后编码必须变化
    ///
    /// 温泉版 `extract_nn_features` 漏掉了这一项，第一代要用 7 个马娘，
    /// 缺了它网络无法区分面板相同、成长率不同的马娘。本测试专防回归。
    #[test]
    fn test_status_bonus_reaches_features() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let game = make_game()?;
        let base = encode(&game)?;
        let mut mutated = game.clone();
        mutated.base.uma.five_status_bonus[0] += 10;
        let after = encode(&mutated)?;
        let diff: Vec<usize> = base
            .iter()
            .zip(after.iter())
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .map(|(i, _)| i)
            .collect();
        println!("改动 five_status_bonus[0] 后变化的特征下标: {diff:?}");
        let mut c = Checks::new();
        c.check(diff.len() == 1, "改成长率应恰好只有一位特征变化");

        let mut limited = game.clone();
        limited.base.uma.five_status_limit[3] += 100;
        let after2 = encode(&limited)?;
        let diff2 = base.iter().zip(after2.iter()).filter(|(x, y)| x != y).count();
        println!("改动 five_status_limit[3] 后变化位数: {diff2}");
        c.check(diff2 == 1, "属性上限也必须进特征");
        c.finish()
    }

    /// 第 1 年 RegionSelect 根：维度不变、未选择时地区效果块为零、
    /// `BeginAfterRegionSelect` 占用预留槽 10
    #[test]
    fn test_year1_region_root_features() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut c = Checks::new();
        let mut unset = make_game()?;
        unset.base.turn = 2;
        unset.stage = RamenStage::RegionSelect;
        let v_unset = encode(&unset)?;
        println!(
            "y1 RegionSelect 未选择: len={} live={:?} yearly={:?}",
            v_unset.len(),
            unset.ramen.selected_regions,
            unset.ramen.yearly_selected_regions
        );
        c.check(v_unset.len() == INPUT_DIM, "y1 RegionSelect 根特征长度 == INPUT_DIM(760)");
        c.check(INPUT_DIM == 760, "schema v2 下 INPUT_DIM 为 760");

        let mut set = unset.clone();
        set.ramen.selected_regions = [0, 1, 2];
        set.ramen.yearly_selected_regions[0] = [0, 1, 2];
        let v_set = encode(&set)?;
        let diffs: Vec<(usize, f32, f32)> = v_unset
            .iter()
            .zip(v_set.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        println!("未选择 vs 已选择差异 {} 位，前几个: {:?}", diffs.len(), diffs.iter().take(6).collect::<Vec<_>>());
        c.check(!diffs.is_empty(), "选择后地区块应与未选择不同");
        c.check(
            diffs.iter().all(|(_, a, _)| *a == 0.0),
            "未选择时地区效果块必须为零（不能编成三份地区 0）"
        );

        let mut after = unset.clone();
        after.stage = RamenStage::BeginAfterRegionSelect;
        let mut begin = unset.clone();
        begin.stage = RamenStage::Begin;
        let v_after = encode(&after)?;
        let v_begin = encode(&begin)?;
        let stage_diffs: Vec<usize> = v_begin
            .iter()
            .zip(v_after.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        println!("Begin vs BeginAfterRegionSelect 差异下标: {stage_diffs:?}");
        c.check(stage_diffs.len() == 2, "阶段 one-hot 应恰好两位变化（槽 0 ↔ 槽 10）");
        if stage_diffs.len() == 2 {
            let gap = stage_diffs[1].abs_diff(stage_diffs[0]);
            println!("槽距 {gap}（期望 10）");
            c.check(gap == 10, "BeginAfterRegionSelect 使用预留槽 10");
        }
        c.finish()
    }

    /// schema v2：超级拉面的窗口位与限制位 multi-hot 必须真的进特征
    ///
    /// 设 `super_ramen = Some(选项二)` 且 turn=72（生效窗口内）：相对未选状态应恰好
    /// 变化 `1(onehot 选项位) + 1(is_some) + 1(窗口) + len(选项二合法位)` 位。
    /// 选项二数据为 [0,1,2,4]，共 4 位合法 → 恰好 7 位变化。专防 v2 两处盲区回归。
    #[test]
    fn test_super_ramen_v2_dims_reach_features() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let base_game = make_game()?;
        let base = encode(&base_game)?;

        let mut chosen = base_game.clone();
        chosen.base.turn = 72; // 生效窗口内
        chosen.ramen.super_ramen = Some(1); // 选项二（FIXED_SUPER_RAMEN_INDEX 同款位置语义）
        let after = encode(&chosen)?;

        let diffs: Vec<usize> = base
            .iter()
            .zip(after.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        println!("选中选项二(turn=72) 变化的特征下标({}): {diffs:?}", diffs.len());

        let mut c = Checks::new();
        c.check(diffs.len() == 7, "应恰好 7 位变化：onehot(1)+is_some(1)+窗口(1)+multi-hot(4)");
        c.check(chosen.is_super_ramen_turn(), "turn=72 应判定为生效窗口");

        // 未选 + 非窗口：新增 6 维必须全部为 0（基线局面即如此）
        let new_dims_all_zero = base
            .iter()
            .skip(GLOBAL_DIM - 0) // 全向量扫描下方的 ramen 块新增段即可，简化为逐位检查
            .count(); // 占位避免 unused；真实检查放在下方显式区间
        let _ = new_dims_all_zero;
        let ramen_new_start = GLOBAL_DIM - G_REGION - G_RACE - G_FRIEND - G_MARK - G_RAMEN + 13; // 块内第 14 位起为 v2 段
        let v2_slice = &base[ramen_new_start..ramen_new_start + 1 + TRAIN_NUM];
        println!("未选状态下 v2 六维 = {v2_slice:?}");
        c.check(v2_slice.iter().all(|&x| x == 0.0), "未选超级拉面时 v2 六维应全 0");
        c.finish()
    }
}
