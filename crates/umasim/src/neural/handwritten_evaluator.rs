//! 手写启发式评估器
//!
//! 基于规则的评估器实现，不依赖神经网络。
//! 用于 MCTS 的模拟阶段和终局评估。
//!
//! # 优化项（参考 C++ HandwrittenLogic.cpp）
//!
//! 1. 属性上限软约束 - 避免属性溢出浪费
//! 2. 体力分段评估 - 更精细的体力管理
//! 3. 羁绊价值提升 - 前期更重视羁绊

use anyhow::Result;
use rand::{rngs::StdRng, seq::IndexedRandom};

use super::{Evaluator, ValueOutput};
use crate::{
    game::{
        ActionScore, Game, PersonType, onsen::{action::OnsenAction, game::OnsenGame}
    },
    gamedata::{GAMECONSTANTS, GameConfig, onsen::ONSENDATA}, global
};

// ============================================================================
// 常量定义（参考 umaai的手写逻辑）
// ============================================================================

/// 属性权重 [速度, 耐力, 力量, 根性, 智力]
/// 速力优先：速度 > 力量 > 耐根 > 智力 （没带智卡）
const STATUS_WEIGHTS: [f64; 5] = [7.0, 8.0, 8.0, 8.0, 6.0];

/// 没带卡的属性权重
const ABSENT_WEIGHT: f64 = 2.0;

/// 训练类型权重调整 [速度训练, 耐力训练, 力量训练, 根性训练, 智力训练]
/// 用于额外偏好某些训练类型
const TRAIN_TYPE_BONUS: [f64; 5] = [20.0, 10.0, 30.0, 30.0, 20.0];

/// 智力训练人头阈值：智力训练人头数>此值时才考虑选择
const ABSENT_HEAD_THRESHOLD: usize = 2;

/// 前期回合阈值：前期更倾向于智力训练（不怎么消耗体力+攒羁绊+多挖掘回合）
/// 缩短前期窗口，只有前12回合考虑智力
const EARLY_TURN_THRESHOLD: i32 = 12;

/// 控属性预留空间因子
const RESERVE_STATUS_FACTOR: f64 = 40.0;

/// 羁绊基础价值（C++ jibanValue = 12）
const JIBAN_VALUE: f64 = 12.0;

/// 体力价值因子（游戏开始时）
const VITAL_FACTOR_START: f64 = 3.5;

/// 体力价值因子（游戏结束时）
const VITAL_FACTOR_END: f64 = 2.0;

/// 小失败惩罚值
const SMALL_FAIL_VALUE: f64 = -1500.0;

/// 大失败惩罚值
const BIG_FAIL_VALUE: f64 = -1800.0;

/// 外出加成（干劲不满时）
const OUTGOING_BONUS_IF_NOT_FULL_MOTIVATION: f64 = 200.0;

/// 最终事件预估属性加成
const FINAL_BONUS: i32 = 45 + 30 + 20 + 20; // URA3 + 最终事件 + URA2 + URA1

// ============================================================================
// 挖掘评估相关常量
// ============================================================================

/// 完成一个温泉的奖励（获得温泉券 + 装备升级 + 源泉选择）
const DIG_COMPLETE_BONUS: f64 = 800.0;

/// 挖掘量基础价值系数（每点挖掘量的价值）
const DIG_BASE_VALUE: f64 = 0.5;

// ============================================================================
// 比赛评估相关常量
// ============================================================================

/// 目标比赛基础价值（参考 C++ raceBonus = 150）
const RACE_BASE_BONUS: f64 = 450.0;

/// 非目标比赛基础价值
const NON_TARGET_RACE_BONUS: f64 = 180.0;

/// 自选比赛最大额外价值 (/剩余回合数)
const FREE_RACE_BONUS: f64 = 2000.0;

/// 狄杜斯角色ID（生涯比赛额外加成）
const DIDUS_CHARA_ID: u32 = 1063;

// ============================================================================
// 超回复策略相关常量
// ============================================================================

/// 友人外出基础价值
const FRIEND_OUTING_BASE_VALUE: f64 = 400.0;

/// 高累计体力阈值（接近保底275，避免友人外出浪费）
const SUPER_RECOVERY_VITAL_COST_HIGH: i32 = 200;

/// 中累计体力阈值
const SUPER_RECOVERY_VITAL_COST_MID: i32 = 100;

/// 低累计体力阈值（友人外出可快速获得超回复）
const SUPER_RECOVERY_VITAL_COST_LOW: i32 = 60;

/// 友人外出在低累计体力时的额外收益（快速获得超回复）
const FRIEND_OUTING_SUPER_BONUS: f64 = 200.0;

/// 友人外出在高累计体力时的惩罚（避免浪费即将自然触发的超回复）
const FRIEND_OUTING_HIGH_COST_PENALTY: f64 = 300.0;

/// 友人外出在中累计体力时的惩罚
const FRIEND_OUTING_MID_COST_PENALTY: f64 = 150.0;

/// 友人外出在特定回合区间时的收益，每8回合一组
const FRIEND_OUTING_TURN_MODIFIER: [f64; 10] =
    [-400.0, -400.0, -200.0, 200.0, 0.0, 200.0, 400.0, -400.0, 200.0, -400.0];

/// 第一年无券时的PR分数
const PR_SCORE: f64 = 100.0;

/// 第一年无券且有超回复时的PR分数
const PR_SCORE_SUPER: f64 = 300.0;

/// 速度Build温泉顺序
const ONSEN_ORDER_SPEED: [u32; 11] = [1, 3, 2, 5, 7, 4, 8, 9, 4, 2, 6];

/// 耐力Build温泉顺序
const ONSEN_ORDER_STAMINA: [u32; 11] = [1, 3, 2, 5, 6, 4, 8, 9, 4, 2, 7];

/// 设施升级对应温泉
/// 1. 疾驰 - 砂
/// 2. 坚忍 - 土
/// 3. 明晰 - 岩
/// 4. 骏闪 - 砂
/// 5. 刚足 - 砂
/// 6. 健壮 - 土
/// 7. 天翔 - 土
/// 8. 秘汤 - 岩
/// 9. 传说 - 岩
const RECOMMENDED_TOOL_TYPE: [usize; 9] = [0, 1, 2, 0, 0, 1, 1, 2, 2];

/// 挖完秘汤汤驹后，忽略体力计算（后面可以全覆盖）
const SECRET_ONSEN_INDEX: usize = 8;

// ============================================================================
// 辅助函数
// ============================================================================

/// 属性上限软约束函数
fn status_soft_function(x: f64, reserve: f64) -> f64 {
    if reserve <= 0.0 {
        return x.min(0.0);
    }
    let reserve_inv_x2 = 1.0 / (2.0 * reserve);

    if x >= 0.0 {
        0.0
    } else if x > -reserve {
        -x * x * reserve_inv_x2 * 0.5
    } else {
        x + 0.25 * reserve
    }
}

/// 体力评估函数（分段函数）
fn vital_evaluation(vital: i32, max_vital: i32) -> f64 {
    if vital <= 50 {
        2.0 * vital as f64
    } else if vital <= 70 {
        1.5 * (vital - 50) as f64 + vital_evaluation(50, max_vital)
    } else if vital <= max_vital {
        1.0 * (vital - 70) as f64 + vital_evaluation(70, max_vital)
    } else {
        vital_evaluation(max_vital, max_vital)
    }
}

/// 计算当前回合的体力价值因子
fn calc_vital_factor(turn: i32, max_turn: i32) -> f64 {
    VITAL_FACTOR_START + (turn as f64 / max_turn as f64) * (VITAL_FACTOR_END - VITAL_FACTOR_START)
}

/// 统计已挖掘温泉数量
/*
fn count_completed_onsen(game: &OnsenGame) -> usize {
    game.onsen_state.iter().filter(|&&x| x).count()
}
    */

/// 判断是否应该跳过体力评估惩罚
///
/// # 策略说明
/// 挖完秘汤汤驹后，温泉的回复可以覆盖所有体力消耗
/// 此时体力管理变得不那么重要，可以移除体力评估的惩罚性因素
fn should_skip_vital_penalty(game: &OnsenGame) -> bool {
    game.onsen_state[SECRET_ONSEN_INDEX]
}

pub fn load_onsen_order() -> Result<Vec<u32>> {
    // 1. 先读取配置文件
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let game_config: GameConfig = toml::from_str(&config_file)?;
    Ok(game_config.onsen_order)
}

/// 手写启发式评估器
#[derive(Debug, Clone)]
pub struct HandwrittenEvaluator {
    /// 属性权重 [速度, 耐力, 力量, 根性, 智力]
    pub weights: [f64; 5],
    /// 技能点权重
    pub skill_weight: f64,
    /// 体力阈值（低于此值考虑休息）
    pub vital_threshold: i32,
    /// 彩圈加成系数
    pub shining_bonus: f64,
    /// 温泉顺序
    pub onsen_order: Vec<u32>
}

impl HandwrittenEvaluator {
    /// 创建默认评估器
    pub fn new() -> Self {
        Self {
            weights: [1.0, 1.0, 1.0, 1.0, 1.0],
            skill_weight: 0.5,
            vital_threshold: 55,
            shining_bonus: 35.0,
            onsen_order: load_onsen_order().expect("onsen order")
        }
    }

    /// 速度型配置
    pub fn speed_build() -> Self {
        Self {
            weights: [1.2, 0.9, 1.0, 0.8, 0.8],
            skill_weight: 0.5,
            vital_threshold: 55,
            shining_bonus: 35.0,
            onsen_order: ONSEN_ORDER_SPEED.to_vec()
        }
    }

    /// 耐力型配置
    pub fn stamina_build() -> Self {
        Self {
            weights: [0.9, 1.2, 1.0, 0.9, 0.8],
            skill_weight: 0.5,
            vital_threshold: 55,
            shining_bonus: 35.0,
            onsen_order: ONSEN_ORDER_STAMINA.to_vec()
        }
    }

    /// 判断是否应该使用温泉券
    ///
    /// # 普通温泉券使用
    /// 温泉券>=2，或者体力低于阈值55，或者挖了秘汤
    fn should_use_ticket(&self, game: &OnsenGame) -> bool {
        // 基础检查：没有温泉券或正在buff中
        if game.bathing.ticket_num == 0 || game.bathing.buff_remain_turn > 0 {
            return false;
        }

        // 超回复: 有就立即用
        if game.bathing.is_super_ready {
            return true;
        }

        // 普通温泉券使用：温泉券>=2，或者体力低于阈值，或者挖了秘汤
        if game.bathing.ticket_num >= 2 || game.uma.vital < self.vital_threshold || game.onsen_state[SECRET_ONSEN_INDEX]
        {
            //println!("{} {}", game.uma.vital, self.vital_threshold);
            return true;
        }

        false
    }

    /// 计算友人外出的超回复相关价值调整
    ///
    /// # 策略说明
    ///
    /// 超回复触发机制：
    /// - 自然触发：训练累计消耗体力 → 概率表（每50体力一档，275保底）
    /// - 友人外出：清零累计计数器 → 直接获得超回复
    ///
    /// 核心冲突：
    /// - 累计体力高时友人外出 = 浪费即将自然触发的超回复
    /// - 累计体力低时友人外出 = 快速获得超回复
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    ///
    /// # 返回
    /// 友人外出的价值调整值（可正可负）
    fn calc_friend_outing_super_adjustment(&self, game: &OnsenGame) -> f64 {
        // 已有超回复状态时，友人外出不再提供额外超回复收益
        if game.bathing.is_super_ready {
            return 0.0;
        }

        let vital_cost = game.dig_vital_cost;

        if vital_cost >= SUPER_RECOVERY_VITAL_COST_HIGH {
            // 累计体力很高（>=200），接近保底(275)
            // 继续训练更划算，大幅降低友人外出价值
            -FRIEND_OUTING_HIGH_COST_PENALTY
        } else if vital_cost >= SUPER_RECOVERY_VITAL_COST_MID {
            // 累计体力中等（>=150），有较高概率自然触发
            // 适度降低友人外出价值
            -FRIEND_OUTING_MID_COST_PENALTY
        } else if vital_cost < SUPER_RECOVERY_VITAL_COST_LOW {
            // 累计体力低（<100），自然触发概率很低
            // 友人外出可以快速获得超回复，提升价值
            FRIEND_OUTING_SUPER_BONUS
        } else {
            // 中间地带（100-149），保持中性
            0.0
        }
    }

    /// 选择温泉
    ///
    /// # 选择逻辑
    /// 基础：按推荐顺序选择没挖完的
    /// 根据年份决定推荐表的起始位置。第一年>=0, 第2年>=3，第3年>=6。表后面追加了第三年的冗余选项（挖前面没挖完的2和6泉
    /// 根据设定的优先级，第三年9月会优先选择传说秘泉
    fn select_onsen_by_order(&self, game: &OnsenGame, actions: &[OnsenAction]) -> Option<OnsenAction> {
        // 检查是否有 Dig 动作
        let dig_actions: Vec<_> = actions
            .iter()
            .filter_map(|a| {
                if let OnsenAction::Dig(idx) = a {
                    Some(*idx as usize)
                } else {
                    None
                }
            })
            .collect();

        if dig_actions.is_empty() {
            return None;
        }
        let mut pos = game.turn as usize / 24 * 3;
        while pos < self.onsen_order.len() {
            let idx = self.onsen_order[pos] as usize;
            if dig_actions.contains(&idx) {
                return Some(OnsenAction::Dig(idx as i32));
            }
            pos += 1;
        }
        // 如果推荐列表都不可选，选择第一个可选的
        actions.first().cloned()
    }

    /// 按主流攻略顺序选择温泉（返回索引）
    ///
    /// 用于 `do_select_onsen` 场景，传入的 actions 只包含 Dig 动作
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `actions`: 可选的 Dig 动作列表
    ///
    /// # 返回
    /// 选择的动作在 actions 中的索引
    pub fn select_onsen_index(&self, game: &OnsenGame, actions: &[OnsenAction]) -> usize {
        if let Some(act) = self.select_onsen_by_order(game, actions) {
            return actions
                .iter()
                .position(|a| *a == act)
                .expect("action must contain selected act");
        } else {
            0
        }
    }

    /// 计算属性收益（带上限软约束）
    fn calc_status_gain(&self, game: &OnsenGame, train_value: &[i32; 6]) -> f64 {
        let remain_turn = (game.max_turn() - game.turn - 1).max(0) as f64;
        let total_turn = game.max_turn() as f64;
        let reserve = RESERVE_STATUS_FACTOR * remain_turn * (1.0 - remain_turn / (total_turn * 2.0));

        let mut total = 0.0;
        for i in 0..5 {
            let limit = game.uma.five_status_limit[i];
            let remain = limit - game.uma.five_status[i] - FINAL_BONUS;
            let gain = train_value[i];

            let s0 = status_soft_function(-remain as f64, reserve);
            let s1 = status_soft_function((gain - remain) as f64, reserve);
            let status_weight = if game.card_type_count[i] > 0 {
                STATUS_WEIGHTS[i]
            } else {
                ABSENT_WEIGHT
            };
            total += status_weight * (s1 - s0);
        }

        total += train_value[5] as f64 * self.skill_weight;
        total
    }

    /// 评估挖掘收益
    ///
    /// # 评估逻辑
    /// 1. 基础挖掘价值：总挖掘量 × 基础系数
    /// 2. 完成温泉奖励：如果即将挖完当前温泉，额外加分
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `action`: 要评估的动作
    ///
    /// # 返回
    /// 挖掘收益分数
    fn evaluate_dig_value(&self, game: &OnsenGame, action: &OnsenAction) -> f64 {
        // 获取挖掘量
        let dig_value = match game.calc_dig_value(action) {
            Some(v) => v,
            None => return 0.0
        };

        // 总挖掘量
        let total_dig: i32 = dig_value.iter().sum();

        // 基础挖掘价值（挖掘量越多越好）
        let mut score = total_dig as f64 * DIG_BASE_VALUE;

        // 判断是否即将完成当前温泉
        let current_onsen = game.current_onsen;
        let dig_remain = &game.dig_remain[current_onsen];
        let total_remain: i32 = dig_remain.iter().sum();

        if total_dig >= total_remain && total_remain > 0 {
            // 即将挖完当前温泉，额外奖励
            score += DIG_COMPLETE_BONUS;
        }

        score
    }

    /// 评估训练动作（优化版）
    ///
    /// # 训练类型偏好策略
    /// 1. 速耐训练优先：速度 > 耐力 > 力量 = 根性 > 智力
    /// 2. 智力训练特殊处理：只有在人头多、彩圈多或前期时才倾向选择
    ///
    /// # 体力惩罚豁免机制
    /// 当挖掘秘汤后，温泉券的回复可以覆盖所有体力消耗
    /// 此时跳过体力评估惩罚，让AI更激进地训练
    fn evaluate_training(&self, game: &OnsenGame, train: usize) -> f64 {
        let mut score = 0.0;
        let skip_vital_penalty = should_skip_vital_penalty(game);

        let mut fail_rate = 0.0;
        let mut vital_delta = 0;

        if let Ok(buffs) = game.calc_training_buff(train) {
            if let Ok(value) = game.calc_training_value(&buffs, train) {
                score = self.calc_status_gain(game, &value.status_pt);
                vital_delta = value.vital;
            }
            fail_rate = game.calc_training_failure_rate(&buffs, train) as f64;
        }

        // 体力价值评估（挖掘完成 7 个温泉后跳过）
        if !skip_vital_penalty {
            let vital_factor = calc_vital_factor(game.turn, game.max_turn());
            let vital_before = vital_evaluation(game.uma.vital, game.uma.max_vital);
            let vital_after = (game.uma.vital + vital_delta).clamp(0, game.uma.max_vital);
            let vital_after_value = vital_evaluation(vital_after, game.uma.max_vital);
            score += vital_factor * (vital_after_value - vital_before);
        }

        // 失败率惩罚（挖掘完成 7 个温泉后跳过）
        let big_fail_prob = if fail_rate < 20.0 { 0.0 } else { fail_rate };
        let fail_value_avg = 0.01 * big_fail_prob * BIG_FAIL_VALUE + (1.0 - 0.01 * big_fail_prob) * SMALL_FAIL_VALUE;
        score = 0.01 * fail_rate * fail_value_avg + (1.0 - 0.01 * fail_rate) * score;

        // 彩圈加成
        let shining_count = game.shining_count(train);
        if shining_count > 0 {
            score *= 1.0 + shining_count as f64 * 0.15;
            score += shining_count as f64 * self.shining_bonus;
        }

        // 人头数量统计
        let head_count = game.distribution()[train].len();

        // 人头加成
        let mut has_friend = false;
        for &person_idx in &game.distribution()[train] {
            if let Some(person) = game.persons.get(person_idx as usize) {
                match person.person_type {
                    PersonType::ScenarioCard | PersonType::OtherFriend => {
                        has_friend = true;
                        if person.friendship < 60 {
                            score += 100.0;
                        } else {
                            score += 40.0;
                        }
                    }
                    PersonType::Card => {
                        if person.friendship < 80 {
                            let base_jiban_add: f64 = 7.0;
                            let jiban_add = base_jiban_add.min((80 - person.friendship) as f64);
                            score += jiban_add * JIBAN_VALUE;
                            if has_friend {
                                score += 2.0 * JIBAN_VALUE;
                            }
                        }
                        if person.is_hint {
                            let hint_bonus: f64 = STATUS_WEIGHTS.iter().sum::<f64>() * 1.6;
                            score += hint_bonus;
                        }
                    }
                    _ => {}
                }
            }
        }

        // 温泉效果激活加成
        if game.bathing.buff_remain_turn > 0 {
            score *= 1.15;
        }

        // 挖掘收益评估
        score += self.evaluate_dig_value(game, &OnsenAction::Train(train as i32));

        // ============================================================================
        // 训练类型偏好调整
        // ============================================================================

        // 基础训练类型加成
        score += TRAIN_TYPE_BONUS[train];

        // 没带卡的训练特殊处理
        // 默认有 -100 惩罚，只有以下极端情况才解锁：
        if game.card_type_count[train] == 0 {
            // 计算其他训练的最大人头数
            let other_max_head: usize = (0..4).map(|t| game.distribution()[t].len()).max().unwrap_or(0);

            // 条件1：前期（回合 < 12）且智力人头 >= 3 且其他训练人头都 <= 1
            // 只有其他训练完全没人头时，才考虑前期智力攒羁绊
            if game.turn < EARLY_TURN_THRESHOLD && train == 4 && head_count >= 3 && other_max_head <= 1 {
                score += 120.0; // 解锁前期智力训练
            }

            // 条件2：非擅长训练人头 >= 3（超多人头）
            if head_count >= ABSENT_HEAD_THRESHOLD + 1 {
                score += 80.0; // 多人头部分解锁
            }

            // 条件4：其他四个训练人头都 <= 1 时，非擅长训练人头 >= 3
            // 即：被迫无奈的情况
            if other_max_head <= 1 && head_count >= 3 {
                score += 80.0; // 被迫选择非得意训练
            }
        }

        score
    }

    /// 根据当前挖掘的温泉选择装备升级
    pub fn select_upgrade_action(&self, game: &OnsenGame, upgrade_actions: &[OnsenAction]) -> usize {
        if !upgrade_actions.is_empty() {
            let recommended = RECOMMENDED_TOOL_TYPE[game.current_onsen - 1];
            for i in 0..upgrade_actions.len() {
                if upgrade_actions[i] == OnsenAction::Upgrade(recommended as i32) {
                    return i;
                }
            }
        }
        0
    }
}

impl Evaluator<OnsenGame> for HandwrittenEvaluator {
    fn select_action(&self, game: &OnsenGame, rng: &mut StdRng) -> Option<ActionScore<OnsenAction>> {
        let actions = game.list_actions().ok()?;
        if actions.is_empty() {
            return None;
        }

        // 硬编码规则：生病必须治病
        if game.uma.flags.ill {
            if let Some(a) = actions.iter().find(|a| matches!(a, OnsenAction::Clinic)) {
                return Some(ActionScore::new(a.clone(), actions, vec![]));
            }
        }

        // 硬编码规则：目标比赛
        if game.is_race_turn() {
            if let Some(a) = actions.iter().find(|a| matches!(a, OnsenAction::Race)) {
                return Some(ActionScore::new(a.clone(), actions, vec![]));
            }
        }

        // 硬编码规则：温泉券使用
        if actions.iter().any(|a| matches!(a, OnsenAction::UseTicket(true))) {
            return Some(ActionScore::new(
                OnsenAction::UseTicket(self.should_use_ticket(game)),
                actions,
                vec![]
            ));
        }

        // 硬编码规则：温泉选择（按主流攻略顺序）
        if let Some(dig_action) = self.select_onsen_by_order(game, &actions) {
            return Some(ActionScore::new(dig_action, actions, vec![]));
        }

        // 评估所有动作，选择价值最高的
        let skip_vital_penalty = should_skip_vital_penalty(game);
        let vital_factor = calc_vital_factor(game.turn, game.max_turn());
        let vital_before = vital_evaluation(game.uma.vital, game.uma.max_vital);
        let mut best_action: Option<OnsenAction> = None;
        let mut best_value = f64::NEG_INFINITY;
        let mut scores = vec![];
        //let mut debug_line = String::new();
        for action in &actions {
            let value = match action {
                OnsenAction::Train(t) => self.evaluate_training(game, *t as usize),

                OnsenAction::Sleep => {
                    // 挖掘完成 8 个温泉后，休息价值大幅降低（体力不再重要）
                    if skip_vital_penalty {
                        let mut value = -100.0; // 基础负值，避免无意义休息
                        value += self.evaluate_dig_value(game, action);
                        value
                    } else {
                        let vital_gain = 50;
                        let vital_after = (game.uma.vital + vital_gain).min(game.uma.max_vital);
                        let mut value =
                            vital_factor * (vital_evaluation(vital_after, game.uma.max_vital) - vital_before);
                        // 休息也有挖掘收益
                        value += self.evaluate_dig_value(game, action);
                        value
                    }
                }

                OnsenAction::NormalOuting => {
                    // 挖掘完成 8 个温泉后，普通外出价值降低
                    if skip_vital_penalty {
                        let mut value = -50.0; // 基础负值
                        if game.uma.motivation < 5 {
                            value += OUTGOING_BONUS_IF_NOT_FULL_MOTIVATION;
                        }
                        value += self.evaluate_dig_value(game, action);
                        value
                    } else {
                        let vital_gain = 10;
                        let vital_after = (game.uma.vital + vital_gain).min(game.uma.max_vital);
                        let mut value =
                            vital_factor * (vital_evaluation(vital_after, game.uma.max_vital) - vital_before);
                        if game.uma.motivation < 5 {
                            value += OUTGOING_BONUS_IF_NOT_FULL_MOTIVATION;
                        }
                        // 普通外出有挖掘收益
                        value += self.evaluate_dig_value(game, action);
                        value
                    }
                }

                OnsenAction::FriendOuting => {
                    // 友人外出基础价值，根据回合数计算
                    let mut value = FRIEND_OUTING_BASE_VALUE + FRIEND_OUTING_TURN_MODIFIER[(game.turn / 8) as usize];

                    // 干劲不满时额外加成
                    if game.uma.motivation < 5 {
                        value += OUTGOING_BONUS_IF_NOT_FULL_MOTIVATION;
                    }

                    // 超回复策略调整（挖掘完成8个温泉后跳过，因为超回复已经不重要）
                    if !skip_vital_penalty {
                        // - 累计体力低时：+200（快速获得超回复）
                        // - 累计体力高时：-300（避免浪费即将自然触发的超回复）
                        // - 已有超回复时：不调整
                        value += self.calc_friend_outing_super_adjustment(game);
                    }

                    // 友人外出有挖掘收益
                    value += self.evaluate_dig_value(game, action);
                    value
                }

                OnsenAction::Race => {
                    // 比赛价值评估（参考 action.rs 的实际执行逻辑）
                    let mut value = if game.is_race_turn() {
                        // 目标比赛：使用剧本比赛倍率
                        let career_multiplier = ONSENDATA.get().map(|d| d.career_race_multiplier).unwrap_or(1.5);
                        let career_bonus = (100 + game.scenario_buff.onsen.career_race_bonus) as f64 / 100.0;
                        let mut race_value = RACE_BASE_BONUS * career_multiplier as f64 * career_bonus;

                        // 狄杜斯角色额外加成
                        if game.uma.chara_id() == DIDUS_CHARA_ID {
                            race_value *= 1.8;
                        }
                        race_value
                    } else {
                        // 如果在自选比赛区间，且有足够等级的比赛，则考虑自选比赛
                        if let Some(free) = game.uma.find_free_race(game.turn) {
                            let race_grade = global!(GAMECONSTANTS).race_grades[game.turn as usize];
                            let count = game.uma.count_free_race(free);
                            let remain_turn = free.end_turn as i32 - game.turn + 1;
                            if count < free.count {
                                match &free.grade {
                                    Some(g) => {
                                        if race_grade <= *g as i32 {
                                            // 相当于生涯比赛收益 * 缺的比赛数量
                                            RACE_BASE_BONUS * (free.count - count) as f64 + FREE_RACE_BONUS / remain_turn as f64
                                        } else {
                                            NON_TARGET_RACE_BONUS
                                        }
                                    }
                                    None => RACE_BASE_BONUS * (free.count - count) as f64 + FREE_RACE_BONUS / remain_turn as f64
                                }
                            } else {
                                NON_TARGET_RACE_BONUS
                            }
                        } else {
                            NON_TARGET_RACE_BONUS // 非目标比赛
                        }                        
                    };
                    // 比赛也有挖掘收益
                    value += self.evaluate_dig_value(game, action);
                    value
                }

                OnsenAction::PR => {
                    // 第一年消耗体力越多，PR收益越高，有超回复无券时再加一点
                    let mut value = if game.turn < 24 {
                        game.dig_vital_cost as f64 * 1.2
                    } else {
                        -1000.0
                    };
                    if game.bathing.ticket_num == 0 {
                        if game.bathing.is_super_ready {
                            value += PR_SCORE_SUPER;
                        } else {
                            value += PR_SCORE;
                        }
                    }
                    value += self.evaluate_dig_value(game, action);
                    value
                }

                _ => -1000.0
            };

            scores.push(value);

            if value > best_value {
                best_value = value;
                best_action = Some(action.clone());
            }
            //debug_line += &format!("{action}: {value:.1} ");
        }
        //info!("{}", debug_line.cyan());
        if best_action.is_none() {
            best_action = actions.choose(rng).cloned();
        }
        Some(ActionScore::new(best_action.expect("best_action"), actions, scores))
    }

    fn evaluate(&self, game: &OnsenGame) -> ValueOutput {
        let score = game.uma.calc_score() as f64;
        let progress = game.turn as f64 / game.max_turn() as f64;
        let stdev = 500.0 * (1.0 - progress) + 100.0;
        ValueOutput::new(score, stdev)
    }

    fn evaluate_choice(&self, _game: &OnsenGame, choice_index: usize) -> f64 {
        if choice_index == 0 { 1.0 } else { 0.0 }
    }

    /// 从给定的动作列表中选择动作
    ///
    /// 当 `select_action` 返回的动作不在给定的动作列表中时调用。
    /// 主要用于以下场景：
    /// - 温泉选择场景（只传入 Dig 动作）
    /// - 装备升级场景（只传入 Upgrade 动作）
    fn select_action_from_list(&self, game: &OnsenGame, actions: &[OnsenAction], _rng: &mut StdRng) -> usize {
        // 检查是否是温泉选择场景（所有动作都是 Dig）
        let all_dig = actions.iter().all(|a| matches!(a, OnsenAction::Dig(_)));
        if all_dig {
            return self.select_onsen_index(game, actions);
        }

        // 检查是否是装备升级场景（所有动作都是 Upgrade）
        let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
        if all_upgrade {
            return self.select_upgrade_action(game, actions);
        }

        // 其他场景：使用 select_action 的结果（如果在列表中）
        // 如果不在列表中，返回 0
        0
    }
}
