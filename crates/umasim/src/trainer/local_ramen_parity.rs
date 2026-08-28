//! 配对指标迭代用训练器包装（深水矿脉版 v3）。
//!
//! 与上游 `RecommendedRamenTrainer` 的关系：本文件按上游 preset 的**原始配方**
//! 等价重建三年策略实例（字段级对齐其 `new()` 内联配置），并把本 workbench
//! 锁定增量与深水矿脉覆盖直接写入构建参数——无需改动上游
//! `local_ramen_trainer.rs`，即可扫描 cook2 / y3 门禁 / 友人权重 / 方案E折扣。
//!
//! 保真约定：`make_year()` 必须与上游 preset 逐字段一致；上游演进时同步本文件。
//!
//! == 决策日志 ==
//! - 锁定层：gap0.75/ov1.00、sac260(老)/180(赛)、hint9、res60、win0.10；
//!   老卡组冠军 `hint9-res60`、比赛卡组 `sac180-win200-hint9-res60`（r800 三绿）
//! - 弃用：bond12 组合塌缩、pt32 全负、rwc 无增益
//! - 本轮新开深水矿脉：cook / y3pre / y3sf / y3hard / stv / prw / capd
//!
//! == 术语订正（2026-08-27，回应"玩家语义不对"）==
//! 以下名称是上游自创剧本的内部代号，**不是赛马娘原作设定**，中文表述按机制本义改写：
//! - `stv`(原"饥饿加成") → **食材告罄补给权重**：万能食材(special_feeling)上限4、
//!   吃面约耗1.5、合宿+2/年末+1/友人外出+2；该权重让"库存见底时那次外出更值得"，
//!   且外出前若有固定发放将补足缺口则不加分（防溢出）。token 沿用上游 starve 缩写以保
//!   历史报告可比性
//! - `prw`(原"主动积极使用") → **友人主动外出保守分**：未来三回合无固定发放且本次
//!   不溢出时给的外出基础加值（体力维持+完链，不等饥饿才动）
//! - `capd`(方案E"残余收益折扣") → **主属性近满时副属性打折系数**
//! 其余字段语义与上游 Rustdoc 一致。
//!
//! 默认＝老卡组冠军全量固化；`RAMEN_VARIANT` 绝对值覆盖 token：
//! - 基础层 `gapNNN` `ovNNN` `winNNN` `sacNNN` `rwcNNN` `bondN` `hintN` `resN` `pt32`
//! - 深水层 `cookNN` `y3preN` `y3sfNNN` `y3hardN` `stvNNN` `prwNNN` `capdNN`
//! 未知 token 直接报错。

use anyhow::{anyhow, Result};
use rand::prelude::StdRng;

use crate::{
    game::{
        ramen::{
            policy::RamenPolicyConfig,
            {RamenAction, RamenGame}
        },
        Game,
        Trainer
    },
    gamedata::{EventChoice, EventData},
    trainer::local_ramen_trainer::{LocalRamenConfig, LocalRamenTrainer}
};

/// 深水矿脉可调覆盖值集合。数值语义与上游配置同名字段一致；
/// Default 即「上游 preset 值 ∪ 本 workbench 已锁定冠军项」。
#[derive(Debug, Clone)]
pub struct VeinOverrides {
    // ==== 已锁定层 ====
    /// 动态属性平衡：短板追赶强度。
    pub status_gap_strength: f32,
    /// 动态属性平衡：近上限衰减强度。
    pub status_overflow_strength: f32,
    /// 长期结构牺牲上限。
    pub max_base_score_sacrifice: f32,
    /// 吃面训练窗口权重。
    pub ramen_window_weight: f32,
    /// 属性预留目标。
    pub status_reserve_max: f32,
    /// 早期羁绊价值。
    pub early_bond_value: f32,
    /// Hint 加成偏好。
    pub hint_bonus: f32,
    /// 地区弱位覆盖加分权重。
    pub region_weak_cover_weight: f32,

    // ==== 深水矿脉层（数值＝上游 preset 当前值）====
    /// Cook2 库存凹函数估值总权重（preset 40）。
    ///
    /// 对 A/B/C 三类食材算 sqrt(吃前+2)-sqrt(吃后+2) 的边际稀缺成本，
    /// 越接近年末/RMJ 达标线该成本越低——控制"囤料还是立刻吃面"的取舍。
    pub cook2_stock_weight: f32,
    /// 吃面前体力软目标（preset 25；三年吃面决策都评估）。
    pub y3_pre_train_vital_target: i32,
    /// 缺口软成本每点（preset 0.5）。
    pub y3_vital_shortfall_weight: f32,
    /// 非智力训练后硬底线（preset 15）。
    pub y3_post_train_hard_floor: i32,
    /// **食材告罄补给权重**（上游内部代号 starve，preset 300）。
    ///
    /// 万能食材被吃面持续消耗；缺口越大，友人外出"+2 补给"越有价值，
    /// 本权重把该价值计入友人外出评分。外出前若近期有固定发放会先扣减，
    /// 防止为即将自然回满的库存付费。
    pub friend_hidden_starve_weight: f32,
    /// **友人主动外出保守分**（上游内部代号 proactive，preset 150）。
    ///
    /// 未来三回合无固定发放、且本次 +2 不溢出时给予的基础加值——
    /// 让策略在体力尚可时也愿意用友人维持体力线并推进完链，
    /// 而不是拖到饥饿或被迫休息才使用。
    pub friend_proactive_weight: f32,
    /// **主属性近满时副属性打折系数**（上游方案 E，preset 1.0）。
    pub cap_discount_weight: f32,
    /// 三年技能 PT 权重。
    pub pt_rates: [f32; 3]
}

impl Default for VeinOverrides {
    fn default() -> Self {
        Self {
            status_gap_strength: 0.75,
            status_overflow_strength: 1.00,
            max_base_score_sacrifice: 260.0,
            ramen_window_weight: 0.10,
            status_reserve_max: 60.0,
            early_bond_value: 8.0,
            hint_bonus: 9.0,
            region_weak_cover_weight: 0.0,
            cook2_stock_weight: 40.0,
            y3_pre_train_vital_target: 25,
            y3_vital_shortfall_weight: 0.5,
            y3_post_train_hard_floor: 15,
            friend_hidden_starve_weight: 300.0,
            friend_proactive_weight: 150.0,
            cap_discount_weight: 1.0,
            pt_rates: [16.0, 64.0, 64.0]
        }
    }
}

/// 等价重建上游 preset 的单年实例并应用覆盖。
///
/// 未列入 [`VeinOverrides`] 的字段逐字复制上游 preset 值：
/// 吃面事务门、动态体力、概率 Hint、期望失败模型、友人 [0,2,5] 节奏、
/// 动态特殊目标等结构逻辑保持不变。
///
/// `eating_rest` 仅第三年为 0（Y3 吃面必成放掉回合门限，Y1/Y2 保持 40）。
fn make_year(pt_rate: f32, vital_rest: i32, eating_rest: i32, ov: &VeinOverrides) -> LocalRamenTrainer {
    let mut policy = RamenPolicyConfig::default();
    policy.pt_rate = pt_rate;
    policy.ramen_pt_weight = 2.0;
    policy.vital_rest = vital_rest;
    policy.vital_rest_eating = eating_rest;
    // 上游 preset：打分用保守基础失败率，规则层仍用真实失败率。
    policy.effective_ramen_failure = false;
    policy.cap_discount_weight = ov.cap_discount_weight;
    policy.region_weak_cover_weight = ov.region_weak_cover_weight;

    let mut local = LocalRamenConfig::default();
    // —— 上游 preset 结构层（不改）——
    local.status_reserve_max = 40.0;
    local.dynamic_vital = true;
    local.probabilistic_hint = true;
    local.expected_fail = true;
    local.max_base_score_sacrifice = 140.0;
    local.ramen_window_weight = 0.10;
    local.ramen_train_coupling_weight = 2.0;
    local.eat_guarantee_weight = 3.0;
    local.friend_hidden_starve_weight = 300.0;
    local.friend_proactive_weight = 150.0;
    local.friend_future_hidden_weight = 0.0;
    local.dynamic_status_balance = true;
    local.status_gap_strength = 0.5;
    local.status_overflow_strength = 0.5;
    local.ramen_lookahead_weight = 0.0;
    local.ramen_lookahead_samples = 1;
    local.effective_ramen_failure = false;
    local.cook2_stock_weight = 40.0;
    local.eat_requires_training = true;
    local.eat_requires_covered_train = true;
    local.y3_pre_train_vital_target = 25;
    local.y3_post_train_vital_target = 0;
    local.y3_vital_shortfall_weight = 0.5;
    local.y3_post_train_hard_floor = 15;
    local.y3_recovery_horizon = true;
    local.friend_outing_replaces_rest = true;
    local.friend_outing3_recovery_vital = 0;
    // v44 千局验证胜出的友人跨年节奏（有守门测试锚定）。
    local.friend_outing_cumulative_caps = [0, 2, 5];
    local.friend_rest_max_special = 4;
    local.deadline_urgency_scale = 0.0;
    local.dynamic_special_targets = true;

    // —— 本 workbench 锁定层 + 深水矿脉覆盖（全部来自 VeinOverrides）——
    local.dynamic_status_balance =
        ov.status_gap_strength != 0.0 || ov.status_overflow_strength != 0.0;
    local.status_gap_strength = ov.status_gap_strength;
    local.status_overflow_strength = ov.status_overflow_strength;
    local.max_base_score_sacrifice = ov.max_base_score_sacrifice;
    local.ramen_window_weight = ov.ramen_window_weight;
    local.status_reserve_max = ov.status_reserve_max;
    local.early_bond_value = ov.early_bond_value;
    local.hint_bonus = ov.hint_bonus;
    local.cook2_stock_weight = ov.cook2_stock_weight;
    local.y3_pre_train_vital_target = ov.y3_pre_train_vital_target;
    local.y3_vital_shortfall_weight = ov.y3_vital_shortfall_weight;
    local.y3_post_train_hard_floor = ov.y3_post_train_hard_floor;
    local.friend_hidden_starve_weight = ov.friend_hidden_starve_weight;
    local.friend_proactive_weight = ov.friend_proactive_weight;

    LocalRamenTrainer::with_configs(policy, local)
}

/// 从正式 preset 等价重建、按 `RAMEN_VARIANT` 覆盖的训练器。
pub struct IterationRamenTrainer {
    years: [LocalRamenTrainer; 3],
    /// 解析出的变体标签（供日志与测试断言）。
    pub variant: String
}

impl IterationRamenTrainer {
    /// 默认：两卡组锁定冠军配置；存在 `RAMEN_VARIANT` 时按 token 覆盖。
    pub fn new() -> Self {
        let variant = std::env::var("RAMEN_VARIANT").unwrap_or_default();
        match Self::from_variant(&variant) {
            Ok(t) => t,
            Err(e) => panic!("RAMEN_VARIANT 无效: {e}")
        }
    }

    /// 按 token 串构造；空串即老卡组冠军版。
    pub fn from_variant(variant: &str) -> Result<Self> {
        let mut ov = VeinOverrides::default();

        for token in variant.split('-').filter(|t| !t.is_empty()) {
            if token == "pt32" {
                ov.pt_rates = [32.0, 32.0, 32.0];
            } else if let Some(pct) = token.strip_prefix("gap") {
                ov.status_gap_strength = Self::parse_percent(token, pct)?;
            } else if let Some(pct) = token.strip_prefix("ov") {
                ov.status_overflow_strength = Self::parse_percent(token, pct)?;
            } else if let Some(per_mille) = token.strip_prefix("win") {
                ov.ramen_window_weight = Self::parse_per_mille(token, per_mille)?;
            } else if let Some(raw) = token.strip_prefix("sac") {
                ov.max_base_score_sacrifice =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(raw) = token.strip_prefix("rwc") {
                ov.region_weak_cover_weight =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(tenths) = token.strip_prefix("bond") {
                ov.early_bond_value =
                    tenths.parse::<f32>().map_err(|_| anyhow!("token {token} 数值段非法: {tenths}"))? / 10.0;
            } else if let Some(tenths) = token.strip_prefix("hint") {
                ov.hint_bonus =
                    tenths.parse::<f32>().map_err(|_| anyhow!("token {token} 数值段非法: {tenths}"))? / 10.0;
            } else if let Some(raw) = token.strip_prefix("res") {
                ov.status_reserve_max =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(raw) = token.strip_prefix("cook") {
                ov.cook2_stock_weight =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(raw) = token.strip_prefix("y3pre") {
                ov.y3_pre_train_vital_target =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(hundredths) = token.strip_prefix("y3sf") {
                ov.y3_vital_shortfall_weight = hundredths
                    .parse::<f32>()
                    .map_err(|_| anyhow!("token {token} 数值段非法: {hundredths}"))?
                    / 100.0;
            } else if let Some(raw) = token.strip_prefix("y3hard") {
                ov.y3_post_train_hard_floor =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(raw) = token.strip_prefix("stv") {
                ov.friend_hidden_starve_weight =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(raw) = token.strip_prefix("prw") {
                ov.friend_proactive_weight =
                    raw.parse().map_err(|_| anyhow!("token {token} 数值段非法: {raw}"))?;
            } else if let Some(hundredths) = token.strip_prefix("capd") {
                ov.cap_discount_weight = hundredths
                    .parse::<f32>()
                    .map_err(|_| anyhow!("token {token} 数值段非法: {hundredths}"))?
                    / 100.0;
            } else {
                return Err(anyhow!("未知 RAMEN_VARIANT token: {token}"));
            }
        }

        Ok(Self {
            // 上游 preset 门限节奏：不吃面回合三年一律 40；
            // 吃面回合仅第三年放掉（fail_rate_drop=100% 必成），Y1/Y2 保留 40。
            years: [
                make_year(ov.pt_rates[0], 40, 40, &ov),
                make_year(ov.pt_rates[1], 40, 40, &ov),
                make_year(ov.pt_rates[2], 40, 0, &ov)
            ],
            variant: variant.to_string()
        })
    }

    /// 解析 NNN% 数值后缀（`gap100` → 1.00）；非法后缀或越界报错。
    fn parse_percent(token: &str, pct: &str) -> Result<f32> {
        let pct_value: f32 = pct.parse().map_err(|_| anyhow!("token {token} 数值段非法: {pct}"))?;
        let value = pct_value / 100.0;
        if !(0.0..=2.0).contains(&value) {
            return Err(anyhow!("token {token} 超出允许区间 [0%,200%]"));
        }
        Ok(value)
    }

    /// 解析 NNN 数值后缀映射到千分比（`win150` → 0.15）；非法后缀报错。
    fn parse_per_mille(token: &str, per_mille: &str) -> Result<f32> {
        let value: f32 =
            per_mille.parse().map_err(|_| anyhow!("token {token} 数值段非法: {per_mille}"))?;
        Ok(value / 1000.0)
    }

    fn current_year(game: &RamenGame) -> usize {
        if game.turn() < 24 {
            0
        } else if game.turn() < 48 {
            1
        } else {
            2
        }
    }
}

impl Trainer<RamenGame> for IterationRamenTrainer {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        self.years[Self::current_year(game)].select_action(game, actions, rng)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.years[Self::current_year(game)].select_choice(game, choices, rng)
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.years[Self::current_year(game)].select_event_choice(game, event, choices, rng)
    }

    fn last_breakdown(&self) -> Option<String> {
        None
    }
}

/// 兼容旧名的历史别名。
pub type RestoredRamenTrainer = IterationRamenTrainer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_variant_token_fails_cleanly() {
        assert!(IterationRamenTrainer::from_variant("nonsense").is_err());
        assert!(IterationRamenTrainer::from_variant("gapx100").is_err());
        assert!(IterationRamenTrainer::from_variant("gap250").is_err());
        assert!(IterationRamenTrainer::from_variant("cookxx").is_err());
        assert!(IterationRamenTrainer::from_variant("capdx").is_err());
    }

    #[test]
    fn locked_recipes_parse_cleanly() {
        assert!(IterationRamenTrainer::from_variant("").is_ok());
        for v in [
            "sac180-win200",
            "sac230-win200",
            "hint9-res60",
            "sac180-win200-hint9-res60",
            "pt32"
        ] {
            assert!(IterationRamenTrainer::from_variant(v).is_ok(), "锁定配方应可解析: {v}");
        }
        assert_eq!(
            IterationRamenTrainer::from_variant("sac180-win200-hint9-res60").unwrap().variant,
            "sac180-win200-hint9-res60"
        );
    }

    #[test]
    fn deep_vein_tokens_parse_cleanly() {
        for v in [
            "cook55",
            "y3pre15-y3sf25-y3hard10",
            "stv400",
            "prw250",
            "capd80",
            "hint9-res60-cook55",
            "sac180-win200-hint9-res60-prw250"
        ] {
            assert!(
                IterationRamenTrainer::from_variant(v).is_ok(),
                "深水组合 token 应可解析: {v}"
            );
        }
        assert!(IterationRamenTrainer::from_variant("stvhigh").is_err());
    }
}
