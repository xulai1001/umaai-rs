//! 手写策略训练员
//!
//! 使用 HandwrittenEvaluator 的决策逻辑作为独立训练员，
//! 不经过 MCTS 搜索，直接使用手写启发式规则选择动作。
//!
//! # 用途
//! - 快速测试手写策略效果（无搜索开销）
//! - 作为 MCTS 的 baseline 对比
//! - 调试和验证策略逻辑

use anyhow::Result;
use log::info;
use rand::prelude::StdRng;

use crate::{
    game::{Trainer, onsen::game::OnsenGame},
    gamedata::ActionValue,
    neural::{Evaluator, HandwrittenEvaluator}
};

/// 手写策略训练员
///
/// 直接使用 HandwrittenEvaluator 的 select_action 逻辑，
/// 不经过 MCTS 搜索，执行速度快。
pub struct HandwrittenTrainer {
    evaluator: HandwrittenEvaluator,
    verbose: bool
}

impl HandwrittenTrainer {
    /// 创建默认手写策略训练员
    pub fn new() -> Self {
        Self {
            evaluator: HandwrittenEvaluator::new(),
            verbose: false
        }
    }

    /// 设置是否输出详细日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 使用速度型配置
    pub fn speed_build() -> Self {
        Self {
            evaluator: HandwrittenEvaluator::speed_build(),
            verbose: false
        }
    }

    /// 使用耐力型配置
    pub fn stamina_build() -> Self {
        Self {
            evaluator: HandwrittenEvaluator::stamina_build(),
            verbose: false
        }
    }
}

impl Default for HandwrittenTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer<OnsenGame> for HandwrittenTrainer {
    fn select_action(
        &self, game: &OnsenGame, actions: &[<OnsenGame as crate::game::Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        use crate::game::onsen::action::OnsenAction;

        // 只有一个动作时直接返回
        if actions.len() <= 1 {
            return Ok(0);
        }

        // 检查是否是温泉选择场景（所有动作都是 Dig）
        let all_dig = actions.iter().all(|a| matches!(a, OnsenAction::Dig(_)));
        if all_dig {
            // 温泉选择：使用硬编码顺序
            let idx = self.evaluator.select_onsen_index(game, actions);
            if self.verbose {
                info!("[回合 {}] 手写策略选择温泉: {}", game.turn + 1, actions[idx]);
            }
            return Ok(idx);
        }

        // 检查是否是装备升级场景（所有动作都是 Upgrade）
        let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
        if all_upgrade {
            // 装备升级：使用智能升级策略
            let idx = self.evaluator.select_upgrade_action(game, actions);
            if self.verbose {
                info!("[回合 {}] 手写策略选择装备升级: {}", game.turn + 1, actions[idx]);
            }
            return Ok(idx);
        }

        // 使用 HandwrittenEvaluator 的 select_action 逻辑
        let selected_action = self.evaluator.select_action(game, rng);

        // 找到选中动作在列表中的索引
        let idx = match &selected_action {
            Some(action) => actions.iter().position(|a| a == action).unwrap_or(0),
            None => 0
        };

        if self.verbose {
            info!("[回合 {}] 手写策略选择: {}", game.turn + 1, actions[idx]);
        }

        Ok(idx)
    }

    fn select_choice(&self, game: &OnsenGame, choices: &[ActionValue], _rng: &mut StdRng) -> Result<usize> {
        // 使用 HandwrittenEvaluator 的 evaluate_choice 逻辑
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (i, _choice) in choices.iter().enumerate() {
            let value = self.evaluator.evaluate_choice(game, i);
            if value > best_value {
                best_value = value;
                best_idx = i;
            }
        }

        if self.verbose {
            info!(
                "[回合 {}] 手写策略选择事件选项: {} (索引 {})",
                game.turn, choices[best_idx], best_idx
            );
        }

        Ok(best_idx)
    }
}
