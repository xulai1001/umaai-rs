//! MCTS 训练员
//!
//! 使用扁平蒙特卡洛搜索进行决策，通过多次模拟评估各决策的价值。
//!
//! # 用途
//! - 高质量决策（比手写策略更优）
//! - 生成高质量训练数据（每个状态有准确的价值估计）
//! - 自对弈训练

use anyhow::Result;
use log::info;
use rand::prelude::StdRng;

use crate::{
    game::{Trainer, onsen::game::OnsenGame},
    gamedata::ActionValue,
    neural::{Evaluator, HandwrittenEvaluator},
    search::{SearchConfig, FlatSearch},
};

/// MCTS 训练员
///
/// 使用扁平蒙特卡洛搜索进行动作选择。
/// 对于温泉选择和装备升级使用手写逻辑（这些场景有固定最优策略），
/// 其他动作使用 MCTS 搜索评估。
pub struct MctsTrainer {
    /// 扁平搜索器
    search: FlatSearch,
    /// 手写评估器（用于温泉/装备等特殊场景）
    evaluator: HandwrittenEvaluator,
    /// 是否输出详细日志
    verbose: bool,
}

impl MctsTrainer {
    /// 创建 MCTS 训练员
    pub fn new(config: SearchConfig) -> Self {
        Self {
            search: FlatSearch::new(config),
            evaluator: HandwrittenEvaluator::new(),
            verbose: false,
        }
    }

    /// 创建默认 MCTS 训练员
    pub fn default_trainer() -> Self {
        Self::new(SearchConfig::default())
    }

    /// 设置是否输出详细日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 获取搜索配置
    pub fn config(&self) -> &SearchConfig {
        self.search.config()
    }
}

impl Default for MctsTrainer {
    fn default() -> Self {
        Self::default_trainer()
    }
}

impl Trainer<OnsenGame> for MctsTrainer {
    fn select_action(
        &self,
        game: &OnsenGame,
        actions: &[<OnsenGame as crate::game::Game>::Action],
        rng: &mut StdRng,
    ) -> Result<usize> {
        use crate::game::onsen::action::OnsenAction;

        // 只有一个动作时直接返回
        if actions.len() <= 1 {
            return Ok(0);
        }

        // 检查是否是温泉选择场景（所有动作都是 Dig）
        let all_dig = actions.iter().all(|a| matches!(a, OnsenAction::Dig(_)));
        if all_dig {
            // 温泉选择：使用手写逻辑（固定最优顺序）
            let idx = self.evaluator.select_onsen_index(game, actions);
            if self.verbose {
                info!(
                    "[回合 {}] MCTS 选择温泉（手写逻辑）: {}",
                    game.turn,
                    actions[idx]
                );
            }
            return Ok(idx);
        }

        // 检查是否是装备升级场景（所有动作都是 Upgrade）
        let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
        if all_upgrade {
            // 装备升级：使用手写逻辑
            let idx = self.evaluator.select_upgrade_action(game, actions);
            if self.verbose {
                info!(
                    "[回合 {}] MCTS 选择装备升级（手写逻辑）: {}",
                    game.turn,
                    actions[idx]
                );
            }
            return Ok(idx);
        }

        // 使用 MCTS 搜索
        let search_output = self.search.search(game, rng)?;
        let best_action = search_output.best_action();

        if self.verbose {
            // 输出搜索结果
            info!(
                "[回合 {}] MCTS 搜索完成: search_n={}, radical_factor={:.1}",
                game.turn,
                self.search.config().search_n,
                search_output.radical_factor
            );

            // 输出各动作的分数
            for (i, action) in search_output.actions.iter().enumerate() {
                let result = &search_output.action_results[i];
                let weighted = result.weighted_mean(search_output.radical_factor);
                let marker = if i == search_output.best_action_idx { " <-- 最优" } else { "" };
                info!(
                    "  {}: mean={:.0}, weighted={:.0}{}",
                    action,
                    result.mean(),
                    weighted,
                    marker
                );
            }
        }

        // 找到最优动作在原列表中的索引
        let idx = actions.iter().position(|a| a == best_action).unwrap_or(0);

        Ok(idx)
    }

    fn select_choice(
        &self,
        game: &OnsenGame,
        choices: &[ActionValue],
        _rng: &mut StdRng,
    ) -> Result<usize> {
        // 事件选择：使用手写逻辑
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
                "[回合 {}] MCTS 选择事件选项（手写逻辑）: {} (索引 {})",
                game.turn,
                choices[best_idx],
                best_idx
            );
        }

        Ok(best_idx)
    }
}

