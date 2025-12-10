//! 数据收集训练员
//!
//! 包装 HandwrittenEvaluator，在做出决策的同时收集训练数据。
//! 用于生成神经网络训练样本。
//!
//! # 用途
//! - 运行大规模模拟收集训练数据
//! - 记录每回合的游戏状态和决策
//! - 生成精英样本（Top N%）用于训练
//!
//! # 探索性策略
//!
//! 为了让神经网络能够学习温泉挖掘顺序的选择，而不是只记住固定的顺序，
//! CollectorTrainer 在温泉选择时引入了探索率机制：
//!
//! - **探索（Exploration）**: 随机选择一个可用温泉，收集多样化的训练数据
//! - **利用（Exploitation）**: 使用 HandwrittenEvaluator 的策略选择温泉
//!
//! 通过精英样本筛选（Top 1%），最终训练数据中会保留那些"探索到更好策略"的样本，
//! 从而让神经网络学习到更优的温泉挖掘顺序。

use std::cell::RefCell;

use anyhow::Result;
use log::debug;
use rand::{Rng, prelude::StdRng};

use crate::{
    game::{Trainer, onsen::game::OnsenGame},
    gamedata::ActionValue,
    neural::{Evaluator, HandwrittenEvaluator},
    sample_collector::{GameSample, SampleCollector}
};

/// 默认探索率（40%）
///
/// 在温泉选择时，40% 概率随机选择，60% 概率使用 handwritten 策略
pub const DEFAULT_EXPLORATION_RATE: f64 = 0.4;

/// 数据收集训练员
///
/// 在执行 HandwrittenEvaluator 决策的同时，收集训练数据。
/// 使用 RefCell 实现内部可变性，以便在 Trainer trait 方法中修改收集器。
///
/// # 探索率
///
/// `exploration_rate` 参数控制温泉选择时的随机性：
/// - 0.0 = 完全使用 handwritten 策略（无探索）
/// - 0.4 = 40% 随机探索 + 60% 策略利用（默认值）
/// - 1.0 = 完全随机选择（纯探索）
///
/// 注意：探索率**仅影响温泉选择**，训练、比赛、休息等动作仍使用 handwritten 策略。
pub struct CollectorTrainer {
    /// 内部评估器
    evaluator: HandwrittenEvaluator,
    /// 样本收集器（使用 RefCell 实现内部可变性）
    collector: RefCell<SampleCollector>,
    /// 温泉选择探索率（0.0 ~ 1.0）
    ///
    /// 仅在温泉选择（all_dig）场景生效：
    /// - 探索率概率下随机选择温泉
    /// - 其余情况使用 handwritten 策略
    exploration_rate: f64,
    /// 是否输出详细日志
    verbose: bool
}

impl CollectorTrainer {
    /// 创建新的数据收集训练员（使用默认探索率 40%）
    pub fn new() -> Self {
        Self {
            evaluator: HandwrittenEvaluator::new(),
            collector: RefCell::new(SampleCollector::new()),
            exploration_rate: DEFAULT_EXPLORATION_RATE,
            verbose: false
        }
    }

    /// 使用指定的评估器创建（使用默认探索率 40%）
    pub fn with_evaluator(evaluator: HandwrittenEvaluator) -> Self {
        Self {
            evaluator,
            collector: RefCell::new(SampleCollector::new()),
            exploration_rate: DEFAULT_EXPLORATION_RATE,
            verbose: false
        }
    }

    /// 设置温泉选择探索率
    ///
    /// # 参数
    /// - `rate`: 探索率（0.0 ~ 1.0）
    ///   - 0.0 = 完全使用 handwritten 策略
    ///   - 0.4 = 40% 随机 + 60% 策略（默认）
    ///   - 1.0 = 完全随机选择
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// 获取当前探索率
    pub fn exploration_rate(&self) -> f64 {
        self.exploration_rate
    }

    /// 设置是否输出详细日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 重置收集器（开始新的一局）
    pub fn reset(&self) {
        *self.collector.borrow_mut() = SampleCollector::new();
    }

    /// 设置最终分数并完成收集
    pub fn set_final_score(&self, score: i32) {
        self.collector.borrow_mut().set_final_score(score);
    }

    /// 获取收集的回合数
    pub fn num_turns(&self) -> usize {
        self.collector.borrow().num_turns()
    }

    /// 取走收集的样本（会重置收集器）
    ///
    /// # 返回
    /// 包含最终分数和所有训练样本的 GameSample
    pub fn take_samples(&self) -> GameSample {
        let collector = std::mem::replace(&mut *self.collector.borrow_mut(), SampleCollector::new());
        GameSample::from_collector(collector)
    }

    /// 完成收集并返回样本（设置分数后调用）
    pub fn finalize(&self, final_score: i32) -> GameSample {
        self.set_final_score(final_score);
        self.take_samples()
    }
}

impl Default for CollectorTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer<OnsenGame> for CollectorTrainer {
    fn select_action(
        &self, game: &OnsenGame, actions: &[<OnsenGame as crate::game::Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        use crate::game::onsen::action::OnsenAction;

        // 提取特征（在选择动作之前）
        let features = game.extract_nn_features(None);

        // 只有一个动作时直接返回
        if actions.len() <= 1 {
            self.collector.borrow_mut().record_turn(features, &actions[0]);
            return Ok(0);
        }

        // 检查是否是温泉选择场景（所有动作都是 Dig）
        // 这里应用探索率：随机选择 vs 使用 handwritten 策略
        let all_dig = actions.iter().all(|a| matches!(a, OnsenAction::Dig(_)));
        if all_dig {
            // 探索 vs 利用
            let idx = if rng.random::<f64>() < self.exploration_rate {
                // 探索：随机选择一个温泉
                let random_idx = rng.random_range(0..actions.len());
                if self.verbose {
                    debug!(
                        "[回合 {}] 探索：随机选择温泉 {} (共 {} 个)",
                        game.turn,
                        actions[random_idx],
                        actions.len()
                    );
                }
                random_idx
            } else {
                // 利用：使用 handwritten 策略
                let strategy_idx = self.evaluator.select_onsen_index(game, actions);
                if self.verbose {
                    debug!("[回合 {}] 利用：策略选择温泉 {}", game.turn, actions[strategy_idx]);
                }
                strategy_idx
            };

            self.collector.borrow_mut().record_turn(features, &actions[idx]);
            return Ok(idx);
        }

        // 检查是否是装备升级场景
        let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
        if all_upgrade {
            let idx = self.evaluator.select_upgrade_action(game, actions);
            self.collector.borrow_mut().record_turn(features, &actions[idx]);
            if self.verbose {
                debug!("[回合 {}] 收集：装备升级 {}", game.turn, actions[idx]);
            }
            return Ok(idx);
        }

        // 使用 HandwrittenEvaluator 的 select_action 逻辑
        let selected_action = self.evaluator.select_action(game, rng);
        let idx = match &selected_action {
            Some(action) => actions.iter().position(|a| a == action).unwrap_or(0),
            None => 0
        };

        // 记录到收集器（使用实际选择的动作）
        self.collector.borrow_mut().record_turn(features, &actions[idx]);

        if self.verbose {
            debug!("[回合 {}] 收集：动作 {}", game.turn, actions[idx]);
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

        // 记录事件选项到收集器
        self.collector.borrow_mut().record_choice(best_idx, choices.len());

        if self.verbose {
            debug!(
                "[回合 {}] 收集：事件选项 {} (共 {} 个)",
                game.turn,
                best_idx,
                choices.len()
            );
        }

        Ok(best_idx)
    }
}
