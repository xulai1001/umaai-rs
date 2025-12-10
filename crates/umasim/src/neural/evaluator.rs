//! 评估器接口定义
//!
//! 提供策略选择和局面评估的统一接口，支持多种实现：
//! - 手写启发式评估器（无需神经网络）
//! - 随机评估器（基准测试）
//! - 神经网络评估器（正在实现）

use rand::rngs::StdRng;

use super::ValueOutput;
use crate::game::Game;

/// 评估器 Trait
///
/// 定义 MCTS 搜索所需的两个核心能力：
/// 1. `select_action`: 策略网络 - 选择下一步动作（用于模拟阶段）
/// 2. `evaluate`: 价值网络 - 评估当前局面（用于终局评估）
///
/// # 类型参数
/// - `G`: 游戏类型，必须实现 `Game` trait
pub trait Evaluator<G: Game>: Send + Sync {
    /// 选择动作（策略）
    ///
    /// 在模拟过程中选择下一步动作。
    /// 可以是基于规则的启发式选择，也可以是神经网络预测的概率分布采样。
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// - `Some(action)`: 选择的动作
    /// - `None`: 无可选动作（游戏结束或异常）
    fn select_action(&self, game: &G, rng: &mut StdRng) -> Option<G::Action>;

    /// 评估局面（价值）
    ///
    /// 估计当前局面的最终分数分布。
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    ///
    /// # 返回
    /// 包含分数均值和标准差的 ValueOutput
    fn evaluate(&self, game: &G) -> ValueOutput;

    /// 评估事件选项（可选覆盖）
    ///
    /// 默认实现：返回 0.0（选择第一个）
    fn evaluate_choice(&self, _game: &G, _choice_index: usize) -> f64 {
        0.0
    }

    /// 从给定的动作列表中选择动作（可选覆盖）
    ///
    /// 当 `select_action` 返回的动作不在给定的动作列表中时调用。
    /// 主要用于温泉选择场景（只传入 Dig 动作）。
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `actions`: 可选动作列表
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// 选择的动作在 actions 中的索引
    fn select_action_from_list(&self, _game: &G, _actions: &[G::Action], _rng: &mut StdRng) -> usize {
        // 默认选择第一个
        0
    }
}

/// 随机评估器（用于基准测试）
///
/// 随机选择动作，使用游戏评分评估局面。
pub struct RandomEvaluator;

impl<G: Game> Evaluator<G> for RandomEvaluator
where
    G::Action: Clone
{
    fn select_action(&self, game: &G, rng: &mut StdRng) -> Option<G::Action> {
        use rand::seq::IndexedRandom;

        let actions = game.list_actions().ok()?;
        if actions.is_empty() {
            return None;
        }
        actions.choose(rng).cloned()
    }

    fn evaluate(&self, game: &G) -> ValueOutput {
        // 使用游戏评分
        let score = game.uma().calc_score() as f64;

        // 根据游戏进度调整标准差
        let progress = game.turn() as f64 / game.max_turn() as f64;
        let stdev = 500.0 * (1.0 - progress) + 100.0;

        ValueOutput::new(score, stdev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_evaluator_send_sync() {
        // 验证 RandomEvaluator 满足 Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RandomEvaluator>();
    }
}
