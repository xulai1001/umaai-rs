//! 神经网络训练员
//!
//! 使用 ONNX 神经网络模型进行决策的训练员。
//! 直接使用神经网络的 Policy 输出选择动作，不进行 MCTS 搜索。

use anyhow::Result;
use log::info;
use rand::rngs::StdRng;

use crate::{
    game::{
        Trainer,
        onsen::{action::OnsenAction, game::OnsenGame}
    },
    gamedata::ActionValue,
    neural::{Evaluator, NeuralNetEvaluator}
};

/// 神经网络训练员
///
/// 使用 ONNX 模型直接进行决策，不使用 MCTS 搜索。
/// 适用于快速推理和性能测试。
pub struct NeuralNetTrainer {
    /// 神经网络评估器
    evaluator: NeuralNetEvaluator,
    /// 是否输出详细日志
    verbose: bool
}

impl NeuralNetTrainer {
    /// 从 ONNX 模型文件创建训练员
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    ///
    /// # 返回
    /// 加载成功返回 NeuralNetTrainer，失败返回错误
    pub fn load(model_path: &str) -> Result<Self> {
        let evaluator = NeuralNetEvaluator::load(model_path)?;
        Ok(Self { evaluator, verbose: false })
    }

    /// 设置是否输出详细日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Trainer<OnsenGame> for NeuralNetTrainer {
    fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng) -> Result<usize> {
        if actions.is_empty() {
            anyhow::bail!("没有可选动作");
        }

        // 使用评估器选择动作
        if let Some(selected_action) = self.evaluator.select_action(game, rng) {
            // 在动作列表中找到选中的动作
            for (idx, action) in actions.iter().enumerate() {
                if *action == selected_action {
                    if self.verbose {
                        info!("神经网络选择动作 {}: {}", idx, action);
                    }
                    return Ok(idx);
                }
            }
        }

        // 如果神经网络选择的动作不在列表中，使用 select_action_from_list
        let idx = self.evaluator.select_action_from_list(game, actions, rng);
        if self.verbose {
            info!("神经网络选择动作 {}: {}", idx, actions[idx]);
        }
        Ok(idx)
    }

    fn select_choice(&self, game: &OnsenGame, choices: &[ActionValue], _rng: &mut StdRng) -> Result<usize> {
        if choices.is_empty() {
            anyhow::bail!("没有可选选项");
        }

        // 使用神经网络的 Choice 输出选择
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (idx, _) in choices.iter().enumerate() {
            let value = self.evaluator.evaluate_choice(game, idx);
            if value > best_value {
                best_value = value;
                best_idx = idx;
            }
        }

        if self.verbose {
            info!("神经网络选择选项 {}: {:?}", best_idx + 1, choices[best_idx]);
        }

        Ok(best_idx)
    }
}
