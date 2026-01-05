//! 神经网络训练员
//!
//! 使用 ONNX 神经网络模型进行决策的训练员。
//! 直接使用神经网络的 Policy 输出选择动作，不进行 MCTS 搜索。

use anyhow::Result;
use log::{info, warn};
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{
    game::{
        Trainer,
        onsen::{action::OnsenAction, game::OnsenGame}
    },
    gamedata::{ActionValue, EventData},
    neural::{Evaluator, HandwrittenEvaluator, NeuralNetEvaluator},
    training_sample::{CHOICE_DIM, POLICY_DIM}
};

// Choice 相关常量（对齐 muxue）
const CHOICE_OFFSET: usize = POLICY_DIM; // 50
const DEFAULT_CHOICE_TEMPERATURE: f64 = 1.0;

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
                if *action == selected_action.selection {
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

        // 默认事件选项用 handwritten
        //
        // 注：如果直接用未充分训练的 choice head，事件选择会接近随机，明显拉低整局分数。
        let evaluator = HandwrittenEvaluator::new();
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (idx, _) in choices.iter().enumerate() {
            let value = evaluator.evaluate_choice(game, idx);
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

    fn select_event_choice(
        &self, game: &OnsenGame, event: &EventData, choices: &[ActionValue], rng: &mut StdRng
    ) -> Result<usize> {
        if choices.is_empty() {
            return Ok(0);
        }

        // ===== Chance 事件：按 random_choice_prob 采样 =====
        if let Some(probs) = &event.random_choice_prob {
            if probs.len() != choices.len() {
                warn!(
                    "[Choice] 事件#{} {} random_choice_prob.len()={} != choices.len()={}，回退均匀随机",
                    event.id,
                    event.name,
                    probs.len(),
                    choices.len()
                );
                return Ok(rng.random_range(0..choices.len()));
            }
            match WeightedIndex::new(probs) {
                Ok(weights) => return Ok(weights.sample(rng)),
                Err(e) => {
                    warn!(
                        "[Choice] 事件#{} {} 权重非法（{}），回退均匀随机",
                        event.id, event.name, e
                    );
                    return Ok(rng.random_range(0..choices.len()));
                }
            }
        }

        // ===== 决策事件 =====
        let n = choices.len();

        // 单选项直接返回
        if n == 1 {
            return Ok(0);
        }

        // 超过 8 个选项：回退 handwritten
        if n > CHOICE_DIM {
            if self.verbose {
                warn!(
                    "[Choice] 事件#{} {} choices.len()={} > {}，回退 handwritten",
                    event.id, event.name, n, CHOICE_DIM
                );
            }
            return self.select_choice(game, choices, rng);
        }

        // 提取特征并推理（需要把 pending_choices 喂进特征）
        let features = game.extract_nn_features(Some(choices));
        let output = match self.evaluator.infer(&features) {
            Ok(o) => o,
            Err(e) => {
                warn!(
                    "[Choice] 事件#{} {} 推理失败: {}，回退 handwritten",
                    event.id, event.name, e
                );
                return self.select_choice(game, choices, rng);
            }
        };

        // 提取 choice logits [CHOICE_OFFSET, CHOICE_OFFSET + n)
        let logits: Vec<f64> = output[CHOICE_OFFSET..CHOICE_OFFSET + n]
            .iter()
            .map(|&x| x as f64)
            .collect();

        // Softmax 采样（带 temperature）
        let temperature = DEFAULT_CHOICE_TEMPERATURE.max(0.01);
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if !max_logit.is_finite() {
            warn!(
                "[Choice] 事件#{} {} logits 非有限值，回退均匀随机",
                event.id, event.name
            );
            return Ok(rng.random_range(0..n));
        }

        let weights: Vec<f64> = logits
            .iter()
            .map(|&logit| ((logit - max_logit) / temperature).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            warn!(
                "[Choice] 事件#{} {} softmax sum 异常，回退均匀随机",
                event.id, event.name
            );
            return Ok(rng.random_range(0..n));
        }

        match WeightedIndex::new(&weights) {
            Ok(dist) => {
                let selected = dist.sample(rng);
                if self.verbose {
                    info!(
                        "[Choice] 事件#{} {}: choice_head selected={}, temperature={:.2}, logits={:?}",
                        event.id, event.name, selected, temperature, logits
                    );
                }
                Ok(selected)
            }
            Err(e) => {
                warn!(
                    "[Choice] 事件#{} {} WeightedIndex 失败: {}，回退均匀随机",
                    event.id, event.name, e
                );
                Ok(rng.random_range(0..n))
            }
        }
    }
}
