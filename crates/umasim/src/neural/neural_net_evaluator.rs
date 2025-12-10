//! 神经网络评估器
//!
//! 使用 ONNX 模型进行策略和价值评估。
//!
//! # 输入输出维度
//! - 输入：590 维特征向量
//! - 输出：58 维 (Policy 50 + Choice 5 + Value 3)
//!
//! # Value 反归一化
//! - scoreMean = VALUE_MEAN + VALUE_SCALE * output[55]
//! - scoreStdev = STDEV_SCALE * output[56]
//! - value = VALUE_MEAN + VALUE_SCALE * output[57]

use std::sync::Arc;

use anyhow::{Context, Result};
use rand::{Rng, rngs::StdRng};
use tract_onnx::prelude::*;

use super::{Evaluator, ValueOutput};
use crate::game::{
    Game,
    onsen::{action::OnsenAction, game::OnsenGame}
};

// ============================================================================
// 常量定义（与 Python config.py 一致）
// ============================================================================

/// 输入维度
const INPUT_DIM: usize = 590;

/// 输出维度
const OUTPUT_DIM: usize = 58;

/// Policy 输出维度
const POLICY_DIM: usize = 50;

/// Choice 输出维度
const CHOICE_DIM: usize = 5;

/// Value 反归一化参数 - 均值
const VALUE_MEAN: f64 = 58000.0;

/// Value 反归一化参数 - 缩放
const VALUE_SCALE: f64 = 300.0;

/// Stdev 反归一化参数 - 缩放
const STDEV_SCALE: f64 = 150.0;

// ============================================================================
// 类型别名
// ============================================================================

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// ============================================================================
// NeuralNetEvaluator
// ============================================================================

/// 神经网络评估器
///
/// 使用 ONNX 模型进行策略和价值评估。
#[derive(Clone)]
pub struct NeuralNetEvaluator {
    /// ONNX 模型（使用 Arc 共享，因为 SimplePlan 不可克隆）
    model: Arc<OnnxModel>
}

impl NeuralNetEvaluator {
    /// 从 ONNX 文件加载模型
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    ///
    /// # 返回
    /// 加载成功返回 NeuralNetEvaluator，失败返回错误
    pub fn load(model_path: &str) -> Result<Self> {
        log::info!("加载 ONNX 模型: {}", model_path);

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("无法读取 ONNX 模型文件")?
            .into_optimized()
            .context("模型优化失败")?
            .into_runnable()
            .context("模型转换失败")?;

        log::info!("ONNX 模型加载成功");

        Ok(Self { model: Arc::new(model) })
    }

    /// 执行神经网络推理
    ///
    /// # 参数
    /// - `features`: 590 维输入特征
    ///
    /// # 返回
    /// 58 维输出向量
    pub fn infer(&self, features: &[f32]) -> Result<Vec<f32>> {
        if features.len() != INPUT_DIM {
            anyhow::bail!("输入维度错误: 期望 {}, 实际 {}", INPUT_DIM, features.len());
        }

        // 创建输入张量 [1, 590]
        let input =
            tract_ndarray::Array2::from_shape_vec((1, INPUT_DIM), features.to_vec()).context("创建输入张量失败")?;

        // 运行推理
        let output = self.model.run(tvec!(input.into_tvalue())).context("推理失败")?;

        // 提取输出
        let output_tensor = output[0].to_array_view::<f32>().context("提取输出张量失败")?;

        let result: Vec<f32> = output_tensor.iter().copied().collect();

        if result.len() != OUTPUT_DIM {
            anyhow::bail!("输出维度错误: 期望 {}, 实际 {}", OUTPUT_DIM, result.len());
        }

        Ok(result)
    }

    /// 从输出中提取 Policy 概率分布
    fn extract_policy(&self, output: &[f32]) -> Vec<f32> {
        output[0..POLICY_DIM].to_vec()
    }

    /// 从输出中提取 Choice 概率分布
    fn extract_choice(&self, output: &[f32]) -> Vec<f32> {
        output[POLICY_DIM..POLICY_DIM + CHOICE_DIM].to_vec()
    }

    /// 从输出中提取 Value（反归一化）
    fn extract_value(&self, output: &[f32]) -> ValueOutput {
        let score_mean = VALUE_MEAN + VALUE_SCALE * output[POLICY_DIM + CHOICE_DIM] as f64;
        let score_stdev = STDEV_SCALE * output[POLICY_DIM + CHOICE_DIM + 1] as f64;
        // output[57] 是 value，但我们使用 score_mean 作为主要评估值
        ValueOutput::new(score_mean, score_stdev.abs())
    }

    /// 根据 Policy 概率分布采样选择动作索引
    fn sample_action_index(&self, policy: &[f32], legal_mask: &[bool], rng: &mut StdRng) -> usize {
        // 应用合法动作掩码并归一化
        let mut probs: Vec<f64> = policy
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if i < legal_mask.len() && legal_mask[i] {
                    p.max(0.0) as f64
                } else {
                    0.0
                }
            })
            .collect();

        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            // 如果没有合法动作，返回第一个合法的
            return legal_mask.iter().position(|&x| x).unwrap_or(0);
        }

        // 归一化
        for p in &mut probs {
            *p /= sum;
        }

        // 采样
        let r: f64 = rng.random();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        // 返回最后一个合法动作
        legal_mask.iter().rposition(|&x| x).unwrap_or(0)
    }

    /// 将动作转换为全局索引
    ///
    /// 与 sample_collector::action_to_global_index 保持一致
    fn action_to_global_index(action: &OnsenAction) -> Option<usize> {
        match action {
            OnsenAction::Train(t) => Some(*t as usize),
            OnsenAction::Sleep => Some(5),
            OnsenAction::NormalOuting => Some(6),
            OnsenAction::FriendOuting => Some(7),
            OnsenAction::Race => Some(8),
            OnsenAction::Clinic => Some(9),
            OnsenAction::PR => Some(10),
            OnsenAction::Dig(idx) => Some(11 + *idx as usize),
            OnsenAction::Upgrade(idx) => Some(21 + *idx as usize),
            OnsenAction::UseTicket(is_super) => Some(if *is_super { 25 } else { 24 })
        }
    }
}

// ============================================================================
// Evaluator trait 实现
// ============================================================================

impl Evaluator<OnsenGame> for NeuralNetEvaluator {
    fn select_action(&self, game: &OnsenGame, rng: &mut StdRng) -> Option<OnsenAction> {
        // 获取可选动作列表
        let actions = game.list_actions().ok()?;
        if actions.is_empty() {
            return None;
        }

        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        let output = match self.infer(&features) {
            Ok(o) => o,
            Err(e) => {
                log::warn!("神经网络推理失败: {}", e);
                // 回退到随机选择
                return actions.first().cloned();
            }
        };

        // 提取 Policy
        let policy = self.extract_policy(&output);

        // 构建合法动作掩码（使用全局动作索引）
        let mut legal_mask = vec![false; POLICY_DIM];
        for action in &actions {
            if let Some(idx) = Self::action_to_global_index(action) {
                if idx < POLICY_DIM {
                    legal_mask[idx] = true;
                }
            }
        }

        // 采样选择全局动作索引
        let global_idx = self.sample_action_index(&policy, &legal_mask, rng);

        // 找到对应的动作
        for action in &actions {
            if let Some(idx) = Self::action_to_global_index(action) {
                if idx == global_idx {
                    return Some(action.clone());
                }
            }
        }

        // 如果没找到，返回第一个动作
        actions.first().cloned()
    }

    fn evaluate(&self, game: &OnsenGame) -> ValueOutput {
        // 如果游戏结束，返回实际分数
        if game.turn >= game.max_turn() {
            let score = game.uma.calc_score() as f64;
            return ValueOutput::new(score, 0.0);
        }

        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        match self.infer(&features) {
            Ok(output) => self.extract_value(&output),
            Err(e) => {
                log::warn!("神经网络推理失败: {}", e);
                // 回退到简单评估
                let score = game.uma.calc_score() as f64;
                let progress = game.turn as f64 / game.max_turn() as f64;
                let stdev = 500.0 * (1.0 - progress) + 100.0;
                ValueOutput::new(score, stdev)
            }
        }
    }

    fn evaluate_choice(&self, game: &OnsenGame, choice_index: usize) -> f64 {
        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        match self.infer(&features) {
            Ok(output) => {
                let choice = self.extract_choice(&output);
                if choice_index < choice.len() {
                    choice[choice_index] as f64
                } else {
                    0.0
                }
            }
            Err(_) => {
                // 默认选择第一个
                if choice_index == 0 { 1.0 } else { 0.0 }
            }
        }
    }

    fn select_action_from_list(&self, game: &OnsenGame, actions: &[OnsenAction], _rng: &mut StdRng) -> usize {
        if actions.is_empty() {
            return 0;
        }

        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        let output = match self.infer(&features) {
            Ok(o) => o,
            Err(_) => return 0
        };

        // 提取 Policy
        let policy = self.extract_policy(&output);

        // 找到给定动作列表中 Policy 值最大的动作（使用全局索引）
        let mut best_idx = 0;
        let mut best_value = f32::NEG_INFINITY;

        for (action_idx, action) in actions.iter().enumerate() {
            if let Some(global_idx) = Self::action_to_global_index(action) {
                if global_idx < policy.len() {
                    let value = policy[global_idx];
                    if value > best_value {
                        best_value = value;
                        best_idx = action_idx;
                    }
                }
            }
        }

        best_idx
    }
}

// ============================================================================
// Send + Sync 实现
// ============================================================================

// NeuralNetEvaluator 通过 Arc 共享模型，是线程安全的
unsafe impl Send for NeuralNetEvaluator {}
unsafe impl Sync for NeuralNetEvaluator {}
