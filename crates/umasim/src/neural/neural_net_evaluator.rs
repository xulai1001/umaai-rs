//! 神经网络评估器
//!
//! 使用 ONNX 模型进行策略和价值评估。
//!
//! # 输入输出维度
//! - 输入：1121 维特征向量 (Global 587 + Card 89*6)
//! - 输出：61 维 (Policy 50 + Choice 8 + Value 3)
//!
//! # Value 反归一化
//! - scoreMean = VALUE_MEAN + VALUE_SCALE * output[58]
//! - scoreStdev = STDEV_SCALE * abs(output[59])
//! - value = VALUE_MEAN + VALUE_SCALE * output[60]

use std::{
    cell::RefCell,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::Instant,
};

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
const INPUT_DIM: usize = 1121;

/// 输出维度
const OUTPUT_DIM: usize = 61;

/// Policy 输出维度
const POLICY_DIM: usize = 50;

/// Choice 输出维度
const CHOICE_DIM: usize = 8;

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

fn load_onnx_model(model_path: &str) -> Result<OnnxModel> {
    tract_onnx::onnx()
        .model_for_path(model_path)
        .context("无法读取 ONNX 模型文件")?
        .into_optimized()
        .context("模型优化失败")?
        .into_runnable()
        .context("模型转换失败")
}

fn extract_value_from_output(output: &[f32]) -> ValueOutput {
    let score_mean = VALUE_MEAN + VALUE_SCALE * output[POLICY_DIM + CHOICE_DIM] as f64;
    let score_stdev = STDEV_SCALE * output[POLICY_DIM + CHOICE_DIM + 1] as f64;
    ValueOutput::new(score_mean, score_stdev.abs())
}

thread_local! {
    static THREAD_LOCAL_MODEL: RefCell<Option<(String, OnnxModel)>> = RefCell::new(None);
}

fn action_to_global_index_v1(action: &OnsenAction) -> Option<usize> {
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
        OnsenAction::UseTicket(is_super) => Some(if *is_super { 25 } else { 24 }),
    }
}

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

        let model = load_onnx_model(model_path)?;

        log::info!("ONNX 模型加载成功");

        Ok(Self { model: Arc::new(model) })
    }

    /// 执行神经网络推理
    ///
    /// # 参数
    /// - `features`: 1121 维输入特征
    ///
    /// # 返回
    /// 61 维输出向量
    pub fn infer(&self, features: &[f32]) -> Result<Vec<f32>> {
        if features.len() != INPUT_DIM {
            anyhow::bail!("输入维度错误: 期望 {}, 实际 {}", INPUT_DIM, features.len());
        }

        // 创建输入张量 [1, 1121]
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
        // output[POLICY_DIM + CHOICE_DIM + 2] 是 value，但我们使用 score_mean 作为主要评估值
        extract_value_from_output(output)
    }

    /// 根据 Policy logits 采样选择动作索引
    ///
    /// 注意：神经网络输出的是 logits（可为负数），不能直接当作概率使用。
    /// 这里对合法动作做 softmax，再按概率采样。
    fn sample_action_index(&self, logits: &[f32], legal_mask: &[bool], rng: &mut StdRng) -> usize {
        // 找到合法动作中最大的 logit（softmax 数值稳定）
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if i < legal_mask.len() && legal_mask[i] && v > max_logit {
                max_logit = v;
            }
        }

        // 没有任何合法动作，回退到第一个合法动作
        if !max_logit.is_finite() {
            return legal_mask.iter().position(|&x| x).unwrap_or(0);
        }

        // 计算 softmax 权重（只对合法动作赋值）
        let mut weights: Vec<f64> = vec![0.0; logits.len()];
        let mut sum: f64 = 0.0;
        for (i, &v) in logits.iter().enumerate() {
            if i < legal_mask.len() && legal_mask[i] {
                let w = ((v - max_logit) as f64).exp();
                weights[i] = w;
                sum += w;
            }
        }

        if sum <= 0.0 || !sum.is_finite() {
            // 数值异常时回退到最后一个合法动作（保持确定性）
            return legal_mask.iter().rposition(|&x| x).unwrap_or(0);
        }

        // 采样：在 [0, sum) 上采样，再落到累计权重区间
        let r: f64 = rng.random::<f64>() * sum;
        let mut acc = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            acc += w;
            if r <= acc {
                return i;
            }
        }

        // 理论上不会走到这里，兜底返回最后一个合法动作
        legal_mask.iter().rposition(|&x| x).unwrap_or(0)
    }

    /// 将动作转换为全局索引
    ///
    /// 与 sample_collector::action_to_global_index 保持一致
    fn action_to_global_index(action: &OnsenAction) -> Option<usize> {
        action_to_global_index_v1(action)
    }
}

// ============================================================================
// ThreadLocalNeuralNetLeafEvaluator
// ============================================================================

/// 搜索 leaf eval 专用：每线程懒加载一份模型，避免跨线程共享 `SimplePlan` 的线程安全风险。
#[derive(Clone)]
pub struct ThreadLocalNeuralNetLeafEvaluator {
    model_path: Arc<String>,
    stats: Arc<ThreadLocalNeuralNetLeafStats>,
}

#[derive(Debug)]
struct ThreadLocalNeuralNetLeafStats {
    model_loads: AtomicU64,
    infer_batches: AtomicU64,
    infer_calls: AtomicU64,
    infer_errors: AtomicU64,
    infer_time_ns_total: AtomicU64,
}

impl ThreadLocalNeuralNetLeafStats {
    fn new() -> Self {
        Self {
            model_loads: AtomicU64::new(0),
            infer_batches: AtomicU64::new(0),
            infer_calls: AtomicU64::new(0),
            infer_errors: AtomicU64::new(0),
            infer_time_ns_total: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThreadLocalNeuralNetLeafStatsSnapshot {
    pub model_loads: u64,
    pub infer_batches: u64,
    pub infer_calls: u64,
    pub infer_errors: u64,
    pub infer_time_ns_total: u64,
}

impl ThreadLocalNeuralNetLeafEvaluator {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: Arc::new(model_path.into()),
            stats: Arc::new(ThreadLocalNeuralNetLeafStats::new()),
        }
    }

    pub fn stats(&self) -> ThreadLocalNeuralNetLeafStatsSnapshot {
        ThreadLocalNeuralNetLeafStatsSnapshot {
            model_loads: self.stats.model_loads.load(Ordering::Relaxed),
            infer_batches: self.stats.infer_batches.load(Ordering::Relaxed),
            infer_calls: self.stats.infer_calls.load(Ordering::Relaxed),
            infer_errors: self.stats.infer_errors.load(Ordering::Relaxed),
            infer_time_ns_total: self.stats.infer_time_ns_total.load(Ordering::Relaxed),
        }
    }

    /// 微批推理：输入 [batch,1121] 的扁平数组，输出 [batch,61] 的扁平数组。
    ///
    /// - 当模型不支持动态 batch 时，会回退为循环 `infer(1)`（并打印 warning）。
    pub fn infer_batch(&self, features_flat: &[f32], batch: usize) -> Result<Vec<f32>> {
        if batch == 0 {
            anyhow::bail!("batch 不能为 0");
        }
        if features_flat.len() != batch * INPUT_DIM {
            anyhow::bail!(
                "输入维度错误: 期望 {} (=batch*INPUT_DIM), 实际 {}",
                batch * INPUT_DIM,
                features_flat.len()
            );
        }

        // 计数：一次 batch 调用，覆盖 batch 个样本
        self.stats.infer_batches.fetch_add(1, Ordering::Relaxed);
        self.stats.infer_calls.fetch_add(batch as u64, Ordering::Relaxed);
        let t0 = Instant::now();

        let run_once = |model: &OnnxModel| -> Result<Vec<f32>> {
            let input = tract_ndarray::Array2::from_shape_vec((batch, INPUT_DIM), features_flat.to_vec())
                .context("创建 batch 输入张量失败")?;
            let output = model.run(tvec!(input.into_tvalue())).context("batch 推理失败")?;
            let output_tensor = output[0].to_array_view::<f32>().context("提取 batch 输出张量失败")?;
            let out: Vec<f32> = output_tensor.iter().copied().collect();
            if out.len() != batch * OUTPUT_DIM {
                anyhow::bail!(
                    "batch 输出维度错误: 期望 {} (=batch*OUTPUT_DIM), 实际 {}",
                    batch * OUTPUT_DIM,
                    out.len()
                );
            }
            Ok(out)
        };

        let result = THREAD_LOCAL_MODEL.with(|slot| -> Result<Vec<f32>> {
            let mut slot = slot.borrow_mut();
            let need_reload = match slot.as_ref() {
                Some((p, _)) => p != self.model_path.as_str(),
                None => true,
            };
            if need_reload {
                log::info!("[NN][leaf] 线程内加载模型: {}", self.model_path.as_str());
                let model = load_onnx_model(self.model_path.as_str())?;
                *slot = Some((self.model_path.as_str().to_string(), model));
                self.stats.model_loads.fetch_add(1, Ordering::Relaxed);
            }

            let (_, model) = slot.as_ref().expect("thread_local model");

            // 先尝试动态 batch（ONNX 导出脚本已开启 dynamic_axes）
            match run_once(model) {
                Ok(v) => Ok(v),
                Err(e) => {
                    // 降级：逐样本 infer(1)，保证功能不挂（但会很慢）
                    log::warn!(
                        "[NN][leaf] batch 推理失败，回退为逐样本 infer(1)（性能受限）。原因: {e}"
                    );
                    let mut out_all = Vec::with_capacity(batch * OUTPUT_DIM);
                    for i in 0..batch {
                        let start = i * INPUT_DIM;
                        let end = start + INPUT_DIM;

                        let input = tract_ndarray::Array2::from_shape_vec((1, INPUT_DIM), features_flat[start..end].to_vec())
                            .context("创建单样本输入张量失败")?;
                        let output = model.run(tvec!(input.into_tvalue())).context("单样本推理失败")?;
                        let output_tensor = output[0].to_array_view::<f32>().context("提取单样本输出张量失败")?;
                        let out: Vec<f32> = output_tensor.iter().copied().collect();
                        if out.len() != OUTPUT_DIM {
                            anyhow::bail!("单样本输出维度错误: 期望 {}, 实际 {}", OUTPUT_DIM, out.len());
                        }
                        out_all.extend_from_slice(&out);
                    }
                    Ok(out_all)
                }
            }
        });

        let elapsed_ns: u64 = t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        self.stats.infer_time_ns_total.fetch_add(elapsed_ns, Ordering::Relaxed);
        if result.is_err() {
            self.stats.infer_errors.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// 给定一批 features，直接返回每个样本的 value 输出（mean/stdev）。
    pub fn evaluate_features_batch(&self, features_flat: &[f32], batch: usize) -> Result<Vec<ValueOutput>> {
        let out = self.infer_batch(features_flat, batch)?;
        let mut values = Vec::with_capacity(batch);
        for i in 0..batch {
            let start = i * OUTPUT_DIM;
            let end = start + OUTPUT_DIM;
            values.push(extract_value_from_output(&out[start..end]));
        }
        Ok(values)
    }

    fn infer(&self, features: &[f32]) -> Result<Vec<f32>> {
        if features.len() != INPUT_DIM {
            anyhow::bail!("输入维度错误: 期望 {}, 实际 {}", INPUT_DIM, features.len());
        }

        self.stats.infer_batches.fetch_add(1, Ordering::Relaxed);
        self.stats.infer_calls.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();

        let result = THREAD_LOCAL_MODEL.with(|slot| -> Result<Vec<f32>> {
            let mut slot = slot.borrow_mut();
            let need_reload = match slot.as_ref() {
                Some((p, _)) => p != self.model_path.as_str(),
                None => true,
            };
            if need_reload {
                log::info!("[NN][leaf] 线程内加载模型: {}", self.model_path.as_str());
                let model = load_onnx_model(self.model_path.as_str())?;
                *slot = Some((self.model_path.as_str().to_string(), model));
                self.stats.model_loads.fetch_add(1, Ordering::Relaxed);
            }

            let (_, model) = slot.as_ref().expect("thread_local model");

            // 创建输入张量 [1, 1121]
            let input = tract_ndarray::Array2::from_shape_vec((1, INPUT_DIM), features.to_vec())
                .context("创建输入张量失败")?;

            // 运行推理
            let output = model.run(tvec!(input.into_tvalue())).context("推理失败")?;

            // 提取输出
            let output_tensor = output[0].to_array_view::<f32>().context("提取输出张量失败")?;
            let out: Vec<f32> = output_tensor.iter().copied().collect();

            if out.len() != OUTPUT_DIM {
                anyhow::bail!("输出维度错误: 期望 {}, 实际 {}", OUTPUT_DIM, out.len());
            }

            Ok(out)
        });

        let elapsed_ns: u64 = t0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        self.stats.infer_time_ns_total.fetch_add(elapsed_ns, Ordering::Relaxed);

        if result.is_err() {
            self.stats.infer_errors.fetch_add(1, Ordering::Relaxed);
        }

        result
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
            if let Some(idx) = action_to_global_index_v1(action) {
                if idx < POLICY_DIM {
                    legal_mask[idx] = true;
                }
            }
        }

        // 采样选择全局动作索引
        let global_idx = self.sample_action_index(&policy, &legal_mask, rng);

        // 找到对应的动作
        for action in &actions {
            if let Some(idx) = action_to_global_index_v1(action) {
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
            if let Some(global_idx) = action_to_global_index_v1(action) {
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

impl Evaluator<OnsenGame> for ThreadLocalNeuralNetLeafEvaluator {
    fn select_action(&self, _game: &OnsenGame, _rng: &mut StdRng) -> Option<OnsenAction> {
        // leaf eval 专用：不在 rollout 过程中使用 NN policy（避免混变量）
        None
    }

    fn evaluate(&self, game: &OnsenGame) -> ValueOutput {
        // 如果游戏结束，返回实际分数
        if game.turn >= game.max_turn() {
            let score = game.uma.calc_score() as f64;
            return ValueOutput::new(score, 0.0);
        }

        let features = game.extract_nn_features(None);
        match self.infer(&features) {
            Ok(output) => extract_value_from_output(&output),
            Err(e) => {
                log::warn!("[NN][leaf] 推理失败: {}", e);
                // 回退到简单评估（不允许 silent fallback：必须有日志）
                let score = game.uma.calc_score() as f64;
                let progress = game.turn as f64 / game.max_turn() as f64;
                let stdev = 500.0 * (1.0 - progress) + 100.0;
                ValueOutput::new(score, stdev)
            }
        }
    }
}

// ============================================================================
// Send + Sync 实现
// ============================================================================

// NeuralNetEvaluator 通过 Arc 共享模型，是线程安全的
unsafe impl Send for NeuralNetEvaluator {}
unsafe impl Sync for NeuralNetEvaluator {}
