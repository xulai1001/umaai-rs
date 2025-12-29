//! 神经网络评估器模块
//!
//! 提供神经网络训练和推理相关的评估器功能。
//!
//! # 模块结构
//!
//! - [`Evaluator`]: 评估器 trait
//! - [`HandwrittenEvaluator`]: 手写启发式评估器（用于数据收集）
//! - [`NeuralNetEvaluator`]: 神经网络评估器（ONNX 推理）
//! - [`RandomEvaluator`]: 随机评估器（基准测试）
//! - [`ValueOutput`]: 评估器输出值
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use umasim::neural::{HandwrittenEvaluator, NeuralNetEvaluator, ValueOutput, Evaluator};
//!
//! // 使用手写评估器
//! let evaluator = HandwrittenEvaluator::new();
//! let action = evaluator.select_action(&game, &mut rng);
//!
//! // 使用神经网络评估器
//! let nn_evaluator = NeuralNetEvaluator::load("model.onnx")?;
//! let action = nn_evaluator.select_action(&game, &mut rng);
//! ```

mod evaluator;
mod handwritten_evaluator;
mod neural_net_evaluator;
mod value_output;

// 公开导出
pub use evaluator::{Evaluator, RandomEvaluator};
pub use handwritten_evaluator::HandwrittenEvaluator;
pub use neural_net_evaluator::{
    NeuralNetEvaluator,
    ThreadLocalNeuralNetLeafEvaluator,
    ThreadLocalNeuralNetLeafStatsSnapshot,
};
pub use value_output::ValueOutput;
