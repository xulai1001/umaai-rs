use std::{
    fs::File,
    io::{BufWriter, Write}
};

use anyhow::Result;
/// 神经网络训练样本模块
///
/// 用于收集和导出训练数据
use serde::{Deserialize, Serialize};

/// 训练样本结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// 神经网络输入特征（590 维）
    ///
    /// 维度分布：
    /// - 全局信息（410 维）
    ///   - 搜索参数（6 维）
    ///   - 回合信息（78 维）
    ///   - 马娘属性（15 维）
    ///   - 体力与干劲（5 维）
    ///   - 训练数值（30 维）
    ///   - 失败率（5 维）
    ///   - 温泉剧本特定（140 维）- 支持温泉选择学习
    ///   - 其他信息（61 维）
    ///   - 事件选项特征（70 维）
    /// - 支援卡信息（30 维 × 6 张 = 180 维）
    pub nn_input: Vec<f32>,

    /// Policy 目标（50 维动作概率分布）
    pub policy_target: Vec<f32>,

    /// Choice 目标（5 维事件选项概率分布）
    pub choice_target: Vec<f32>,

    /// Value 目标（3 维：scoreMean, scoreStdev, value）
    pub value_target: Vec<f32>
}

/// 神经网络输入维度常量
pub const NN_INPUT_DIM: usize = 590;

impl TrainingSample {
    /// 创建新的训练样本
    pub fn new(nn_input: Vec<f32>, policy_target: Vec<f32>, choice_target: Vec<f32>, value_target: Vec<f32>) -> Self {
        assert_eq!(nn_input.len(), NN_INPUT_DIM, "nn_input 必须是 {} 维", NN_INPUT_DIM);
        assert_eq!(policy_target.len(), 50, "policy_target 必须是 50 维");
        assert_eq!(choice_target.len(), 5, "choice_target 必须是 5 维");
        assert_eq!(value_target.len(), 3, "value_target 必须是 3 维");

        Self {
            nn_input,
            policy_target,
            choice_target,
            value_target
        }
    }

    /// 创建空的 choice_target（无事件选项时使用）
    pub fn empty_choice_target() -> Vec<f32> {
        vec![0.0; 5]
    }
}

/// 训练样本批次（用于批量保存）
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingSampleBatch {
    pub samples: Vec<TrainingSample>
}

impl TrainingSampleBatch {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn add(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// 保存为 JSON 文件
    pub fn save_json(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// 保存为二进制文件（更紧凑）
    pub fn save_binary(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, self)?;
        Ok(())
    }

    /// 从 JSON 文件加载
    pub fn load_json(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let batch = serde_json::from_reader(file)?;
        Ok(batch)
    }

    /// 从二进制文件加载
    pub fn load_binary(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let batch = bincode::deserialize_from(file)?;
        Ok(batch)
    }

    /// 追加保存到文件（JSONL 格式，每行一个样本）
    pub fn append_jsonl(&self, path: &str) -> Result<()> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);

        for sample in &self.samples {
            serde_json::to_writer(&mut writer, sample)?;
            writeln!(writer)?;
        }

        Ok(())
    }
}

impl Default for TrainingSampleBatch {
    fn default() -> Self {
        Self::new()
    }
}
