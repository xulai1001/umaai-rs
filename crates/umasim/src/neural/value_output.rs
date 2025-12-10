//! 评估器输出值定义
//!
//! 表示局面评估的结果，包含分数均值、标准差和加权值。

/// 评估器输出值
///
/// 用于表示局面评估的结果，支持分数分布建模。
///
/// # 字段说明
///
/// - `score_mean`: 预测的平均分数
/// - `score_stdev`: 预测分数的标准差（不确定性度量）
/// - `value`: 综合考虑激进因子后的加权值
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ValueOutput {
    /// 预测分数均值
    pub score_mean: f64,
    /// 预测分数标准差
    pub score_stdev: f64,
    /// 加权后的价值（考虑激进因子）
    pub value: f64
}

impl ValueOutput {
    /// 创建新的 ValueOutput
    pub fn new(score_mean: f64, score_stdev: f64) -> Self {
        Self {
            score_mean,
            score_stdev,
            value: score_mean // 默认 value = mean
        }
    }

    /// 使用激进因子计算加权值
    ///
    /// # 参数
    /// - `radical_factor`: 激进因子，越大越倾向高风险高回报
    ///
    /// # 公式
    /// `value = score_mean + radical_factor * score_stdev`
    pub fn with_radical_factor(mut self, radical_factor: f64) -> Self {
        self.value = self.score_mean + radical_factor * self.score_stdev;
        self
    }

    /// 表示非法动作的特殊值
    pub const ILLEGAL: ValueOutput = ValueOutput {
        score_mean: f64::NEG_INFINITY,
        score_stdev: 0.0,
        value: f64::NEG_INFINITY
    };

    /// 检查是否为非法值
    pub fn is_illegal(&self) -> bool {
        self.score_mean == f64::NEG_INFINITY
    }
}
