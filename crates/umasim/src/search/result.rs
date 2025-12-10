//! 搜索结果
//!
//! 定义分数分布统计和搜索输出结构。

use std::cell::Cell;
use crate::game::onsen::action::OnsenAction;
use crate::game::onsen::game::OnsenGame;
use crate::sample_collector::action_to_global_index;
use crate::training_sample::TrainingSample;
use super::SearchConfig;

/// 最大分数（用于分布直方图）
const MAX_SCORE: usize = 100000;

/// 单个动作的搜索结果
///
/// 统计多次模拟的分数分布，支持计算均值、标准差和加权平均分。
#[derive(Debug, Clone)]
pub struct ActionResult {
    /// 分数分布直方图
    /// distribution[score] = 该分数出现的次数
    distribution: Vec<u32>,

    /// 模拟次数
    num: u32,

    /// 分数总和（用于计算均值）
    sum: f64,

    /// 分数平方和（用于计算方差）
    sum_sq: f64,

    /// 最小分数
    min_score: f64,

    /// 最大分数
    max_score: f64,

    // ========== 缓存 ==========

    /// 缓存的加权平均分结果: (radical_factor, result)
    cached_weighted: Cell<Option<(f64, f64)>>,
}

impl Default for ActionResult {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionResult {
    /// 创建新的搜索结果
    pub fn new() -> Self {
        Self {
            distribution: vec![0; MAX_SCORE],
            num: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min_score: f64::MAX,
            max_score: f64::MIN,
            cached_weighted: Cell::new(None),
        }
    }

    /// 添加一次模拟结果
    ///
    /// # 参数
    /// - `score`: 模拟得到的最终分数
    pub fn add(&mut self, score: f64) {
        self.num += 1;
        self.sum += score;
        self.sum_sq += score * score;

        // 更新最小最大值
        self.min_score = self.min_score.min(score);
        self.max_score = self.max_score.max(score);

        // 更新分布直方图
        let idx = (score as usize).clamp(0, MAX_SCORE - 1);
        self.distribution[idx] += 1;

        // 清除缓存（数据已更新）
        self.cached_weighted.set(None);
    }

    /// 获取模拟次数
    pub fn count(&self) -> u32 {
        self.num
    }

    /// 计算均值
    pub fn mean(&self) -> f64 {
        if self.num == 0 {
            return 0.0;
        }
        self.sum / self.num as f64
    }

    /// 计算标准差
    pub fn stdev(&self) -> f64 {
        if self.num <= 1 {
            return 0.0;
        }
        let n = self.num as f64;
        let variance = (self.sum_sq - self.sum * self.sum / n) / (n - 1.0);
        variance.max(0.0).sqrt()
    }

    /// 计算加权平均分
    ///
    /// 使用排名加权的方式计算，激进度越高越偏向高分。
    /// 结果会被缓存，相同的 radical_factor 不会重复计算。
    ///
    /// # 参数
    /// - `radical_factor`: 激进度因子
    ///
    /// # 算法
    /// 对于每个分数 s：
    /// - rank_ratio = 累计到 s 的样本比例
    /// - weight = rank_ratio^radical_factor
    /// - weighted_sum += weight * count * s
    pub fn weighted_mean(&self, radical_factor: f64) -> f64 {
        if self.num == 0 {
            return 0.0;
        }

        // 激进度为 0 时直接返回均值
        if radical_factor.abs() < 1e-6 {
            return self.mean();
        }

        // 检查缓存
        if let Some((cached_rf, cached_result)) = self.cached_weighted.get() {
            if (cached_rf - radical_factor).abs() < 1e-9 {
                return cached_result;
            }
        }

        // 计算加权平均分
        let result = self.compute_weighted_mean(radical_factor);

        // 更新缓存
        self.cached_weighted.set(Some((radical_factor, result)));

        result
    }

    /// 内部计算加权平均分（不使用缓存）
    fn compute_weighted_mean(&self, radical_factor: f64) -> f64 {
        let n = self.num as f64;
        let n_inv = 1.0 / n;

        let mut cumulative = 0.0;
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for (score, &count) in self.distribution.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let c = count as f64;
            // 排名比例（累计到当前分数的样本比例）
            let rank_ratio = (cumulative + 0.5 * c) * n_inv;
            // 按排名加权
            let weight = rank_ratio.powf(radical_factor);

            weighted_sum += weight * c * score as f64;
            weight_total += weight * c;
            cumulative += c;
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            self.mean()
        }
    }

    /// 获取最小分数
    pub fn min(&self) -> f64 {
        if self.num == 0 {
            0.0
        } else {
            self.min_score
        }
    }

    /// 获取最大分数
    pub fn max(&self) -> f64 {
        if self.num == 0 {
            0.0
        } else {
            self.max_score
        }
    }
}

/// 搜索输出
///
/// 包含所有动作的搜索结果和最优动作信息。
#[derive(Debug, Clone)]
pub struct SearchOutput {
    /// 动作列表
    pub actions: Vec<OnsenAction>,

    /// 各动作的搜索结果
    pub action_results: Vec<ActionResult>,

    /// 最优动作索引
    pub best_action_idx: usize,

    /// 本次搜索使用的激进度因子
    pub radical_factor: f64,
}

impl SearchOutput {
    /// 创建搜索输出
    pub fn new(
        actions: Vec<OnsenAction>,
        action_results: Vec<ActionResult>,
        radical_factor: f64,
    ) -> Self {
        // 找到加权平均分最高的动作
        let best_action_idx = action_results
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let wa = a.weighted_mean(radical_factor);
                let wb = b.weighted_mean(radical_factor);
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        Self {
            actions,
            action_results,
            best_action_idx,
            radical_factor,
        }
    }

    /// 获取最优动作
    pub fn best_action(&self) -> &OnsenAction {
        &self.actions[self.best_action_idx]
    }

    /// 获取最优动作的搜索结果
    pub fn best_result(&self) -> &ActionResult {
        &self.action_results[self.best_action_idx]
    }

    /// 导出训练样本
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `config`: 搜索配置
    pub fn export_sample(&self, game: &OnsenGame, config: &SearchConfig) -> TrainingSample {
        // 1. 提取特征
        let features = game.extract_nn_features(None);

        // 2. Value Target: 最优动作的搜索结果
        let best = self.best_result();
        let value_target = vec![
            (best.mean() / 1000.0) as f32,                              // 归一化均值
            (best.stdev() / 150.0) as f32,                              // 归一化标准差
            (best.weighted_mean(self.radical_factor) / 1000.0) as f32,  // 归一化加权值
        ];

        // 3. Policy Target: softmax(各动作 weighted)
        let policy_target = self.calc_policy_target(config.policy_delta);

        // 4. Choice Target: 暂时为空
        let choice_target = vec![0.0_f32; 5];

        TrainingSample::new(features, policy_target, choice_target, value_target)
    }

    /// 计算 Policy Target
    ///
    /// 将各动作的加权平均分通过 softmax 转换为概率分布。
    fn calc_policy_target(&self, policy_delta: f64) -> Vec<f32> {
        // 计算各动作的加权平均分
        let values: Vec<f64> = self
            .action_results
            .iter()
            .map(|r| r.weighted_mean(self.radical_factor))
            .collect();

        // 找到最大值（用于数值稳定性）
        let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // 计算 exp((v - max) / delta)
        let exp_values: Vec<f64> = values
            .iter()
            .map(|v| ((v - max_v) / policy_delta).exp())
            .collect();
        let sum: f64 = exp_values.iter().sum();

        // 创建 50 维 policy target
        let mut policy = vec![0.0_f32; 50];

        // 将概率分配到对应的全局索引
        for (i, action) in self.actions.iter().enumerate() {
            if let Some(global_idx) = action_to_global_index(action) {
                if global_idx < 50 {
                    policy[global_idx] = (exp_values[i] / sum) as f32;
                }
            }
        }

        policy
    }

    /// 打印搜索结果摘要
    pub fn print_summary(&self) {
        println!("=== 搜索结果 (radical_factor={:.1}) ===", self.radical_factor);
        for (i, (action, result)) in self.actions.iter().zip(self.action_results.iter()).enumerate() {
            let mark = if i == self.best_action_idx { "*" } else { " " };
            println!(
                "{} {:12}: mean={:.0}, stdev={:.0}, weighted={:.0}, n={}",
                mark,
                format!("{}", action),
                result.mean(),
                result.stdev(),
                result.weighted_mean(self.radical_factor),
                result.count()
            );
        }
    }
}


