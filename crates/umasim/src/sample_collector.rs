/// 样本收集器模块
///
/// 用于在模拟过程中收集训练数据
/// 每回合记录游戏状态、选择的动作、事件选项等信息
/// 游戏结束后根据最终分数生成训练样本
use crate::game::onsen::action::OnsenAction;
use crate::training_sample::{NN_INPUT_DIM, TrainingSample};

/// Policy 输出维度
pub const POLICY_DIM: usize = 50;

/// 全局动作类型索引映射
///
/// 将 OnsenAction 转换为固定的全局索引，确保同一动作类型
/// 在不同回合始终对应相同的索引。
///
/// # 索引定义
/// - 0-4: Train(0-4) 五种训练
/// - 5: Sleep 休息
/// - 6: NormalOuting 普通外出
/// - 7: FriendOuting 友人外出
/// - 8: Race 比赛
/// - 9: Clinic 就医
/// - 10: PR 练习赛
/// - 11-20: Dig(0-9) 挖掘温泉
/// - 21-23: Upgrade(0-2) 装备升级
/// - 24-25: UseTicket(false/true) 使用温泉券
/// - 26-49: 保留
pub fn action_to_global_index(action: &OnsenAction) -> Option<usize> {
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

/// 单回合数据
#[derive(Debug, Clone)]
pub struct TurnData {
    /// 590 维特征向量
    pub features: Vec<f32>,
    /// 选择的动作的**全局索引**（使用 action_to_global_index 转换）
    pub global_action_idx: usize,
    /// 事件选项索引（如果有）
    pub choice_idx: Option<usize>,
    /// 事件选项数量
    pub num_choices: usize
}

/// 样本收集器
///
/// 在游戏过程中收集每回合的决策数据
/// 游戏结束后生成训练样本
#[derive(Debug, Clone)]
pub struct SampleCollector {
    /// 每回合的数据
    turn_data: Vec<TurnData>,
    /// 最终分数
    final_score: i32,
    /// 是否已完成
    is_finished: bool
}

impl SampleCollector {
    /// 创建新的样本收集器
    pub fn new() -> Self {
        Self {
            turn_data: Vec::with_capacity(78), // 预分配 78 回合
            final_score: 0,
            is_finished: false
        }
    }

    /// 记录一个回合的动作选择
    ///
    /// # 参数
    /// - `features`: 当前游戏状态的特征向量（590 维）
    /// - `action`: 选择的动作
    pub fn record_turn(&mut self, features: Vec<f32>, action: &OnsenAction) {
        debug_assert_eq!(features.len(), NN_INPUT_DIM, "特征维度必须是 {}", NN_INPUT_DIM);

        // 使用全局动作索引
        let global_action_idx = action_to_global_index(action).unwrap_or(0);

        self.turn_data.push(TurnData {
            features,
            global_action_idx,
            choice_idx: None,
            num_choices: 0
        });
    }

    /// 记录事件选项选择
    ///
    /// 在 `record_turn` 之后调用，为当前回合添加事件选项信息
    ///
    /// # 参数
    /// - `choice_idx`: 选择的事件选项索引
    /// - `num_choices`: 可选事件选项数量
    pub fn record_choice(&mut self, choice_idx: usize, num_choices: usize) {
        if let Some(last) = self.turn_data.last_mut() {
            debug_assert!(choice_idx < num_choices, "选项索引超出范围");
            last.choice_idx = Some(choice_idx);
            last.num_choices = num_choices;
        }
    }

    /// 设置最终分数
    pub fn set_final_score(&mut self, score: i32) {
        self.final_score = score;
        self.is_finished = true;
    }

    /// 获取最终分数
    pub fn final_score(&self) -> i32 {
        self.final_score
    }

    /// 获取回合数
    pub fn num_turns(&self) -> usize {
        self.turn_data.len()
    }

    /// 是否已完成
    pub fn is_finished(&self) -> bool {
        self.is_finished
    }

    /// 生成训练样本
    ///
    /// 将收集的回合数据转换为训练样本
    /// 每个回合生成一个样本，使用最终分数作为 value target
    ///
    /// # 返回
    /// 训练样本列表，每个回合一个样本
    pub fn finalize(self) -> Vec<TrainingSample> {
        if !self.is_finished {
            log::warn!("SampleCollector 未设置最终分数，使用默认值 0");
        }

        let final_score = self.final_score as f32;

        self.turn_data
            .into_iter()
            .map(|turn| {
                // Policy target: one-hot 编码选择的动作（使用全局索引）
                let mut policy_target = vec![0.0_f32; POLICY_DIM];
                if turn.global_action_idx < POLICY_DIM {
                    policy_target[turn.global_action_idx] = 1.0;
                }

                // Choice target: one-hot 编码选择的事件选项
                let mut choice_target = vec![0.0_f32; 5];
                if let Some(idx) = turn.choice_idx {
                    if idx < 5 {
                        choice_target[idx] = 1.0;
                    }
                }

                // Value target: [scoreMean, scoreStdev, value]
                // 使用最终分数作为均值，固定标准差 500
                let value_target = vec![
                    final_score / 1000.0, // 归一化分数
                    0.5,                  // 标准差 (500 / 1000)
                    final_score / 1000.0, // 价值
                ];

                TrainingSample::new(turn.features, policy_target, choice_target, value_target)
            })
            .collect()
    }
}

impl Default for SampleCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// 游戏样本（包含所有回合的样本和最终分数）
///
/// 用于批量收集时按分数排序
#[derive(Debug, Clone)]
pub struct GameSample {
    /// 最终分数
    pub final_score: i32,
    /// 所有回合的训练样本
    pub samples: Vec<TrainingSample>
}

impl GameSample {
    /// 从 SampleCollector 创建
    pub fn from_collector(collector: SampleCollector) -> Self {
        let final_score = collector.final_score();
        let samples = collector.finalize();
        Self { final_score, samples }
    }
}
