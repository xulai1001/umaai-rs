use std::collections::HashMap;

use anyhow::Result;
use log::info;
use serde::{Deserialize, Serialize};

use crate::{
    game::onsen::OnsenOrder,
    gamedata::load_json,
    utils::{Array5, Array6}
};

/// 训练基础值表格
/// - 外层 Vec: 训练类型（通常5种，可扩展）
/// -中层 Vec: 等级（通常Lv1-5，可扩展）
/// - 内层 [i32; 7]: 7种属性（速耐力根智 + SP + 体力消耗）
pub type TrainingBasicTable = Vec<Vec<[i32; 7]>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GameConstants {
    /// 基础属性上限, 1200不减半
    pub five_status_limit_base: [i32; 5],
    /// 训练名字
    pub train_names: Vec<String>,
    /// 心情名字
    pub motivation_names: Vec<String>,
    /// 训练会失败的体力阈值，拟合的
    pub training_vital_threshold: Vec<Vec<f32>>,
    /// 团队卡Buff解除概率
    pub group_buff_end_prob: Vec<f64>,
    // 评分相关
    /// 每pt对应分数
    pub pt_score_rate: f32,
    /// 每级hint对应的pt
    pub hint_pt_rate: f32,
    /// 每点属性对应的评分 ~2000(翻倍2800)
    pub five_status_final_score: Vec<i32>,
    /// 评价档次
    pub rank_scores: Vec<i32>,
    /// 评价名字
    pub rank_names: Vec<String>,
    /// 事件出现概率
    pub event_probs: HashMap<String, f64>,
    /// 不能出现随机事件的回合
    pub no_event_turns: Vec<i32>,
    /// 基础Hint率
    pub base_hint_rate: f64,
    // ========== 步骤 1 迁出字段（Phase 2） ==========
    // 以下三个字段已从 `gamedata/constants.json` 迁出到 `gamedata/default_config.toml`
    // （顶层 + `game_config.toml` 可覆盖）。`#[serde(default, skip)]` 表示：
    // - `skip`：serde 反序列化时忽略（避免旧 `constants.json` 残留字段引发反序列化错误）
    // - `default`：若字段缺失给一个类型默认值（Vec→空 vec，f32/i32→0）
    // - **运行时注入**：`init_global_with_config(&GameConfig)` 在 `constants.json` 加载后，
    //   把 `GameConfig` 中用户可调值（mcts_turn_bonus / pt_favor_rate / race_grades）
    //   注入到这里。所有现有 `global!(GAMECONSTANTS).race_grades` 等引用点保持不变。
    // - **后续步骤**：若需要彻底从 `GameConstants` 移除这三个字段（改为独立全局或扩展其他结构），
    //   再统一迁移引用点（mcts_trainer、uma.rs:calc_score_with_pt_favor、base/action.rs:race_grade 等）。
    /// 每回合的比赛等级（72 项；URA 回合 72-77 固定 G1 不在此表）
    /// 默认值来源："game_config.toml" → "default_config.toml" → `default_race_grades()`
    #[serde(default, skip)]
    pub race_grades: Vec<i32>,
    /// 休息结果分布 +30=18%,+50=57%,+70=25%
    pub rest_probs: Vec<i32>,
    /// 红点属性
    pub hint_event_value: Vec<Array6>,
    /// 每张卡最大提供Hint等级
    pub max_hint_per_card: i32,
    /// PT 特化时，PT 评分倍数（与 `mcts_selection` 联用）
    /// 默认值来源："game_config.toml" → "default_config.toml" → `default_pt_favor_rate()`
    #[serde(default, skip)]
    pub pt_favor_rate: f32,
    /// PT特化时，超过1200的属性压缩系数
    pub five_status_favor_rate: Vec<f32>,
    /// 蒙特卡洛每回合比手写逻辑增加的分数（搜索启发式）
    /// 默认值来源："game_config.toml" → "default_config.toml" → `default_mcts_turn_bonus()`
    #[serde(default, skip)]
    pub mcts_turn_bonus: i32
}

impl GameConstants {
    pub fn load() -> Result<Self> {
        info!("载入游戏数据");
        load_json("gamedata/constants.json")
    }

    /// 查五维属性对应的终局评分，越界饱和到表末
    ///
    /// `five_status_final_score` 长度有限（当前 3399 档），而五维上限 =
    /// 剧本基值 + 继承三次，蓝因子拉满时会超出表长。此前三处消费点行为各不相同：
    /// 裸下标会 panic，`unwrap_or(0)` 会静默把分数算成 0（进而让手写策略
    /// 误判该维收益为极大负值、永久回避它）。统一走本方法，越界一律饱和。
    pub fn status_final_score(&self, status: i32) -> i32 {
        // 不用 `get().unwrap_or(0)` 兜底：那正是缺陷 C 的形态——下标算错时静默返回 0，
        // 把「查不到」伪装成「零分」。这里下标已饱和到 len-1，非空表必然命中；
        // 空表是数据损坏，返回 0 是唯一合理的降级。
        match self.five_status_final_score.last() {
            None => 0,
            Some(&last) => {
                let idx = (status.max(0) as usize).min(self.five_status_final_score.len() - 1);
                if idx == self.five_status_final_score.len() - 1 {
                    last
                } else {
                    self.five_status_final_score[idx]
                }
            }
        }
    }

    pub fn get_rank_name(&self, score: i32) -> String {
        self.rank_scores
            .iter()
            .enumerate()
            .find_map(|(i, x)| {
                if score.max(0) < *x {
                    Some(self.rank_names[i - 1].clone())
                } else {
                    None
                }
            })
            .unwrap_or("US9".to_string())
    }

    /// 随机事件为支援卡，马娘，掉心情和不发生的分布
    pub fn get_event_distribution(&self) -> [f64; 4] {
        let probs = &self.event_probs;
        let mut ret = [probs["card_event"], probs["uma_event"], probs["drop_motivation"], 0.0];
        ret[3] = 1.0 - ret[0] - ret[1] - ret[2];
        ret
    }
}

/// MCTS 搜索配置
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MctsConfig {
    /// 每个动作的搜索次数
    #[serde(default = "default_mcts_search_n")]
    pub search_n: usize,
    /// 激进度因子最大值
    #[serde(default = "default_mcts_radical_factor_max")]
    pub radical_factor_max: f64,
    /// 最大搜索深度（0 = 搜到游戏结束）
    #[serde(default = "default_mcts_max_depth")]
    pub max_depth: usize,
    /// P3-MVP：leaf eval 评估器开关（用于 A/B 对照）
    ///
    /// 重要约定（避免混变量）：
    /// - MVP 阶段 rollout 过程的动作选择固定使用 HandwrittenEvaluator（不引入 NN policy）
    /// - 该字段仅控制：当 `max_depth>0` 截断 rollout 且未终局时，leaf 估值使用：
    ///   - `"handwritten"`：HandwrittenEvaluator::evaluate
    ///   - `"nn"`：NeuralNetEvaluator::evaluate（要求 `GameConfig.neuralnet_model_path` 可用；无效时应直接报错退出）
    #[serde(default = "default_mcts_rollout_evaluator")]
    pub rollout_evaluator: String,
    /// E4：leaf eval 微批大小（仅在 max_depth>0 && rollout_evaluator="nn" 时生效）
    ///
    /// **当前未接线**：本字段只被写入、从未被搜索逻辑读取。
    /// 经验值：32（与默认 search_group_size 对齐），后续可按模型/CPU 调整。
    #[serde(default = "default_mcts_rollout_batch_size")]
    pub rollout_batch_size: usize,
    /// Policy softmax 温度（分数每降低多少，概率变成 1/e 倍）
    #[serde(default = "default_mcts_policy_delta")]
    pub policy_delta: f64,
    /// 拉面杯：哪些阶段走搜索（逗号分隔，见 `RamenSearchStages::parse`）
    ///
    /// 只对 `scenario = "ramen"` + `trainer = "mcts"` 生效。拉面一局约 173 个
    /// 决策点，全搜代价过高，故按阶段挑。缺省 `train,ramen`：实测只搜 `train`
    /// 时那 61 个 `RamenSelect` 点一次 rollout 都不跑，补上后 42 局配对
    /// +2306 分（t=7.94）；同一笔算力加到 `train` 的 `search_n` 上只值 +39
    /// 分（t=0.14），即 `train` 一侧已饱和。
    #[serde(default = "default_mcts_ramen_search_stages")]
    pub ramen_search_stages: String,

    // ========== UCB 搜索分配参数 ==========
    /// 是否启用 UCB 搜索分配
    #[serde(default = "default_mcts_use_ucb")]
    pub use_ucb: bool,
    /// UCB 每组搜索次数（UCB 每组步长）。
    ///
    /// 首组大小会被 clamp 进 `search_n`，避免 `search_group_size > search_n` 时首组溢出。
    #[serde(default = "default_mcts_search_group_size")]
    pub search_group_size: usize,
    /// UCB 探索常数 (cpuct)
    #[serde(default = "default_mcts_search_cpuct")]
    pub search_cpuct: f64,
    /// UCB 探索项的缩放标尺（量纲与 score 相同），不是实测统计量。
    ///
    /// 生产 toml 取 15000 是有意调猛的探索强度；`SearchConfig::default()` 的 2200
    /// 是 C++ UmaAi 默认值。二者服务不同场景，不需要对齐。
    #[serde(default = "default_mcts_expected_search_stdev")]
    pub expected_search_stdev: f64,
    /// 是否按 `(回合, 阶段)` 重新播种 rollout 随机流（外挂 CRN，仅 onsen 生效）
    ///
    /// 拉面规则层已由无状态流接管（RNG Refactor Plan v2 §5.2），不受此开关影响。
    #[serde(default = "default_mcts_crn_stage_reseed")]
    pub crn_stage_reseed: bool,
    /// 决策理由分差门限（保留字段，不再用作触发器）
    ///
    /// 当前每回合都输出决策理由；分差仅用于选择其他候选的显示颜色档位
    ///（与最优项的差距 <30/<100/<300/其余 → 亮绿/绿/黄/灰）。字段值仍
    /// 写入 [`DecisionReasonData::threshold`]，供下游兼容与对照。
    #[serde(default = "default_mcts_reason_gap_threshold")]
    pub reason_gap_threshold: f64,
    /// 决策理由最多显示选项数
    ///
    /// 全部候选按评分降序只取前 N 个进入显示与分析，其余**直接排除**：
    /// 不显示内容、不做原因分析。中选者一般也在前 N 内；若不在，渲染时
    /// 仍按"首选"在第 1 行单独显示。
    #[serde(default = "default_mcts_reason_max_display")]
    pub reason_max_display: usize
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            search_n: default_mcts_search_n(),
            radical_factor_max: default_mcts_radical_factor_max(),
            max_depth: default_mcts_max_depth(),
            rollout_evaluator: default_mcts_rollout_evaluator(),
            rollout_batch_size: default_mcts_rollout_batch_size(),
            policy_delta: default_mcts_policy_delta(),
            ramen_search_stages: default_mcts_ramen_search_stages(),
            use_ucb: default_mcts_use_ucb(),
            search_group_size: default_mcts_search_group_size(),
            search_cpuct: default_mcts_search_cpuct(),
            expected_search_stdev: default_mcts_expected_search_stdev(),
            crn_stage_reseed: default_mcts_crn_stage_reseed(),
            reason_gap_threshold: default_mcts_reason_gap_threshold(),
            reason_max_display: default_mcts_reason_max_display()
        }
    }
}

fn default_mcts_search_n() -> usize {
    10240 // 默认搜索次数
}

fn default_mcts_radical_factor_max() -> f64 {
    2.0 // 默认激进度最大值
}

fn default_mcts_max_depth() -> usize {
    0 // 搜到游戏结束
}

fn default_mcts_rollout_evaluator() -> String {
    "handwritten".to_string()
}

fn default_mcts_rollout_batch_size() -> usize {
    32
}

/// `ramen_search_stages` 缺省值：训练 + 吃面
fn default_mcts_ramen_search_stages() -> String {
    "train,ramen".to_string()
}

fn default_mcts_policy_delta() -> f64 {
    100.0
}

fn default_mcts_use_ucb() -> bool {
    true // 默认使用UCB分配
}

fn default_mcts_search_group_size() -> usize {
    512
}

fn default_mcts_search_cpuct() -> f64 {
    1.0
}

fn default_mcts_expected_search_stdev() -> f64 {
    2200.0
}

/// [`MctsConfig::crn_stage_reseed`] 默认值
fn default_mcts_crn_stage_reseed() -> bool {
    true
}

/// `reason_gap_threshold` 缺省值：保留兼容值，分差档位（<30/<100/<300/其余）
/// 在 `reason.rs` 内硬编码；颜色档位不再依赖此门限
fn default_mcts_reason_gap_threshold() -> f64 {
    150.0
}

/// `reason_max_display` 缺省值：理由最多显示/分析评分前 5 个选项
fn default_mcts_reason_max_display() -> usize {
    5
}

/// 训练数据生成（collector）配置
///
/// 说明：
/// - 该配置主要服务于“按样本 scoreMean 筛选”的 mean-filter 数据生成器（P0/P1）。
/// - 为了避免重复配置，搜索相关字段允许为空（None），实现侧可回退到 `mcts` 段或 `SearchConfig::default()`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// 目标 accepted 样本数
    #[serde(default = "default_collector_target_samples")]
    pub target_samples: usize,
    /// 最大模拟局数（阈值过高时避免无限跑）
    #[serde(default = "default_collector_max_games")]
    pub max_games: usize,

    /// 样本筛选阈值：scoreMean >= threshold（scoreMean = value_target[0]）
    #[serde(default = "default_collector_score_mean_threshold")]
    pub score_mean_threshold: f64,
    /// 是否丢弃 scoreMean==0 的样本（即使 threshold=0 也可过滤）
    #[serde(default = "default_collector_drop_zero_mean")]
    pub drop_zero_mean: bool,

    // ========== Choice 样本（P2）==========
    /// 是否采集 decision event 的 choice 样本
    #[serde(default = "default_collector_collect_choice")]
    pub collect_choice: bool,
    /// choice 评估：每个选项的 rollout 次数（方案 A）
    #[serde(default = "default_collector_choice_rollouts_per_option")]
    pub choice_rollouts_per_option: usize,
    /// choice softmax 温度（越小越尖锐）
    #[serde(default = "default_collector_choice_policy_delta")]
    pub choice_policy_delta: f64,
    /// choice gate 阈值：scoreMean >= threshold；None 则回退到 score_mean_threshold
    #[serde(default)]
    pub choice_score_mean_threshold: Option<f64>,
    /// 跳过 choices.len() > CHOICE_DIM 的事件（避免特征/label 维度不一致）
    #[serde(default = "default_collector_choice_skip_if_too_many")]
    pub choice_skip_if_too_many: bool,

    /// choice 样本是否跟随 action 的采样回合范围（turn_min/turn_max/turn_stride）
    #[serde(default = "default_collector_choice_follow_action_turn_range")]
    pub choice_follow_action_turn_range: bool,

    /// 当 choice_follow_action_turn_range=true 且当前回合不采样时：是否仍使用 rollout 决策（否则回退 select_choice，成本更低但轨迹分布会变化）
    #[serde(default = "default_collector_choice_rollout_on_uncollected_turns")]
    pub choice_rollout_on_uncollected_turns: bool,

    /// 达到 target_samples 后是否切换为“快速完成”（不再跑 FlatSearch/choice rollouts，直接用手写策略推进）
    #[serde(default = "default_collector_fast_after_target")]
    pub fast_after_target: bool,

    /// 采样回合范围（按人类回合 1..=78；内部会用 human_turn = turn+1 做判断）
    #[serde(default = "default_collector_turn_min")]
    pub turn_min: i32,
    /// 采样回合范围（按人类回合 1..=78；内部会用 human_turn = turn+1 做判断）
    #[serde(default = "default_collector_turn_max")]
    pub turn_max: i32,
    /// 采样步长（stride=2 表示每隔 1 回合采 1 条）
    #[serde(default = "default_collector_turn_stride")]
    pub turn_stride: i32,

    /// 输出目录（P1：分片写盘）
    #[serde(default = "default_collector_output_dir")]
    pub output_dir: String,
    /// 输出名称（可选）：若非空，则实际输出目录会变为 `output_dir/output_name`（再按需追加时间戳）
    ///
    /// 典型用法：
    /// - output_dir = "training_data"
    /// - output_name = "p2_60k_s128_r2"
    #[serde(default = "default_collector_output_name")]
    pub output_name: String,
    /// 是否自动在输出目录名后追加时间戳（避免每次手动改目录名）
    ///
    /// - true: 输出到 `.../<name>_<timestamp>/`
    /// - false: 输出到 `.../<name>/`
    #[serde(default = "default_collector_output_append_timestamp")]
    pub output_append_timestamp: bool,
    /// 时间戳格式（chrono strftime）
    ///
    /// 注意：Windows 路径不允许 `:` 等字符，建议使用 `_` 分隔，例如 `%Y%m%d_%H%M%S`
    #[serde(default = "default_collector_output_timestamp_format")]
    pub output_timestamp_format: String,
    /// 每个分片的样本数
    #[serde(default = "default_collector_shard_size")]
    pub shard_size: usize,
    /// manifest 文件名（输出目录内）
    #[serde(default = "default_collector_manifest_name")]
    pub manifest_name: String,
    /// scoreMean values 文件名（输出目录内，append-only，用于精确分位数）
    #[serde(default = "default_collector_score_mean_values_name")]
    pub score_mean_values_name: String,
    /// 是否允许 resume（输出目录存在时从已有 part 继续）
    #[serde(default = "default_collector_resume")]
    pub resume: bool,
    /// 是否允许覆盖输出目录（危险操作，需显式开启）
    #[serde(default = "default_collector_overwrite")]
    pub overwrite: bool,

    /// 并行线程数（外层顺序跑 game；FlatSearch 内部用 rayon）
    #[serde(default = "default_collector_threads")]
    pub threads: usize,

    /// 进度输出间隔（按局数）
    #[serde(default = "default_collector_progress_interval")]
    pub progress_interval: usize,

    // ========== SearchConfig 覆盖（可选）==========
    /// 覆盖 search_n（None 则回退到 mcts/search 默认）
    #[serde(default)]
    pub search_n: Option<usize>,
    /// 覆盖 max_depth（None 则回退）
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// 覆盖 radical_factor_max（None 则回退）
    #[serde(default)]
    pub radical_factor_max: Option<f64>,
    /// 覆盖 policy_delta（None 则回退）
    #[serde(default)]
    pub policy_delta: Option<f64>,

    /// 覆盖 use_ucb（None 则回退）
    #[serde(default)]
    pub use_ucb: Option<bool>,
    /// 覆盖 search_group_size（None 则回退）
    #[serde(default)]
    pub search_group_size: Option<usize>,
    /// 覆盖 search_cpuct（None 则回退）
    #[serde(default)]
    pub search_cpuct: Option<f64>,
    /// 覆盖 expected_search_stdev（None 则回退）
    #[serde(default)]
    pub expected_search_stdev: Option<f64>
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            target_samples: default_collector_target_samples(),
            max_games: default_collector_max_games(),
            score_mean_threshold: default_collector_score_mean_threshold(),
            drop_zero_mean: default_collector_drop_zero_mean(),
            collect_choice: default_collector_collect_choice(),
            choice_rollouts_per_option: default_collector_choice_rollouts_per_option(),
            choice_policy_delta: default_collector_choice_policy_delta(),
            choice_score_mean_threshold: None,
            choice_skip_if_too_many: default_collector_choice_skip_if_too_many(),
            choice_follow_action_turn_range: default_collector_choice_follow_action_turn_range(),
            choice_rollout_on_uncollected_turns: default_collector_choice_rollout_on_uncollected_turns(),
            fast_after_target: default_collector_fast_after_target(),
            turn_min: default_collector_turn_min(),
            turn_max: default_collector_turn_max(),
            turn_stride: default_collector_turn_stride(),
            output_dir: default_collector_output_dir(),
            output_name: default_collector_output_name(),
            output_append_timestamp: default_collector_output_append_timestamp(),
            output_timestamp_format: default_collector_output_timestamp_format(),
            shard_size: default_collector_shard_size(),
            manifest_name: default_collector_manifest_name(),
            score_mean_values_name: default_collector_score_mean_values_name(),
            resume: default_collector_resume(),
            overwrite: default_collector_overwrite(),
            threads: default_collector_threads(),
            progress_interval: default_collector_progress_interval(),
            search_n: None,
            max_depth: None,
            radical_factor_max: None,
            policy_delta: None,
            use_ucb: None,
            search_group_size: None,
            search_cpuct: None,
            expected_search_stdev: None
        }
    }
}

fn default_collector_target_samples() -> usize {
    100000
}

fn default_collector_max_games() -> usize {
    50000
}

fn default_collector_score_mean_threshold() -> f64 {
    60000.0
}

fn default_collector_drop_zero_mean() -> bool {
    true
}

fn default_collector_collect_choice() -> bool {
    true
}

fn default_collector_choice_rollouts_per_option() -> usize {
    8
}

fn default_collector_choice_policy_delta() -> f64 {
    50.0
}

fn default_collector_choice_skip_if_too_many() -> bool {
    true
}

fn default_collector_choice_follow_action_turn_range() -> bool {
    true
}

fn default_collector_choice_rollout_on_uncollected_turns() -> bool {
    false
}

fn default_collector_fast_after_target() -> bool {
    true
}

fn default_collector_turn_min() -> i32 {
    1
}

fn default_collector_turn_max() -> i32 {
    78
}

fn default_collector_turn_stride() -> i32 {
    1
}

fn default_collector_output_dir() -> String {
    "training_data/mean_filtered".to_string()
}

fn default_collector_output_name() -> String {
    "".to_string()
}

fn default_collector_output_append_timestamp() -> bool {
    false
}

fn default_collector_output_timestamp_format() -> String {
    "%Y%m%d_%H%M%S".to_string()
}

fn default_collector_shard_size() -> usize {
    4096
}

fn default_collector_manifest_name() -> String {
    "manifest.json".to_string()
}

fn default_collector_score_mean_values_name() -> String {
    "score_mean_values.bin".to_string()
}

fn default_collector_resume() -> bool {
    true
}

fn default_collector_overwrite() -> bool {
    false
}

fn default_collector_threads() -> usize {
    24
}

fn default_collector_progress_interval() -> usize {
    100
}

/// 运行配置（临时）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameConfig {
    /// 剧本类型: "basic" | "onsen" | "ramen"（ramen 为当前主线）
    #[serde(default = "default_scenario")]
    pub scenario: String,
    /// 日志级别: "debug" (完整显示) | "off" (全部关闭)
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// 训练员类型: "manual" | "random" | "handwritten" | "collector" | "neuralnet" | "mcts"
    #[serde(default = "default_trainer")]
    pub trainer: String,
    /// neuralnet ONNX 模型路径（仅 trainer="neuralnet" / "nn" 生效）
    #[serde(default = "default_neuralnet_model_path")]
    pub neuralnet_model_path: String,
    /// 模拟次数（默认1次，设置大于1可多次模拟并统计）
    #[serde(default = "default_simulation_count")]
    pub simulation_count: usize,
    /// 马娘ID
    pub uma: u32,
    /// 卡组（ID，突破等级）
    pub cards: [u32; 6],
    /// 种马蓝因子个数
    pub blue_count: Array5,
    /// 种马额外属性
    pub extra_count: Array6,
    /// 温泉顺序
    pub onsen_order: OnsenOrder,
    /// collector 配置（用于训练数据生成工具）
    #[serde(default)]
    pub collector: CollectorConfig,
    /// MCTS 配置（可选）
    #[serde(default)]
    pub mcts: MctsConfig,
    /// 允许MCTS自由选择温泉
    #[serde(default)]
    pub mcts_selected_onsen: bool,
    /// 蒙特卡洛输出评分还是PT重视结果
    pub mcts_selection: String,
    /// 蒙特卡洛每回合期望得分加成（搜索启发式参数；Phase 2 步骤 1 从 constants.json 迁出）
    #[serde(default = "default_mcts_turn_bonus")]
    pub mcts_turn_bonus: i32,
    /// PT 偏好评分倍率（与 mcts_selection 联用；Phase 2 步骤 1 从 constants.json 迁出）
    #[serde(default = "default_pt_favor_rate")]
    pub pt_favor_rate: f32,
    /// 比赛等级表（72 项，对应回合 0-71；URA 回合 72-77 固定 G1 不在此表）
    /// 默认值与迁出前 constants.json 一致；用户可在 game_config.toml 顶层覆盖
    #[serde(default = "default_race_grades")]
    pub race_grades: Vec<i32>,
    /// 拉面杯**第3年**地区选择策略（Phase 2 步骤 5 接入）
    /// - "all"：枚举第3年所有合法组合（120 个）交给 Trainer（默认）
    /// - "fixed"：按 `ramen_region_fixed[0]` 单组合，跳过枚举
    /// 第1/2年固定走 all 枚举（不在本策略范围内）
    #[serde(default)]
    pub ramen_region_strategy: RamenRegionStrategy,
    /// 第3年固定地区组合（`Fixed` 策略时生效；长度必须 = 1）
    ///
    /// 例如 `[[10, 12, 14]]`：第3年固定选 [10,12,14]
    #[serde(default)]
    pub ramen_region_fixed: Option<Vec<[usize; 3]>>,
    /// 拉面杯客户端动作决策由谁负责（默认 `mcts`，见 [`RamenTrainerPolicy`]）
    #[serde(default)]
    pub ramen_trainer_policy: RamenTrainerPolicy,
    /// 拉面杯网络模型路径（`ramen_trainer_policy` 不是 `mcts` 时生效）
    ///
    /// 需同目录存在同名 `.json` 旁车（`<model>.onnx.json`）。与温泉的
    /// [`Self::neuralnet_model_path`] 是两个不同的模型，不可混用。
    #[serde(default)]
    pub ramen_nn_model_path: Option<String>,
    /// 在线决策记录开关（默认开；仅 umaai 实时运行使用，离线 sim/bench 不读）
    ///
    /// 开时按局落盘 `logs/game{id}/`：`thisTurn.json` 原文 + `decisions.csv` +
    /// `meta.json`（详见 `umaai::decision::record` 与 issues.md 对应规划条目）。
    #[serde(default = "default_luck_record")]
    pub luck_record: bool,
    /// 拉面杯：友人出行是否必须走完 5 次（默认开）。
    ///
    /// 开时手写策略（及以手写为基策的 MCTS）对友人出行施加**完成硬门限**：
    /// 剩余出行次数达到"剩余可出行回合数"时就强制出行（不再受隐藏风味闸门约束），
    /// 保证 5 次走完。关时回到纯动态估值口径（可能主动跳过第 5 次）。
    ///
    /// 依据：第三年可出行回合被必赛/夏合宿压缩，纯估值口径下第 5 次约有
    /// 8%~20% 的对局走不完（赛程越差越明显）；硬门限的代价见 experiments 记录。
    #[serde(default = "default_friend_complete")]
    pub friend_complete_required: bool,
}

fn default_friend_complete() -> bool {
    true
}

fn default_luck_record() -> bool {
    true
}

fn default_mcts_turn_bonus() -> i32 {
    70
}

fn default_pt_favor_rate() -> f32 {
    // 与 gamedata/default_config.toml 保持一致（2.0 = 2026-09-15 扫参 + N=4096 深测定档）
    2.0
}

fn default_race_grades() -> Vec<i32> {
    // 与迁出前 constants.json race_grades 一致（72 项）
    // 注意：此默认值与 gamedata/default_config.toml 中的 race_grades 保持一致；
    // 仅当 default_config.toml 缺该字段时（异常情况）才使用此处兜底
    vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 4, 3, 3, 4, 3, 3, 2, 3, 1, 1, 3, 4, 3, 4, 2, 2, 1, 2, 1, 1, 1, 1, 1,
        3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 3, 3, 3, 3, 3, 1, 2, 1, 1, 1, 2, 1,
    ]
}

impl GameConfig {
    /// 为 `init_global()` 兜底提供的默认值（不依赖任何 TOML 文件）
    ///
    /// 仅在测试或调试场景无 `load_game_config()` 结果时使用；正常入口
    /// 必须先 `load_game_config()` 拿到完整 `GameConfig` 后调用
    /// `init_global_with_config(&config)`。
    ///
    /// 注意：部分字段（uma/cards/blue_count/extra_count/onsen_order/mcts_selection）
    /// 没有合理兜底值，调用方应确保只在已显式构造过 `GameConfig` 的场景才用本函数。
    pub fn default_for_init() -> Self {
        Self {
            scenario: default_scenario(),
            log_level: default_log_level(),
            trainer: default_trainer(),
            neuralnet_model_path: default_neuralnet_model_path(),
            simulation_count: default_simulation_count(),
            // 无合理兜底的字段：用 0/默认值占位，调用方不应依赖这些值
            uma: 0,
            cards: [0; 6],
            blue_count: [0; 5],
            extra_count: [0; 6],
            onsen_order: OnsenOrder::default(),
            collector: CollectorConfig::default(),
            mcts: MctsConfig::default(),
            mcts_selected_onsen: false,
            mcts_selection: "score".to_string(),
            mcts_turn_bonus: default_mcts_turn_bonus(),
            pt_favor_rate: default_pt_favor_rate(),
            race_grades: default_race_grades(),
            ramen_region_strategy: RamenRegionStrategy::default(),
            ramen_region_fixed: None,
            ramen_trainer_policy: RamenTrainerPolicy::default(),
            ramen_nn_model_path: None,
            luck_record: default_luck_record(),
            friend_complete_required: default_friend_complete()
        }
    }

    /// 仿真参数：剧本、训练员、马娘、卡组、模拟次数
    pub fn simulation(&self) -> SimulationConfig {
        SimulationConfig {
            scenario: self.scenario.clone(),
            trainer: self.trainer.clone(),
            uma: self.uma,
            cards: self.cards,
            blue_count: self.blue_count,
            extra_count: self.extra_count,
            simulation_count: self.simulation_count
        }
    }

    /// 搜索参数：MCTS、神经网络、用户可调搜索项（mcts_turn_bonus / pt_favor_rate / race_grades）
    pub fn search(&self) -> SearchConfig {
        SearchConfig {
            mcts: self.mcts.clone(),
            mcts_selection: self.mcts_selection.clone(),
            neuralnet_model_path: self.neuralnet_model_path.clone(),
            mcts_turn_bonus: self.mcts_turn_bonus,
            pt_favor_rate: self.pt_favor_rate,
            race_grades: self.race_grades.clone()
        }
    }

    /// 策略参数：拉面杯地区/超级拉面选择策略等（步骤 5 接入）
    pub fn policy(&self) -> PolicyConfig {
        PolicyConfig {
            ramen_region_strategy: self.ramen_region_strategy,
            ramen_region_fixed: self.ramen_region_fixed.clone()
        }
    }

    /// 输出参数：日志级别、统计级别等
    pub fn output(&self) -> OutputConfig {
        OutputConfig {
            log_level: self.log_level.clone() // 步骤 3 后续：统计级别（None / Summary / Turn / Detailed）等
        }
    }

    /// 开发者参数：collector 数据收集、线程数等
    pub fn dev(&self) -> DeveloperConfig {
        DeveloperConfig {
            collector: self.collector.clone(),
            num_threads: self.collector.threads
        }
    }
}

// ========== 五个子配置结构（Phase 2 步骤 2+3） ==========
//
// 设计目的：业务代码可按需通过 `game_config.simulation()` / `.search()` 等访问子配置，
// 避免直接依赖整个 `GameConfig`。渐进式保留 `GameConfig` 聚合壳，子结构由方法按需构造（拷贝）。
// 后续步骤可逐步将业务模块从 `game_config.xxx` 迁移到 `game_config.simulation().xxx`。

/// 仿真参数（剧本/训练员/马娘/卡组/模拟次数）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// 剧本类型: "basic" | "onsen" | "ramen"
    pub scenario: String,
    /// 训练员类型: "manual" | "random" | "handwritten" | "collector" | "neuralnet" | "mcts"
    pub trainer: String,
    /// 马娘 ID
    pub uma: u32,
    /// 卡组（6 张支援卡 ID）
    pub cards: [u32; 6],
    /// 种马蓝因子个数
    pub blue_count: Array5,
    /// 种马额外属性
    pub extra_count: Array6,
    /// 模拟次数（默认 1 次）
    pub simulation_count: usize
}

/// 搜索参数（MCTS、神经网络、用户可调搜索项）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    /// MCTS 详细参数
    pub mcts: MctsConfig,
    /// MCTS 优先选择评分还是 PT
    pub mcts_selection: String,
    /// neuralnet ONNX 模型路径
    pub neuralnet_model_path: String,
    /// 蒙特卡洛每回合期望得分加成（用户可调）
    pub mcts_turn_bonus: i32,
    /// PT 偏好评分倍率（用户可调）
    pub pt_favor_rate: f32,
    /// 比赛等级表 72 项（用户可调）
    pub race_grades: Vec<i32>
}

/// 拉面杯地区选择策略（Phase 2 步骤 5 接入）
///
/// 仅对**第3年**生效（解决 C(10,3)=120 组合的性能问题）。第1/2年固定走 all 枚举。
///
/// - `All`：枚举第3年所有合法组合（120 个）交给 Trainer（默认）
/// - `Fixed`：按 `ramen_region_fixed[0]` 指定的3个地区选区，跳过枚举
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum RamenRegionStrategy {
    #[default]
    #[serde(rename = "all")]
    All,
    #[serde(rename = "fixed")]
    Fixed
}

/// 拉面杯客户端动作决策由谁负责
///
/// 只作用于 umaai 客户端的实际对局；事件选项在三种取值下都走手写策略。
///
/// - `mcts`（默认）：既有逻辑，按 `[mcts]` 配置搜索。
/// - `mcts_nn_hint`：执行与 `mcts` 完全相同，另外在每个动作决策上显示网络的推荐作参考。
/// - `nn`：全部动作决策由 `ramen_nn_model_path` 指定的网络直接给出，不做任何搜索；
///   比搜索快得多、分数更低，是以分数换速度的选项。只适用于训练覆盖的卡组构成，
///   见 `gamedata/default_config.toml` 的说明。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RamenTrainerPolicy {
    /// 既有搜索逻辑
    #[default]
    #[serde(rename = "mcts")]
    Mcts,
    /// 执行既有搜索逻辑，另外显示网络推荐
    #[serde(rename = "mcts_nn_hint")]
    MctsNnHint,
    /// 全部动作决策由网络直接给出
    #[serde(rename = "nn")]
    Nn
}

impl RamenTrainerPolicy {
    /// 是否需要加载网络模型（除 `mcts` 外都需要）
    pub fn needs_model(self) -> bool {
        !matches!(self, Self::Mcts)
    }
}

/// 策略参数（手写/未来模型策略参数）
///
/// 当前承载拉面杯第3年地区选择策略；后续扩展可加入超级拉面选择策略、未来模型策略参数等。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// 拉面杯第3年地区选择策略
    pub ramen_region_strategy: RamenRegionStrategy,
    /// 第3年固定地区组合（`Fixed` 策略时生效；长度必须 = 1，每项为 3 个地区 id）
    ///
    /// 例如 `[[10, 12, 14]]`：第3年固定选 [10,12,14]
    pub ramen_region_fixed: Option<Vec<[usize; 3]>>
}

/// 输出参数（日志、统计级别等）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    /// 日志级别: "debug" | "off" | "info" | "trace"
    pub log_level: String // 步骤 3 后续：统计级别（None / Summary / Turn / Detailed）
}

/// 开发者参数（collector、线程数、调试开关）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeveloperConfig {
    /// 训练数据收集（collector）配置
    pub collector: CollectorConfig,
    /// 线程数
    pub num_threads: usize
}

fn default_scenario() -> String {
    "basic".to_string()
}

fn default_log_level() -> String {
    "debug".to_string()
}

fn default_trainer() -> String {
    "manual".to_string()
}

fn default_neuralnet_model_path() -> String {
    // 默认路径
    "saved_models/onsen_v4/model.onnx".to_string()
}

fn default_simulation_count() -> usize {
    1
}

/// 用户覆盖配置（对应 `game_config.toml`）。
///
/// `[mcts]` 为可选覆盖层：省略整段时全部字段为 `None`，merge 后保留
/// `default_config.toml` 的生产参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverrideGameConfig {
    /// 温泉顺序（可选；game_config.toml 可省略 `[onsen_order]` 段，使用 default）
    #[serde(default)]
    pub onsen_order: OnsenOrder,
    pub config_override: OverrideConfig,
    /// MCTS 可选覆盖（`None` 字段不改写 default）
    #[serde(default)]
    pub mcts: OverrideMctsConfig,
    /// 拉面杯第3年地区选择策略（顶层覆盖；对应 `GameConfig::ramen_region_strategy`）
    ///
    /// `None` = 不覆盖 default_config.toml。注意：game_config.toml 顶层写
    /// `ramen_region_strategy = "fixed"`/`ramen_region_fixed = [[...]]` 才走这里；
    /// `[config_override]` 段内没有这两个字段（地区策略不是"常用覆盖项"）。
    #[serde(default)]
    pub ramen_region_strategy: Option<RamenRegionStrategy>,
    /// 第3年固定地区组合（顶层覆盖；仅 `ramen_region_strategy=fixed` 时生效）
    ///
    /// `None` = 不覆盖 default；写 `ramen_region_fixed = [[.., .., ..]]` 即覆盖；
    /// 要显式清空 default 的 fixed 组合可写空数组 `[]`。
    #[serde(default)]
    pub ramen_region_fixed: Option<Vec<[usize; 3]>>,
    /// 动作决策来源（顶层覆盖；对应 `GameConfig::ramen_trainer_policy`）
    ///
    /// `None` = 不覆盖 default_config.toml（即保持 `mcts`）。
    #[serde(default)]
    pub ramen_trainer_policy: Option<RamenTrainerPolicy>,
    /// 网络模型路径（顶层覆盖；对应 `GameConfig::ramen_nn_model_path`）
    ///
    /// `None` = 不覆盖 default。模型不随仓库发布，填本机路径（绝对路径或相对
    /// workspace 根目录）。
    #[serde(default)]
    pub ramen_nn_model_path: Option<String>
}

/// MCTS 覆盖配置：每个字段都是可选覆盖（`None` = 不覆盖 `default_config.toml`）。
///
/// `Default` 必须是全 `None`。若任何字段缺省成 `Some(MctsConfig 的代码缺省值)`，
/// 整段省略 `[mcts]` 时会把 10240/2.0/512/2200/32 打到生产配置上。
/// `deny_unknown_fields` 让 `[mcts]` 里拼错的键显式报错。
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverrideMctsConfig {
    /// 每个动作的搜索次数（可选覆盖）
    #[serde(default)]
    pub search_n: Option<usize>,
    /// 激进度因子最大值（可选覆盖）
    #[serde(default)]
    pub radical_factor_max: Option<f64>,
    /// 最大搜索深度（可选覆盖）
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// leaf eval 评估器开关（可选覆盖）
    #[serde(default)]
    pub rollout_evaluator: Option<String>,
    /// leaf eval 微批大小（可选覆盖）
    #[serde(default)]
    pub rollout_batch_size: Option<usize>,
    /// Policy softmax 温度（可选覆盖）
    #[serde(default)]
    pub policy_delta: Option<f64>,
    /// 拉面杯搜索阶段（可选覆盖）
    #[serde(default)]
    pub ramen_search_stages: Option<String>,
    /// 是否启用 UCB 搜索分配（可选覆盖）
    #[serde(default)]
    pub use_ucb: Option<bool>,
    /// UCB 每组搜索次数（可选覆盖）
    #[serde(default)]
    pub search_group_size: Option<usize>,
    /// UCB 探索常数 cpuct（可选覆盖）
    #[serde(default)]
    pub search_cpuct: Option<f64>,
    /// UCB 探索项缩放标尺（可选覆盖）
    #[serde(default)]
    pub expected_search_stdev: Option<f64>,
    /// 是否按阶段重播种 rollout 随机流（可选覆盖）
    #[serde(default)]
    pub crn_stage_reseed: Option<bool>,
    /// 决策理由分差门限（可选覆盖；`0` 等价禁用理由输出）
    #[serde(default)]
    pub reason_gap_threshold: Option<f64>,
    /// 决策理由最多显示选项数（可选覆盖）
    #[serde(default)]
    pub reason_max_display: Option<usize>
}

/// 简化的覆盖配置 - GameConfig部分
///
/// 所有字段均为可选覆盖（`None` = 不覆盖 default 值），对应 game_config.toml
/// 「只写你要改的项」的语义；`deny_unknown_fields` 让拼错/未支持的字段显式报错，
/// 避免静默忽略。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverrideConfig {
    /// 马娘 ID（可选覆盖）
    #[serde(default)]
    pub uma: Option<u32>,
    /// 卡组（可选覆盖）
    #[serde(default)]
    pub cards: Option<[u32; 6]>,
    /// 种马蓝因子个数（可选覆盖）
    #[serde(default)]
    pub blue_count: Option<Array5>,
    /// 种马额外属性（可选覆盖）
    #[serde(default)]
    pub extra_count: Option<Array6>,
    /// 训练员类型（可选覆盖；`"manual" | "random" | "handwritten" | "collector" | "neuralnet" | "mcts"`）
    ///
    /// `None` = 不覆盖 `default_config.toml`（仍是 `handwritten`）。
    /// `trainer` 是 `GameConfig` 顶层字段、原本不在 `OverrideConfig` 里，
    /// 此处加进来让 `game_config.toml` 能顶层切换 trainer 跑特定场景（不污染 default）。
    #[serde(default)]
    pub trainer: Option<String>,
    /// 温泉选择是否使用蒙特卡洛（可选覆盖）
    #[serde(default)]
    pub mcts_selected_onsen: Option<bool>,
    /// 日志级别（可选覆盖）
    #[serde(default)]
    pub log_level: Option<String>,
    /// 线程数（可选覆盖）
    #[serde(default)]
    pub num_threads: Option<usize>,
    /// 蒙特卡洛每回合期望得分加成（可选覆盖；Phase 2 步骤 1 引入）
    #[serde(default)]
    pub mcts_turn_bonus: Option<i32>,
    /// PT 偏好评分倍率（可选覆盖；Phase 2 步骤 1 引入）
    #[serde(default)]
    pub pt_favor_rate: Option<f32>,
    /// 比赛等级表 72 项（可选覆盖；Phase 2 步骤 1 引入）
    #[serde(default)]
    pub race_grades: Option<Vec<i32>>,
    /// 在线决策记录开关（可选覆盖；默认开）
    #[serde(default)]
    pub luck_record: Option<bool>,
    /// 友人出行是否必须走完 5 次（可选覆盖；默认开）
    #[serde(default)]
    pub friend_complete_required: Option<bool>
}

impl OverrideGameConfig {
    pub fn merge(self, game: &GameConfig) -> GameConfig {
        let mut ret = game.clone();

        ret.onsen_order = self.onsen_order;
        let o = self.config_override;
        if let Some(v) = o.uma {
            ret.uma = v;
        }
        if let Some(v) = o.cards {
            ret.cards = v;
        }
        if let Some(v) = o.blue_count {
            ret.blue_count = v;
        }
        if let Some(v) = o.extra_count {
            ret.extra_count = v;
        }
        if let Some(v) = o.log_level {
            ret.log_level = v;
        }
        if let Some(v) = o.trainer {
            ret.trainer = v;
        }
        if let Some(v) = o.mcts_selected_onsen {
            ret.mcts_selected_onsen = v;
        }
        if let Some(v) = o.num_threads {
            ret.collector.threads = v;
        }
        if let Some(v) = o.mcts_turn_bonus {
            ret.mcts_turn_bonus = v;
        }
        if let Some(v) = o.pt_favor_rate {
            ret.pt_favor_rate = v;
        }
        if let Some(v) = o.race_grades {
            ret.race_grades = v;
        }
        if let Some(v) = o.friend_complete_required {
            ret.friend_complete_required = v;
        }
        if let Some(v) = o.luck_record {
            ret.luck_record = v;
        }
        let m = self.mcts;
        if let Some(v) = m.search_n {
            ret.mcts.search_n = v;
        }
        if let Some(v) = m.radical_factor_max {
            ret.mcts.radical_factor_max = v;
        }
        if let Some(v) = m.max_depth {
            ret.mcts.max_depth = v;
        }
        if let Some(v) = m.rollout_evaluator {
            ret.mcts.rollout_evaluator = v;
        }
        if let Some(v) = m.rollout_batch_size {
            ret.mcts.rollout_batch_size = v;
        }
        if let Some(v) = m.policy_delta {
            ret.mcts.policy_delta = v;
        }
        if let Some(v) = m.ramen_search_stages {
            ret.mcts.ramen_search_stages = v;
        }
        if let Some(v) = m.use_ucb {
            ret.mcts.use_ucb = v;
        }
        if let Some(v) = m.search_group_size {
            ret.mcts.search_group_size = v;
        }
        if let Some(v) = m.search_cpuct {
            ret.mcts.search_cpuct = v;
        }
        if let Some(v) = m.expected_search_stdev {
            ret.mcts.expected_search_stdev = v;
        }
        if let Some(v) = m.crn_stage_reseed {
            ret.mcts.crn_stage_reseed = v;
        }
        if let Some(v) = m.reason_gap_threshold {
            ret.mcts.reason_gap_threshold = v;
        }
        if let Some(v) = m.reason_max_display {
            ret.mcts.reason_max_display = v;
        }
        if let Some(v) = self.ramen_region_strategy {
            ret.ramen_region_strategy = v;
        }
        if let Some(v) = self.ramen_region_fixed {
            ret.ramen_region_fixed = Some(v);
        }
        if let Some(v) = self.ramen_trainer_policy {
            ret.ramen_trainer_policy = v;
        }
        if let Some(v) = self.ramen_nn_model_path {
            ret.ramen_nn_model_path = Some(v);
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use anyhow::ensure;
    use fs_err::read_to_string;

    use super::*;
    use crate::{
        gamedata::{onsen::ONSENDATA, ramen::RAMENDATA},
        global,
        trainer::RamenSearchStages,
        utils::Checks
    };

    /// 构造全 None 的覆盖配置（= 不覆盖任何字段）。
    fn empty_override() -> OverrideConfig {
        OverrideConfig {
            uma: None,
            cards: None,
            blue_count: None,
            extra_count: None,
            trainer: None,
            mcts_selected_onsen: None,
            log_level: None,
            num_threads: None,
            mcts_turn_bonus: None,
            pt_favor_rate: None,
            race_grades: None,
            luck_record: None,
            friend_complete_required: None
        }
    }

    fn wrap(cfg: OverrideConfig) -> OverrideGameConfig {
        OverrideGameConfig {
            onsen_order: OnsenOrder::default(),
            config_override: cfg,
            mcts: OverrideMctsConfig::default(),
            ramen_region_strategy: None,
            ramen_region_fixed: None,
            ramen_trainer_policy: None,
            ramen_nn_model_path: None
        }
    }

    /// 从 workspace 根目录加载真实的 `gamedata/default_config.toml`。
    fn load_real_default() -> Result<GameConfig> {
        let path = crate::utils::get_workspace_root()?.join("gamedata/default_config.toml");
        let text = read_to_string(&path)?;
        Ok(toml::from_str(&text)?)
    }

    /// 打印 12 个 MCTS 字段，便于验收对照。
    fn dump_mcts(label: &str, m: &MctsConfig) {
        println!(
            "{label}: search_n={} radical_factor_max={} max_depth={} rollout_evaluator={} rollout_batch_size={} policy_delta={} ramen_search_stages={} use_ucb={} search_group_size={} search_cpuct={} expected_search_stdev={} crn_stage_reseed={}",
            m.search_n,
            m.radical_factor_max,
            m.max_depth,
            m.rollout_evaluator,
            m.rollout_batch_size,
            m.policy_delta,
            m.ramen_search_stages,
            m.use_ucb,
            m.search_group_size,
            m.search_cpuct,
            m.expected_search_stdev,
            m.crn_stage_reseed
        );
    }

    /// 全 None 不覆盖：merge 后与 default 完全一致。
    #[test]
    fn test_override_merge_all_none_keeps_default() -> Result<()> {
        let base = GameConfig::default_for_init();
        let merged = wrap(empty_override()).merge(&base);
        println!(
            "全 None merge: uma={} cards={:?} blue_count={:?} extra_count={:?}",
            merged.uma, merged.cards, merged.blue_count, merged.extra_count
        );
        ensure!(merged.uma == base.uma, "uma 不应被覆盖");
        ensure!(merged.cards == base.cards, "cards 不应被覆盖");
        ensure!(merged.blue_count == base.blue_count, "blue_count 不应被覆盖");
        ensure!(merged.extra_count == base.extra_count, "extra_count 不应被覆盖");
        Ok(())
    }

    /// 部分覆盖：uma/cards/blue_count/extra_count 生效，其余字段保留 default。
    #[test]
    fn test_override_merge_partial_overrides() -> Result<()> {
        let base = GameConfig::default_for_init();
        let mut o = empty_override();
        o.uma = Some(100901);
        o.cards = Some([302424, 302894, 303044, 302924, 303024, 303054]);
        o.blue_count = Some([15, 0, 0, 0, 3]);
        o.extra_count = Some([10, 10, 20, 20, 20, 40]);
        let merged = wrap(o).merge(&base);
        println!(
            "部分覆盖 merge: uma={} cards={:?} blue_count={:?} extra_count={:?}",
            merged.uma, merged.cards, merged.blue_count, merged.extra_count
        );
        ensure!(merged.uma == 100901, "uma 应被覆盖为 100901");
        ensure!(merged.cards[0] == 302424, "cards 应被覆盖");
        ensure!(merged.blue_count == [15, 0, 0, 0, 3], "blue_count 应被覆盖");
        ensure!(merged.extra_count == [10, 10, 20, 20, 20, 40], "extra_count 应被覆盖");
        // 未覆盖字段保留 default
        ensure!(merged.scenario == base.scenario, "scenario 应保留 default");
        Ok(())
    }

    /// deny_unknown_fields：未知字段解析显式报错，不再静默忽略。
    #[test]
    fn test_override_config_denies_unknown_fields() -> Result<()> {
        let text = r#"
[config_override]
uma = 100901
bogus_field = 1

[onsen_order]
year1 = [1, 2, 3]
year2 = []
year3 = []
"#;
        let result: Result<OverrideGameConfig, _> = toml::from_str(text);
        println!("未知字段解析结果 = {}", result.is_err());
        ensure!(result.is_err(), "未知字段应报错而非静默忽略");
        Ok(())
    }

    /// 对照：同样的 toml 去掉 `bogus_field`（且无 `[mcts]`）必须解析成功。
    ///
    /// 锁住「报错原因是未知字段，不是缺 `[mcts]`」。
    #[test]
    fn test_override_config_parses_without_mcts_or_bogus() -> Result<()> {
        let text = r#"
[config_override]
uma = 100901

[onsen_order]
year1 = [1, 2, 3]
year2 = []
year3 = []
"#;
        let result: Result<OverrideGameConfig, _> = toml::from_str(text);
        println!("无 bogus / 无 [mcts] 解析 = {:?}", result.as_ref().map(|_| "ok"));
        let mut c = Checks::new();
        c.check(result.is_ok(), "去掉 bogus_field 后应解析成功（[mcts] 可省略）");
        c.finish()
    }

    /// 顶层覆盖：game_config.toml 顶层写 `ramen_region_strategy` / `ramen_region_fixed` 必须生效
    /// （回归：此前顶层字段被 serde 静默忽略，解除 game_config.toml 注释后无法切换第3年固定选区。
    /// 注意 TOML 结构：顶层字段必须写在所有 `[...]` 段**之前**，否则会被归入前一个段。
    /// 用户 game_config.toml 曾把这两个字段写在 `[mcts]` 段后 → 被 `[mcts]` 的 unknown field 吸收。）
    #[test]
    fn test_top_level_region_override_takes_effect() -> Result<()> {
        let base = GameConfig::default_for_init();
        // 模拟 game_config.toml 顶层（段之前）写区域策略
        let text = r#"
ramen_region_strategy = "fixed"
ramen_region_fixed = [[10, 12, 14]]

[config_override]
uma = 100901
"#;
        let mut c = Checks::new();
        match toml::from_str::<OverrideGameConfig>(text) {
            Ok(ov) => {
                println!(
                    "顶层 region 解析: strategy={:?} fixed={:?}",
                    ov.ramen_region_strategy, ov.ramen_region_fixed
                );
                let merged = ov.merge(&base);
                println!(
                    "merge 后: strategy={:?} fixed={:?}",
                    merged.ramen_region_strategy, merged.ramen_region_fixed
                );
                c.check(
                    matches!(
                        merged.ramen_region_strategy,
                        RamenRegionStrategy::Fixed
                    ),
                    "顶层 ramen_region_strategy=fixed 应覆盖进 GameConfig",
                );
                c.check(
                    merged.ramen_region_fixed.as_deref() == Some(&[[10, 12, 14]]),
                    "顶层 ramen_region_fixed 应覆盖进 GameConfig",
                );
            }
            Err(e) => {
                println!("顶层 region 解析失败: {e:?}");
                c.check(false, "顶层 region 字段应可解析");
            }
        }
        c.check(
            RamenRegionStrategy::default() == RamenRegionStrategy::All,
            "RamenRegionStrategy 缺省应为 All（对照）",
        );
        c.finish()
    }

    /// 在线决策记录开关：默认开 + `[config_override]` 可关（merge 生效）
    #[test]
    fn test_luck_record_default_and_override() -> Result<()> {
        let base = load_real_default()?;
        let mut c = Checks::new();
        c.check(base.luck_record, "default_config.toml 缺省应为开（true）");

        // 不写 luck_record → merge 保留默认开
        let ov_plain: OverrideGameConfig = toml::from_str(r#"[config_override]
uma = 101901"#)?;
        c.check(
            ov_plain.clone().merge(&base).luck_record,
            "未覆盖时 luck_record 保持默认开",
        );

        // [config_override] 写 luck_record = false → merge 后关
        let ov_off: OverrideGameConfig = toml::from_str(r#"[config_override]
luck_record = false"#)?;
        let merged_off = ov_off.merge(&base);
        println!("luck_record=false merge 后: {}", merged_off.luck_record);
        c.check(!merged_off.luck_record, "luck_record = false 应覆盖为关");
        c.finish()
    }

    /// 日常路径回归：真实 default_config.toml + 只覆盖 search_n / radical_factor_max。
    ///
    /// 若 merge 改成整段赋值，`search_group_size` 会变成 512、`expected_search_stdev`
    /// 变成 2200，本测试必须红。
    #[test]
    fn test_mcts_override_daily_path_keeps_production() -> Result<()> {
        let base = load_real_default()?;
        let text = r#"
[config_override]
uma = 101901

[onsen_order]
year1 = [1, 3, 2]
year2 = [7, 5, 4, 2]
year3 = [8, 9, 4, 2, 6]

[mcts]
search_n = 12288
radical_factor_max = 1.4
"#;
        let ov: OverrideGameConfig = toml::from_str(text)?;
        let merged = ov.merge(&base);
        dump_mcts("日常路径 merge 后", &merged.mcts);
        let mut c = Checks::new();
        c.check(merged.mcts.search_group_size == 512, "search_group_size == 512");
        c.check(
            merged.mcts.expected_search_stdev == 15000.0,
            "expected_search_stdev == 15000.0"
        );
        c.check(merged.mcts.rollout_batch_size == 64, "rollout_batch_size == 64");
        c.check(merged.mcts.search_n == 12288, "search_n == 12288");
        c.check(merged.mcts.radical_factor_max == 1.4, "radical_factor_max == 1.4");
        c.finish()
    }

    /// 整段省略 `[mcts]`：解析成功，12 个字段全部保留 default_config.toml。
    ///
    /// 若 `OverrideMctsConfig` 某字段 Default 改成 `Some(缺省值)`，本测试必须红。
    #[test]
    fn test_mcts_override_omitted_section_keeps_all_twelve() -> Result<()> {
        let base = load_real_default()?;
        let text = r#"
[config_override]
uma = 101901

[onsen_order]
year1 = [1, 3, 2]
year2 = [7, 5, 4, 2]
year3 = [8, 9, 4, 2, 6]
"#;
        let ov: OverrideGameConfig = toml::from_str(text)?;
        let mut c = Checks::new();
        // 不与 OverrideMctsConfig::default() 比较：那是同义反复，Default 被改坏也发现不了
        c.check(
            ov.mcts.search_n.is_none()
                && ov.mcts.radical_factor_max.is_none()
                && ov.mcts.max_depth.is_none()
                && ov.mcts.rollout_evaluator.is_none()
                && ov.mcts.rollout_batch_size.is_none()
                && ov.mcts.policy_delta.is_none()
                && ov.mcts.ramen_search_stages.is_none()
                && ov.mcts.use_ucb.is_none()
                && ov.mcts.search_group_size.is_none()
                && ov.mcts.search_cpuct.is_none()
                && ov.mcts.expected_search_stdev.is_none()
                && ov.mcts.crn_stage_reseed.is_none(),
            "省略 [mcts] 时 12 个覆盖字段全为 None"
        );
        let merged = ov.merge(&base);
        dump_mcts("省略 [mcts] merge 后", &merged.mcts);
        dump_mcts("default_config.toml 原文", &base.mcts);
        c.check(merged.mcts.search_n == base.mcts.search_n, "search_n 保留 toml");
        c.check(
            merged.mcts.radical_factor_max == base.mcts.radical_factor_max,
            "radical_factor_max 保留 toml"
        );
        c.check(merged.mcts.max_depth == base.mcts.max_depth, "max_depth 保留 toml");
        c.check(
            merged.mcts.rollout_evaluator == base.mcts.rollout_evaluator,
            "rollout_evaluator 保留 toml"
        );
        c.check(
            merged.mcts.rollout_batch_size == base.mcts.rollout_batch_size,
            "rollout_batch_size 保留 toml"
        );
        c.check(
            merged.mcts.policy_delta == base.mcts.policy_delta,
            "policy_delta 保留 toml"
        );
        c.check(
            merged.mcts.ramen_search_stages == base.mcts.ramen_search_stages,
            "ramen_search_stages 保留 toml"
        );
        c.check(merged.mcts.use_ucb == base.mcts.use_ucb, "use_ucb 保留 toml");
        c.check(
            merged.mcts.search_group_size == base.mcts.search_group_size,
            "search_group_size 保留 toml"
        );
        c.check(
            merged.mcts.search_cpuct == base.mcts.search_cpuct,
            "search_cpuct 保留 toml"
        );
        c.check(
            merged.mcts.expected_search_stdev == base.mcts.expected_search_stdev,
            "expected_search_stdev 保留 toml"
        );
        c.check(
            merged.mcts.crn_stage_reseed == base.mcts.crn_stage_reseed,
            "crn_stage_reseed 保留 toml"
        );
        c.finish()
    }

    /// `[mcts]` 里写未知键必须解析报错。
    #[test]
    fn test_mcts_override_denies_unknown_fields() -> Result<()> {
        let text = r#"
[config_override]
uma = 101901

[onsen_order]
year1 = [1]
year2 = []
year3 = []

[mcts]
bogus = 1
"#;
        let result: Result<OverrideGameConfig, _> = toml::from_str(text);
        println!("[mcts] 未知键解析结果 = {}", result.is_err());
        if let Err(e) = &result {
            println!("[mcts] 未知键错误: {e}");
        }
        let mut c = Checks::new();
        c.check(result.is_err(), "[mcts] 未知键应报错");
        c.finish()
    }

    /// 逐字段覆盖真的生效。
    #[test]
    fn test_mcts_override_partial_fields_apply() -> Result<()> {
        let base = load_real_default()?;
        let text = r#"
[config_override]
uma = 101901

[onsen_order]
year1 = [1]
year2 = []
year3 = []

[mcts]
search_group_size = 777
crn_stage_reseed = false
ramen_search_stages = "train,region"
"#;
        let ov: OverrideGameConfig = toml::from_str(text)?;
        let merged = ov.merge(&base);
        dump_mcts("逐字段覆盖 merge 后", &merged.mcts);
        let mut c = Checks::new();
        c.check(merged.mcts.search_group_size == 777, "search_group_size 覆盖为 777");
        c.check(!merged.mcts.crn_stage_reseed, "crn_stage_reseed 覆盖为 false");
        c.check(
            merged.mcts.ramen_search_stages == "train,region",
            "ramen_search_stages 覆盖为 train,region"
        );
        c.check(merged.mcts.search_n == base.mcts.search_n, "未写的 search_n 保留 toml");
        c.check(
            merged.mcts.expected_search_stdev == base.mcts.expected_search_stdev,
            "未写的 expected_search_stdev 保留 toml"
        );
        c.finish()
    }

    /// 生产缺省必须搜 `ramen`：42 局配对实测 `train,ramen` 比只搜 `train` 高 +2306 分
    /// （t=7.94），而同算力加到 `train` 的 `search_n` 上只值 +39 分。漏掉它分数上
    /// 看不出是配置的锅。serde 缺省与 `default_config.toml` 两个真值源都要钉。
    #[test]
    fn test_production_default_searches_ramen_stage() -> Result<()> {
        let base = load_real_default()?;
        let serde_default = MctsConfig::default();
        println!("toml       ramen_search_stages = {:?}", base.mcts.ramen_search_stages);
        println!("serde 缺省 ramen_search_stages = {:?}", serde_default.ramen_search_stages);

        // 解析成结构体再判，避免把 "train,ramen" 写死成字符串比较：
        // 上游哪天改成 "ramen,train" 或加空格，语义没变就不该红。
        let from_toml = RamenSearchStages::parse(&base.mcts.ramen_search_stages)?;
        let from_serde = RamenSearchStages::parse(&serde_default.ramen_search_stages)?;

        let mut c = Checks::new();
        c.check(from_toml.train, "toml 缺省搜 train");
        c.check(from_toml.ramen_select, "toml 缺省搜 ramen（值 +2306 分）");
        c.check(from_serde.train, "serde 缺省搜 train");
        c.check(from_serde.ramen_select, "serde 缺省搜 ramen（值 +2306 分）");
        c.finish()
    }

    /// [`GameConstants::status_final_score`] 越界必须**饱和**，不得 panic、不得返回 0
    ///
    /// 守缺陷 C 的回归：五维上限 = 剧本基值 + 继承三次，蓝因子拉满会超出评分表长度。
    /// 此前三处消费点行为不一致——裸下标 panic、`unwrap_or(0)` 静默返回 0。
    /// 后者最坏：`status_gain` 会算出约 -19000 的巨大负分，手写策略永久回避该维且不报错。
    /// 本测试专门锁「越界返回表末且非 0」，改回 `unwrap_or(0)` 式实现即红。
    #[test]
    fn test_status_final_score_saturates_out_of_range() -> anyhow::Result<()> {
        std::env::set_current_dir(crate::utils::get_workspace_root()?)?;
        let cons = GameConstants::load()?;
        let table = &cons.five_status_final_score;
        let len = table.len();
        let last = *table.last().ok_or_else(|| anyhow::anyhow!("评分表为空，constants.json 数据损坏"))?;
        println!("评分表长度={len} 表末={last}");

        let mut c = Checks::new();
        c.check(cons.status_final_score(-1) == table[0], "负值查表 = 表首");
        c.check(cons.status_final_score(0) == table[0], "0 查表 = 表首");
        c.check(cons.status_final_score(2800) == table[2800], "表内值逐位相同");
        c.check(cons.status_final_score(len as i32 - 1) == last, "最后一档 = 表末");

        // 越界：必须饱和到表末，而不是 0
        for over in [len as i32, len as i32 + 100, i32::MAX] {
            let got = cons.status_final_score(over);
            println!("越界 {over} -> {got}（期望 {last}）");
            c.check(got == last, &format!("越界 {over} 饱和到表末"));
            c.check(got != 0, &format!("越界 {over} 不得返回 0"));
        }
        c.finish()
    }

    /// 各剧本五维上限基值的**数据契约**：JSON 里的权威值不得被无意改动
    ///
    /// 与 `test_newgame_status_limit_is_scenario_base_plus_inherit` 分工：那条守
    /// **代码路径**（基值有没有被事后补丁擦掉），期望值从 JSON 推导，因此改数据不误报；
    /// 本条守**数据本身**。两条都在，才既能抓住「代码改坏」也能抓住「数据漂移」。
    ///
    /// 另一个只有本条能抓的回归：拉面 JSON 与 `constants.json` 当前数值恰好相同，
    /// 若有人把拉面误接成全局常量，路径测试仍会绿。
    #[test]
    fn test_scenario_status_limit_base_contract() -> anyhow::Result<()> {
        std::env::set_current_dir(crate::utils::get_workspace_root()?)?;
        let _ = crate::gamedata::init_global();
        let ramen = global!(RAMENDATA).status_limit_base();
        let onsen = global!(ONSENDATA).status_limit_base();
        println!("拉面基值={ramen:?} 温泉基值={onsen:?}");
        let mut c = Checks::new();
        c.check(ramen == [3100, 2400, 2200, 2200, 2400], "拉面杯五维上限基值");
        c.check(onsen == [3000, 2200, 2200, 2200, 2400], "温泉杯五维上限基值");
        c.check(ramen != onsen, "两剧本基值必须不同（否则剧本隔离形同虚设）");
        c.finish()
    }
}
