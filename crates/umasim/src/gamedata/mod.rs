use std::{
    collections::BTreeMap,
    fmt::Display,
    sync::{Mutex, OnceLock}
};

use anyhow::{Result, anyhow};
use flexi_logger::LoggerHandle;
use hashbrown::HashMap;
use log::info;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    explain::Explain,
    global,
    utils::{Array5, Array6}
};

pub mod onsen;

/// 自由比赛区间数据
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FreeRaceData {
    /// 开始回合(从0开始)
    pub start_turn: u32,
    // 结束回合
    pub end_turn: u32,
    /// 比赛次数
    pub count: u32,
    /// 比赛等级, 可选
    pub grade: Option<u32>,
    /// 比赛掩码，json里不存在，载入时计算
    #[serde(default)]
    pub mask: u64
}

impl FreeRaceData {
    /// 能打的比赛设为1，其他为0
    pub fn update_turn_mask(&mut self) {
        let mut ret = 0;
        let race_grades = &global!(GAMECONSTANTS).race_grades;
        for i in self.start_turn..=self.end_turn {
            if let Some(g) = &self.grade {
                if race_grades[i as usize] <= *g as i32 {
                    ret |= 1 << (i - 11);
                }
            } else {
                ret |= 1 << (i - 11);
            }
        }
        self.mask = ret;
    }
}

/// 马娘数据 UmaDB.json
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UmaData {
    /// 马娘ID
    pub game_id: u32,
    /// 星数
    pub star: u32,
    /// 名字
    pub name: String,
    /// 五维加成
    pub five_status_bonus: Array5,
    /// 初始五维
    pub five_status_initial: Array5,
    /// 比赛回合
    pub races: Vec<i32>,
    /// 自由比赛回合
    pub free_races: Vec<FreeRaceData>
}

impl UmaData {
    pub fn short_name(&self) -> &str {
        self.name.split("]").last().unwrap_or(&self.name)
    }

    /// 把比赛回合压缩进u64位段 对应11-71回合
    pub fn zip_races(&self) -> u64 {
        let mut ret = 0;
        for race in &self.races {
            ret |= 1 << (race - 11);
        }
        ret
    }
}
/// 支援卡数据 CardDB.json
/// 支援卡具体数值
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CardValue {
    /// 友情
    #[serde(default, rename = "youQing")]
    pub youqing: f32,
    /// 干劲
    #[serde(default, rename = "ganJing")]
    pub ganjing: i32,
    /// 训练
    #[serde(default, rename = "xunLian")]
    pub xunlian: i32,
    /// 赛后
    #[serde(default, rename = "saiHou")]
    pub saihou: i32,
    /// 得意率
    #[serde(default, rename = "deYiLv")]
    pub deyilv: f32,
    /// 初始羁绊
    #[serde(default, rename = "initialJiBan")]
    pub initial_jiban: i32,
    /// 启发等级
    #[serde(default)]
    pub hint_level: i32,
    /// 启发概率
    #[serde(default)]
    pub hint_prob_increase: i32,
    /// 智训练体力恢复
    #[serde(default)]
    pub wiz_vital_bonus: i32,
    /// 失败率下降
    #[serde(default)]
    pub fail_rate_drop: f32,
    /// 体力消耗降低
    #[serde(default)]
    pub vital_cost_drop: f32,
    /// 事件效果提高
    #[serde(default)]
    pub event_effect_up: i32,
    /// 事件回复量提高
    #[serde(default)]
    pub event_recovery_amount_up: i32,
    /// 副属性
    pub bonus: Array6,
    /// 初始属性
    pub initial_bonus: Array6,
    /// 启发收益
    pub hint_bonus: Array6
}

/// 支援卡数据 CardDB.json
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SupportCardData {
    /// 支援卡ID
    pub card_id: u32,
    /// 角色ID
    pub chara_id: u32,
    /// 卡名
    pub card_name: String,
    /// 全名
    pub full_name: String,
    /// 稀有度，123
    pub rarity: u32,
    /// 卡类型 0速1耐2力3根4智5团队6友人
    pub card_type: i32,
    /// 数值
    pub card_value: Vec<CardValue>,
    /// 固有类型
    #[serde(default)]
    pub unique_effect_type: u32,
    /// 固有描述
    pub unique_effect_summary: Option<String>,
    /// 固有数值
    #[serde(default)]
    pub unique_effect_param: Vec<i32>
}

impl SupportCardData {
    pub fn short_name(&self) -> String {
        let parts: Vec<_> = self.card_name.split(']').collect();
        let left = parts[0];
        let right_short: String = parts[1].chars().take(2).collect();
        format!("{left}]{right_short}")
    }
}

/// 训练或事件数值
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionValue {
    /// 基础属性
    #[serde(default)]
    pub status_pt: Array6,
    /// 体力
    #[serde(default)]
    pub vital: i32,
    /// 最大体力
    #[serde(default)]
    pub max_vital: i32,
    /// 干劲
    #[serde(default)]
    pub motivation: i32,
    /// Hint等级
    #[serde(default)]
    pub hint_level: i32,
    /// 羁绊
    #[serde(default)]
    pub friendship: i32
}

impl ActionValue {
    pub fn explain(&self) -> String {
        let mut s = Explain::status_with_pt(&self.status_pt);
        if self.vital != 0 {
            s += &format!(" 体力{}", self.vital);
        }
        if self.max_vital != 0 {
            s += &format!(" 最大体力+{}", self.max_vital);
        }
        if self.friendship != 0 {
            s += &format!(" 羁绊+{}", self.friendship);
        }
        if self.motivation != 0 {
            s += &format!(" 干劲{}", self.motivation);
        }
        if self.hint_level != 0 {
            s += &format!(" Hint+{}", self.hint_level);
        }
        s
    }

    pub fn map_status<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(i32) -> i32
    {
        for i in 0..6 {
            self.status_pt[i] = f(self.status_pt[i]);
        }
        self
    }
}

impl Display for ActionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.explain())
    }
}

/// 剧本事件信息，也用于临时生成一些固定事件如赛后
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EventData {
    /// ID, 必须不同, 游戏内ID为9位数，自定义的位数更少
    pub id: u32,
    /// 名字
    pub name: String,
    /// 对应第几张卡或者理事长记者，计算时随机指定，不在数据里
    #[serde(default)]
    pub person_index: Option<i32>,
    /// 可以触发的开始回合
    #[serde(default)]
    pub start_turn: i32,
    /// 结束回合, 不包含
    #[serde(default)]
    pub end_turn: i32,
    /// 概率, 100为必发
    pub prob: i32,
    /// 为Some时，选项完全随机并按概率分布; 为None时选项交给玩家选择
    #[serde(default)]
    pub random_choice_prob: Option<Vec<f32>>,
    /// 最大触发次数, 0为无限
    pub max_trigger_time: u32,
    /// 属性奖励(随机改为平均) 速耐力根智pt，体力
    #[serde(default)]
    pub choices: Vec<ActionValue>
}

impl EventData {
    /// 红点属性事件
    pub fn hint_attr_event(train: usize, person_index: usize) -> Result<Self> {
        if train < 5 {
            let train_name = global!(GAMECONSTANTS).train_names[train].clone();
            let value = ActionValue {
                status_pt: global!(GAMECONSTANTS).hint_event_value[train],
                friendship: 5,
                ..Default::default()
            };
            Ok(Self {
                id: 101,
                name: format!("Hint - {train_name}属性"),
                person_index: Some(person_index as i32),
                choices: vec![value],
                ..Default::default()
            })
        } else {
            Err(anyhow!("train越界: {train}"))
        }
    }

    /// 红点技能事件
    pub fn hint_skill_event(hint_level: i32, person_index: usize) -> Self {
        let value = ActionValue {
            status_pt: [0, 0, 0, 0, 0, 0],
            hint_level,
            friendship: 5,
            ..Default::default()
        };
        Self {
            id: 101,
            name: format!("Hint - 技能"),
            person_index: Some(person_index as i32),
            choices: vec![value],
            ..Default::default()
        }
    }

    /// 加练事件
    pub fn extra_training_event(train: usize) -> Self {
        let mut ret = global!(GAMEDATA).events.system_events["extra_train"].clone();
        ret.choices[0].status_pt[train] = 5;
        ret
    }
}

/// 事件数据表
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventCollection {
    /// 剧本必发事件
    pub story_events: Vec<EventData>,
    /// 马娘正面事件
    pub uma_events: Vec<EventData>,
    /// 支援卡连续事件
    pub card_events: Vec<EventData>,
    /// 友人事件
    pub friend_events: HashMap<String, EventData>,
    /// 系统事件
    pub system_events: HashMap<String, EventData>
}
#[derive(Clone, Debug)]
pub struct GameData {
    pub uma: BTreeMap<String, UmaData>,
    pub card: BTreeMap<String, SupportCardData>,
    pub text: BTreeMap<String, BTreeMap<String, String>>,
    pub events: EventCollection
}

pub fn load_json<T: DeserializeOwned>(path: &str) -> Result<T> {
    info!("载入数据 {path}");
    Ok(serde_json::from_str(&fs_err::read_to_string(path)?)?)
}

impl GameData {
    pub fn load() -> Result<Self> {
        let mut uma: BTreeMap<String, UmaData> = load_json("gamedata/umaDB.json")?;
        let card: BTreeMap<_, _> = load_json("gamedata/cardDB.json")?;
        let text = load_json("gamedata/text_data_dict.json")?;
        let events = load_json("gamedata/events.json")?;
        info!("载入 {} 马娘, {} 支援卡", uma.len(), card.len());
        // 处理free race mask
        for uma in uma.values_mut() {
            for f in uma.free_races.iter_mut() {
                f.update_turn_mask();
            }
        }
        Ok(Self { uma, card, text, events })
    }

    pub fn get_uma(&self, id: u32) -> Result<&UmaData> {
        self.uma
            .get(&id.to_string())
            .ok_or_else(|| anyhow!("未找到 id={id} 的马娘，需要更新数据"))
    }

    pub fn get_card(&self, id: u32) -> Result<&SupportCardData> {
        self.card
            .get(&id.to_string())
            .ok_or_else(|| anyhow!("未找到 id={id} 的支援卡，需要更新数据"))
    }

    pub fn get_chara_name(&self, chara_id: u32) -> &str {
        self.text["6"]
            .get(&chara_id.to_string())
            .map(|x| x.as_str())
            .unwrap_or("未知")
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GameConstants {
    /// 训练基础值 训练类型 等级 速耐力根智pt体力
    pub training_basic_value: Vec<Vec<Vec<i32>>>,
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
    /// 每回合的比赛等级
    pub race_grades: Vec<i32>,
    /// 休息结果分布 +30=18%,+50=57%,+70=25%
    pub rest_probs: Vec<i32>,
    /// 红点属性
    pub hint_event_value: Vec<Array6>,
    /// 每张卡最大提供Hint等级
    pub max_hint_per_card: i32,
    /// PT特化时，PT评分倍数
    pub pt_favor_rate: f32,
    /// PT特化时，超过1200的属性压缩系数
    pub five_status_favor_rate: Vec<f32>,
    /// 蒙特卡洛每回合比手写逻辑增加的分数, 用于修正估分
    pub mcts_turn_bonus: i32
}

impl GameConstants {
    pub fn load() -> Result<Self> {
        info!("载入游戏数据");
        load_json("gamedata/constants.json")
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
    pub fn get_event_distribution(&self) -> Vec<f64> {
        let probs = &self.event_probs;
        let mut ret = vec![probs["card_event"], probs["uma_event"], probs["drop_motivation"]];
        ret.push(1.0 - ret[0] - ret[1] - ret[2]);
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
    /// 经验值：32（与默认 search_group_size 对齐），后续可按模型/CPU 调整。
    #[serde(default = "default_mcts_rollout_batch_size")]
    pub rollout_batch_size: usize,
    /// Policy softmax 温度（分数每降低多少，概率变成 1/e 倍）
    #[serde(default = "default_mcts_policy_delta")]
    pub policy_delta: f64,

    // ========== UCB 搜索分配参数 ==========
    /// 是否启用 UCB 搜索分配
    #[serde(default = "default_mcts_use_ucb")]
    pub use_ucb: bool,
    /// UCB 每组搜索次数
    #[serde(default = "default_mcts_search_group_size")]
    pub search_group_size: usize,
    /// UCB 探索常数 (cpuct)
    #[serde(default = "default_mcts_search_cpuct")]
    pub search_cpuct: f64,
    /// 预期搜索标准差
    #[serde(default = "default_mcts_expected_search_stdev")]
    pub expected_search_stdev: f64
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
            use_ucb: default_mcts_use_ucb(),
            search_group_size: default_mcts_search_group_size(),
            search_cpuct: default_mcts_search_cpuct(),
            expected_search_stdev: default_mcts_expected_search_stdev()
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
    /// 剧本类型: "basic" | "onsen"
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
    pub onsen_order: Vec<u32>,
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
    pub mcts_selection: String
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use anyhow::Result;

    use super::*;
    use crate::utils::{init_logger, make_table};

    #[test]
    fn test_uma_data() -> Result<()> {
        let uma_data: HashMap<String, UmaData> = serde_json::from_str(&fs_err::read_to_string("gamedata/umaDB.json")?)?;
        let umas: Vec<_> = uma_data.values().take(10).collect();
        println!("{}", make_table(&umas)?);
        Ok(())
    }

    #[test]
    fn test_support_data() -> Result<()> {
        let support_data: HashMap<String, SupportCardData> =
            serde_json::from_str(&fs_err::read_to_string("gamedata/cardDB.json")?)?;
        let cards: Vec<_> = support_data.values().skip(300).take(10).collect();
        println!("{:#?}", cards);
        Ok(())
    }

    #[test]
    fn test_consts() -> Result<()> {
        init_logger("test", "info")?;
        let consts = GameConstants::load()?;
        println!("{:?}", consts);

        println!("{}", consts.get_rank_name(63399));
        Ok(())
    }

    #[test]
    fn test_turn_mask() -> Result<()> {
        GAMECONSTANTS.set(GameConstants::load()?).expect("global constants");
        init_logger("test", "info")?;
        let mut free_race = FreeRaceData {
            start_turn: 24,
            end_turn: 47,
            count: 1,
            grade: Some(1),
            mask: 0
        };
        free_race.update_turn_mask(); // 只有G1会被标1
        println!("{:b}", free_race.mask); // 10111010000111110100000000000000000000
        Ok(())
    }
}

pub static GAMEDATA: OnceLock<GameData> = OnceLock::new();
pub static GAMECONSTANTS: OnceLock<GameConstants> = OnceLock::new();
pub static LOGGER: OnceLock<Mutex<LoggerHandle>> = OnceLock::new();

pub fn init_global() -> Result<()> {
    GAMECONSTANTS.set(GameConstants::load()?).expect("global constants");
    GAMEDATA.set(GameData::load()?).expect("global gamedata");
    onsen::init_onsen_data()?;
    Ok(())
}
