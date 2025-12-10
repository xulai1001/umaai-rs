use std::{collections::BTreeMap, fmt::Display, sync::OnceLock};

use anyhow::{Result, anyhow};
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
    pub count: u32
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
        let uma: BTreeMap<_, _> = load_json("gamedata/umaDB.json")?;
        let card: BTreeMap<_, _> = load_json("gamedata/cardDB.json")?;
        let text = load_json("gamedata/text_data_dict.json")?;
        let events = load_json("gamedata/events.json")?;
        info!("载入 {} 马娘, {} 支援卡", uma.len(), card.len());
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
    pub max_hint_per_card: i32
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
    pub expected_search_stdev: f64,
    /// 是否启用激进度随回合调整
    #[serde(default = "default_mcts_adjust_radical_by_turn")]
    pub adjust_radical_by_turn: bool,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            search_n: default_mcts_search_n(),
            radical_factor_max: default_mcts_radical_factor_max(),
            max_depth: default_mcts_max_depth(),
            policy_delta: default_mcts_policy_delta(),
            use_ucb: default_mcts_use_ucb(),
            search_group_size: default_mcts_search_group_size(),
            search_cpuct: default_mcts_search_cpuct(),
            expected_search_stdev: default_mcts_expected_search_stdev(),
            adjust_radical_by_turn: default_mcts_adjust_radical_by_turn(),
        }
    }
}

fn default_mcts_search_n() -> usize {
    1024 // 默认搜索次数
}

fn default_mcts_radical_factor_max() -> f64 {
    50.0 // 默认激进度最大值
}

fn default_mcts_max_depth() -> usize {
    0  // 搜到游戏结束
}

fn default_mcts_policy_delta() -> f64 {
    100.0 
}

fn default_mcts_use_ucb() -> bool {
    true // 默认使用UCB分配
}

fn default_mcts_search_group_size() -> usize {
    128
}

fn default_mcts_search_cpuct() -> f64 {
    1.0
}

fn default_mcts_expected_search_stdev() -> f64 {
    2200.0 
}

fn default_mcts_adjust_radical_by_turn() -> bool {
    true  // 默认启用激进度调整
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
    /// MCTS 配置（可选）
    #[serde(default)]
    pub mcts: MctsConfig,
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
        init_logger("debug")?;
        let consts = GameConstants::load()?;
        println!("{:?}", consts);

        println!("{}", consts.get_rank_name(63399));
        Ok(())
    }
}

pub static GAMEDATA: OnceLock<GameData> = OnceLock::new();
pub static GAMECONSTANTS: OnceLock<GameConstants> = OnceLock::new();

pub fn init_global() -> Result<()> {
    GAMEDATA.set(GameData::load()?).expect("global gamedata");
    GAMECONSTANTS.set(GameConstants::load()?).expect("global constants");
    onsen::init_onsen_data()?;
    Ok(())
}
