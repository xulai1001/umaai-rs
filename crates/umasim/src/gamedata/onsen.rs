use std::{collections::BTreeMap, fmt::Display, sync::OnceLock};

use anyhow::Result;
use hashbrown::HashMap;
use log::info;
use serde::{Deserialize, Serialize};

use crate::{
    game::CardTrainingEffect,
    gamedata::{ActionValue, EventData, load_json},
    global,
    utils::{Array5, AttributeArray}
};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OnsenScenarioData {
    /// 剧本ID = 12
    pub scenario_id: i32,
    /// 链接角色ID
    pub link_chara_id: Vec<i32>,
    /// 链接角色对应哪种挖掘加成
    pub link_effect: HashMap<String, i32>,
    /// pr训练基础值
    pub pr_base_value: ActionValue,
    /// 温泉信息
    pub onsen_info: Vec<OnsenInfo>,
    /// 超回复效果
    pub super_effect: OnsenEffect,
    /// 旅馆效果
    pub hotel_effect: Vec<HotelEffect>,
    /// 挖掘工具等级对应挖掘加成
    pub dig_tool_level: Vec<i32>,
    /// 蓝因子对应挖掘加成
    pub dig_blue_bonus: Vec<Vec<i32>>,
    /// 属性对应挖掘加成的阈值
    pub dig_stat_ranks: Vec<i32>,
    /// 属性对应挖掘加成 [ type, rank ]
    pub dig_stat_bonus: Vec<Vec<i32>>,
    /// 不同挖掘时，分别是哪个属性对应挖掘加成
    pub dig_stat_bonus_types: Vec<Vec<i32>>,
    /// 不同挖掘时，每回合的固定属性加成(不含3体力降低)
    pub dig_fixed_stat: Vec<Array5>,
    /// 挖掘奖励: [速度, 耐力, 力量, 根性, 智力, 体力变化]
    /// 砂层: [2, 1, 0, 0, 2, -3]
    /// 土层: [2, 0, 1, 2, 0, -3]
    /// 岩层: [0, 1, 2, 0, 2, -3]
    pub dig_bonus: Vec<[i32; 6]>,
    /// 超回复触发概率
    pub super_probs: Vec<i32>,
    /// 剧本事件
    pub scenario_events: Vec<EventData>
}

impl OnsenScenarioData {
    pub fn load() -> Result<Self> {
        load_json("gamedata/scenario_onsen.json")
    }
}

/// 温泉信息
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OnsenInfo {
    pub id: i32,
    pub name: String,
    /// 解锁回合
    pub unlock_turn: i32,
    /// 挖掘量
    pub dig_volume: Vec<i32>,
    /// 效果
    pub effect: OnsenEffect
}

/// 温泉效果词条
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OnsenEffect {
    /// 体力
    #[serde(default)]
    pub vital: i32,
    /// 羁绊
    #[serde(default)]
    pub friendship: i32,
    /// 增加羁绊的人数
    #[serde(default)]
    pub friendship_count: i32,
    /// 干劲
    #[serde(default)]
    pub motivation: i32,
    /// 友情加成
    #[serde(default)]
    pub youqing: Array5,
    /// 训练
    #[serde(default)]
    pub xunlian: i32,
    /// 生涯比赛加成
    #[serde(default)]
    pub career_race_bonus: i32,
    /// Hint加成
    #[serde(default)]
    pub hint_bonus: i32,
    /// 体力减少
    #[serde(default)]
    pub vital_cost_drop: i32,
    /// 失败率下降
    #[serde(default)]
    pub fail_rate_drop: f32,
    /// 分身
    #[serde(default)]
    pub split: i32,
    /// 技能点
    #[serde(default)]
    pub pt: i32,
    /// Hint数量
    #[serde(default)]
    pub hint: i32,
    /// 临时体力
    #[serde(default)]
    pub temp_max_vital: i32
}

impl OnsenEffect {
    pub fn add_eq(&mut self, other: &Self) -> &mut Self {
        self.vital += other.vital;
        self.friendship += other.friendship;
        self.friendship_count += other.friendship_count;
        self.motivation += other.motivation;
        self.youqing.add_eq(&other.youqing);
        self.xunlian += other.xunlian;
        self.career_race_bonus += other.career_race_bonus;
        self.hint_bonus += other.hint_bonus;
        self.vital_cost_drop += other.vital_cost_drop;
        self.split += other.split;
        self.pt += other.pt;
        self.hint += other.hint;
        self.temp_max_vital += other.temp_max_vital;
        self.fail_rate_drop += other.fail_rate_drop;
        self
    }

    pub fn to_training_effect(&self, train: usize) -> CardTrainingEffect {
        CardTrainingEffect {
            youqing: self.youqing[train] as f32,
            xunlian: self.xunlian,
            vital_cost_drop: self.vital_cost_drop as f32,
            fail_rate_drop: self.fail_rate_drop,
            ..Default::default()
        }
    }
}

/// 旅馆效果词条
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HotelEffect {
    /// 几个温泉时发动
    pub onsen_count: i32,
    /// 得意率
    pub deyilv: i32,
    /// Hint数量
    pub hint: i32,
    /// 必定超回复
    #[serde(default)]
    pub must_super: bool
}

pub static ONSENDATA: OnceLock<OnsenScenarioData> = OnceLock::new();

pub fn init_onsen_data() -> Result<()> {
    ONSENDATA.set(OnsenScenarioData::load()?).expect("onsen data");
    Ok(())
}
