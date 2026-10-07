//! 拉面杯剧本数据

use std::{collections::HashMap, sync::OnceLock};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{
    gamedata::{EventData, GAMECONSTANTS, TrainingBasicTable, load_json},
    global,
    utils::Array5
};

/// 拉面基础效果
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RamenBasicEffect {
    /// 训练加成
    pub xunlian: i32,
    /// 友情训练加成
    pub youqing: i32,
    /// 得意率（本剧本无此效果）
    pub deyilv: i32,
    /// 失败率下降
    pub fail_rate_drop: i32,
    /// 羁绊增加
    pub friendship: i32,
    /// 上段数值上限增加（「獲得上限アップ」：属性与 PT 同值生效）
    pub status_limit: i32,
    /// 仅第三年生效的特殊hint效果
    pub hint_special: bool
}

/// 地区拉面效果
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RegionEffect {
    /// 地区 ID
    pub id: usize,
    /// 地区名称
    pub name: String,
    /// 训练加成
    #[serde(default)]
    pub xunlian: i32,
    /// 友情训练加成
    #[serde(default)]
    pub youqing: i32,
    /// PT 加成
    #[serde(default)]
    pub pt_bonus: i32,
    /// 发动 Hint 数量
    #[serde(default)]
    pub hint_count: i32,
    /// 生效的训练位置
    #[serde(default)]
    pub at_trains: Vec<i32>
}

/// 超级拉面基础效果
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FinalsBaseEffect {
    /// 体力恢复
    pub vital: i32,
    /// 干劲提升
    pub motivation: i32,
    /// 赛后加成
    pub saihou: i32,
    /// 友情加成
    pub youqing: i32,
    /// hint等级
    pub hint_level: i32
}

/// 超级拉面额外效果（大成功时）
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FinalsExtraEffect {
    /// PT加成
    pub pt_bonus: i32,
    /// PT上限增加
    pub pt_limit: i32,
    /// 分身数量
    pub clone_count: i32
}

/// 超级拉面效果
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FinalsEffect {
    /// 基础效果
    pub base: FinalsBaseEffect,
    /// 额外效果（大成功时）
    pub extra: FinalsExtraEffect,
    /// 训练限制选项（三个选项对应的训练位置）
    pub training_limit_options: Vec<Vec<i32>>
}

/// RMJ成功/失败效果
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RmjEffect {
    /// 友情加成
    pub youqing: i32,
    /// 得意率加成
    pub deyilv: i32,
    /// hint出现率加成
    pub hint: i32
}

/// 剧本PT常驻加成
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PtEffect {
    /// 最低PT要求
    pub pt_min: i32,
    /// 训练加成
    pub xunlian: i32,
    /// 得意率加成
    pub deyilv: i32,
    /// hint出现率加成
    pub hint: i32
}

/// 拉面杯剧本数据
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RamenScenarioData {
    /// 剧本ID = 14
    pub scenario_id: i32,
    /// 该剧本的友人首次点击事件 ID（真实游戏 story_id）
    ///
    /// 用于把小黑板 `keyEvents` 里的真实事件 ID 归一到 umasim 的事件历史；
    /// 与模拟内部事件 ID（`RamenScenarioData::friend_events["first"].id`）不是一个概念。
    /// 缺字段时为 0，表示无法匹配。
    #[serde(default)]
    pub friend_first_event: i32,
    /// 链接角色ID
    pub link_chara_id: Vec<i32>,
    /// RMJ结算阈值（按年份）
    #[serde(default)]
    pub ramen_success_pt: Vec<i32>,
    /// 每次吃面基础PT增量（按年份）
    #[serde(default)]
    pub gain_pt_base: Vec<i32>,
    /// 随吃面次数叠加的PT修正值（按年份）
    #[serde(default)]
    pub gain_pt_delta: Vec<i32>,
    /// 诀窍槽基础值总和（无友人/旧友人/新友人）
    #[serde(default)]
    pub feeling_gauge_gain_base: Vec<i32>,
    /// 支援卡隐藏风味
    #[serde(default)]
    pub card_special_feeling: HashMap<String, i32>,
    /// 训练基础值表格
    pub training_basic_value: TrainingBasicTable,
    /// 拉面基础效果（按年份）
    pub ramen_basic_effect: Vec<RamenBasicEffect>,
    /// 超级拉面效果
    #[serde(default)]
    pub finals_effect: FinalsEffect,
    /// RMJ成功效果（按年份）
    #[serde(default)]
    pub ramen_success_effect: Vec<RmjEffect>,
    /// RMJ失败效果（按年份）
    #[serde(default)]
    pub ramen_fail_effect: Vec<RmjEffect>,
    /// 剧本PT常驻加成
    #[serde(default)]
    pub ramen_pt_effect: Vec<PtEffect>,
    /// 地区诀窍配方
    #[serde(default)]
    pub region_feeling: Vec<[i32; 3]>,
    /// 地区词条加成档位
    #[serde(default)]
    pub region_bonus: Vec<i32>,
    /// 地区拉面效果
    #[serde(default)]
    pub ramen_region_effect: Vec<RegionEffect>,
    /// 剧本事件
    #[serde(default)]
    pub scenario_events: Vec<EventData>,
    /// 友人事件
    #[serde(default)]
    pub friend_events: HashMap<String, EventData>,
    /// 拉面杯剧本的五维属性上限基值（不含继承）
    ///
    /// 每个剧本的上限基值都不同，由各自的 `scenario_*.json` 提供；
    /// `constants.json` 的同名字段只作 basic 剧本与缺字段时的兜底。
    /// 读取请走 [`RamenScenarioData::status_limit_base`]，不要直接用本字段。
    #[serde(default)]
    pub five_status_limit_base: Option<Array5>
}

impl RamenScenarioData {
    /// 从 JSON 文件加载拉面杯剧本数据
    pub fn load() -> Result<Self> {
        load_json("gamedata/scenario_ramen.json")
    }

    /// 解析拉面杯的五维上限基值：剧本 JSON 未提供时回退到全局默认值
    pub fn status_limit_base(&self) -> Array5 {
        self.five_status_limit_base
            .unwrap_or_else(|| global!(GAMECONSTANTS).five_status_limit_base)
    }
}

/// 全局拉面杯剧本数据
pub static RAMENDATA: OnceLock<RamenScenarioData> = OnceLock::new();

/// 初始化拉面杯剧本数据
pub fn init_ramen_data() -> Result<()> {
    // 幂等：已初始化过则直接返回
    if RAMENDATA.get().is_some() {
        return Ok(());
    }
    let _ = RAMENDATA.set(RamenScenarioData::load()?);
    Ok(())
}
