use std::sync::Arc;

use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use umasim::{
    game::{BaseGame, FriendOutState, FriendState, InheritInfo, SupportCard, TurnStage, Uma, UmaFlags},
    gamedata::{GAMEDATA, GameConfig},
    global,
    utils::Array5
};

pub mod onsen;
pub mod urafile;
pub use onsen::*;

/// 描述不同剧本的通信状态，需要能转为对应的Game结构
pub trait GameStatus: DeserializeOwned {
    type Game;

    fn scenario_id() -> u32;

    fn into_game(self) -> Result<Self::Game>;
}

/// 从小黑板接收的基础人头信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BasePersonStatus {
    /// 人头类型
    pub person_type: u32,
    /// 角色ID
    pub chara_id: u32,
    /// 羁绊
    pub friendship: i32,
    /// 是否叹号
    pub is_hint: bool
}

/// 从小黑板接收的数据的baseGame字段
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusBase {
    /// 剧本ID
    pub scenario_id: u32,
    /// 马娘ID
    pub uma_id: u32,
    /// 马娘星数
    pub uma_star: u32,
    /// 回合(0-77)
    pub turn: i32,
    /// 体力
    pub vital: i32,
    /// 最大体力
    pub max_vital: i32,
    /// 干劲 [1, 5]
    pub motivation: i32,
    /// 当前属性。1200以上不减半
    pub five_status: Array5,
    /// 属性上限
    pub five_status_limit: Array5,
    /// 技能点
    pub skill_pt: i32,
    /// 已学习技能分数
    pub skill_score: i32,
    /// 总Hint等级
    pub total_hints: i32,
    /// 训练设施等级
    pub train_level_count: Array5,
    /// PT系数
    pub pt_score_rate: f32,
    /// 失败率修正值
    pub failure_rate_bias: i32,
    /// 是否生病
    pub is_ill: bool,
    /// 是否切者
    #[serde(rename = "isQieZhe")]
    pub is_qiezhe: bool,
    /// 是否爱娇
    #[serde(rename = "isAiJiao")]
    pub is_aijiao: bool,
    /// 是否正向思考
    pub is_positive_thinking: bool,
    /// 是否有休息心得
    pub is_refresh_mind: bool,
    /// 是否有幸运体质
    pub is_lucky: bool,
    /// 种马蓝因子数量
    #[serde(rename = "zhongMaBlueCount")]
    pub zhongma_blue_count: Array5,
    /// 是否生涯比赛状态
    pub is_racing: bool,
    /// 卡组
    pub card_id: Vec<u32>,
    /// 人头
    pub persons: Vec<BasePersonStatus>,
    /// 人头分布
    pub person_distribution: Vec<Vec<i32>>,
    /// 是否锁定到某个训练
    pub locked_training_id: i32,
    /// 理事长羁绊
    #[serde(rename = "friendship_noncard_yayoi")]
    pub friendship_noncard_yayoi: i32,
    /// 记者羁绊
    #[serde(rename = "friendship_noncard_reporter")]
    pub friendship_noncard_reporter: i32,
    /// 友人解锁阶段
    #[serde(rename = "friend_stage")]
    pub friend_stage: i32,
    /// 友人出行阶段
    #[serde(rename = "friend_outgoingUsed")]
    pub friend_outgoing_used: i32,
    /// 回合状态
    #[serde(rename = "playing_state")]
    pub playing_state: i32,
    /// 胜场信息
    pub race_history: Vec<i32>
}

impl GameStatusBase {
    pub fn parse_uma(&self) -> Result<Uma> {
        let data = global!(GAMEDATA).get_uma(self.uma_id)?;
        let flags = UmaFlags {
            ill: self.is_ill,
            lucky: self.is_lucky,
            qiezhe: self.is_qiezhe,
            aijiao: self.is_aijiao,
            good_trainer: self.failure_rate_bias > 0,
            bad_trainer: self.failure_rate_bias < 0,
            positive_thinking: self.is_positive_thinking,
            refresh_mind: self.is_refresh_mind as i32,
            ..Default::default()
        };

        let mut ret = Uma {
            uma_id: self.uma_id,
            vital: self.vital,
            max_vital: self.max_vital,
            motivation: self.motivation,
            five_status: self.five_status.clone(),
            five_status_bonus: data.five_status_bonus.clone(),
            five_status_limit: self.five_status_limit,
            skill_pt: self.skill_pt,
            skill_score: self.skill_score,
            total_hints: self.total_hints,
            race_bonus: 0,
            flags,
            career_races: data.zip_races(),
            win_races: 0
        };
        // 设置比赛状态
        for t in &self.race_history {
            ret.set_race(*t);
        }
        //if ret.win_races != 0 {
        //    info!("win_races: {:b}", ret.win_races);
       // }
        Ok(ret)
    }

    pub fn parse_friend(&self, scenario_friend_chara_id: u32) -> Result<FriendState> {
        for (index, id) in self.card_id.iter().enumerate() {
            let card = SupportCard::new(*id)?;
            if card.card_type >= 5 && card.data.chara_id == scenario_friend_chara_id {
                let friend_id = Some(*id);
                let friend_index = index;
                let mut friend = FriendState::new(friend_id, friend_index)?;
                friend.out_state = match self.friend_stage {
                    0 => FriendOutState::UnClicked,
                    1 => FriendOutState::BeforeUnlock,
                    2 => FriendOutState::AfterUnlock,
                    _ => FriendOutState::Away
                };
                for i in 0..self.friend_outgoing_used {
                    friend.out_used[i as usize] = true;
                }
                return Ok(friend);
            }
        }
        // 如果没找到友人
        warn!("没带剧本友人? AI可能无法正常工作。请检查卡组");
        Ok(FriendState::default())
    }

    pub fn parse_inherit(&self) -> Result<InheritInfo> {
        // 1. 先读取配置文件
        let config_file = fs_err::read_to_string("game_config.toml")?;
        let game_config: GameConfig = toml::from_str(&config_file)?;
        Ok(InheritInfo {
            blue_count: self.zhongma_blue_count.clone(),
            extra_count: game_config.extra_count.clone()
        })
    }

    pub fn parse_basegame(&self, scenario_friend_chara_id: u32) -> Result<BaseGame> {
        let inherit = self.parse_inherit()?;
        let mut uma = self.parse_uma()?;
        // 检查是否有赛程信息
        if self.turn > 12 && self.race_history.is_empty() {
            warn!("未接收到胜场信息，自选比赛计算可能出错；需要更新小黑板插件");
        }
        let friend = self.parse_friend(scenario_friend_chara_id)?;
        let mut deck = vec![];
        let mut card_type_count = [0; 7];
        for (index, id) in self.card_id.iter().enumerate() {
            let mut card = SupportCard::new(*id)?;
            card.friendship = self.persons[index].friendship;
            uma.race_bonus += card.effect.saihou;
            if card.card_type < 7 {
                card_type_count[card.card_type as usize] += 1;
            }
            deck.push(card);
        }
        Ok(BaseGame {
            turn: self.turn,
            stage: TurnStage::Train, // 随便列一个
            uma,
            deck,
            inherit: Arc::new(inherit),
            friend,
            train_level_count: self.train_level_count.clone(),
            distribution: self.person_distribution.clone(),
            card_type_count: Arc::new(card_type_count),
            ..Default::default()
        })
    }
}
