pub mod base;
pub mod inherit;
pub mod onsen;
pub mod simulator;
pub mod support_card;
pub mod traits;
pub mod uma;

use std::{default::Default, fmt::Display};

use anyhow::Result;
pub use base::*;
use enum_iterator::Sequence;
pub use inherit::*;
use serde::{Deserialize, Serialize};
pub use support_card::*;
pub use traits::*;
pub use uma::*;

use crate::{gamedata::GAMEDATA, global};

/// 回合阶段
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, Sequence)]
pub enum TurnStage {
    /// 1. 回合开始
    #[default]
    Begin,
    /// 2. 分配人头前，随机事件
    Distribute,
    // --- 可操作部分
    /// 3. 选择剧本buff前
    StoryShop,
    /// 4. 选择训练或比赛
    Train,
    /// 5. 回合后事件
    AfterTrain,
    // ---
    /// 6. 结束，剧本结算
    End
}

/// 训练人头类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PersonType {
    /// 携带的支援卡
    #[default]
    Card,
    /// 剧本友人
    ScenarioCard,
    /// NPC
    Npc,
    /// 理事长
    Yayoi,
    /// 记者
    Reporter,
    /// 其他友人
    OtherFriend,
    /// 团队卡
    TeamCard
}

/// 回合阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FriendCardState {
    /// 没带
    #[default]
    Empty,
    /// 3星友人
    SSR,
    /// 1星友人
    R
}

impl Display for FriendCardState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "没带"),
            Self::SSR => write!(f, "SSR"),
            Self::R => write!(f, "R")
        }
    }
}

/// 友人出行阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FriendOutState {
    /// 未点击
    #[default]
    UnClicked,
    /// 已点击，未出行
    BeforeUnlock,
    /// 已出行
    AfterUnlock,
    /// 离开
    Away
}

impl FriendOutState {
    pub fn explain(&self) -> String {
        match self {
            Self::UnClicked => "未点击".to_string(),
            Self::BeforeUnlock => "未出行".to_string(),
            Self::AfterUnlock => "已出行".to_string(),
            Self::Away => "离开".to_string()
        }
    }
}

/// 友人出行信息，剧本通用
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FriendState {
    /// 剧本友人携带状态
    pub card_state: FriendCardState,
    /// 友人出行阶段
    pub out_state: FriendOutState,
    /// 友人的出行用了哪几段
    pub out_used: Vec<bool>,
    /// 团队卡Buff已经持续几回合，0为未开启
    pub group_buff_turn: u32,
    /// 友人在Persons中是哪一个
    pub person_index: usize,
    /// 友人事件体力回复量加成
    pub vital_bonus: i32,
    /// 友人事件效果加成
    pub event_bonus: i32
}

impl FriendState {
    pub fn new(idrank: Option<u32>, index: usize) -> Result<Self> {
        let data = global!(GAMEDATA);
        let mut ret = FriendState::default();
        ret.out_used = vec![false, false, false, false, false];
        if let Some(i) = idrank {
            let (id, rank) = (i / 10, i % 10);
            SupportCard::ensure_valid_rank(rank)?;
            let card = data.get_card(id)?;
            ret.card_state = match card.rarity {
                3 => FriendCardState::SSR,
                _ => FriendCardState::R
            };
            ret.vital_bonus = card.card_value[rank as usize].event_recovery_amount_up;
            ret.event_bonus = card.card_value[rank as usize].event_effect_up;
        }
        ret.person_index = index;
        Ok(ret)
    }

    pub fn explain(&self) -> String {
        let mut ret = self.out_state.explain();
        for (i, b) in self.out_used.iter().enumerate() {
            if *b {
                ret += &(i + 1).to_string();
            }
        }
        if self.group_buff_turn > 0 {
            ret += &format!(" 团队Buff已持续 {} 回合", self.group_buff_turn);
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gamedata::init_global, init_logger};

    #[test]
    fn test_friend() -> Result<()> {
        init_logger("debug")?;
        init_global()?;
        let friend = FriendState::new(Some(302574), 3)?;
        println!("{friend:#?} {}", friend.explain());
        Ok(())
    }
}
