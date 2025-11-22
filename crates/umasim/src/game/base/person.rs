use std::default::Default;

use serde::{Deserialize, Serialize};

use crate::{game::*, gamedata::GAMEDATA, global};

/// 训练人头信息（动态）
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BasePerson {
    /// 人头顺序 0-5为支援卡 >6理事长 NPC和记者
    pub person_index: i32,
    /// 人头类型
    pub person_type: PersonType,
    /// 得意训练类型，0-5:速耐力根智团 一部分npc也有 -1为没有
    pub train_type: i32,
    /// 角色ID
    pub chara_id: u32,
    /// 羁绊
    pub friendship: i32,
    /// 是否有叹号
    pub is_hint: bool,
    /// 支援卡信息
    pub card_id: Option<u32>
}

impl BasePerson {
    pub fn short_name(&self) -> String {
        let gamedata = global!(GAMEDATA);
        match self.person_type {
            PersonType::Npc => {
                let short_chara_name: String = gamedata.get_chara_name(self.chara_id).chars().take(2).collect();
                format!("[NPC]{short_chara_name}")
            }
            PersonType::Yayoi => "理事长".to_string(),
            PersonType::Reporter => "记者".to_string(),
            _ => {
                if let Some(Ok(support)) = self.card_id.map(|id| gamedata.get_card(id)) {
                    support.short_name()
                } else {
                    let short_chara_name: String = gamedata.get_chara_name(self.chara_id).chars().take(2).collect();
                    format!("[???]{short_chara_name}")
                }
            }
        }
    }

    pub fn explain(&self) -> String {
        let mut ret = self.short_name();
        if self.friendship > 0 && self.friendship < 100 {
            ret = format!("{}{}", ret, self.friendship);
        }
        if self.is_hint {
            ret = format!("{}{ret}", "!");
        }
        ret
    }

    pub fn yayoi() -> Self {
        BasePerson {
            person_index: 6,
            person_type: PersonType::Yayoi,
            train_type: -1,
            chara_id: 9002,
            friendship: 0,
            is_hint: false,
            card_id: None
        }
    }

    pub fn reporter() -> Self {
        BasePerson {
            person_index: 7,
            person_type: PersonType::Reporter,
            train_type: -1,
            chara_id: 9003,
            friendship: 0,
            is_hint: false,
            card_id: None
        }
    }
}

impl Person for BasePerson {
    fn person_type(&self) -> PersonType {
        self.person_type
    }
    fn person_index(&self) -> i32 {
        self.person_index
    }
    fn train_type(&self) -> i32 {
        self.train_type
    }
    fn friendship(&self) -> i32 {
        self.friendship
    }
    fn set_hint(&mut self, hint: bool) {
        self.is_hint = hint;
    }
    fn hint(&self) -> bool {
        self.is_hint
    }
}

impl TryFrom<&SupportCard> for BasePerson {
    type Error = anyhow::Error;
    fn try_from(card: &SupportCard) -> Result<Self> {
        let person_type = match card.card_type {
            0..=4 => PersonType::Card,
            5 => PersonType::ScenarioCard,
            6 => PersonType::TeamCard,
            _ => PersonType::Card
        };
        Ok(BasePerson {
            person_index: 0,
            person_type,
            train_type: card.card_type,
            chara_id: card.get_data()?.chara_id,
            friendship: card.friendship,
            is_hint: false,
            card_id: Some(card.card_id)
        })
    }
}
