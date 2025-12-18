use std::ops::Deref;

use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use umasim::{
    game::{
        BasePerson,
        PersonType,
        onsen::{BathingInfo, OnsenBuff, OnsenTurnStage, game::OnsenGame}
    },
    gamedata::onsen::ONSENDATA,
    global
};

use crate::protocol::{GameStatus, GameStatusBase};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BathingStatus {
    pub ticket_num: i32,
    pub buff_remain_turn: i32,
    pub is_super_ready: bool
}

impl From<&BathingStatus> for BathingInfo {
    fn from(status: &BathingStatus) -> Self {
        BathingInfo {
            ticket_num: status.ticket_num,
            buff_remain_turn: status.buff_remain_turn,
            is_super: false,
            is_super_ready: status.is_super_ready
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OnsenStatus {
    /// 当前挖掘的温泉
    pub current_onsen: usize,
    /// 温泉Buff信息
    pub bathing: BathingStatus,
    /// 温泉状态
    pub onsen_state: Vec<bool>,
    /// 当前每个温泉的剩余挖掘量
    pub dig_remain: Vec<[i32; 3]>,
    /// 已挖掘的温泉数
    pub dig_count: i32,
    /// 挖掘力加成
    pub dig_power: [i32; 3],
    /// 挖掘工具等级
    pub dig_level: [i32; 3],
    /// 挖掘花费的体力
    pub dig_vital_cost: i32,
    /// 是否需要选择温泉
    pub pending_selection: bool
}

/// 从小黑板接收的数据
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusOnsen {
    pub base_game: GameStatusBase,
    pub onsen: OnsenStatus
}

impl Deref for GameStatusOnsen {
    type Target = GameStatusBase;
    fn deref(&self) -> &Self::Target {
        &self.base_game
    }
}

impl GameStatus for GameStatusOnsen {
    type Game = OnsenGame;

    fn scenario_id() -> u32 {
        12
    }

    fn into_game(self) -> Result<Self::Game> {
        let mut base = self.parse_basegame(9050)?;
        // 默认状态为选择温泉券前
        let mut stage = OnsenTurnStage::Bathing;
        let mut pending = false;
        if base.turn < 2 || self.onsen.bathing.buff_remain_turn > 0 || self.onsen.bathing.ticket_num == 0 {
            // 不能用温泉券时，不进入Bathing状态
            stage = OnsenTurnStage::Train;
        }
        if self.playing_state == 36 {
            // 温泉选择状态
            stage = OnsenTurnStage::Begin;
            pending = true;
            // 调整回合状态到新回合开始(仅开局不用调)
            if base.turn != 2 {
                base.turn += 1;
                info!("调整为回合: {}", base.turn+1);
            }
        } else if self.playing_state != 1 {
            warn!("未知回合状态: {}", self.playing_state);
        }
        // 初始化人头
        let mut persons = vec![];
        for (i, card) in base.deck.iter().enumerate() {
            let mut person = BasePerson::try_from(card)?;
            // 对默认转换再过滤一下非剧本友人
            if person.person_type == PersonType::ScenarioCard {
                if person.chara_id != 9050 {
                    person.person_type = PersonType::OtherFriend;
                }
            }
            person.friendship = self.persons[i].friendship;
            person.is_hint = self.persons[i].is_hint;
            persons.push(person);
        }
        // 添加理事长,记者(记者在12回合才会出现)
        let mut yayoi = BasePerson::yayoi();
        let mut reporter = BasePerson::reporter();
        yayoi.friendship = self.friendship_noncard_yayoi;
        reporter.friendship = self.friendship_noncard_reporter;
        persons.push(yayoi);
        persons.push(reporter);

        // 温泉信息
        let dig_blue_count = base
            .inherit
            .blue_count
            .iter()
            .map(|x| (*x as f32 / 3.0).ceil() as i32)
            .collect::<Vec<_>>();
        // 计算挖掘进度
        let mut dig_progress = vec![];
        for i in 0..self.onsen.dig_remain.len() {
            let mut progress = global!(ONSENDATA).onsen_info[i].dig_volume.clone();
            for j in 0..3 {
                progress[j] -= self.onsen.dig_remain[i][j];
            }
            dig_progress.push(progress.try_into().expect("Array3"));
        }
        // 携带5种卡以上才能分身
        let deck_can_split = base.card_type_count.iter().filter(|x| **x > 0).count() >= 5;
        let mut ret = OnsenGame {
            base,
            stage,
            persons,
            current_onsen: self.onsen.current_onsen,
            bathing: BathingInfo::from(&self.onsen.bathing),
            onsen_state: self.onsen.onsen_state,
            dig_remain: self.onsen.dig_remain,
            dig_count: self.onsen.dig_count + 1,    // 包含默认泉
            dig_power: self.onsen.dig_power,
            dig_level: self.onsen.dig_level,
            dig_vital_cost: self.onsen.dig_vital_cost,
            dig_blue_count: dig_blue_count.try_into().expect("Array5"),
            dig_progress,
            scenario_buff: OnsenBuff::default(),
            pending_selection: pending,
            deck_can_split
        };
        // 刷新温泉Buff
        ret.update_scenario_buff(true);
        Ok(ret)
    }
}
