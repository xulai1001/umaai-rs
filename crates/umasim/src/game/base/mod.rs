pub mod action;
pub mod basic;
pub mod person;
use std::{default::Default, sync::Arc};

pub use action::*;
use anyhow::Result;
use hashbrown::HashMap;
use log::info;
pub use person::*;
use rand::{rngs::StdRng, seq::IndexedRandom};

use crate::{explain::Explain, game::*, gamedata::EventData, utils::*};

/// 一局游戏的基本状态，剧本通用，用于计算，不用于通信(例如通信只传递卡组id)  
/// 不包含人头信息(Person类型可能不同)，实际的剧本对象需要补上Vec<Person>才能实现Game Trait    
/// 需要频繁clone，一部分不变量需要引用  
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BaseGame {
    /// 回合数 [0, 77]
    pub turn: i32,
    /// 回合阶段
    pub stage: TurnStage,
    /// 马娘信息
    pub uma: Uma,
    /// 卡组信息
    pub deck: Vec<SupportCard>,
    /// 继承因子信息，在育成中不变但是要随时取
    pub inherit: Arc<InheritInfo>,
    /// 友人数据
    pub friend: FriendState,
    /// 设施等级计数 (设施等级x4)
    pub train_level_count: Array5,
    /// 人头分布 [训练, persons_index]. -1为不在    
    /// 使用index而非引用
    pub distribution: Vec<Vec<i32>>,
    /// 不在率下降，处理成加算
    pub absent_rate_drop: i32,
    /// 已经触发的事件id和次数
    pub events: HashMap<u32, u32>,
    /// 本回合内还没触发的事件(Hint, 点击友人等)
    pub unresolved_events: Vec<EventData>
}

impl BaseGame {
    pub fn explain(&self) -> Result<String> {
        let mut lines = vec![];
        lines.push(format!(
            "回合: {}-{:?} 设施等级: {} 友人: {}",
            self.turn + 1,
            self.stage,
            Explain::train_level_count(&self.train_level_count),
            self.friend.explain()
        ));
        lines.push(self.uma.explain()?);
        Ok(lines.join("\n"))
    }

    /// 建立游戏对象
    pub fn new(uma_id: u32, deck_ids: &[u32; 6], inherit: InheritInfo) -> Result<Self> {
        let mut uma = Uma::new(uma_id)?;
        info!("{}", uma.explain()?);
        let mut deck = vec![];
        let mut friend_id = None;
        let mut friend_index = 0;
        // 支援卡
        for (index, id) in deck_ids.iter().enumerate() {
            let card = SupportCard::new(*id)?;
            // 初始属性
            let initial = card.initial_bonus()?;
            let race_bonus = card.effect.saihou;
            if !initial.is_default() {
                info!("{} +初始属性 {initial:?} 赛后{race_bonus}", card.short_name()?);
            } else {
                info!("{} 赛后{race_bonus}", card.short_name()?);
            }
            let (initial, pt) = split_status(initial)?;
            uma.five_status.add_eq(initial);
            uma.skill_pt += pt;
            uma.race_bonus += race_bonus;
            // 友人. 暂时不处理多个友人
            if card.card_type >= 5 {
                friend_id = Some(*id);
                friend_index = index;
            }
            deck.push(card);
        }
        // 继承
        let newgame_inherit = inherit.inherit_newgame();
        info!("+继承: {newgame_inherit:?}");
        uma.five_status.add_eq(&newgame_inherit);

        Ok(Self {
            turn: 0,
            stage: TurnStage::Begin,
            uma,
            deck,
            inherit: Arc::new(inherit),
            friend: FriendState::new(friend_id, friend_index)?,
            train_level_count: [0; 5],
            distribution: vec![],
            events: HashMap::new(),
            absent_rate_drop: 0,
            unresolved_events: vec![]
        })
    }

    pub fn base_train_level(&self, train: usize) -> usize {
        (self.train_level_count[train] / 4 + 1).max(0).min(5) as usize
    }
    /// 随机选择一个能发生的事件
    pub fn random_select_event(&self, events: &[EventData], rng: &mut StdRng) -> Option<EventData> {
        let available_events: Vec<_> = events
            .iter()
            .filter(|e| e.max_trigger_time == 0 || *self.events.get(&e.id).unwrap_or(&0) < e.max_trigger_time)
            .collect();
        available_events.choose(rng).map(|e| (*e).clone())
    }
    /// 使事件生效，无交互或随机判定（训练员无法选择）。如果需要随机判定或选择，需要把事件加入unresolved_events让他在回合后调用
    pub fn apply_event(&mut self, event: &EventData, choice: usize) {
        self.events.entry(event.id).and_modify(|x| *x += 1).or_insert(1);
        if !event.choices.is_empty() {
            self.uma.add_value(&event.choices[choice]);
        }
    }

    pub fn is_xiahesu(&self) -> bool {
        (self.turn >= 36 && self.turn < 40) || (self.turn >= 60 && self.turn < 64)
    }

    pub fn generate_card_event(&self, person_index: i32, rng: &mut StdRng) -> Option<EventData> {
        // 支援卡事件. 再精细一点模拟 后一段事件发生次数不能多于前一段事件
        let card_event_times: Vec<_> = vec![8001, 8002, 8003]
            .iter()
            .map(|x| *self.events.get(x).unwrap_or(&0))
            .collect();
        let mut available_events = vec![];
        if card_event_times[0] < 5 {
            available_events.push(0);
        }
        if card_event_times[1] < card_event_times[0] {
            available_events.push(1);
        }
        if card_event_times[2] < card_event_times[1] {
            available_events.push(2);
        }
        if let Some(index) = available_events.choose(rng) {
            let mut event = global_events().card_events[*index].clone();

            event.person_index = Some(person_index);
            Some(event)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::{gamedata::*, utils::init_logger};

    #[test]
    fn test_explain() -> Result<()> {
        init_logger("debug")?;
        init_global()?;
        let mut game = BaseGame::default();
        game.uma.uma_id = 101901;
        game.uma.motivation = 5;
        game.uma.flags.qiezhe = true;
        println!("{}", game.explain()?);

        Ok(())
    }

    #[test]
    fn test_newgame() -> Result<()> {
        init_logger("debug")?;
        init_global()?;
        let game = BaseGame::new(101901, &[302424, 302464, 302484, 302564, 302574, 302644], InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        })?;
        println!("{}", game.explain()?);
        let score = game.uma.calc_score();
        println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);
        Ok(())
    }
}
