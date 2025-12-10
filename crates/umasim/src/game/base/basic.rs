//! 基础（无剧本）游戏，用于测试特性

use std::{
    fmt::Display,
    ops::{Deref, DerefMut}
};

use anyhow::{Result, anyhow};
use comfy_table::{ColumnConstraint, Table, Width};
use enum_iterator::Sequence;
use log::{info, warn};
use rand::{Rng, rngs::StdRng, seq::IndexedRandom};
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{
    game::{
        BaseAction::{self, *},
        BaseGame,
        BasePerson,
        FriendOutState,
        InheritInfo,
        PersonType,
        SupportCard,
        TurnStage,
        Uma,
        traits::*
    },
    gamedata::*,
    global,
    utils::{AttributeArray, global_events, system_event, system_event_prob}
};

#[derive(Debug, Clone, PartialEq)]
pub struct BasicAction(BaseAction);

impl Deref for BasicAction {
    type Target = BaseAction;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for BasicAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ActionEnum for BasicAction {
    type Game = BasicGame;
    // Train需要在当前类型里实现，其他操作可以使用BaseAction里默认的实现
    fn apply(&self, game: &mut Self::Game, rng: &mut StdRng) -> Result<()> {
        match &self.0 {
            BaseAction::Train(train) => self.do_train(game, *train as usize, rng),
            _ => self.0.apply(game, rng)
        }
    }

    fn as_base_action(&self) -> Option<BaseAction> {
        self.0.as_base_action()
    }
}

impl BasicAction {
    pub fn do_train(&self, game: &mut BasicGame, train: usize, rng: &mut StdRng) -> Result<()> {
        // sanity check
        if train >= 5 {
            return Err(anyhow!("训练等级越界: {train}"));
        }
        info!(
            ">> {}训练 等级 {}",
            global!(GAMECONSTANTS).train_names[train],
            game.train_level(train)
        );
        let buffs = game.calc_training_buff(train)?;
        let failure_rate = game.calc_training_failure_rate(&buffs, train) / 100.0;

        if rng.random_bool(failure_rate as f64) {
            // 再判断一次，如果还失败就是大失败
            if rng.random_bool(failure_rate as f64) {
                warn!("训练大失败!");
                game.apply_event(system_event("training_fail_low")?, 0, rng)?;
                game.uma.flags.ill = true;
                game.uma.flags.bad_trainer = true;
            } else {
                warn!("训练失败!");
                game.apply_event(system_event("training_fail")?, 0, rng)?;
            }
        } else {
            let value = game.calc_training_value(&buffs, train)?;
            game.uma.add_value(&value);
            // 增加训练次数
            game.train_level_count[train] += 1;
            // 增加羁绊
            let f = if game.uma.flags.aijiao { 9 } else { 7 };
            let mut hint_persons = vec![];
            let mut friend_clicked = false;
            let mut yayoi_clicked = false;
            let mut reporter_clicked = false;
            for person_index in game.distribution[train].clone() {
                // 跳过空位（-1 表示空位）
                if person_index < 0 {
                    continue;
                }
                game.add_friendship(person_index as usize, f);
                if game.persons[person_index as usize].is_hint {
                    hint_persons.push(person_index);
                }
                match game.persons[person_index as usize].person_type {
                    PersonType::ScenarioCard => friend_clicked = true,
                    PersonType::Yayoi => yayoi_clicked = true,
                    PersonType::Reporter => reporter_clicked = true,
                    _ => {}
                };
            }
            // 生成加练或者红点事件
            if let Some(p) = hint_persons.choose(rng) {
                let attr_prob = system_event_prob("hint_attr")?;
                let hint_level = if *p < 6 {
                    1 + game.deck[*p as usize].card_value()?.hint_level
                } else {
                    1
                };
                let mut hint_event = if rng.random_bool(attr_prob as f64) {
                    // 红点提供属性
                    EventData::hint_attr_event(game.persons[*p as usize].train_type as usize, *p as usize)?
                } else {
                    // 红点提供技能
                    EventData::hint_skill_event(hint_level, *p as usize)
                };
                hint_event.name = format!("{} - {}", hint_event.name, game.deck[*p as usize].short_name()?);
                game.unresolved_events.push(hint_event);
            }
            let extra_train_prob = system_event_prob("extra_train")?;
            if !game.is_xiahesu() && rng.random_bool(extra_train_prob as f64) {
                game.unresolved_events.push(EventData::extra_training_event(train));
            }
            // 更新友人状态
            if friend_clicked {
                match game.friend.out_state {
                    FriendOutState::UnClicked => {
                        game.friend.out_state = FriendOutState::BeforeUnlock;
                        let mut event = global_events().friend_events["first"].clone();
                        event.person_index = Some(game.friend.person_index as i32); // 设置增加羁绊的目标为友人
                        game.unresolved_events.push(event);
                    }
                    _ => {
                        let mut event = global_events().friend_events["click"].clone();
                        event.person_index = Some(game.friend.person_index as i32);
                        game.unresolved_events.push(event);
                    }
                }
            }
            if yayoi_clicked {
                game.unresolved_events.push(system_event("yayoi_click")?.clone());
            }
            if reporter_clicked {
                let mut event = system_event("reporter_click")?.clone();
                event.choices[0].status_pt[train] = 2;
                game.unresolved_events.push(event);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BasicGame {
    pub base: BaseGame,
    pub persons: Vec<BasePerson>
}

impl Deref for BasicGame {
    type Target = BaseGame;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for BasicGame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl BasicGame {
    pub fn add_person(&mut self, mut person: BasePerson) {
        info!("新训练角色: {}", person.explain());
        person.person_index = self.persons.len() as i32;
        self.persons.push(person);
    }

    pub fn is_race_turn(&self) -> Result<bool> {
        self.uma.is_race_turn(self.turn)
    }

    pub fn newgame(uma_id: u32, deck_ids: &[u32; 6], inherit: InheritInfo) -> Result<Self> {
        let mut ret = BasicGame {
            base: BaseGame::new(uma_id, deck_ids, inherit)?,
            persons: vec![]
        };
        ret.init_persons()?;
        Ok(ret)
    }

    pub fn add_friendship(&mut self, person_index: usize, value: i32) {
        if person_index < self.persons.len() {
            let new_value = (self.persons[person_index].friendship + value).min(100);
            self.persons[person_index].friendship = new_value;
            if person_index < 6 {
                self.deck[person_index].friendship = new_value;
            }
            info!(
                "{} 羁绊+{} (={})",
                self.persons[person_index].short_name(),
                value,
                new_value
            );
        }
    }
}

impl Game for BasicGame {
    type Person = BasePerson;
    type Action = BasicAction;
    fn init_persons(&mut self) -> Result<()> {
        let persons = self
            .deck
            .iter()
            .map(|card| BasePerson::try_from(card))
            .collect::<Result<Vec<_>>>()?;
        for p in persons {
            self.add_person(p);
        }
        // 添加理事长
        self.add_person(BasePerson::yayoi());
        Ok(())
    }
    fn next(&mut self) -> bool {
        if let Some(stage) = self.stage.next() {
            // 回合内，下一个阶段
            self.stage = stage;
        } else if self.turn < self.max_turn() {
            // 下一个回合
            self.turn += 1;
            self.stage = TurnStage::Begin;
        } else {
            return false;
        }
        true
    }

    fn list_actions(&self) -> Result<Vec<Self::Action>> {
        let mut actions = vec![];
        if self.is_race_turn()? {
            Ok(vec![BasicAction(Race)])
        } else {
            actions = vec![
                BasicAction(Train(0)),
                BasicAction(Train(1)),
                BasicAction(Train(2)),
                BasicAction(Train(3)),
                BasicAction(Train(4)),
            ];
            if self.is_xiahesu() {
                actions.push(BasicAction(Race));
                actions.push(BasicAction(Sleep));
            } else {
                // 普通训练
                actions.push(BasicAction(Sleep));
                actions.push(BasicAction(NormalOuting));
                if self.turn > 13 && self.turn < 72 {
                    actions.push(BasicAction(Race));
                }
                if self.uma.flags.ill {
                    actions.push(BasicAction(Clinic));
                }
                if self.friend.out_state == FriendOutState::AfterUnlock && self.turn < 72 {
                    if !self.friend.out_used.iter().all(|used| *used) {
                        actions.push(BasicAction(FriendOuting));
                    }
                }
            }
            Ok(actions)
        }
    }

    fn generate_events(&self, rng: &mut StdRng) -> Vec<EventData> {
        let mut events = vec![];
        let no_event_turns = &global!(GAMECONSTANTS).no_event_turns;
        // 剧本事件
        let story_events = global_events()
            .story_events
            .iter()
            .filter_map(|e| {
                if e.start_turn == self.turn {
                    Some(e.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if !story_events.is_empty() {
            // 如果有剧本事件，则返回剧本事件
            story_events
        } else if !no_event_turns.contains(&self.turn) {
            // 否则如果是允许发生随机事件的回合
            // 先判断友人出门事件
            if self.friend.out_state == FriendOutState::BeforeUnlock {
                let friendship = self.persons[self.friend.person_index as usize].friendship;
                let out_prob = if friendship < 60 {
                    system_event_prob("friend_unlock_low")
                } else {
                    system_event_prob("friend_unlock_high")
                }
                .expect("friend_unlock_* prob key not found");
                if rng.random_bool(out_prob) {
                    events.push(global_events().friend_events["out"].clone());
                    return events;
                }
            }
            // 之后处理一般随机事件
            let weights = WeightedIndex::new(global!(GAMECONSTANTS).get_event_distribution()).expect("event weights");
            match weights.sample(rng) {
                0 => {
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
                        let mut p = rng.random_range(0..6);
                        while !self.person_is_available(p as usize) {
                            p = rng.random_range(0..6);
                        }
                        event.person_index = Some(p);
                        events.push(event);
                    }
                }
                1 => {
                    // 马娘事件
                    if let Some(event) = self.random_select_event(&global_events().uma_events, rng) {
                        events.push(event);
                    }
                }
                2 => {
                    // 掉心情事件
                    if self.turn >= 12 {
                        events.push(system_event("drop_motivation").expect("掉心情事件").clone());
                    }
                }
                _ => {
                    // 无事发生
                }
            }
            events
        } else {
            vec![]
        }
    }

    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        //let events = self.list_events();
        //info!("-- Turn {}-{:?} --", self.turn, self.stage);
        match self.stage {
            TurnStage::Begin => {
                println!("-----------------------------------------");
                info!("{}", self.explain()?);
                let mut events = self.generate_events(rng);
                // 友人强制事件
                if self.friend.out_state == FriendOutState::AfterUnlock {
                    if self.turn == 24 {
                        events.push(global_events().friend_events["newyear"].clone());
                    } else if self.turn == 77 {
                        self.unresolved_events
                            .push(global_events().friend_events["end"].clone());
                    }
                }
                // 抽签事件
                if self.turn == 48 {
                    self.unresolved_events.push(system_event("ticket")?.clone());
                }
                // 育成结束事件
                if self.turn == 77 {
                    self.unresolved_events.push(system_event("ending")?.clone());
                }
                // 执行回合前事件
                for event in &events {
                    self.run_event(event, trainer, rng)?;
                }
            }
            TurnStage::Distribute => {
                if self.is_race_turn()? {
                    self.reset_distribution();
                } else {
                    self.distribute_all(rng)?;
                    self.distribute_hint(rng)?;
                    info!("训练:\n{}", self.explain_distribution()?);
                }
            }
            TurnStage::Train => {
                let actions = self.list_actions()?;
                let selection = trainer.select_action(self, &actions, rng)?;
                //info!("玩家选择: {:?}", actions[selection]);
                self.apply_action(&actions[selection], rng)?;
            }
            TurnStage::AfterTrain => {
                let after_events = std::mem::take(&mut self.unresolved_events);
                for event in &after_events {
                    self.run_event(event, trainer, rng)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// 使事件生效（无选项）。修改羁绊和特殊事件的部分需要在当前类型里完成
    fn apply_event(&mut self, event: &EventData, choice: usize, rng: &mut StdRng) -> Result<()> {
        self.base.apply_event(event, choice);
        if let (Some(person_index), Some(value)) = (&event.person_index, event.choices.get(choice)) {
            if value.friendship != 0 {
                self.add_friendship(*person_index as usize, value.friendship);
            }
        }
        // 判断特殊事件
        match event.id {
            4012 | 4013 => {
                // 继承
                let inherit_value = ActionValue {
                    status_pt: self.inherit.inherit(rng),
                    ..Default::default()
                };
                let inherit_limit = self.inherit.inherit_limit(rng);
                self.uma.add_value(&inherit_value);
                self.uma.five_status_limit.add_eq(&inherit_limit);
            }
            5007 => {
                // 大成功事件
                if rng.random_bool(system_event_prob("qiezhe_normal")?) {
                    warn!(">> 获得【切者】");
                    self.uma.flags.qiezhe = true;
                }
            }
            809050004 => {
                // 友人出门事件
                info!(">> 友人出行已解锁");
                self.friend.out_state = FriendOutState::AfterUnlock;
            }
            _ => {}
        }
        Ok(())
    }

    fn deyilv(&mut self, person_index: i32) -> Result<f32> {
        if person_index < 6 {
            let (eff, lock) = self.deck[person_index as usize].calc_training_effect(self, 0)?;
            if lock {
                self.deck[person_index as usize].is_locked = true;
            }
            Ok(eff.deyilv)
        } else {
            Ok(0.0)
        }
    }
    fn explain_distribution(&self) -> Result<String> {
        let headers = vec!["速", "耐", "力", "根", "智"];
        let dist = &self.distribution;
        let mut rows = vec![];
        for i in 0..6 {
            let mut row = vec![];
            for train in 0..5 {
                if let Some(id) = dist[train].get(i) {
                    let mut text = self.persons[*id as usize].explain();
                    if self.is_shining_at(*id as usize, train) {
                        text = format!("+{text}+");
                    }
                    row.push(text);
                } else {
                    row.push("".to_string());
                }
            }
            rows.push(row)
        }
        let mut table = Table::new();
        table.set_header(headers.clone()).add_rows(rows).set_width(80);
        for col in table.column_iter_mut() {
            col.set_constraint(ColumnConstraint::Absolute(Width::Percentage(20)));
        }
        let mut lines = vec![table.to_string()];
        for train in 0..5 {
            let buffs = self.calc_training_buff(train)?;
            let fail_rate = self.calc_training_failure_rate(&buffs, train);
            let value = self.calc_training_value(&buffs, train)?;
            if fail_rate > 0.0 {
                lines.push(format!("{} {} 失败率: {}%", headers[train], value.explain(), fail_rate));
            } else {
                lines.push(format!("{} {}", headers[train], value.explain()));
            }
        }
        Ok(lines.join("\n"))
    }
    // getters
    fn persons(&self) -> &[Self::Person] {
        &self.persons
    }
    fn persons_mut(&mut self) -> &mut [Self::Person] {
        &mut self.persons
    }
    fn absent_rate_drop(&self) -> i32 {
        self.absent_rate_drop
    }
    fn turn(&self) -> i32 {
        self.turn
    }
    fn max_turn(&self) -> i32 {
        77
    }
    fn uma(&self) -> &Uma {
        &self.uma
    }
    fn uma_mut(&mut self) -> &mut Uma {
        &mut self.uma
    }
    fn deck(&self) -> &Vec<SupportCard> {
        &self.deck
    }
    fn distribution(&self) -> &Vec<Vec<i32>> {
        &self.distribution
    }
    fn distribution_mut(&mut self) -> &mut Vec<Vec<i32>> {
        &mut self.distribution
    }
    fn has_group_buff(&self) -> bool {
        self.friend.group_buff_turn > 0
    }
    fn train_level(&self, train: usize) -> usize {
        if self.is_xiahesu() {
            5
        } else {
            (self.train_level_count[train] as usize / 4 + 1).min(5).max(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rand::SeedableRng;

    use super::*;
    use crate::{global, trainer::RandomTrainer, utils::*};

    #[test]
    fn test_newgame() -> Result<()> {
        init_logger("debug")?;
        init_global()?;
        let mut game = BasicGame::newgame(101901, &[302424, 302464, 302484, 302564, 302574, 302644], InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        })?;
        println!("{}", game.explain()?);
        let score = game.uma.calc_score();
        println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);
        let trainer = RandomTrainer {};
        let mut rng = StdRng::from_os_rng();
        game.run_full_game(&trainer, &mut rng)?;
        info!("育成结束！");
        let score = game.uma.calc_score();
        println!(
            "评分: {} {}, PT: {}",
            global!(GAMECONSTANTS).get_rank_name(score),
            score,
            game.uma.total_pt()
        );
        Ok(())
    }
}
