use std::ops::{Deref, DerefMut};

use anyhow::{Result, anyhow};
use colored::Colorize;
use comfy_table::{ColumnConstraint, Table, Width};
use log::{info, warn};
use rand::{Rng, rngs::StdRng, seq::IndexedRandom};
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{
    explain::Explain,
    game::{
        BaseGame,
        BasePerson,
        CardTrainingEffect,
        FriendCardState,
        FriendOutState,
        Game,
        InheritInfo,
        Person,
        PersonType,
        SupportCard,
        Trainer,
        Uma,
        onsen::{action::OnsenAction, *}
    },
    gamedata::{ActionValue, EventData, GAMECONSTANTS, onsen::ONSENDATA},
    global,
    utils::{Array5, Array6, AttributeArray, global_events, system_event, system_event_prob}
};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OnsenGame {
    pub base: BaseGame,
    /// 回合阶段 (覆盖base.stage)
    pub stage: OnsenTurnStage,
    pub persons: Vec<BasePerson>,
    /// 当前剧本buff，更新状态时重新计算
    pub scenario_buff: OnsenBuff,
    /// 当前温泉id
    pub current_onsen: usize,
    /// 温泉buff信息
    pub bathing: BathingInfo,
    /// 温泉是否已挖掘
    pub onsen_state: Vec<bool>,
    /// 当前每个温泉的剩余挖掘量
    pub dig_remain: Vec<[i32; 3]>,
    /// 挖了几个温泉
    pub dig_count: i32,
    /// 当前挖掘力加成，更新状态时重新计算
    pub dig_power: [i32; 3],
    /// 当前设施等级
    pub dig_level: [i32; 3],
    /// 蓝因子数量
    pub dig_blue_count: Array5,
    /// 挖掘消耗的体力
    pub dig_vital_cost: i32
}

impl Deref for OnsenGame {
    type Target = BaseGame;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for OnsenGame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl OnsenGame {
    pub fn explain(&self) -> Result<String> {
        let mut lines = vec![];
        lines.push(format!(
            "回合: {} - {:?} 训练等级: {} 友人: {}",
            self.turn + 1,
            self.stage,
            Explain::train_level_count(&self.train_level_count),
            self.friend.explain()
        ));
        lines.push(self.uma.explain()?);
        lines.push(self.bathing.explain());
        lines.push(format!(
            "已挖掘: {} {:?}, 当前挖掘: {}, 剩余: {:?}",
            self.dig_count, self.onsen_state, self.current_onsen, self.dig_remain[self.current_onsen]
        ));
        lines.push(format!(
            "挖掘等级: {:?}, 挖掘力加成: {:?}, 挖掘消耗体力: {}",
            self.dig_level, self.dig_power, self.dig_vital_cost
        ));
        Ok(lines.join("\n"))
    }

    pub fn add_person(&mut self, mut person: BasePerson) {
        info!("新人物: {:?} - {}", person.person_type, person.explain());
        person.person_index = self.persons.len() as i32;
        self.persons.push(person);
    }

    pub fn is_race_turn(&self) -> Result<bool> {
        self.uma.is_race_turn(self.turn)
    }

    pub fn newgame(uma_id: u32, deck_ids: &[u32; 6], inherit: InheritInfo) -> Result<Self> {
        let mut ret = OnsenGame {
            base: BaseGame::new(uma_id, deck_ids, inherit)?,
            stage: OnsenTurnStage::Begin,
            persons: vec![],
            onsen_state: vec![true, false, false, false, false, false, false, false, false, false],
            dig_level: [1, 1, 1],
            ..Default::default()
        };
        // 蓝因子
        for i in 0..5 {
            ret.dig_blue_count[i] = (ret.inherit.blue_count[i] as f32 / 3.0).ceil() as i32;
        }
        // 温泉剩余量
        let onsen_info = &global!(ONSENDATA).onsen_info;
        for i in 0..onsen_info.len() {
            ret.dig_remain
                .push(onsen_info[i].dig_volume.clone().try_into().expect("dig_volume"));
        }
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

    /// 获得当前挖掘的是哪种土质
    pub fn current_dig_type(&self) -> Option<usize> {
        if self.turn < 2 || self.turn >= 72 || self.is_xiahesu() {
            None
        } else if self.dig_remain[self.current_onsen][0] > 0 {
            Some(0)
        } else if self.dig_remain[self.current_onsen][1] > 0 {
            Some(1)
        } else {
            Some(2)
        }
    }

    pub fn next_dig_type(&self) -> Option<usize> {
        if let Some(mut ty) = self.current_dig_type() {
            while ty < 3 && self.dig_remain[self.current_onsen][ty] == 0 {
                ty += 1;
            }
            if ty < 3 {
                return Some(ty);
            }
        }
        None
    }

    /// 获得温泉券
    pub fn add_ticket(&mut self, num: i32) {
        self.bathing.ticket_num = (self.bathing.ticket_num + num).min(3);
        info!("{}", format!("+ 温泉券+{num}, 当前 {}", self.bathing.ticket_num).cyan());
    }
    /// 获得超回复
    pub fn add_super(&mut self) {
        if self.bathing.is_super_ready {
            warn!("无法重复获得超回复");
        } else {
            info!("+ 超回复预备");
            self.bathing.is_super_ready = true;
            self.dig_vital_cost = 0;
            if self.bathing.ticket_num == 0 {
                warn!("获得超回复状态，但暂时没有温泉券");
            }
        }
    }

    /// 使用温泉券
    /*
    pub fn use_ticket(&mut self) -> Result<()> {
        // sanity check
        if self.bathing.ticket_num == 0 || self.bathing.buff_remain_turn > 0 {
            Err(anyhow!("无法使用温泉券: {:?}", self.bathing))
        } else {
            if self.bathing.is_super_ready {
                info!("使用温泉券(超回复)");
                // 增加超回复效果
                self.bathing.is_super = true;
            }
            self.bathing.ticket_num -= 1;
            self.bathing.buff_remain_turn = 2;
            Ok(())
        }
    }
    */

    /// 判断友人出门事件
    pub fn check_friend_out_event(&self, rng: &mut StdRng) -> Option<EventData> {
        if self.friend.out_state == FriendOutState::BeforeUnlock {
            let friendship = self.persons[self.friend.person_index as usize].friendship;
            let out_prob = if friendship < 60 {
                system_event_prob("friend_unlock_low")
            } else {
                system_event_prob("friend_unlock_high")
            }
            .expect("friend_unlock_* prob key not found");
            if rng.random_bool(out_prob) {
                return Some(global_events().friend_events["out"].clone());
            }
        }
        None
    }

    /// 计算剧本Buff
    pub fn update_scenario_buff(&mut self) {
        let onsen_data = global!(ONSENDATA);
        let mut onsen_buff = OnsenBuff::default();
        self.dig_count = 0;
        for i in 0..self.onsen_state.len() {
            if self.onsen_state[i] {
                onsen_buff.onsen.add_eq(&onsen_data.onsen_info[i].effect);
                self.dig_count += 1;
            }
        }
        // 旅馆加成
        let hotel_effect = &onsen_data.hotel_effect;
        let mut j = hotel_effect.len() - 1;
        while self.dig_count < hotel_effect[j].onsen_count {
            if j > 0 {
                j -= 1;
            } else {
                break;
            }
        }
        if self.dig_count >= hotel_effect[j].onsen_count {
            onsen_buff.hotel = hotel_effect[j].clone();
        } else {
            onsen_buff.hotel = HotelEffect::default();
        }
        // 挖掘力加成
        let mut dig_blue_bonus = vec![0, 0, 0];
        let mut dig_stat_bonus = vec![0, 0, 0];
        for stat in 0..5 {
            // 计算属性是哪一档
            let mut stat_rank = onsen_data.dig_stat_ranks.len() - 1;
            while self.uma.five_status[stat] < onsen_data.dig_stat_ranks[stat_rank] {
                stat_rank -= 1;
            }
            for dig_type in 0..3 {
                // 因子加成
                dig_blue_bonus[dig_type] += self.dig_blue_count[stat] * onsen_data.dig_blue_bonus[stat][dig_type];
                // 对挖掘类型dig_type，当前属性加成对应第几个表
                for which in 0..3 {
                    if onsen_data.dig_stat_bonus_types[dig_type][which] == (stat + 1) as i32 {
                        // 在dig_stat_bonus中查到当前属性档位对应的加成
                        dig_stat_bonus[dig_type] += onsen_data.dig_stat_bonus[which][stat_rank];
                    }
                }
            }
        }
        let dig_tool_bonus = self
            .dig_level
            .iter()
            .map(|x| onsen_data.dig_tool_level[*x as usize])
            .collect::<Vec<_>>();
        info!("挖掘力加成: 因子 {dig_blue_bonus:?}, 属性 {dig_stat_bonus:?}, 工具 {dig_tool_bonus:?}");
        for i in 0..3 {
            self.dig_power[i] = dig_blue_bonus[i] + dig_stat_bonus[i] + dig_tool_bonus[i];
        }
        // 固定属性
        if let Some(ty) = self.current_dig_type() {
            onsen_buff.fixed_stat = onsen_data.dig_fixed_stat[ty].clone();
        } else {
            onsen_buff.fixed_stat = Array5::default();
        }
        self.scenario_buff = onsen_buff;
    }

    /// 计算不同指令的挖掘量
    pub fn calc_dig_value(&self, action: &OnsenAction) -> Option<[i32; 3]> {
        let mut ret = [0, 0, 0];
        let mut link_bonus = [0, 0, 0];
        let link_effect = &global!(ONSENDATA).link_effect;
        // Link角色加成
        if let Some(ty) = link_effect.get(&self.uma.chara_id().to_string()) {
            link_bonus[(*ty - 1) as usize] += 10;
        }
        // 基础挖掘量
        let base_dig_value = match action {
            OnsenAction::Train(t) => {
                let mut person_count = 0;
                for p in &self.distribution[*t as usize] {
                    // 人头数排除记者理事长
                    if *p != 6 && *p != 7 {
                        person_count += 1;
                    }
                    // link卡加成
                    if let Some(ty) = link_effect.get(&self.persons[*p as usize].chara_id.to_string()) {
                        link_bonus[(*ty - 1) as usize] += 10;
                    }
                }
                Some(25 + person_count)
            }
            OnsenAction::PR => Some(10),
            OnsenAction::Race => {
                if self.uma.is_race_turn(self.turn).unwrap_or(false) {
                    Some(25)
                } else {
                    Some(15)
                }
            }
            OnsenAction::Sleep | OnsenAction::NormalOuting => Some(15),
            OnsenAction::FriendOuting => Some(25),
            OnsenAction::Clinic => Some(0),
            _ => None
        };
        if let (Some(base), Some(ty)) = (base_dig_value, self.current_dig_type()) {
            if link_bonus[ty] > 0 {
                info!("Link加成(#{ty}): {}%", link_bonus[ty]);
            }
            let current_dig_rate = (100.0 + self.dig_power[ty] as f32) * (100.0 + link_bonus[ty] as f32) / 10000.0;
            let dig_value = (base as f32 * current_dig_rate).floor() as i32;
            // 如果触发挖掘下一层，要重新按照下层挖掘力计算剩余挖掘量
            if dig_value > self.dig_remain[self.current_onsen][ty]
                && let Some(next_ty) = self.next_dig_type()
            {
                ret[ty] = self.dig_remain[self.current_onsen][ty];
                let base_remain = base as f32 - self.dig_remain[self.current_onsen][ty] as f32 / current_dig_rate;
                let next_dig_rate = (100.0 + self.dig_power[next_ty] as f32) / 100.0;
                ret[next_ty] = (base_remain * next_dig_rate).floor() as i32;
            } else {
                ret[ty] = dig_value.min(self.dig_remain[self.current_onsen][ty]);
            }
            Some(ret)
        } else {
            None
        }
    }

    /// 计算超回复触发概率
    /// （1）每档体力为50（不带友人）42.5（带友人）  
    /// （2）档位增加时 超回复几率为记载的值  
    /// （3）档位没增加时触发超回复的几率是当前概率/4  
    /// https://github.com/mee1080/umasim/blob/main/core/src/commonMain/kotlin/io/github/mee1080/umasim/scenario/onsen/OnsenCalculator.kt#L329
    pub fn calc_super_prob(&self, vital_cost: i32) -> f64 {
        let threshold = if self.friend.card_state == FriendCardState::Empty {
            50.0
        } else {
            42.5
        };
        let old_rank = ((self.dig_vital_cost as f64 / threshold).floor() as usize).min(5);
        let new_rank = (((self.dig_vital_cost + vital_cost) as f64 / threshold).floor() as usize).min(5);
        let ret = global!(ONSENDATA).super_probs[new_rank] as f64 / 100.0;
        if new_rank > old_rank { ret } else { ret / 4.0 }
    }
    /// 执行挖掘, true为挖完了
    pub fn do_dig(&mut self, value: &[i32; 3]) -> bool {
        for i in 0..3 {
            self.dig_remain[self.current_onsen][i] -= value[i];
        }
        if self.dig_remain[self.current_onsen].iter().all(|x| *x <= 0) {
            self.dig_remain[self.current_onsen] = [0, 0, 0];
            // 挖完了
            self.onsen_state[self.current_onsen] = true;
            self.add_ticket(self.grant_ticket_num());
            true
        } else {
            false
        }
    }

    /// 执行选择温泉
    pub fn do_select_onsen<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let onsen_info = &global!(ONSENDATA).onsen_info;
        let actions = self
            .onsen_state
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                if !*b && self.turn >= onsen_info[i].unlock_turn {
                    Some(OnsenAction::Dig(i as i32))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        info!("选择要挖掘的温泉:");
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action(&actions[selection], rng)
    }

    /// 赠送票数
    pub fn grant_ticket_num(&self) -> i32 {
        if self.turn == 2 {
            // 开局均为2张
            2
        } else if self.turn == 72 {
            // 72回合 SSR=2 其他为1
            if self.friend.card_state == FriendCardState::SSR {
                2
            } else {
                1
            }
        } else {
            // 其他情况
            match self.friend.card_state {
                FriendCardState::SSR => 2,
                FriendCardState::R => 1,
                _ => 0
            }
        }
    }

    /// 使用Hint buff后再对没有叹号的卡判定一次Hint
    pub fn redistribute_hint(&mut self, rng: &mut StdRng) -> Result<()> {
        let base_hint_rate = global!(GAMECONSTANTS).base_hint_rate / 100.0;
        let hint_probs: Vec<_> = self
            .deck()
            .iter()
            .map(|card| card.card_value().expect("card_value").hint_prob_increase + self.scenario_buff.onsen.hint_bonus)
            .collect();
        for person in self.persons_mut() {
            if person.person_type == PersonType::Card && !person.is_hint {
                let hint_prob = base_hint_rate * ((100 + hint_probs[person.person_index() as usize]) as f32 / 100.0);
                person.set_hint(rng.random_bool(hint_prob as f64));
            }
        }
        Ok(())
    }
}

impl Game for OnsenGame {
    type Person = BasePerson;
    type Action = OnsenAction;

    fn init_persons(&mut self) -> Result<()> {
        let mut persons = vec![];
        let mut friend_index = None;
        let mut friend_state = FriendCardState::Empty;

        for (i, card) in self.deck.iter().enumerate() {
            let mut person = BasePerson::try_from(card)?;
            // 对默认转换再过滤一下非剧本友人
            if person.person_type == PersonType::ScenarioCard {
                if person.chara_id == 9050 {
                    friend_state = match card.get_data()?.rarity {
                        1 => FriendCardState::R,
                        3 => FriendCardState::SSR,
                        _ => return Err(anyhow!("invalid friend rarity"))
                    };
                    friend_index = Some(i);
                    info!("剧本友人: {friend_state}, index = {i}");
                } else {
                    person.person_type = PersonType::OtherFriend;
                }
            }
            persons.push(person);
        }
        if let Some(i) = friend_index {
            self.friend.person_index = i;
            self.friend.card_state = friend_state;
        }
        for p in persons {
            self.add_person(p);
        }
        // 添加理事长,记者(记者在12回合才会出现)
        self.add_person(BasePerson::yayoi());
        self.add_person(BasePerson::reporter());
        Ok(())
    }

    fn next(&mut self) -> bool {
        if let Some(stage) = self.stage.next() {
            // 回合内，下一个阶段(不包含OnsenSelect)
            self.stage = stage;
        } else if self.turn < self.max_turn() {
            // 下一个回合
            self.turn += 1;
            self.stage = OnsenTurnStage::Begin;
        } else {
            return false;
        }
        true
    }

    fn list_actions(&self) -> Result<Vec<Self::Action>> {
        let mut actions = vec![];
        match self.stage {
            OnsenTurnStage::Bathing => {
                // sanity check
                if self.bathing.buff_remain_turn > 0 || self.bathing.ticket_num == 0 {
                    Err(anyhow!("invalid bathing state: {:?}", self.bathing))
                } else {
                    Ok(vec![OnsenAction::UseTicket(true), OnsenAction::UseTicket(false)])
                }
            }
            OnsenTurnStage::Train => {
                if self.is_race_turn()? {
                    Ok(vec![OnsenAction::Race])
                } else {
                    actions = vec![
                        OnsenAction::Train(0),
                        OnsenAction::Train(1),
                        OnsenAction::Train(2),
                        OnsenAction::Train(3),
                        OnsenAction::Train(4),
                    ];
                    if self.is_xiahesu() {
                        actions.push(OnsenAction::Race);
                        actions.push(OnsenAction::Sleep);
                    } else {
                        // 普通训练
                        if self.turn < 72 {
                            actions.push(OnsenAction::PR);
                        }
                        actions.push(OnsenAction::Sleep);
                        actions.push(OnsenAction::NormalOuting);
                        if self.turn > 13 && self.turn < 72 {
                            actions.push(OnsenAction::Race);
                        }
                        if self.uma.flags.ill {
                            actions.push(OnsenAction::Clinic);
                        }
                        if self.friend.out_state == FriendOutState::AfterUnlock && self.turn < 72 {
                            if !self.friend.out_used.iter().all(|used| *used) {
                                actions.push(OnsenAction::FriendOuting);
                            }
                        }
                    }
                    Ok(actions)
                }
            }
            _ => Err(anyhow!("当前阶段不允许进行Actions: {:?}", self.stage))
        }
    }

    fn generate_events(&self, rng: &mut StdRng) -> Vec<EventData> {
        let no_event_turns = &global!(GAMECONSTANTS).no_event_turns;
        // 通用剧本事件
        let mut scenario_events = self.list_turn_events(&global_events().story_events);
        scenario_events.extend(self.list_turn_events(&global!(ONSENDATA).scenario_events));
        if !scenario_events.is_empty() {
            scenario_events
        } else if !no_event_turns.contains(&self.turn) {
            // 判断友人出门事件
            if let Some(event) = self.check_friend_out_event(rng) {
                vec![event]
            } else {
                // 一般随机事件
                let weights =
                    WeightedIndex::new(global!(GAMECONSTANTS).get_event_distribution()).expect("event weights");
                let event = match weights.sample(rng) {
                    0 => {
                        // 支援卡连续事件
                        let available_indices = (0..6)
                            .filter(|x| {
                                self.persons[*x].person_type == PersonType::Card && self.person_is_available(*x)
                            })
                            .collect::<Vec<_>>();
                        if let Some(index) = available_indices.choose(rng) {
                            self.generate_card_event(*index as i32, rng)
                        } else {
                            None
                        }
                    }
                    1 => {
                        // 马娘事件
                        self.random_select_event(&global_events().uma_events, rng)
                    }
                    2 => {
                        // 掉心情
                        if self.turn >= 12 {
                            system_event("drop_motivation").ok().cloned()
                        } else {
                            None
                        }
                    }
                    _ => None
                };
                event.map(|x| vec![x]).unwrap_or_default()
            }
        } else {
            vec![]
        }
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
            809050011..=809050014 => {
                // 温泉券+1，超回复
                self.add_ticket(1);
                self.add_super();
            }
            809050015 => {
                // 温泉券+1或2，超回复
                self.add_ticket(self.grant_ticket_num());
                self.add_super();
            }
            400012005 | 400012011 => {
                // 第1,2年底增加全部训练等级
                info!(">> 全部训练等级+1");
                self.train_level_count.add_eq(&[4, 4, 4, 4, 4]);
            }
            400012017 => {
                // 3年底再给一次票
                self.add_ticket(self.grant_ticket_num());
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
            Ok(eff.deyilv + self.scenario_buff.hotel.deyilv as f32)
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
            let dig = self
                .calc_dig_value(&OnsenAction::Train(train as i32))
                .map(|x| format!("挖: {x:?}"))
                .unwrap_or_default();
            if fail_rate > 0.0 {
                lines.push(format!(
                    "{} {} 失败率: {}% {dig}",
                    headers[train],
                    value.explain(),
                    fail_rate
                ));
            } else {
                lines.push(format!("{} {} {dig}", headers[train], value.explain()));
            }
        }
        let mut others = vec![];
        for action in [
            OnsenAction::FriendOuting,
            OnsenAction::NormalOuting,
            OnsenAction::Race,
            OnsenAction::Sleep,
            OnsenAction::PR
        ] {
            if let Some(dig) = self.calc_dig_value(&action) {
                others.push(format!("{} 挖: {:?}", action, dig));
            }
        }
        if !others.is_empty() {
            lines.push(others.join(" | "));
        }
        Ok(lines.join("\n"))
    }

    fn calc_training_value(&self, buffs: &CardTrainingEffect, train: usize) -> Result<ActionValue> {
        // 计算下层值
        let mut base_value = self.default_calc_training_value(buffs, train)?;
        // 下层不超过100
        for i in 0..6 {
            base_value.status_pt[i] = base_value.status_pt[i].min(100);
        }
        if self.bathing.buff_remain_turn > 0 {
            // 温泉buff加成
            let mut onsen_buff = self.scenario_buff.onsen.to_training_effect(train);
            if buffs.youqing == 0.0 {
                onsen_buff.youqing = 0.0;
            }
            let total_buff = buffs.add(&onsen_buff);
            let mut total_value = self.default_calc_training_value(&total_buff, train)?;
            // 土质固定属性加成
            if self.current_dig_type().is_some() {
                for i in 0..5 {
                    total_value.status_pt[i] += self.scenario_buff.fixed_stat[i];
                }
                total_value.vital -= 3; // 额外-3体力
            }
            let mut upper_value = Array6::default();
            // 计算上层值=总-下层，不超过100
            for i in 0..6 {
                upper_value[i] = (total_value.status_pt[i] - base_value.status_pt[i]).min(100);
                total_value.status_pt[i] = base_value.status_pt[i] + upper_value[i];
            }
            info!(
                "训练: {train}, 下层: {:?}, 上层: {:?}",
                base_value.status_pt, upper_value
            );
            Ok(total_value)
        } else {
            Ok(base_value)
        }
    }

    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        info!("-- Turn {}-{:?} --", self.turn, self.stage);
        match self.stage {
            OnsenTurnStage::Begin => {
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
                // 开局和合宿送券
                if [2, 36, 60].contains(&self.turn) {
                    self.add_ticket(self.grant_ticket_num());
                }
                // 执行回合前事件
                for event in &events {
                    self.run_event(event, trainer, rng)?;
                }
            }
            OnsenTurnStage::Distribute => {
                self.update_scenario_buff();
                if self.is_race_turn()? {
                    self.reset_distribution();
                } else {
                    self.distribute_all(rng)?;
                    self.distribute_hint(rng)?;
                    info!("训练:\n{}", self.explain_distribution()?);
                }
            }
            OnsenTurnStage::Train => {
                self.list_and_apply_action(trainer, rng)?;
            }
            OnsenTurnStage::AfterTrain => {
                let after_events = std::mem::take(&mut self.unresolved_events);
                for event in &after_events {
                    self.run_event(event, trainer, rng)?;
                }
            }
            OnsenTurnStage::Bathing => {
                if self.turn >= 2 && self.bathing.buff_remain_turn == 0 && self.bathing.ticket_num > 0 {
                    self.list_and_apply_action(trainer, rng)?;
                }
            }
        }
        Ok(())
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
