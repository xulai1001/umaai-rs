use std::ops::{Deref, DerefMut};

use anyhow::{Result, anyhow};
use colored::Colorize;
use comfy_table::{ColumnConstraint, Table, Width};
use log::{info, warn};
use rand::{
    Rng,
    rngs::StdRng,
    seq::{IndexedRandom, IteratorRandom}
};
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
    /// 当前每个温泉的每层累计挖掘进度（用于计算奖励）
    pub dig_progress: Vec<[i32; 3]>,
    /// 挖了几个温泉
    pub dig_count: i32,
    /// 当前挖掘力加成，更新状态时重新计算
    pub dig_power: [i32; 3],
    /// 当前设施等级
    pub dig_level: [i32; 3],
    /// 蓝因子数量
    pub dig_blue_count: Array5,
    /// 挖掘消耗的体力
    pub dig_vital_cost: i32,
    /// 挖掘完成后待处理选择（装备升级+源泉选择）
    pub pending_selection: bool,
    /// 是否能触发分身
    pub deck_can_split: bool
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

    /// 显示本回合人头分布（给 umaai 调试/日志使用）
    ///
    /// 注意：底层实现来自 `Game` trait 的 `explain_distribution()`。
    pub fn explain_distribution(&self) -> Result<String> {
        <Self as crate::game::Game>::explain_distribution(self)
    }

    pub fn add_person(&mut self, mut person: BasePerson) {
        info!("新人物: {:?} - {}", person.person_type, person.explain());
        person.person_index = self.persons.len() as i32;
        self.persons.push(person);
    }

    pub fn is_race_turn(&self) -> bool {
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
        // 上限规范化
        for i in 0..5 {
            ret.uma.five_status_limit[i] = ret.uma.five_status_limit[i].min(2800);
        }
        // 蓝因子
        for i in 0..5 {
            ret.dig_blue_count[i] = (ret.inherit.blue_count[i] as f32 / 3.0).ceil() as i32;
        }
        // 温泉剩余量和进度
        let onsen_info = &global!(ONSENDATA).onsen_info;
        for i in 0..onsen_info.len() {
            if i == 0 {
                // 初始温泉已挖掘完成，剩余量为0，避免挖掘完成初始温泉给俩票的问题出现
                ret.dig_remain.push([0, 0, 0]);
            } else {
                ret.dig_remain
                    .push(onsen_info[i].dig_volume.clone().try_into().expect("dig_volume"));
            }
            ret.dig_progress.push([0, 0, 0]); // 初始进度为0
        }
        ret.init_persons()?;
        // 携带5种卡以上才能分身
        ret.deck_can_split = ret.card_type_count.iter().filter(|x| **x > 0).count() >= 5;
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

    /// 获得当前层之后的下一个有剩余的层
    pub fn next_dig_type(&self) -> Option<usize> {
        if let Some(current_ty) = self.current_dig_type() {
            // 从下一层开始找
            let mut ty = current_ty + 1;
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
            info!("{}", "触发超回复".bright_yellow());
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

    /// 计算剧本Buff use_ticket为false时不更新温泉券buff(锁面板)
    pub fn update_scenario_buff(&mut self, use_ticket: bool) {
        let onsen_data = global!(ONSENDATA);
        let onsen_buff = &mut self.scenario_buff;
        // use_ticket=True 或者vital=0（没计算）时更新温泉券buff
        if use_ticket || onsen_buff.onsen.vital == 0 {
            onsen_buff.onsen = OnsenEffect::default();
            for i in 0..self.onsen_state.len() {
                if self.onsen_state[i] {
                    onsen_buff.onsen.add_eq(&onsen_data.onsen_info[i].effect);
                }
            }
        }
        self.dig_count = self.onsen_state.iter().filter(|x| **x).count() as i32;

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
            while stat_rank > 0 && self.uma.five_status[stat] < onsen_data.dig_stat_ranks[stat_rank] {
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
        //info!("挖掘力加成: 因子 {dig_blue_bonus:?}, 属性 {dig_stat_bonus:?}, 工具 {dig_tool_bonus:?}");
        for i in 0..3 {
            self.dig_power[i] = dig_blue_bonus[i] + dig_stat_bonus[i] + dig_tool_bonus[i];
        }
    }

    /// 计算不同指令的挖掘量
    pub fn calc_dig_value(&self, action: &OnsenAction) -> Option<[i32; 3]> {
        let mut ret = [0, 0, 0];
        let mut link_bonus = [0, 0, 0];
        let link_effect = &global!(ONSENDATA).link_effect;
        // Link角色加成（马娘）
        if let Some(ty) = link_effect.get(&self.uma.chara_id()) {
            link_bonus[(*ty - 1) as usize] += 10;
        }
        // Link支援卡加成（全局，只要携带就生效）
        for i in 0..6 {
            if i < self.persons.len() {
                if let Some(ty) = link_effect.get(&self.persons[i].chara_id) {
                    link_bonus[(*ty - 1) as usize] += 10;
                }
            }
        }
        // 基础挖掘量
        let base_dig_value = match action {
            OnsenAction::Train(t) => {
                let mut person_count = 0;
                for p in &self.distribution[*t as usize] {
                    // 人头数排除记者理事长
                    if *p >= 0 && *p != 6 && *p != 7 {
                        person_count += 1;
                    }
                }
                Some(25 + person_count)
            }
            OnsenAction::PR => Some(10 + 1), // 以1个人头计算
            OnsenAction::Race => {
                if self.uma.is_race_turn(self.turn) {
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
            // if link_bonus[ty] > 0 {
            //     info!("Link加成(#{ty}): {}%", link_bonus[ty]);
            // }
            let current_dig_rate = (100.0 + self.dig_power[ty] as f32) * (100.0 + link_bonus[ty] as f32) / 10000.0;
            let dig_value = (base as f32 * current_dig_rate).floor() as i32;
            // 如果触发挖掘下一层，要重新按照下层挖掘力计算剩余挖掘量
            if dig_value > self.dig_remain[self.current_onsen][ty]
                && let Some(next_ty) = self.next_dig_type()
            {
                ret[ty] = self.dig_remain[self.current_onsen][ty];
                let base_remain = base as f32 - self.dig_remain[self.current_onsen][ty] as f32 / current_dig_rate;
                let next_dig_rate = (100.0 + self.dig_power[next_ty] as f32) / 100.0;
                // 限制不超过下一层剩余量
                let next_dig_value = (base_remain * next_dig_rate).floor() as i32;
                ret[next_ty] = next_dig_value.min(self.dig_remain[self.current_onsen][next_ty]);
            } else {
                ret[ty] = dig_value.min(self.dig_remain[self.current_onsen][ty]);
            }
            Some(ret)
        } else {
            None
        }
    }

    /// 计算超回复触发概率
    /// 每档体力为50；超回复几率为记载的值；保底275  
    pub fn calc_super_prob(&self, vital_cost: i32) -> f64 {
        let threshold = 50.0;
        if self.dig_vital_cost + vital_cost >= 275 {
            1.0
        } else {
            let rank = (((self.dig_vital_cost + vital_cost) as f64 / threshold).floor() as usize).min(6);
            global!(ONSENDATA).super_probs[rank] as f64 / 100.0
        }
    }

    /// 训练后超回复触发判定
    /// 在训练消耗体力后调用，判定是否触发超回复
    pub fn update_super_on_vital_cost(&mut self, vital_cost: i32, rng: &mut StdRng) {
        if !self.bathing.is_super_ready && self.turn >= 2 {
            // 计算触发概率
            let prob = self.calc_super_prob(vital_cost);
            // 累加体力消耗
            self.dig_vital_cost += vital_cost;
            // 随机判定
            if rng.random_bool(prob) {
                info!("当前消耗体力: {}, 概率: {prob}", self.dig_vital_cost);
                self.add_super();
            }
        }
    }
    /// 执行挖掘, true为挖完了
    ///
    /// 每30点进度触发一次奖励，奖励内容根据地层类型决定：
    /// - 砂层: 速+2, 耐+1, 智+2, 体-3
    /// - 土层: 速+2, 力+1, 根+2, 体-3
    /// - 岩层: 耐+1, 力+2, 智+2, 体-3
    pub fn do_dig(&mut self, value: &[i32; 3], rng: &mut StdRng) -> bool {
        let dig_bonus = &global!(ONSENDATA).dig_bonus;
        let mut total_bonus = Array6::default();

        for i in 0..3 {
            if value[i] > 0 {
                // 计算奖励次数: floor((进度+挖掘量)/30) - floor(进度/30)
                let old_count = self.dig_progress[self.current_onsen][i] / 30;
                let new_count = (self.dig_progress[self.current_onsen][i] + value[i]) / 30;
                let bonus_count = new_count - old_count;

                // 累加奖励
                if bonus_count > 0 {
                    let bonus = &dig_bonus[i];
                    for j in 0..6 {
                        total_bonus[j] += bonus[j] * bonus_count;
                    }
                }

                // 更新进度
                self.dig_progress[self.current_onsen][i] += value[i];
            }
            self.dig_remain[self.current_onsen][i] -= value[i];
        }

        // 应用挖掘奖励
        if total_bonus != Array6::default() {
            let bonus_count = -total_bonus[5] / 3;
            let vital = total_bonus[5];
            total_bonus[5] = 0;
            info!("挖掘奖励x{bonus_count} >>");
            self.uma.add_value(&ActionValue {
                status_pt: total_bonus,
                vital,
                ..Default::default()
            });
            // 判断超回复
            self.update_super_on_vital_cost(-vital, rng);
        }

        if self.dig_remain[self.current_onsen].iter().all(|x| *x <= 0) {
            self.dig_remain[self.current_onsen] = [0, 0, 0];
            // 检查是否是新完成的温泉（避免初始温泉重复给奖励）
            if !self.onsen_state[self.current_onsen] {
                // 挖完了
                self.onsen_state[self.current_onsen] = true;
                self.add_ticket(self.grant_ticket_num());
                // 设置待处理选择标志（装备升级+源泉选择）
                self.pending_selection = true;
                info!("温泉挖掘完成，待处理装备升级和源泉选择");
            }
            true
        } else {
            false
        }
    }

    /// 生成Hint事件
    pub fn do_hint(&mut self, person_index: usize, has_friendship: bool, rng: &mut StdRng) -> Result<()> {
        let attr_prob = system_event_prob("hint_attr")?;
        let max_hint_per_card = global!(GAMECONSTANTS).max_hint_per_card;
        let hint_level = if person_index < 6 {
            (1 + self.deck[person_index].card_value().hint_level)
                .min(5) // <= 5
                .min(max_hint_per_card - self.deck[person_index].total_hints) // 单卡最大hint等级限制
        } else {
            1
        };

        let mut hint_event = if rng.random_bool(attr_prob as f64) || hint_level == 0 {
            // 红点提供属性. 超过最大Hint等级限制时只能提供属性
            EventData::hint_attr_event(self.persons[person_index].train_type as usize, person_index)?
        } else {
            // 红点提供技能
            self.deck[person_index].total_hints += hint_level;
            EventData::hint_skill_event(hint_level, person_index)
        };
        //hint_event.name = format!("{} - {}", hint_event.name, self.deck[person_index].short_name()?); // short_name is slow
        if !has_friendship {
            hint_event.choices[0].friendship = 0;
        }
        self.unresolved_events.push(hint_event);
        Ok(())
    }

    /// 执行训练，不包括剧本操作
    /// 返回训练是否成功，体力消耗
    pub fn do_train(&mut self, train: usize, rng: &mut StdRng) -> Result<(bool, i32)> {
        // sanity check 训练等级越界
        if train >= 5 {
            return Err(anyhow!("训练等级越界: {train}"));
        }
        info!(
            ">> {}训练 等级 {}",
            global!(GAMECONSTANTS).train_names[train],
            self.train_level(train)
        );

        let vital_before = self.uma.vital;
        let buffs = self.calc_training_buff(train)?;
        let failure_rate = self.calc_training_failure_rate(&buffs, train) / 100.0;

        if rng.random_bool(failure_rate as f64) {
            // 再判断一次，如果还失败就是大失败
            if rng.random_bool(failure_rate as f64) {
                warn!("训练大失败!");
                self.apply_event(system_event("training_fail_low")?, 0, rng)?;
                self.uma.flags.ill = true;
                self.uma.flags.bad_trainer = true;
            } else {
                warn!("训练失败!");
                self.apply_event(system_event("training_fail")?, 0, rng)?;
            }
            let vital_cost = (vital_before - self.uma.vital).max(0);
            return Ok((false, vital_cost));
        }

        // 训练成功
        let value = self.calc_training_value(&buffs, train)?;
        self.uma.add_value(&value);
        // 以计算值计算超回复消耗的体力，而非实际消耗值
        let vital_cost = -value.vital;
        // 增加训练次数
        self.train_level_count[train] += 1;
        // 增加羁绊
        let f = if self.uma.flags.aijiao { 9 } else { 7 };
        let mut hint_persons = vec![];
        let mut friend_clicked = false;
        let mut yayoi_clicked = false;
        let mut reporter_clicked = false;

        for person_index in self.distribution[train].clone() {
            // 跳过空位（-1 表示空位）
            if person_index < 0 {
                continue;
            }
            // 点击记者/理事长只加2羁绊，其他7
            if matches!(
                self.persons[person_index as usize].person_type,
                PersonType::Yayoi | PersonType::Reporter
            ) {
                self.add_friendship(person_index as usize, 2);
            } else {
                self.add_friendship(person_index as usize, f);
            }
            if self.persons[person_index as usize].is_hint {
                hint_persons.push(person_index);
            }
            match self.persons[person_index as usize].person_type {
                PersonType::ScenarioCard => friend_clicked = true,
                PersonType::Yayoi => yayoi_clicked = true,
                PersonType::Reporter => reporter_clicked = true,
                _ => {}
            };
        }

        // 生成加练或者红点事件
        if let Some(p) = hint_persons.choose(rng) {
            self.do_hint(*p as usize, true, rng)?;
        }
        let extra_train_prob = system_event_prob("extra_train")?;
        if !self.is_xiahesu() && rng.random_bool(extra_train_prob as f64) {
            self.unresolved_events.push(EventData::extra_training_event(train));
        }

        // 更新友人状态
        if friend_clicked {
            match self.friend.out_state {
                FriendOutState::UnClicked => {
                    self.friend.out_state = FriendOutState::BeforeUnlock;
                    let mut event = global_events().friend_events["first"].clone();
                    event.person_index = Some(self.friend.person_index as i32);
                    self.unresolved_events.push(event);
                }
                _ => {
                    let mut event = global_events().friend_events["click"].clone();
                    event.person_index = Some(self.friend.person_index as i32);
                    self.unresolved_events.push(event);
                }
            }
        }
        if yayoi_clicked {
            self.unresolved_events.push(system_event("yayoi_click")?.clone());
        }
        if reporter_clicked {
            let mut event = system_event("reporter_click")?.clone();
            event.choices[0].status_pt[train] = 2;
            self.unresolved_events.push(event);
        }
        return Ok((true, vital_cost));
    }

    /// 列出可选温泉Action，与list_actions类似
    pub fn list_actions_onsen_select(&self) -> Vec<OnsenAction> {
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
        actions
    }
    /// 执行选择温泉
    pub fn do_select_onsen<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions_onsen_select();
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

    /// 考虑剧本加成的Hint判定，也可以用于使用Hint buff后再对没有叹号的卡判定一次Hint
    pub fn onsen_distribute_hint(&mut self, rng: &mut StdRng) -> Result<()> {
        let base_hint_rate = global!(GAMECONSTANTS).base_hint_rate / 100.0;
        let hint_probs: Vec<_> = self
            .deck()
            .iter()
            .map(|card| card.card_value().hint_prob_increase + self.scenario_buff.onsen.hint_bonus)
            .collect();
        for person in self.persons_mut() {
            if person.person_type == PersonType::Card && !person.is_hint {
                let hint_prob = base_hint_rate * ((100 + hint_probs[person.person_index() as usize]) as f64 / 100.0);
                person.set_hint(rng.random_bool(hint_prob as f64));
            }
        }
        Ok(())
    }

    /// 执行PR动作
    ///
    /// # 效果
    /// 使用 pr_base_value 配置：五维各+6，技能点+15，体力-15
    ///
    /// # 返回
    /// 体力消耗值（用于超回复判定）
    pub fn do_pr(&mut self, _rng: &mut StdRng) -> Result<i32> {
        info!(">> PR");
        let pr_value = &global!(ONSENDATA).pr_base_value;
        self.uma.add_value(pr_value);
        self.add_ticket(1);
        Ok(-pr_value.vital)
    }

    /// 执行使用温泉券（温泉入浴）
    ///
    /// # 效果
    /// 1. 随机选择3张支援卡羁绊各+10
    /// 2. 体力恢复（根据温泉效果配置）
    /// 3. 干劲提升（根据温泉效果配置）
    /// 4. 超回复效果应用（如果 is_super_ready）
    /// 5. 消耗温泉券，设置 buff_remain_turn = 2
    pub fn do_use_ticket(&mut self, rng: &mut StdRng) -> Result<()> {
        // 检查温泉券
        if self.bathing.ticket_num <= 0 {
            return Err(anyhow!("没有温泉券"));
        }
        if self.bathing.buff_remain_turn > 0 {
            return Err(anyhow!("温泉效果尚未结束"));
        }

        info!(">> 使用温泉券");
        // 更新温泉Buff
        self.update_scenario_buff(true);
        // 1. 取羁绊最低的3人羁绊各+10
        let mut cards = self.persons[..6].to_vec();
        cards.sort_by_key(|x| x.friendship);
        for i in 0..3 {
            if cards[i].friendship < 100 {
                self.add_friendship(cards[i].person_index as usize, 10);
            }
        }

        // 2. 体力恢复（根据温泉效果配置）
        // Bug 1 修复：先判断超回复状态决定体力上限，再一次性应用所有恢复
        let vital_bonus = self.scenario_buff.onsen.vital;
        let super_effect = &global!(ONSENDATA).super_effect;

        // 计算体力上限和总恢复量
        // 注：max_vital 可能被友人事件提升（如112），超回复时使用临时上限150
        let (max_hp, total_recovery) = if self.bathing.is_super_ready {
            (super_effect.temp_max_vital, vital_bonus + super_effect.vital)
        } else {
            (self.uma.max_vital, vital_bonus)
        };

        // 一次性应用体力恢复
        self.uma.vital = (self.uma.vital + total_recovery).min(max_hp);
        info!("  体力+{} (={}, 上限{})", total_recovery, self.uma.vital, max_hp);

        // 3. 干劲提升（根据温泉效果配置）
        let motivation_bonus = self.scenario_buff.onsen.motivation;
        if motivation_bonus > 0 {
            self.uma.motivation = (self.uma.motivation + motivation_bonus).min(5);
            info!("  干劲+{} (={})", motivation_bonus, self.uma.motivation);
        }

        // 4. 超回复额外效果应用（体力已在上面处理）
        if self.bathing.is_super_ready {
            info!("{}", ">> 超回复发动！".cyan());

            // 技能点
            self.uma.skill_pt += super_effect.pt;
            info!("  超回复: 技能点+{} (={})", super_effect.pt, self.uma.skill_pt);

            // Hint加成（得到hint个叹号加成）
            for _i in 0..self.scenario_buff.hotel.hint + 2 {
                let person_index = (0..6)
                    .filter(|x| self.persons[*x].person_type == PersonType::Card)
                    .collect::<Vec<_>>();
                self.do_hint(*person_index.choose(rng).unwrap_or(&0), false, rng)?;
            }

            // 标记超回复已使用
            self.bathing.is_super = true;
            // Bug 3 修复：如果挖了传说泉，保持超回复可用
            if self.onsen_state[9] {
                info!(" 传说秘泉效果: 超回复状态保持");
            } else {
                self.bathing.is_super_ready = false;
            }
        }

        // 5. 消耗温泉券，设置buff持续2回合
        self.bathing.ticket_num -= 1;
        self.bathing.buff_remain_turn = 2;
        info!(
            "  温泉券剩余: {}, Buff持续: {}回合",
            self.bathing.ticket_num, self.bathing.buff_remain_turn
        );

        // 6. 秘汤汤驹效果：追加支援卡（split效果）
        if self.scenario_buff.onsen.split > 0 && self.deck_can_split {
            // 立即分配并显示更新的训练详情
            self.distribute_extra_supports(rng)?;
        }
        // 7. 如果有Hint率提升效果，重新对没有叹号的卡判定Hint
        if self.scenario_buff.onsen.hint_bonus > 0 {
            self.onsen_distribute_hint(rng)?;
        }
        info!("使用温泉券后训练:\n{}", self.explain_distribution()?);

        Ok(())
    }

    /// 分配追加支援卡到额外训练位置（秘汤汤驹效果）
    ///
    /// 在 Distribute 阶段调用，为 extra_support_indices 中的卡分配额外训练位置
    /// 每张卡分配到其当前未在的随机一个训练位置（不超过5人）
    pub fn distribute_extra_supports(&mut self, rng: &mut StdRng) -> Result<()> {
        // 检查温泉效果是否激活且有追加支援卡效果
        if self.bathing.buff_remain_turn == 0 || self.scenario_buff.onsen.split == 0 || !self.deck_can_split {
            return Ok(());
        }

        // 选择split张卡进行分身
        let split_person_indices = (0..6)
            .filter(|x|  // 是普通卡且已经出现在任意训练中
                self.persons[*x].person_type == PersonType::Card &&
                self.at_trains(*x as i32).iter().any(|t| *t))
            .choose_multiple(rng, self.scenario_buff.onsen.split as usize);

        if !split_person_indices.is_empty() {
            // 为每张选中的卡分配额外位置（计算得意率，存疑）
            for p in &split_person_indices {
                self.distribute_person(*p as i32, false, rng)?;
            }
            info!(">> 秘汤汤驹效果: 追加分配 {} 人", split_person_indices.len());
        }
        Ok(())
    }

    /// 执行选择挖掘温泉
    ///
    /// # 参数
    /// - `onsen_index`: 温泉索引
    pub fn do_select_dig(&mut self, onsen_index: usize) -> Result<()> {
        let onsen_info = &global!(ONSENDATA).onsen_info;
        // sanity check
        if onsen_index >= onsen_info.len() {
            return Err(anyhow!("温泉索引越界: {}", onsen_index));
        }
        if self.onsen_state[onsen_index] {
            return Err(anyhow!("该温泉已被挖掘完成: {}", onsen_info[onsen_index].name));
        }
        if self.turn < onsen_info[onsen_index].unlock_turn {
            return Err(anyhow!("该温泉尚未解锁: {}", onsen_info[onsen_index].name));
        }

        info!(">> 选择挖掘温泉: {}", onsen_info[onsen_index].name.cyan());
        self.current_onsen = onsen_index;

        Ok(())
    }

    /// 获取可升级的装备列表
    ///
    /// # 返回
    /// 等级小于6的装备类型索引列表 (0=砂, 1=土, 2=岩)
    /// 等级范围: 1-6，对应 dig_tool_level 的索引1-6
    pub fn get_upgradeable_equipment(&self) -> Vec<usize> {
        (0..3).filter(|&i| self.dig_level[i] < 6).collect()
    }

    /// 执行装备升级
    ///
    /// # 参数
    /// - `dig_type`: 装备类型 (0=砂, 1=土, 2=岩)
    ///
    /// 等级范围: 1-6，最大等级为6
    pub fn do_upgrade_equipment(&mut self, dig_type: usize) -> Result<()> {
        let dig_type_names = ["砂", "土", "岩"];
        // sanity check
        if dig_type >= 3 {
            return Err(anyhow!("装备类型越界: {}", dig_type));
        }
        if self.dig_level[dig_type] >= 6 {
            return Err(anyhow!("装备已达最高等级: {}", dig_type_names[dig_type]));
        }

        let old_level = self.dig_level[dig_type];
        self.dig_level[dig_type] += 1;
        info!(
            ">> 装备升级: {} Lv.{} -> Lv.{}",
            dig_type_names[dig_type], old_level, self.dig_level[dig_type]
        );

        Ok(())
    }

    /// 处理挖掘完成后的选择流程（装备升级+源泉选择）
    pub fn handle_pending_selection<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        if !self.pending_selection {
            return Ok(());
        }
        // 1. 源泉选择
        self.do_select_onsen(trainer, rng)?;

        // 2. 升级装备
        let upgradeable = self.get_upgradeable_equipment();

        if !upgradeable.is_empty() {
            let actions = upgradeable
                .iter()
                .map(|x| OnsenAction::Upgrade(*x as i32))
                .collect::<Vec<_>>();
            info!("选择要升级的装备:");
            let selection = trainer.select_action(self, &actions, rng)?;
            self.do_upgrade_equipment(upgradeable[selection])?;
        }

        // 3. 重置标志
        self.pending_selection = false;

        Ok(())
    }

    /// 更新休息心得
    pub fn update_refresh_mind(&mut self, rng: &mut StdRng) {
        let t = self.uma.flags.refresh_mind as usize;
        if t > 0 {
            info!("休息心得已持续 {t} 回合 -->");
            self.uma.add_value(&ActionValue { vital: 5, ..Default::default() });
            self.uma.flags.refresh_mind += 1;
            let end_prob = global!(GAMECONSTANTS).group_buff_end_prob[t.min(6)];
            if rng.random_bool(end_prob) {
                info!("{}", "休息心得结束".yellow());
                self.uma.flags.refresh_mind = 0;
            }
        }
    }

    /// 提取神经网络输入特征（1121 维）
    ///
    /// # 参数
    /// - `pending_choices`: 可选的事件选项列表（用于提取事件选项特征）
    ///
    /// # 返回
    /// 1121 维 f32 向量：
    /// - 全局信息（587 维）
    ///   - 搜索参数（6 维）
    ///   - 回合信息（78 维）
    ///   - 马娘属性（15 维）
    ///   - 体力与干劲（5 维）
    ///   - 训练数值（30 维）
    ///   - 失败率（5 维）
    ///   - **温泉剧本特定（140 维）** - 支持温泉选择学习
    ///   - 其他信息（61 维）
    ///   - 事件选项特征（113 维 = 1 + 8*14）
    ///   - 动作合法掩码（50 维）
    ///   - 比赛回合标记（78 维）
    ///   - 预留（6 维）
    /// - 支援卡信息（89 维 × 6 张 = 534 维）
    pub fn extract_nn_features(&self, pending_choices: Option<&[ActionValue]>) -> Vec<f32> {
        use crate::training_sample::{CHOICE_DIM, NN_CARD_DIM, NN_INPUT_DIM, POLICY_DIM};

        let mut features = vec![0.0_f32; NN_INPUT_DIM];
        let mut idx = 0;

        // ========== 全局信息（587 维） ==========

        // 1. 搜索参数（6 维）- 预留，当前填充 0
        idx += 6;

        // 2. 回合信息（78 维）- One-Hot 编码
        if (self.turn as usize) < 78 {
            features[idx + self.turn as usize] = 1.0;
        }
        idx += 78;

        // 3. 马娘属性（15 维）
        // 五维属性（归一化到 0-1）
        for i in 0..5 {
            features[idx + i] = self.uma.five_status[i] as f32 / 50.0;
        }
        // 五维属性上限
        for i in 0..5 {
            features[idx + 5 + i] = self.uma.five_status_limit[i] as f32 / 50.0;
        }
        // 剩余空间（limit - current）
        for i in 0..5 {
            let remain = self.uma.five_status_limit[i] - self.uma.five_status[i];
            features[idx + 10 + i] = remain as f32 / 50.0;
        }
        idx += 15;

        // 4. 体力与干劲（5 维）
        features[idx] = self.uma.vital as f32 / 100.0;
        features[idx + 1] = self.uma.max_vital as f32 / 100.0;
        features[idx + 2] = self.uma.motivation as f32 / 4.0;
        // 3-4: 预留
        idx += 5;

        // 5. 训练数值（30 维）- 5种训练 × 6维收益
        for train in 0..5 {
            if let Ok(buffs) = self.calc_training_buff(train) {
                if let Ok(gain) = self.calc_training_value(&buffs, train) {
                    features[idx + train * 6 + 0] = gain.status_pt[0] as f32 / 10.0; // 速度
                    features[idx + train * 6 + 1] = gain.status_pt[1] as f32 / 10.0; // 耐力
                    features[idx + train * 6 + 2] = gain.status_pt[2] as f32 / 10.0; // 力量
                    features[idx + train * 6 + 3] = gain.status_pt[3] as f32 / 10.0; // 根性
                    features[idx + train * 6 + 4] = gain.status_pt[4] as f32 / 10.0; // 智力
                    features[idx + train * 6 + 5] = gain.status_pt[5] as f32 / 10.0; // 技能点
                }
            }
        }
        idx += 30;

        // 6. 失败率（5 维）
        for train in 0..5 {
            if let Ok(buffs) = self.calc_training_buff(train) {
                let fail_rate = self.calc_training_failure_rate(&buffs, train);
                features[idx + train] = fail_rate / 100.0;
            }
        }
        idx += 5;

        // 7. 温泉剧本特定（140 维）- 支持温泉选择学习
        // 获取温泉静态数据
        let onsen_data = crate::gamedata::onsen::ONSENDATA.get();

        // ========== 7.1 全局温泉状态（20 维） ==========

        // 当前温泉索引 One-Hot（10 维）
        if self.current_onsen < 10 {
            features[idx + self.current_onsen] = 1.0;
        }

        // 已完成温泉数量（1 维）
        let completed_count = self.onsen_state.iter().filter(|&&x| x).count();
        features[idx + 10] = completed_count as f32 / 10.0;

        // 装备等级 [砂/土/岩]（3 维）
        features[idx + 11] = self.dig_level[0] as f32 / 6.0;
        features[idx + 12] = self.dig_level[1] as f32 / 6.0;
        features[idx + 13] = self.dig_level[2] as f32 / 6.0;

        // 挖掘力加成（3 维）- 根据五维属性计算
        // 参考 dig_stat_bonus_types: [[1,5,2], [4,1,3], [3,5,2]]
        // 砂层: 耐力(1) + 技能(5) + 力量(2)
        // 土层: 根性(4) + 耐力(1) + 根性(3)
        // 岩层: 根性(3) + 技能(5) + 力量(2)
        features[idx + 14] = self.dig_power[0] as f32 / 100.0;
        features[idx + 15] = self.dig_power[1] as f32 / 100.0;
        features[idx + 16] = self.dig_power[2] as f32 / 100.0;

        // 温泉券数量（1 维）
        features[idx + 17] = self.bathing.ticket_num as f32 / 5.0;

        // 超回复状态（1 维）
        features[idx + 18] = if self.bathing.is_super { 1.0 } else { 0.0 };

        // 累计体力消耗（1 维）- 用于超回复判断
        features[idx + 19] = self.dig_vital_cost as f32 / 200.0;

        idx += 20;

        // ========== 7.2 每个温泉的详细状态（12 维 × 10 = 120 维） ==========

        for i in 0..10 {
            let base = idx + i * 12;

            // 获取静态数据（如果可用）
            let (unlock_turn, dig_volume, effect_value, main_effect_type) = if let Some(data) = onsen_data {
                if let Some(info) = data.onsen_info.get(i) {
                    // 计算主效果类型（友情加成最高的属性）
                    let youqing = &info.effect.youqing;
                    let main_type = youqing
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, v)| *v)
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    // 计算效果价值（综合评估）
                    let value = info.effect.vital as f32 * 2.0
                        + youqing.iter().sum::<i32>() as f32
                        + info.effect.career_race_bonus as f32 * 0.5
                        + info.effect.hint_bonus as f32 * 0.3
                        + info.effect.split as f32 * 20.0;

                    (info.unlock_turn, info.dig_volume.clone(), value, main_type)
                } else {
                    (0, vec![0, 0, 0], 0.0, 0)
                }
            } else {
                (0, vec![0, 0, 0], 0.0, 0)
            };

            // 1. 是否已解锁（1 维）
            features[base] = if self.turn >= unlock_turn { 1.0 } else { 0.0 };

            // 2. 是否已完成（1 维）
            features[base + 1] = if self.onsen_state[i] { 1.0 } else { 0.0 };

            // 3. 当前挖掘进度 [砂/土/岩]（3 维）- 归一化到 0-1
            let progress = &self.dig_progress[i];
            for j in 0..3 {
                let vol = dig_volume.get(j).copied().unwrap_or(0);
                if vol > 0 {
                    features[base + 2 + j] = (progress[j] as f32 / vol as f32).min(1.0);
                } else {
                    features[base + 2 + j] = 1.0; // 无需挖掘视为已完成
                }
            }

            // 4. 剩余挖掘量 [砂/土/岩]（3 维）- 归一化
            for j in 0..3 {
                let vol = dig_volume.get(j).copied().unwrap_or(0);
                let remain = (vol - progress[j]).max(0);
                features[base + 5 + j] = remain as f32 / 500.0;
            }

            // 5. 主效果类型（1 维）- 归一化到 0-1
            features[base + 8] = main_effect_type as f32 / 5.0;

            // 6. 效果价值评估（1 维）- 归一化
            features[base + 9] = effect_value / 200.0;

            // 7. 挖掘难度评估（1 维）- 根据当前装备等级和挖掘量计算
            let total_vol: i32 = dig_volume.iter().sum();
            let avg_dig_power = (self.dig_power[0] + self.dig_power[1] + self.dig_power[2]) / 3;
            let difficulty = if avg_dig_power > 0 {
                total_vol as f32 / avg_dig_power as f32 / 20.0
            } else {
                1.0
            };
            features[base + 10] = difficulty.min(1.0);

            // 8. 预计完成回合数（1 维）- 根据当前挖掘力估算
            let total_remain: i32 = (0..3)
                .map(|j| {
                    let vol = dig_volume.get(j).copied().unwrap_or(0);
                    (vol - progress[j]).max(0)
                })
                .sum();
            let est_turns = if avg_dig_power > 0 {
                total_remain as f32 / avg_dig_power as f32
            } else {
                20.0
            };
            features[base + 11] = (est_turns / 20.0).min(1.0);
        }

        idx += 120;

        // 8. 其他全局信息（61 维）
        // 技能点
        features[idx] = self.uma.skill_pt as f32 / 100.0;
        // 干劲等级
        features[idx + 1] = self.uma.motivation as f32 / 4.0;
        // 挖掘数量
        features[idx + 2] = self.dig_count as f32 / 10.0;
        // 友人外出状态
        let friend_available = matches!(self.friend.out_state, crate::game::FriendOutState::AfterUnlock);
        features[idx + 3] = if friend_available { 1.0 } else { 0.0 };
        features[idx + 4] = self.friend.vital_bonus as f32 / 100.0;
        // 超回复准备状态
        features[idx + 5] = if self.bathing.is_super_ready { 1.0 } else { 0.0 };
        // Buff 剩余回合
        features[idx + 6] = self.bathing.buff_remain_turn as f32 / 10.0;
        // 预留 (7-60)
        idx += 61;

        // 9. 事件选项特征（113 维 = 1 + 8*14）
        if let Some(choices) = pending_choices {
            features[idx] = choices.len() as f32 / CHOICE_DIM as f32;

            for (i, choice) in choices.iter().take(CHOICE_DIM).enumerate() {
                let base = idx + 1 + i * 14;

                // 五维属性收益
                for j in 0..5 {
                    features[base + j] = choice.status_pt[j] as f32 / 10.0;
                }
                // 技能点收益
                features[base + 5] = choice.status_pt[5] as f32 / 10.0;
                // 体力变化
                features[base + 6] = choice.vital as f32 / 50.0;
                // 最大体力变化
                features[base + 7] = choice.max_vital as f32 / 10.0;
                // 干劲变化
                features[base + 8] = choice.motivation as f32 / 2.0;
                // 羁绊变化
                features[base + 9] = choice.friendship as f32 / 10.0;
                // Hint 等级变化
                features[base + 10] = choice.hint_level as f32 / 5.0;
                // 是否有属性收益
                let has_status = choice.status_pt.iter().take(5).any(|&x| x != 0);
                features[base + 11] = if has_status { 1.0 } else { 0.0 };
                // 是否有体力收益
                features[base + 12] = if choice.vital > 0 { 1.0 } else { 0.0 };
                // 预留
                features[base + 13] = 0.0;
            }
        }
        idx += 1 + CHOICE_DIM * 14; // 113 维

        // 10. 动作合法掩码（50 维）
        // 标记当前可选的动作
        if let Ok(actions) = self.list_actions() {
            for action in &actions {
                if let Some(action_idx) = crate::sample_collector::action_to_global_index(action) {
                    if action_idx < POLICY_DIM {
                        features[idx + action_idx] = 1.0;
                    }
                }
            }
        }
        idx += POLICY_DIM; // 50 维

        // 11. 比赛回合标记（78 维）
        // 标记所有比赛回合
        for turn in 0..78 {
            if self.uma.is_race_turn(turn as i32) {
                features[idx + turn] = 1.0;
            }
        }
        idx += 78;

        // 12. 预留（6 维）
        idx += 6;

        // ========== 支援卡信息（89 维 × 6 张 = 534 维） ==========
        // 布局（对齐 C++ getCardParamNNInputV1）：
        // - Person 信息（12 维）
        // - Card 参数（77 维）

        for card_idx in 0..6 {
            let base = idx + card_idx * NN_CARD_DIM;

            if let Some(person) = self.persons.get(card_idx) {
                // ===== Person 信息（12 维） =====

                // 羁绊状态（6 维）
                features[base] = person.friendship as f32 / 100.0;
                features[base + 1] = if person.friendship >= 60 { 1.0 } else { 0.0 };
                features[base + 2] = if person.friendship >= 80 { 1.0 } else { 0.0 };
                features[base + 3] = if person.friendship >= 100 { 1.0 } else { 0.0 };
                features[base + 4] = if person.is_hint { 1.0 } else { 0.0 };
                // 是否发光（检查当前回合是否有金技能闪光）
                let is_shining = (0..5).any(|train| self.is_shining_at(card_idx, train));
                features[base + 5] = if is_shining { 1.0 } else { 0.0 };

                // 位置信息（5 维）- 在哪个训练
                for train in 0..5 {
                    if self.distribution()[train].contains(&(card_idx as i32)) {
                        features[base + 6 + train] = 1.0;
                    }
                }

                // 预留（1 维）
                // features[base + 11] = 0.0;
            }

            // ===== Card 参数（77 维，从 base+12 开始） =====
            let card_base = base + 12;

            if let Some(card) = self.deck.get(card_idx) {
                let data = &card.data;
                let effect = &card.effect;

                // 卡片类型 One-Hot（7 维：0-4 对应速耐力根智，5=友人，6=其他）
                let card_type = data.card_type.min(6) as usize;
                if card_type < 7 {
                    features[card_base + card_type] = 1.0;
                }

                // 计算后的属性加成（5 维）- 使用 CardTrainingEffect
                features[card_base + 7] = effect.youqing / 100.0; // 友情加成
                features[card_base + 8] = effect.ganjing as f32 / 100.0; // 干劲加成
                features[card_base + 9] = effect.xunlian as f32 / 100.0; // 训练加成
                features[card_base + 10] = effect.wiz_vital_bonus as f32 / 100.0; // 智力体力恢复
                features[card_base + 11] = effect.deyilv / 100.0; // 得意率

                // Hint 相关（2 维）- 从 card_value 取基础值
                if let Some(cv) = data.card_value.first() {
                    features[card_base + 12] = cv.hint_level as f32 / 5.0;
                    features[card_base + 13] = cv.hint_prob_increase as f32 / 100.0;
                }

                // 失败率/体力消耗下降（2 维）
                features[card_base + 14] = effect.fail_rate_drop / 100.0;
                features[card_base + 15] = effect.vital_cost_drop / 100.0;

                // 副属性加成（6 维）
                for i in 0..6 {
                    features[card_base + 16 + i] = effect.bonus[i] as f32 / 10.0;
                }

                // 固有效果类型 One-Hot（35 维）- 简化处理
                // 使用 unique_effect_type 字段
                let unique_type = data.unique_effect_type as usize;
                if unique_type > 0 && unique_type < 35 {
                    features[card_base + 22 + unique_type] = 1.0;
                }
                // 22-56: 35 种固有效果类型

                // 固有效果数值（最多 5 维）
                for (i, &param) in data.unique_effect_param.iter().take(5).enumerate() {
                    features[card_base + 57 + i] = param as f32 / 100.0;
                }
                // 57-61: 固有效果参数

                // 稀有度和突破等级（2 维）
                features[card_base + 62] = data.rarity as f32 / 3.0;
                features[card_base + 63] = card.rank as f32 / 4.0;

                // 五维属性得意类型（5 维）
                for i in 0..5 {
                    if data.card_type == i {
                        features[card_base + 64 + i as usize] = 1.0;
                    }
                }

                // 是否友人卡（1 维）
                features[card_base + 69] = if data.card_type == 5 { 1.0 } else { 0.0 };

                // 事件效果提高（2 维）
                features[card_base + 70] = effect.event_effect_up as f32 / 100.0;
                features[card_base + 71] = effect.event_recovery_amount_up as f32 / 100.0;

                // 赛后加成（1 维）
                features[card_base + 72] = effect.saihou as f32 / 100.0;

                // 预留（4 维）
                // features[card_base + 73..77] 预留
            }
        }

        features
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
                    friend_state = match card.data.rarity {
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
            true
        } else if self.turn < self.max_turn() {
            // 下一个回合
            self.turn += 1;
            self.stage = OnsenTurnStage::Begin;
            // 检查自选比赛
            self.check_free_race()
        } else {
            false
        }
    }

    fn list_actions(&self) -> Result<Vec<Self::Action>> {
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
                if self.is_race_turn() {
                    Ok(vec![OnsenAction::Race])
                } else {
                    let mut actions = vec![
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
                        if self.turn >= 2 && self.turn < 72 {
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
            _ => Ok(vec![])
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
        // 判定超回复
        if let Some(vital_cost) = event.choices.get(choice).map(|value| value.vital)
            && vital_cost < 0
        {
            self.update_super_on_vital_cost(-vital_cost, rng);
        }
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
                info!("当前 limit: {:?}", self.uma.five_status_limit);
                for i in 0..5 {
                    self.uma.five_status_limit[i] = self.uma.five_status_limit[i].min(2800);
                }
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
                self.uma.flags.refresh_mind = 1;
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
        for i in 0..5 {
            let mut row = vec![];
            for train in 0..5 {
                if let Some(id) = dist[train].get(i)
                    && *id >= 0
                {
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
        let mut lines = vec![self.explain()?, table.to_string()];
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
            let mut upper_value = Array6::default();
            // 计算上层值=总-下层，不超过100
            for i in 0..6 {
                upper_value[i] = (total_value.status_pt[i] - base_value.status_pt[i]).min(100);
                total_value.status_pt[i] = base_value.status_pt[i] + upper_value[i];
            }
            /*
            info!(
                "训练: {train}, 下层: {:?}, 上层: {:?}",
                base_value.status_pt, upper_value
            );
            */
            Ok(total_value)
        } else {
            Ok(base_value)
        }
    }

    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        info!("-- 回合 {}-{:?} --", self.turn + 1, self.stage);
        match self.stage {
            OnsenTurnStage::Begin => {
                //println!("-----------------------------------------");
                //info!("{}", self.explain()?);
                // 处理上一回合挖掘完成后的选择（装备升级+源泉选择）
                if self.pending_selection && self.turn < 72 {
                    self.handle_pending_selection(trainer, rng)?;
                }
                // 年初，第三年九月下选择温泉（再次触发）
                if [24, 48, 65].contains(&self.turn) {
                    self.pending_selection = true;
                    self.handle_pending_selection(trainer, rng)?;
                }
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
                // 第二回合：初始温泉视为完成，触发装备升级+源泉选择
                if self.turn == 2 && self.current_onsen == 0 && self.onsen_state[0] {
                    // 复用 handle_pending_selection 的完整流程
                    self.pending_selection = true;
                    self.handle_pending_selection(trainer, rng)?;
                }
                // 新温泉解锁回合：触发装备升级+源泉选择
                let unlock_turns = [24, 48, 65];
                if unlock_turns.contains(&self.turn) {
                    info!("新温泉解锁，触发选择流程");
                    self.handle_pending_selection(trainer, rng)?;
                }
                // 休息心得
                if self.uma.flags.refresh_mind > 0 {
                    self.update_refresh_mind(rng);
                }
                // 执行回合前事件
                for event in &events {
                    self.run_event(event, trainer, rng)?;
                }
            }
            OnsenTurnStage::Distribute => {
                self.update_scenario_buff(false);
                if self.is_race_turn() {
                    self.reset_distribution();
                } else {
                    self.distribute_all(rng)?;
                    self.onsen_distribute_hint(rng)?;

                    // 秘汤汤驹效果：额外分配支援卡并重新显示训练详情
                    self.distribute_extra_supports(rng)?;
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
                // buff回合倒计时
                if self.bathing.buff_remain_turn > 0 {
                    self.bathing.buff_remain_turn -= 1;
                    if self.bathing.buff_remain_turn == 0 {
                        self.bathing.is_super = false;
                        info!("温泉效果已结束");
                    }
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

    fn on_simulation_end<T: Trainer<Self>>(&mut self, _trainer: &T, _rng: &mut StdRng) -> Result<()> {
        info!(">> 育成结束，触发最终奖励事件");
        /*
        // 查找回合78事件 (ぴょいや！大宴会！)
        let onsen_data = global!(ONSENDATA);
        if let Some(event) = onsen_data.scenario_events.iter().find(|e| e.id == 400012021) {
            self.run_event(event, trainer, rng)?;
        } else {
            warn!("未找到育成结束事件 (id=400012021)");
        }
        */
        Ok(())
    }
}

