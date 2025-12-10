use std::fmt::{Debug, Display};

use anyhow::{Result, anyhow};
use log::{info, warn};
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

use super::PersonType;
use crate::{
    game::{BaseAction, CardTrainingEffect, SupportCard, Uma},
    gamedata::{ActionValue, EventData, GAMECONSTANTS},
    global
};
// Game为核心特性，
// ActionEnum 执行动作，修改Game状态
// Trainer 选择动作
// 对事件的处理由Game自己进行

/// 训练人头特性，用于随机分配
pub trait Person: Debug + Clone + PartialEq + Default {
    /// person type getter
    fn person_type(&self) -> PersonType;

    /// person index getter
    fn person_index(&self) -> i32;

    /// train type getter
    fn train_type(&self) -> i32;

    /// friendship getter
    fn friendship(&self) -> i32;

    /// hint getter
    fn hint(&self) -> bool;

    /// hint setter
    fn set_hint(&mut self, hint: bool);

    /// provided: 是否为友人，团队，记者或者理事长
    fn is_friend(&self) -> bool {
        self.train_type() > 4 || matches!(self.person_type(), PersonType::Reporter | PersonType::Yayoi)
    }

    /// 是否为剧本友人
    fn is_scenario_card(&self) -> bool {
        self.person_type() == PersonType::ScenarioCard
    }
}

/// 会改变Game状态的主动选项
pub trait ActionEnum: Debug + Display + Clone + PartialEq {
    /// 操作的对象类型，不一定要实现Game Trait
    type Game;

    /// visitor，调用具体动作
    fn apply(&self, game: &mut Self::Game, rng: &mut StdRng) -> Result<()>;

    /// 尝试转变为BaseAction以获取基础行动类型
    fn as_base_action(&self) -> Option<BaseAction> {
        None
    }
}

/// 游戏状态类型需要实现的Trait，不包括初始化
pub trait Game: Clone {
    type Person: Person;
    type Action: ActionEnum<Game = Self>;

    // 回合相关
    fn turn(&self) -> i32;
    /// 最大回合数 getter
    fn max_turn(&self) -> i32;
    /// 下一阶段。如果已经结束，返回false
    fn next(&mut self) -> bool;
    /// 模拟当前Stage
    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()>;
    /// provided: 模拟到游戏结束
    fn run_full_game<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        self.run_stage(trainer, rng)?;
        while self.next() {
            self.run_stage(trainer, rng)?;
        }
        // 触发育成结束奖励
        self.on_simulation_end(trainer, rng)?;
        Ok(())
    }
    /// 育成结束时的处理（如最终奖励）
    /// 默认实现为空，由具体剧本覆盖
    fn on_simulation_end<T: Trainer<Self>>(&mut self, _trainer: &T, _rng: &mut StdRng) -> Result<()> {
        Ok(())
    }
    // 动作相关
    /// 获取当前可能的可控行动
    fn list_actions(&self) -> Result<Vec<Self::Action>>;
    /// 生成当前回合的事件
    fn generate_events(&self, rng: &mut StdRng) -> Vec<EventData>;
    /// 应用事件效果，一些特殊事件需要用到rng和Result
    fn apply_event(&mut self, event: &EventData, choice: usize, rng: &mut StdRng) -> Result<()>;
    /// 执行事件，如果有选项，交给Trainer去决定
    fn run_event<T: Trainer<Self>>(&mut self, event: &EventData, trainer: &T, rng: &mut StdRng) -> Result<()> {
        info!("+ 事件: #{} {}", event.id, event.name);
        if event.choices.len() > 1 {
            let selection = if let Some(probs) = &event.random_choice_prob {
                // 随机选择选项
                let weights = WeightedIndex::new(probs)?;
                weights.sample(rng)
            } else {
                // 训练员选择选项
                for (index, choice) in event.choices.iter().enumerate() {
                    info!("选项 {}: {}", index + 1, choice.explain());
                }
                trainer.select_choice(self, &event.choices, rng)?
            };
            self.apply_event(&event, selection, rng)
        } else {
            self.apply_event(&event, 0, rng)
        }
    }
    /// provided: 列出本回合触发的事件
    fn list_turn_events(&self, events: &[EventData]) -> Vec<EventData> {
        events
            .iter()
            .filter_map(|e| {
                if e.start_turn == self.turn() {
                    Some(e.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    /// provided: 执行指定动作
    fn apply_action(&mut self, action: &Self::Action, rng: &mut StdRng) -> Result<()> {
        action.apply(self, rng)
    }
    /// provided: 列出动作，交给训练员判定并执行
    fn list_and_apply_action<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions()?;
        if !actions.is_empty() {
            let selection = trainer.select_action(self, &actions, rng)?;
            self.apply_action(&actions[selection], rng)?;
        }
        Ok(())
    }
    // 人头分配相关
    /// persons getter
    fn persons(&self) -> &[Self::Person];
    /// persons mut
    fn persons_mut(&mut self) -> &mut [Self::Person];
    /// 初始化人头
    fn init_persons(&mut self) -> Result<()>;
    /// 已经初始化的人头是否能出现在训练（如记者）
    fn person_is_available(&self, person_index: usize) -> bool {
        match self.persons()[person_index].person_type() {
            PersonType::ScenarioCard => self.turn() >= 2,
            PersonType::Reporter => self.turn() >= 13,
            _ => true
        }
    }
    /// distribution getter
    fn distribution(&self) -> &Vec<Vec<i32>>;
    /// distribution mut
    fn distribution_mut(&mut self) -> &mut Vec<Vec<i32>>;
    /// absent_rate_drop getter
    fn absent_rate_drop(&self) -> i32;
    /// 计算得意率，同时修改卡片计算状态所以要mut
    fn deyilv(&mut self, person_index: i32) -> Result<f32>;
    /// 团队卡是否可以闪彩，不考虑多个团卡的情况
    fn has_group_buff(&self) -> bool;
    /// 显示分布信息
    fn explain_distribution(&self) -> Result<String>;
    /// 重置分布和叹号
    fn reset_distribution(&mut self) {
        self.distribution_mut().clear();
        for _ in 0..5 {
            self.distribution_mut().push(vec![]);
        }
        for p in self.persons_mut() {
            p.set_hint(false);
        }
    }
    /// 追加分配一个在persons里已经存在的人头, -1为不在
    /// 如果要新加角色 需要手动添加到persons里
    fn distribute_person(&mut self, person_index: i32, allow_absent: bool, rng: &mut StdRng) -> Result<i32> {
        let person = self.persons()[person_index as usize].clone();
        let train_type = person.train_type() as usize;
        // 计算不在率
        let mut absent_rate = match person.person_type() {
            PersonType::Card => 50 - self.absent_rate_drop(),
            PersonType::Yayoi | PersonType::Reporter => 200,
            _ => 100 - self.absent_rate_drop()
        };
        if !allow_absent {
            absent_rate = 0;
        }
        // 计算得意率权重
        let mut weights = [100, 100, 100, 100, 100, absent_rate];
        let mut real_deyilv = 0;
        if train_type <= 4 {
            real_deyilv = self.deyilv(person_index)? as i32;
            weights[train_type] += real_deyilv;
        }
        let weights_sum = 500 + absent_rate + real_deyilv;
        // 先判断是否不在
        if rng.random_bool(absent_rate as f64 / weights_sum as f64) {
            Ok(-1)
        } else {
            let dist = WeightedIndex::new(&weights[0..5])?;
            // 尝试分配
            let d = self.distribution();
            let mut ok = false;
            let mut retries = 0;
            let mut train = 0;
            while !ok && retries < 10 {
                train = dist.sample(rng);
                retries += 1;
                // 不能多于5人或出现同样人头
                if d[train].len() >= 5 || d[train].contains(&person_index) {
                    continue;
                }
                // 每个训练只能出现一个友人
                if person.is_friend() && d[train].iter().any(|index| self.persons()[*index as usize].is_friend()) {
                    continue;
                }
                ok = true;
            }
            if !ok {
                warn!("分配角色#{person_index}失败");
                Ok(-1)
            } else {
                self.distribution_mut()[train as usize].push(person_index);
                Ok(train as i32)
            }
        }
    }
    /// 重新分配所有人头
    fn distribute_all(&mut self, rng: &mut StdRng) -> Result<()> {
        let sequence = vec![
            PersonType::Yayoi,
            PersonType::Reporter,
            PersonType::ScenarioCard,
            PersonType::TeamCard,
            PersonType::Card,
            PersonType::Npc,
        ];
        self.reset_distribution();
        for ty in sequence {
            for i in 0..self.persons().len() {
                if self.persons()[i].person_type() == ty && self.person_is_available(i) {
                    self.distribute_person(i as i32, true, rng)?;
                }
            }
        }
        Ok(())
    }
    /// 分配Hint. 注意同一个卡的不同分身会同时触发Hint
    fn distribute_hint(&mut self, rng: &mut StdRng) -> Result<()> {
        let base_hint_rate = global!(GAMECONSTANTS).base_hint_rate / 100.0;
        let hint_probs: Vec<_> = self
            .deck()
            .iter()
            .map(|card| card.card_value().expect("card_value").hint_prob_increase)
            .collect();
        for person in self.persons_mut() {
            if person.person_type() == PersonType::Card {
                let hint_prob = base_hint_rate * ((100 + hint_probs[person.person_index() as usize]) as f64 / 100.0);
                person.set_hint(rng.random_bool(hint_prob as f64));
            }
        }
        Ok(())
    }
    // provided: 指定人头出现在训练中的位置
    fn at_trains(&self, person_index: i32) -> Vec<bool> {
        self.distribution()
            .iter()
            .map(|train| train.contains(&person_index))
            .collect()
    }
    /// provided: 指定人头如果在指定位置是否会闪彩 train 0-4 速耐力根智 >=5暂时不考虑  
    /// 非默认实现需要依赖于一部分剧本Buff，所以要在Game里判断
    fn is_shining_at(&self, person_index: usize, train: usize) -> bool {
        let person = &self.persons()[person_index];
        match person.person_type() {
            PersonType::Card => person.train_type() == train as i32 && person.friendship() >= 80,
            PersonType::TeamCard => self.has_group_buff(),
            // 默认实现中其他卡不能闪彩
            _ => false
        }
    }
    /// provided: 指定训练的彩圈个数
    fn shining_count(&self, train: usize) -> usize {
        self.distribution()[train]
            .iter()
            .filter(|index| **index >= 0)  // 过滤掉 -1（空位）
            .filter(|index| self.is_shining_at(**index as usize, train))
            .count()
    }
    // 训练相关
    /// 设施等级 getter
    fn train_level(&self, train: usize) -> usize;
    /// uma getter
    fn uma(&self) -> &Uma;
    /// uma mut getter
    fn uma_mut(&mut self) -> &mut Uma;
    /// deck getter
    fn deck(&self) -> &Vec<SupportCard>;
    /// provided: 计算来自支援卡的训练buff
    fn calc_training_buff(&self, train: usize) -> Result<CardTrainingEffect> {
        self.default_calc_training_buff(train)
    }

    /// 如果calc_training_buff被重写，仍然可以调用这里的默认方法
    fn default_calc_training_buff(&self, train: usize) -> Result<CardTrainingEffect> {
        let mut ret = CardTrainingEffect::default();
        if train >= 5 {
            return Err(anyhow!("训练类型错误: {train}"));
        }
        for index in &self.distribution()[train] {
            if *index >= 0 && *index < 6 {
                let card = &self.deck()[*index as usize];
                let (mut effect, _) = card.calc_training_effect(self, train as i32)?;
                if !self.is_shining_at(*index as usize, train) {
                    effect.youqing = 0.0;
                }
                ret = ret.add(&effect);
            }
        }
        Ok(ret)
    }

    /// 可重写: 计算训练属性
    fn calc_training_value(&self, buffs: &CardTrainingEffect, train: usize) -> Result<ActionValue> {
        self.default_calc_training_value(buffs, train)
    }
    /// provided: 计算训练属性
    fn default_calc_training_value(&self, buffs: &CardTrainingEffect, train: usize) -> Result<ActionValue> {
        let cons = global!(GAMECONSTANTS);
        let train_level = self.train_level(train) - 1; // 返回1-5处理成0-4
        if train >= 5 {
            return Err(anyhow!("训练类型错误: {train}"));
        }
        // 人数, 排除掉理事长和记者
        let person_count = self.distribution()[train]
            .iter()
            .filter(|p| **p >= 0 && **p != 6 && **p != 7)
            .count();
        // 基础值
        let basic_value = &cons.training_basic_value[train][train_level];
        let basic_motivation = ((self.uma().motivation - 3) * 10) as f32;
        // 成长率
        let b = &self.uma().five_status_bonus;
        let status_bonus = [b[0], b[1], b[2], b[3], b[4], 0];
        let mut ret = ActionValue::default();
        // 副属性
        for i in 0..6 {
            if basic_value[i] > 0 {
                ret.status_pt[i] = basic_value[i] + buffs.bonus[i];
            }
        }
        ret.vital = basic_value[6];
        // 直接计算。假设buffs里已经算好中间加成
        for i in 0..6 {
            if basic_value[i] > 0 {
                let real_value = ret.status_pt[i] as f32
                    * (1.0 + 0.01 * buffs.youqing as f32)
                    * (1.0 + 0.01 * basic_motivation * (1.0 + 0.01 * buffs.ganjing as f32))
                    * (1.0 + 0.01 * buffs.xunlian as f32)
                    * (1.0 + 0.05 * person_count as f32)
                    * (1.0 + 0.01 * status_bonus[i] as f32);
                ret.status_pt[i] = real_value.floor() as i32;
                //warn!("Train: {train}, i: {i}, real: {real_value}, ret: {}", ret.status_pt[i]);
            }
        }
        // 智力回体
        if train == 4 && buffs.youqing > 0.0 {
            ret.vital += buffs.wiz_vital_bonus;
        }
        // 体力消耗降低
        if ret.vital < 0 {
            ret.vital = (ret.vital as f32 * (1.0 - 0.01 * buffs.vital_cost_drop)) as i32;
        }
        //warn!("Train: {train}, buffs: {}, basic_value: {basic_value:?}, status_bonus: {status_bonus:?}, ret: {ret:?}", buffs.explain());
        Ok(ret)
    }

    // 粗略拟合的训练失败率，二次函数 A*(x0-x)^2+B*(x0-x)
    fn calc_training_failure_rate(&self, buffs: &CardTrainingEffect, train: usize) -> f32 {
        let x0 = global!(GAMECONSTANTS).training_vital_threshold[train][self.train_level(train) - 1];
        let vital = self.uma().vital as f32;
        // 失败率修正
        let bias = if self.uma().flags.good_trainer {
            -2.0
        } else if self.uma().flags.bad_trainer {
            2.0
        } else {
            0.0
        };
        // 原始失败率最大99%
        let mut f = if vital < x0 {
            (100.0 - vital) * (x0 - vital) / 40.0
        } else {
            0.0
        }
        .min(99.0)
        .max(0.0);
        // 如果有不擅长练习，失败率可能达到100%
        f = (f * (100.0 - buffs.fail_rate_drop) / 100.0 + bias).min(100.0).max(0.0);
        f
    }
}

pub trait Trainer<G: Game> {
    /// 选择动作
    fn select_action(&self, game: &G, actions: &[<G as Game>::Action], rng: &mut StdRng) -> Result<usize>;
    /// 选择事件选项
    fn select_choice(&self, game: &G, choices: &[ActionValue], rng: &mut StdRng) -> Result<usize>;
}
