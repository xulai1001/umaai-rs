use std::fmt::Display;

use anyhow::{Result, anyhow};
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};
use serde::{Deserialize, Serialize};

use crate::{
    game::{base::*, *},
    gamedata::{ActionValue, GAMECONSTANTS}
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BaseAction {
    /// 训练
    Train(i32),
    /// 比赛
    Race,
    /// 休息
    Sleep,
    /// 友人出行
    FriendOuting,
    /// 普通出行
    NormalOuting,
    /// 治病
    Clinic
}

impl Display for BaseAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BaseAction::Train(x) => write!(f, "{}训练", global!(GAMECONSTANTS).train_names[*x as usize]),
            BaseAction::Race => write!(f, "比赛"),
            BaseAction::Sleep => write!(f, "休息"),
            BaseAction::FriendOuting => write!(f, "友人出行"),
            BaseAction::NormalOuting => write!(f, "普通出行"),
            BaseAction::Clinic => write!(f, "治病")
        }
    }
}

// 具体的操作都是 &Game -> Result<Game>的映射，产生一个新Game对象
// 在具体类型中可以直接调用BaseAction的方法得到BaseGame的变换，然后再把结果转为自己的Game类型
// 但是do_train需要具体类型自己实现，因为BaseGame没有实现Game Trait，不能使用Game Trait里计算训练数值的方法
impl BaseAction {
    pub fn do_race(game: &mut BaseGame, _rng: &mut StdRng) -> Result<()> {
        let race_bonus = (100 + game.uma.race_bonus) as f32 / 100.0;
        if game.uma.is_race_turn(game.turn) {
            info!(">> 生涯比赛 - 比赛加成: {}", game.uma.race_bonus);
            let mut event = system_event("race_career")?.clone();
            // 事件面板乘算比赛加成
            event.choices[0].map_status(|x| (x as f32 * race_bonus).round() as i32);
            game.unresolved_events.push(event);
            game.uma.set_race(game.turn);
        } else {
            let grade = global!(GAMECONSTANTS).race_grades[game.turn as usize];
            info!(">> 自选比赛 G{grade} - 比赛加成: {}", game.uma.race_bonus);
            let event_name = format!("race_g{grade}");
            let mut event = system_event(&event_name)?.clone();
            // 事件面板乘算比赛加成
            event.choices[0].map_status(|x| (x as f32 * race_bonus).round() as i32);
            game.unresolved_events.push(event);
            game.uma.set_race(game.turn);
        }
        Ok(())
    }

    pub fn do_sleep(game: &mut BaseGame, rng: &mut StdRng) -> Result<()> {
        if game.is_xiahesu() {
            let value = ActionValue {
                vital: 40,
                motivation: 1,
                ..Default::default()
            };
            game.uma.add_value(&value);
        } else {
            let weights = WeightedIndex::new(&global!(GAMECONSTANTS).rest_probs)?;
            let mut value = ActionValue::default();
            match weights.sample(rng) {
                0 => {
                    info!(">> 休息 - 寝不足");
                    value.vital = 30;
                    game.uma.add_value(&value);
                }
                1 => {
                    info!(">> 休息 - 正常");
                    value.vital = 50;
                    game.uma.add_value(&value);
                }
                _ => {
                    info!(">> 休息 - 大成功");
                    value.vital = 70;
                    game.uma.add_value(&value);
                }
            }
        }
        Ok(())
    }

    pub fn do_friend_outing(game: &mut BaseGame, _rng: &mut StdRng) -> Result<()> {
        let mut which = 0;
        while which < 5 && game.friend.out_used[which] {
            which += 1;
        }
        if which < 5 {
            info!(">> 友人出行 #{}", which + 1);
            let mut event = global_events().friend_events[&(1 + which).to_string()].clone();
            event.person_index = Some(game.friend.person_index as i32);
            game.friend.out_used[which] = true;
            game.unresolved_events.push(event); // 不直接启动事件，而是放到回合后用run_event执行以带入Trainer选择
            Ok(())
        } else {
            Err(anyhow!("友人出行越界: {which}"))
        }
    }

    pub fn do_normal_outing(game: &mut BaseGame, rng: &mut StdRng) -> Result<()> {
        let event_name = format!("normal_outing_{}", rng.random_range(1..=3));
        game.unresolved_events.push(system_event(&event_name)?.clone());
        // 抓娃娃
        if game.turn >= 24 && !game.uma.flags.doll && rng.random_bool(system_event_prob("doll")?) {
            game.unresolved_events.push(system_event("normal_outing_doll")?.clone());
            game.uma.flags.doll = true;
        }

        Ok(())
    }

    pub fn do_clinic(game: &mut BaseGame, _rng: &mut StdRng) -> Result<()> {
        info!(">> 治病");
        let value = ActionValue { vital: 20, ..Default::default() };
        game.uma.flags.ill = false;
        game.uma.flags.bad_trainer = false;
        game.uma.add_value(&value);
        Ok(())
    }
}

/// 实现基础动作对基础游戏状态的变换，作为实际剧本动作的一部分
impl ActionEnum for BaseAction {
    type Game = BaseGame;
    fn apply(&self, game: &mut Self::Game, rng: &mut StdRng) -> Result<()> {
        match self {
            BaseAction::Train(_) => Err(anyhow!("BaseGame没有实现训练，需要在具体类型中重新实现")),
            BaseAction::Race => BaseAction::do_race(game, rng),
            BaseAction::Sleep => BaseAction::do_sleep(game, rng),
            BaseAction::FriendOuting => BaseAction::do_friend_outing(game, rng),
            BaseAction::NormalOuting => BaseAction::do_normal_outing(game, rng),
            BaseAction::Clinic => BaseAction::do_clinic(game, rng)
        }
    }

    fn as_base_action(&self) -> Option<BaseAction> {
        Some(self.clone())
    }
}
