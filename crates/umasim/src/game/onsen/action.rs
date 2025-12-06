use std::fmt::Display;

use anyhow::Result;
use colored::Colorize;
use log::info;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::{
    game::{base::*, onsen::game::OnsenGame, *},
    gamedata::{GAMECONSTANTS, onsen::ONSENDATA},
    utils::system_event
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OnsenAction {
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
    Clinic,
    /// PR
    PR,
    /// 使用/不使用温泉券
    UseTicket(bool),
    /// 选择温泉
    Dig(i32),
    /// 升级工具
    Upgrade(i32)
}

impl Display for OnsenAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnsenAction::Train(x) => write!(f, "{}训练", global!(GAMECONSTANTS).train_names[*x as usize]),
            OnsenAction::Race => write!(f, "比赛"),
            OnsenAction::Sleep => write!(f, "休息"),
            OnsenAction::FriendOuting => write!(f, "友人出行"),
            OnsenAction::NormalOuting => write!(f, "普通出行"),
            OnsenAction::Clinic => write!(f, "治病"),
            OnsenAction::PR => write!(f, "PR"),
            OnsenAction::UseTicket(b) => {
                if *b {
                    write!(f, "使用温泉券")
                } else {
                    write!(f, "不使用温泉券")
                }
            }
            OnsenAction::Dig(x) => {
                let name = &global!(ONSENDATA).onsen_info[*x as usize].name;
                write!(f, "挖掘 {}", name.cyan())
            }
            OnsenAction::Upgrade(x) => {
                let name = &global!(ONSENDATA).dig_tool_name[*x as usize];
                write!(f, "升级 {}", name.cyan())
            }
        }
    }
}

impl ActionEnum for OnsenAction {
    type Game = OnsenGame;

    fn apply(&self, game: &mut Self::Game, rng: &mut StdRng) -> Result<()> {
        match self {
            // ========== 训练动作 ==========
            OnsenAction::Train(train_type) => {
                let (success, vital_cost) = game.do_train(*train_type as usize, rng)?;
                if success {
                    // 判定超回复。失败时已经在事件里判定了
                    if vital_cost > 0 {
                        game.update_super_on_vital_cost(vital_cost, rng);
                    }
                    // 训练成功后执行挖掘
                    if let Some(dig_value) = game.calc_dig_value(self) {
                        game.do_dig(&dig_value, rng);
                    }
                }
                Ok(())
            }
            // ========== 比赛动作 ==========
            OnsenAction::Race => {
                let race_bonus = (100 + game.uma.race_bonus) as f32 / 100.0;
                let buff_bonus = if game.bathing.buff_remain_turn > 0 {
                    (100 + game.scenario_buff.onsen.career_race_bonus) as f32 / 100.0
                } else {
                    1.0
                };
                if game.uma.is_race_turn(game.turn)? {
                    let mut scenario_bonus = if game.turn < 72 {
                        global!(ONSENDATA).career_race_multiplier * buff_bonus
                    } else {
                        1.0
                    };
                    if game.uma.chara_id() == 1063 && game.turn > 12 {
                        // 狄杜斯出道赛后：再增加50%
                        scenario_bonus *= 1.5;
                    }
                    info!(
                        ">> 生涯比赛 - 比赛加成: {}, 剧本加成: {scenario_bonus}x",
                        game.uma.race_bonus
                    );
                    let mut event = system_event("race_career")?.clone();
                    // 事件面板乘算比赛加成
                    event.choices[0].map_status(|x| (x as f32 * race_bonus * scenario_bonus).round() as i32);
                    game.unresolved_events.push(event);
                } else {
                    let grade = global!(GAMECONSTANTS).race_grades[game.turn as usize];
                    info!(">> 自选比赛 G{grade} - 比赛加成: {}", game.uma.race_bonus);
                    let event_name = format!("race_g{grade}");
                    let mut event = system_event(&event_name)?.clone();
                    // 事件面板乘算比赛加成
                    event.choices[0].map_status(|x| (x as f32 * race_bonus).round() as i32);
                    game.unresolved_events.push(event);
                }
                // 执行挖掘（目标比赛25点，非目标比赛15点）
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value, rng);
                }
                // 注：比赛的体力效果通过事件处理，超回复已经在事件里判定
                Ok(())
            }
            // ========== 休息/普通外出/治病 ==========
            OnsenAction::Sleep | OnsenAction::NormalOuting | OnsenAction::Clinic => {
                self.as_base_action().expect("as_base_action").apply(game, rng)?;
                // 执行挖掘 (Clinic 挖掘值为0，但仍需调用)
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value, rng);
                }
                Ok(())
            }
            // ========== 友人外出动作 ==========
            OnsenAction::FriendOuting => {
                // 调用基础友人外出逻辑（事件放入unresolved_events队列）
                BaseAction::do_friend_outing(&mut game.base, rng)?;
                // 执行挖掘（基础点25）
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value, rng);
                }
                // 注：友人外出的超回复在 apply_event 中通过 add_super() 直接获得
                Ok(())
            }
            // ========== PR动作 ==========
            OnsenAction::PR => {
                let vital_cost = game.do_pr(rng)?;
                // 超回复判定
                game.update_super_on_vital_cost(vital_cost, rng);
                // 执行挖掘（基础点10）
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value, rng);
                }
                Ok(())
            }
            // ========== 使用温泉券动作 ==========
            OnsenAction::UseTicket(use_ticket) => {
                if *use_ticket {
                    game.do_use_ticket(rng)?;
                }
                // 不使用温泉券时直接跳过
                Ok(())
            }
            // ========== 挖掘动作（选择温泉） ==========
            OnsenAction::Dig(onsen_index) => game.do_select_dig(*onsen_index as usize),
            // ========== 升级工具动作 ==========
            OnsenAction::Upgrade(tool) => game.do_upgrade_equipment(*tool as usize)
        }
    }

    fn as_base_action(&self) -> Option<BaseAction> {
        match self {
            OnsenAction::Train(x) => Some(BaseAction::Train(*x)),
            OnsenAction::Race => Some(BaseAction::Race),
            OnsenAction::Sleep => Some(BaseAction::Sleep),
            OnsenAction::FriendOuting => Some(BaseAction::FriendOuting),
            OnsenAction::NormalOuting => Some(BaseAction::NormalOuting),
            OnsenAction::Clinic => Some(BaseAction::Clinic),
            _ => None
        }
    }
}
