use std::fmt::Display;

use anyhow::Result;
use colored::Colorize;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::{
    game::{base::*, onsen::game::OnsenGame, *},
    gamedata::{GAMECONSTANTS, onsen::ONSENDATA}
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
    Dig(i32)
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
                    // 训练成功后执行挖掘
                    if let Some(dig_value) = game.calc_dig_value(self) {
                        game.do_dig(&dig_value);
                    }
                }
                // 超回复判定
                if vital_cost > 0 {
                    game.update_super_on_vital_cost(vital_cost, rng);
                }
                Ok(())
            }
            // ========== 比赛动作 ==========
            OnsenAction::Race => {
                // 调用基础比赛逻辑
                BaseAction::do_race(&mut game.base, rng)?;
                // 执行挖掘（目标比赛25点，非目标比赛15点）
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value);
                }
                // 注：比赛的体力效果通过事件处理，apply时不触发超回复
                Ok(())
            }
            // ========== 休息/普通外出/治病 ==========
            OnsenAction::Sleep | OnsenAction::NormalOuting | OnsenAction::Clinic => {
                self.as_base_action().expect("as_base_action").apply(game, rng)?;
                // 执行挖掘 (Clinic 挖掘值为0，但仍需调用)
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value);
                }
                Ok(())
            }
            // ========== 友人外出动作 ==========
            OnsenAction::FriendOuting => {
                // 调用基础友人外出逻辑（事件放入unresolved_events队列）
                BaseAction::do_friend_outing(&mut game.base, rng)?;
                // 执行挖掘（基础点25）
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value);
                }
                // 注：友人外出的超回复是通过友人事件触发的（事件ID 809050011-809050015）
                // 不在这里处理，而是在 apply_event 中通过 add_super() 直接获得
                Ok(())
            }
            // ========== PR动作 ==========
            OnsenAction::PR => {
                let vital_cost = game.do_pr(rng)?;
                // 执行挖掘（基础点10）
                if let Some(dig_value) = game.calc_dig_value(self) {
                    game.do_dig(&dig_value);
                }
                // 超回复判定
                if vital_cost > 0 {
                    game.update_super_on_vital_cost(vital_cost, rng);
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
            OnsenAction::Dig(onsen_index) => {
                game.do_select_dig(*onsen_index as usize)?;
                Ok(())
            }
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
