use std::fmt::Display;

use anyhow::{Result, anyhow};
use colored::Colorize;
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};
use serde::{Deserialize, Serialize};

use crate::{
    game::{base::*, onsen::game::OnsenGame, *},
    gamedata::{ActionValue, GAMECONSTANTS, onsen::ONSENDATA}
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
            OnsenAction::Sleep | OnsenAction::NormalOuting | OnsenAction::Clinic => {
                self.as_base_action().expect("as_base_action").apply(game, rng)?;
                // todo: 增加挖掘值
                Ok(())
            }
            _ => unimplemented!()
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
