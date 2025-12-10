use anyhow::Result;
use inquire::Select;
use log::info;
use rand::{Rng, prelude::StdRng, seq::SliceRandom};

use crate::{
    game::{ActionEnum, BaseAction, Game, Trainer},
    gamedata::ActionValue
};

// 导出手写逻辑训练员、数据收集训练员、神经网络训练员和 MCTS 训练员
pub mod collector_trainer;
pub mod handwritten_trainer;
pub mod mcts_trainer;
pub mod neural_net_trainer;

pub use collector_trainer::CollectorTrainer;
pub use handwritten_trainer::HandwrittenTrainer;
pub use mcts_trainer::MctsTrainer;
pub use neural_net_trainer::NeuralNetTrainer;

/// 猴子训练师
pub struct RandomTrainer;

impl<G: Game> Trainer<G> for RandomTrainer {
    fn select_action(&self, game: &G, actions: &[<G as Game>::Action], rng: &mut StdRng) -> Result<usize> {
        let mut random_index: Vec<_> = (0..actions.len()).collect();
        let mut ret = 0;
        random_index.shuffle(rng);
        for i in random_index {
            // 优先休息，回心情，训练。都不满足就随机第一个
            if game.uma().vital < 45 {
                if actions[i].as_base_action() == Some(BaseAction::Sleep) {
                    ret = i;
                    break;
                }
            } else if game.uma().motivation < 5 {
                if matches!(
                    actions[i].as_base_action(),
                    Some(BaseAction::NormalOuting) | Some(BaseAction::FriendOuting)
                ) {
                    ret = i;
                    break;
                }
            } else {
                if matches!(actions[i].as_base_action(), Some(BaseAction::Train(_))) {
                    ret = i;
                    break;
                }
            }
        }
        info!("吗喽训练员选择：{:?}", actions[ret]);
        Ok(ret)
    }

    fn select_choice(&self, _game: &G, choices: &[ActionValue], rng: &mut StdRng) -> Result<usize> {
        let ret = rng.random_range(0..choices.len());
        info!("当前选项: {:?}, 随机选择选项 {}", choices, ret + 1);
        Ok(ret)
    }
}

/// 手动训练师
pub struct ManualTrainer;

impl<G: Game> Trainer<G> for ManualTrainer {
    fn select_action(&self, _game: &G, actions: &[<G as Game>::Action], _rng: &mut StdRng) -> Result<usize> {
        let selected = Select::new("请选择:", actions.to_vec())
            .with_page_size(actions.len())
            .prompt()?;
        actions
            .iter()
            .position(|x| *x == selected)
            .ok_or_else(|| anyhow::anyhow!("未找到该动作: {selected}"))
    }

    fn select_choice(&self, _game: &G, choices: &[ActionValue], _rng: &mut StdRng) -> Result<usize> {
        let selected = Select::new("请选择:", choices.to_vec()).prompt()?;
        choices
            .iter()
            .position(|x| *x == selected)
            .ok_or_else(|| anyhow::anyhow!("未找到该选项: {selected}"))
    }
}
