use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use anyhow::Result;
#[cfg(feature = "cli")]
use inquire::Select;
use log::info;
use rand::{Rng, prelude::StdRng, seq::SliceRandom};

use crate::{
    game::{ActionEnum, BaseAction, Game, Trainer},
    gamedata::EventChoice
};

pub mod handwritten_trainer;
pub mod local_ramen_parity;
pub mod local_ramen_trainer;
pub mod logging_trainer;
pub mod mcts_trainer;
pub mod ramen_handwritten_trainer;
pub mod ramen_mcts_trainer;
//pub mod mean_filter_collector_trainer;
//pub mod neural_net_trainer;

pub use handwritten_trainer::HandwrittenTrainer;
pub use local_ramen_parity::RestoredRamenTrainer as RecommendedRamenTrainer;
pub use local_ramen_trainer::LocalRamenTrainer;
pub use logging_trainer::LoggingTrainer;
pub use mcts_trainer::MctsTrainer;
pub use ramen_handwritten_trainer::RamenHandwrittenTrainer;
pub use ramen_mcts_trainer::{RamenMctsTrainer, RamenSearchStages, RamenSelection};
//pub use mean_filter_collector_trainer::MeanFilterCollectorTrainer;
//pub use neural_net_trainer::NeuralNetTrainer;

/// 猴子训练师
pub struct RandomTrainer;

impl<G: Game> Trainer<G> for RandomTrainer {
    fn select_action(&self, game: &G, actions: &[<G as Game>::Action], rng: &mut StdRng) -> Result<usize> {
        let mut random_index: Vec<_> = (0..actions.len()).collect();
        let mut ret = None;
        random_index.shuffle(rng);
        for i in &random_index {
            if game.uma().vital < 45 {
                if actions[*i].as_base_action() == Some(BaseAction::Sleep) {
                    ret = Some(*i);
                    break;
                }
            } else if game.uma().motivation < 5 {
                if matches!(
                    actions[*i].as_base_action(),
                    Some(BaseAction::NormalOuting) | Some(BaseAction::FriendOuting)
                ) {
                    ret = Some(*i);
                    break;
                }
            } else if matches!(actions[*i].as_base_action(), Some(BaseAction::Train(_))) {
                ret = Some(*i);
                break;
            }
        }
        if ret.is_none() {
            for i in &random_index {
                if let Some(ra) = any_ramen_action(&actions[*i]) {
                    if ra.ramen.is_some() || ra.special_targets.is_some_and(|t| t.iter().any(|&x| x > 0)) {
                        ret = Some(*i);
                        break;
                    }
                }
            }
        }
        let ret = ret.unwrap_or(random_index[0]);
        info!("吗喽训练员选择：{:?}", actions[ret]);
        Ok(ret)
    }

    fn select_choice(&self, _game: &G, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        let ret = rng.random_range(0..choices.len());
        let explain: Vec<String> = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect();
        info!("当前选项: {}, 随机选择选项 {}", explain.join(" / "), ret + 1);
        Ok(ret)
    }
}

fn any_ramen_action<A>(_action: &A) -> Option<&crate::game::ramen::RamenAction> {
    None
}

pub struct ManualTrainer {
    pub mock_inputs: Rc<RefCell<VecDeque<String>>>,
    pub fallback: FallbackMode
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackMode {
    Interactive,
    PickFirst
}

impl Default for ManualTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl ManualTrainer {
    pub fn new() -> Self {
        Self {
            mock_inputs: Rc::new(RefCell::new(VecDeque::new())),
            fallback: FallbackMode::Interactive
        }
    }
    pub fn with_mock_inputs(inputs: Vec<String>) -> Self {
        Self {
            mock_inputs: Rc::new(RefCell::new(inputs.into_iter().collect())),
            fallback: FallbackMode::PickFirst
        }
    }
    fn pop_mock_input(&self) -> Option<String> {
        self.mock_inputs.borrow_mut().pop_front()
    }
    fn fallback_pick_first(&self, len: usize, item_desc: &str) -> Result<usize> {
        if len == 0 {
            return Err(anyhow::anyhow!("{item_desc} 候选为空"));
        }
        Ok(0)
    }
}

impl<G: Game> Trainer<G> for ManualTrainer {
    fn select_action(&self, _game: &G, actions: &[<G as Game>::Action], _rng: &mut StdRng) -> Result<usize> {
        if let Some(input) = self.pop_mock_input() {
            return actions
                .iter()
                .position(|x| x.to_string() == input)
                .ok_or_else(|| anyhow::anyhow!("mock 输入未匹配到候选动作: {input}"));
        }
        match self.fallback {
            FallbackMode::PickFirst => self.fallback_pick_first(actions.len(), "动作"),
            #[cfg(feature = "cli")]
            FallbackMode::Interactive => {
                let selected = Select::new("请选择:", actions.to_vec())
                    .with_page_size(actions.len())
                    .prompt()?;
                actions
                    .iter()
                    .position(|x| *x == selected)
                    .ok_or_else(|| anyhow::anyhow!("未找到该动作: {selected}"))
            }
            #[cfg(not(feature = "cli"))]
            FallbackMode::Interactive => Err(anyhow::anyhow!(
                "ManualTrainer::Interactive 需要 cli feature；请改用 with_mock_inputs"
            ))
        }
    }

    fn select_choice(&self, _game: &G, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        let explain: Vec<String> = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect();
        if let Some(input) = self.pop_mock_input() {
            return explain
                .iter()
                .position(|x| x == &input)
                .ok_or_else(|| anyhow::anyhow!("mock 输入未匹配到候选选项: {input}"));
        }
        match self.fallback {
            FallbackMode::PickFirst => self.fallback_pick_first(explain.len(), "事件选项"),
            #[cfg(feature = "cli")]
            FallbackMode::Interactive => {
                let selected = Select::new("请选择:", explain.clone()).prompt()?;
                explain
                    .iter()
                    .position(|x| *x == selected)
                    .ok_or_else(|| anyhow::anyhow!("未找到该动作: {selected}"))
            }
            #[cfg(not(feature = "cli"))]
            FallbackMode::Interactive => Err(anyhow::anyhow!(
                "ManualTrainer::Interactive 需要 cli feature；请改用 with_mock_inputs"
            ))
        }
    }
}
