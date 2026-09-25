//! `mcts_nn_hint`：搜索照常执行，网络只给参考（`onnx` feature）
//!
//! 包着一个 [`RamenMctsTrainer`]：所有调用原样转发，执行结果、随机流、输出与 `mcts`
//! 完全一致。每个多候选动作决策上，网络在随机流副本上另算一次推荐，挂到决策的
//! `scenario_extra.nn_hint`。网络推理失败只打警告、不挂参考，不影响执行。

use std::sync::Mutex;

use anyhow::{Result, anyhow};
use log::warn;
use rand::prelude::StdRng;
use serde_json::{Map, Value, json};
use umasim::{
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{EventChoice, EventData},
    output::{DecisionInfo, decision::NN_HINT_KEY},
    trainer::{RamenMctsTrainer, RamenNnTrainer}
};

use crate::scenario::ramen::{fallback_decision, synthesizes_fallback};

/// 搜索执行、网络给参考的客户端决策器
pub struct NnHintTrainer {
    /// 真正执行决策的搜索训练员
    mcts: RamenMctsTrainer,
    /// 只给参考的网络
    nn: RamenNnTrainer,
    /// 最近一次动作决策挂好参考后的摘要
    ///
    /// 为 `None` 时 [`Trainer::last_decision`] 直接读搜索训练员的摘要。每次调用开头先清空，
    /// 避免上一步的参考串到这一步。
    hinted: Mutex<Option<DecisionInfo>>
}

impl NnHintTrainer {
    /// 装配参考决策器
    pub fn new(mcts: RamenMctsTrainer, nn: RamenNnTrainer) -> Self {
        Self {
            mcts,
            nn,
            hinted: Mutex::new(None)
        }
    }

    /// 清空参考摘要槽（锁被毒化时照清）
    fn clear_hinted(&self) {
        let mut slot = match self.hinted.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner()
        };
        *slot = None;
    }

    /// 把网络的参考推荐挂到决策的 `scenario_extra.nn_hint`
    ///
    /// 只记录网络选了什么、是否与执行推荐一致，不改任何评分字段。
    pub fn attach_hint(mut info: DecisionInfo, actions: &[RamenAction], nn_idx: usize, picked: usize) -> DecisionInfo {
        let payload = json!({
            "choice": actions.get(nn_idx).map(ToString::to_string),
            "same_as_executed": nn_idx == picked
        });
        match info.scenario_extra {
            Some(Value::Object(ref mut map)) => {
                map.insert(NN_HINT_KEY.to_string(), payload);
            }
            _ => {
                let mut map = Map::new();
                map.insert(NN_HINT_KEY.to_string(), payload);
                info.scenario_extra = Some(Value::Object(map));
            }
        }
        info
    }
}

impl Trainer<RamenGame> for NnHintTrainer {
    /// 搜索训练员吃真实随机流执行；多候选时网络在副本上另算一次参考
    ///
    /// 参考只挂在「`mcts` 本来就会输出决策」的地方：搜索给了摘要，或调用方本来就会合成
    /// 手写 fallback 的阶段。其余阶段 `mcts` 不输出，这里也不输出。
    ///
    /// # 错误
    ///
    /// 搜索训练员报错时原样返回，或参考摘要锁被毒化时报错。网络推理失败不报错。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        self.clear_hinted();
        let picked = self.mcts.select_action(game, actions, rng)?;
        if actions.len() < 2 {
            return Ok(picked);
        }
        let base = self
            .mcts
            .last_decision()
            .or_else(|| synthesizes_fallback(game).then(|| fallback_decision(actions, picked, &game.stage)));
        let Some(base) = base else {
            return Ok(picked);
        };
        let mut hint_rng = rng.clone();
        match self.nn.select_action_labeled(game, actions, &mut hint_rng) {
            Ok(nn_pick) => {
                *self
                    .hinted
                    .lock()
                    .map_err(|_| anyhow!("网络参考摘要锁被毒化"))? =
                    Some(Self::attach_hint(base, actions, nn_pick.index, picked));
            }
            Err(e) => warn!("神经网络参考计算失败，本步不显示参考：{e:#}")
        }
        Ok(picked)
    }

    /// 事件选项转发搜索训练员
    ///
    /// # 错误
    ///
    /// 转发的训练员报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_hinted();
        self.mcts.select_choice(game, choices, rng)
    }

    /// 事件选项（新接口）转发搜索训练员
    ///
    /// # 错误
    ///
    /// 转发的训练员报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_hinted();
        self.mcts.select_event_choice(game, event, choices, rng)
    }

    /// 挂了参考时返回挂好参考的摘要，否则返回搜索训练员自己的摘要
    fn last_decision(&self) -> Option<DecisionInfo> {
        match self.hinted.lock().ok()?.clone() {
            Some(info) => Some(info),
            None => self.mcts.last_decision()
        }
    }

    /// 透传搜索训练员的评分分解
    fn last_breakdown(&self) -> Option<String> {
        self.mcts.last_breakdown()
    }
}
