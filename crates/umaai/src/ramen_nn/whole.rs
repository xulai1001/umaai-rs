//! `nn`：全部动作决策由网络直接给出，不做任何搜索（`onnx` feature）
//!
//! [`RamenNnTrainer`] 本身不提供 [`Trainer::last_decision`]，客户端没有摘要就不输出普通
//! 训练回合的推荐。本层只补这一件事：记下每次动作决策的下标、候选描述与真实来源。
//!
//! 摘要不带评分（`candidate_scores` 为空）：policy logits 不是终局分，填进评分字段会让
//! luck 行读出没有量纲的数。事件选项与友人事件仍由 [`RamenNnTrainer`] 内部的手写策略处理。

use std::sync::Mutex;

use anyhow::{Result, anyhow};
use rand::prelude::StdRng;
use umasim::{
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{EventChoice, EventData},
    output::{
        DecisionInfo,
        decision::{SOURCE_RAMEN_HANDWRITTEN_STAGE, SOURCE_RAMEN_NN, SOURCE_RAMEN_RACE_GATE}
    },
    trainer::{NnVia, RamenNnTrainer}
};

use crate::scenario::ramen::fallback_decision;

/// 整局直接走网络的客户端决策器
pub struct WholeNnTrainer {
    /// 真正做决策的网络训练员
    nn: RamenNnTrainer,
    /// 最近一次动作决策的摘要；每次调用开头先清空，任何早退都不会留下上一步的摘要
    last: Mutex<Option<DecisionInfo>>
}

impl WholeNnTrainer {
    /// 包装一个已加载好的网络训练员
    pub fn new(nn: RamenNnTrainer) -> Self {
        Self {
            nn,
            last: Mutex::new(None)
        }
    }

    /// 清空摘要槽（锁被毒化时照清）
    fn clear_last(&self) {
        let mut slot = match self.last.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner()
        };
        *slot = None;
    }

    /// 定案来源 → 协议上的来源标签
    pub fn source_of(via: NnVia) -> &'static str {
        match via {
            NnVia::Network => SOURCE_RAMEN_NN,
            NnVia::RaceGate => SOURCE_RAMEN_RACE_GATE,
            NnVia::Handwritten => SOURCE_RAMEN_HANDWRITTEN_STAGE
        }
    }
}

impl Trainer<RamenGame> for WholeNnTrainer {
    /// 直接走网络（或自选比赛守门），不跑搜索，并记下本次决策的摘要
    ///
    /// # 错误
    ///
    /// 候选为空、推理失败、任一候选无法落格，或摘要锁被毒化时报错，不会退回搜索或手写。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        self.clear_last();
        let pick = self.nn.select_action_labeled(game, actions, rng)?;
        let info = fallback_decision(actions, pick.index, &game.stage).with_source(Self::source_of(pick.via));
        *self
            .last
            .lock()
            .map_err(|_| anyhow!("整局网络决策摘要锁被毒化"))? = Some(info);
        Ok(pick.index)
    }

    /// 事件选项转发网络训练员内部的手写策略
    ///
    /// # 错误
    ///
    /// 内部策略报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_last();
        self.nn.select_choice(game, choices, rng)
    }

    /// 事件选项（新接口）转发网络训练员内部的手写策略
    ///
    /// # 错误
    ///
    /// 内部策略报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_last();
        self.nn.select_event_choice(game, event, choices, rng)
    }

    /// 本次动作决策的摘要；刚转发过事件时为 `None`
    fn last_decision(&self) -> Option<DecisionInfo> {
        self.last.lock().ok()?.clone()
    }

    /// 恒为 `None`：没有搜索就没有评分分解
    fn last_breakdown(&self) -> Option<String> {
        None
    }
}
