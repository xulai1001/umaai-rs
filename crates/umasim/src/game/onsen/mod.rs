use colored::Colorize;
use enum_iterator::Sequence;
use serde::{Deserialize, Serialize};

use crate::gamedata::onsen::{HotelEffect, OnsenEffect};

pub mod action;
pub mod game;

/// 温泉剧本的局中Buff
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OnsenBuff {
    /// 当前的温泉Buff组合
    pub onsen: OnsenEffect,
    /// 当前的旅馆效果
    pub hotel: HotelEffect
}

/// 温泉buff信息
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BathingInfo {
    /// 温泉券数量
    pub ticket_num: i32,
    /// buff剩余回合数
    pub buff_remain_turn: i32,
    /// buff是否超回复
    pub is_super: bool,
    /// 下一个buff是否超回复
    pub is_super_ready: bool
}

impl BathingInfo {
    pub fn explain(&self) -> String {
        let buff_text = if self.buff_remain_turn > 0 {
            format!("Buff剩余回合: {}, 超回复: {}", self.buff_remain_turn, self.is_super)
        } else {
            "Buff未生效".to_string()
        };
        let super_text = if self.is_super_ready {
            "超回复".bright_yellow()
        } else {
            "普通".bright_black()
        };
        format!("温泉券: {}, {}, {}", self.ticket_num, buff_text, super_text)
    }
}

/// 回合阶段，选择温泉不作为回合阶段
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, Sequence)]
pub enum OnsenTurnStage {
    /// 1. 回合开始，随机事件
    #[default]
    Begin,
    /// 2. 分配人头
    Distribute,
    // --- 可操作部分
    /// 3. 选择使用温泉券
    Bathing,
    /// 4. 选择训练或比赛
    Train,
    /// 5. 回合后事件
    AfterTrain
}
