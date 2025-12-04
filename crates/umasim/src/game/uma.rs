use std::default::Default;

use anyhow::Result;
use colored::Colorize;
use log::info;
use serde::{Deserialize, Serialize};

use crate::{
    explain::Explain,
    gamedata::{ActionValue, GAMECONSTANTS, GAMEDATA, UmaData},
    global,
    utils::*
};

/// 训练中的马娘状态，剧本通用
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UmaFlags {
    /// 切者
    pub qiezhe: bool,
    /// 爱娇
    pub aijiao: bool,
    /// 擅长训练
    pub good_trainer: bool,
    /// 不擅长训练
    pub bad_trainer: bool,
    /// 正向思考
    pub positive_thinking: bool,
    /// 休息心得
    pub refresh_mind: bool,
    /// 幸运体质
    pub lucky: bool,
    /// 是否抓过娃娃
    pub doll: bool,
    /// 是否生病
    pub ill: bool
}

impl UmaFlags {
    pub fn explain(&self) -> String {
        let mut s = String::new();
        if self.qiezhe {
            s += &format!("{}", "切者 ".bright_green());
        }
        if self.aijiao {
            s += "爱娇 ";
        }
        if self.good_trainer {
            s += "擅长训练 ";
        }
        if self.bad_trainer {
            s += "不擅长训练 ";
        }
        if self.positive_thinking {
            s += "正向思考 ";
        }
        if self.refresh_mind {
            s += "休息心得 ";
        }
        if self.lucky {
            s += "幸运体质 ";
        }
        if self.doll {
            s += "抓过娃娃 ";
        }
        if self.ill {
            s += "*生病 ";
        }
        s
    }
}

/// 训练中的马娘信息，剧本通用（固定为5星）
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Uma {
    /// 马娘编号
    pub uma_id: u32,
    /// 体力
    pub vital: i32,
    /// 最大体力
    pub max_vital: i32,
    /// 干劲 [1, 5]
    pub motivation: i32,
    /// 当前属性。1200以上不减半
    pub five_status: Array5,
    /// 属性加成
    pub five_status_bonus: Array5,
    /// 属性上限
    pub five_status_limit: Array5,
    /// 剩余技能点
    pub skill_pt: i32,
    /// 已学技能评分
    pub skill_score: i32,
    /// 总共打折级数
    pub total_hints: i32,
    /// 比赛加成
    pub race_bonus: i32,
    /// Buff状态
    pub flags: UmaFlags
}

impl Uma {
    pub fn get_data(&self) -> Result<&UmaData> {
        global!(GAMEDATA).get_uma(self.uma_id)
    }

    /// 角色ID（高4位）
    pub fn chara_id(&self) -> u32 {
        self.uma_id / 100
    }

    pub fn explain(&self) -> Result<String> {
        let data = self.get_data()?;
        let ret = format!(
            "{} 体力 {}/{} {} {} {}PT{} Hint{} 赛后{}",
            data.short_name(),
            self.vital,
            self.max_vital,
            Explain::motivation(self.motivation),
            self.flags.explain(),
            Explain::five_status_cutted(&self.five_status),
            self.skill_pt,
            self.total_hints,
            self.race_bonus
        );
        Ok(ret)
    }

    pub fn new(id: u32) -> Result<Self> {
        let gamedata = global!(GAMEDATA);
        let cons = global!(GAMECONSTANTS);
        let data = gamedata.get_uma(id)?;
        Ok(Self {
            uma_id: id,
            vital: 100,
            max_vital: 100,
            motivation: 3,
            five_status: data.five_status_initial.clone(),
            five_status_bonus: data.five_status_bonus.clone(),
            five_status_limit: cons.five_status_limit_base.clone(),
            skill_score: 510, // 固有按5星计算,
            total_hints: 21,  // 按全部初始技能3级打折计算
            ..Default::default()
        })
    }

    pub fn is_race_turn(&self, turn: i32) -> Result<bool> {
        Ok(self.get_data()?.races.contains(&turn) || vec![73, 75, 77].contains(&turn))
    }

    /// 计算技能点和总Hint等级换算得到的总pt数，不包括已学习的技能
    pub fn total_pt(&self) -> i32 {
        (self.skill_pt as f32 + self.total_hints as f32 * global!(GAMECONSTANTS).hint_pt_rate).floor() as i32
    }

    pub fn calc_score(&self) -> i32 {
        let cons = global!(GAMECONSTANTS);
        // 技能分
        let mut score = self.skill_score + (self.total_pt() as f32 * cons.pt_score_rate) as i32;
        for i in 0..5 {
            let status = self.five_status[i].min(self.five_status_limit[i]).max(0) as usize;
            score += cons.five_status_final_score[status];
        }
        score
    }

    pub fn add_value(&mut self, action: &ActionValue) -> &mut Self {
        info!("{}", action.explain().bright_black());
        for i in 0..5 {
            self.five_status[i] = (self.five_status[i] + action.status_pt[i]).min(self.five_status_limit[i]);
        }
        self.skill_pt += action.status_pt[5];
        self.motivation = (self.motivation + action.motivation).max(1).min(5);
        self.max_vital += action.max_vital;
        self.vital = (self.vital + action.vital).min(self.max_vital).max(0);
        self.total_hints += action.hint_level;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gamedata::init_global;

    #[test]
    fn test_uma() -> Result<()> {
        init_logger("debug")?;
        init_global()?;

        let uma = Uma::new(101901)?;
        println!("{}", uma.explain()?);
        Ok(())
    }
}
