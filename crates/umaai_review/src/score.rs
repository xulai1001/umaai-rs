//! 终局评分与等级换算（文档 §3.4）
//!
//! 快照里没有最终评分（`skillScore` 恒 0），需自行计算：**五维查表 + PT 折算**。
//! 口径与 `umasim::game::Uma::calc_score` 同源（`status_final_score` 查表 +
//! `total_pt × pt_score_rate` + `skill_score`），等级换算直接复用
//! `GameConstants::get_rank_name`（与 `rank.csv` 同表）。
//!
//! 切者（能人，状态 7）×1.1 / 小切（勤勉好学，状态 40）×1.04 的 PT 项加成
//! 与模拟侧同步计入（2026-10-01 起 `calc_score` 生效）：快照路径直接读
//! `baseGame.isQieZhe` / `isXiaoQie`；终局帧不带该标志，由调用方从末快照
//! 代入（见 `digest::build`）。
//!
//! ⚠ 已学技能分数无法从包内还原（`skillScore` 恒 0）→ `final_score` 是 **AI 端
//! 按实际终局五维与技能点的估算**：PT 折算高于实际买技能的得分，故通常
//! **略高于**小黑板与实际分数（用户口径 2026-10-07）；该前提写进 digest.context。

use umaai::protocol::{FinalScorePayload, GameStatusBase};
use umasim::{game::{Uma, UmaFlags}, global, gamedata::GAMECONSTANTS, utils::Array5};

/// 终局评分（`Uma::calc_score` 同源口径；需已 `gdata::init`）
///
/// 切者/小切按 `baseGame.isQieZhe` / `isXiaoQie` 计入 PT 项加成。
pub fn final_score(base: &GameStatusBase) -> i32 {
    let uma = Uma {
        five_status: base.five_status,
        five_status_limit: base.five_status_limit,
        skill_pt: base.skill_pt,
        skill_score: base.skill_score,
        total_hints: base.total_hints,
        flags: UmaFlags {
            qiezhe: base.is_qiezhe,
            xiaoqie: base.is_xiao_qie,
            ..Default::default()
        },
        ..Default::default()
    };
    uma.calc_score()
}

/// 终局评分（真机终局帧口径；需已 `gdata::init`）
///
/// 输入是「育成结束·点技能前」那一帧（`game{id}_final.json`）：五维已含全部结局
/// 事件（育成结束 `401407` / 通用 `5011` / 友人结束）与末回合比赛奖励，故比分末
/// 快照估算高约 2700 分。`skill_score` 仍暂未随帧下发（恒 0）；`total_hints`
/// 自插件 2026-10 起随帧下发（旧帧缺省 0），与 [`final_score`] 同口径。
///
/// 帧内**无切者/小切标志**：由调用方从末份快照的 `isQieZhe` / `isXiaoQie` 代入
/// （快照缺失时传 `(false, false)`，加成不计）。
pub fn final_score_from_frame(p: &FinalScorePayload, qiezhe: bool, xiaoqie: bool) -> i32 {
    let uma = Uma {
        five_status: p.five_status,
        five_status_limit: p.five_status_limit,
        skill_pt: p.skill_pt,
        skill_score: 0,
        total_hints: p.total_hints,
        flags: UmaFlags { qiezhe, xiaoqie, ..Default::default() },
        ..Default::default()
    };
    uma.calc_score()
}

/// 评分 → 等级名（`GameConstants::get_rank_name`，与 rank.csv 同表；需已 init）
pub fn rank_name(score: i32) -> String {
    global!(GAMECONSTANTS).get_rank_name(score)
}

/// 显示值减半阈值（小黑板口径：真实值超过该值的部分减半显示）
const DISPLAY_STATUS_THRESHOLD: i32 = 1200;

/// 单维显示值换算（小黑板口径）：真实值 > 1200 时超出部分减半
///
/// `display = (real - 1200) / 2 + 1200` iff real > 1200，否则 display = real；
/// 整除向下取整。评分与运气分不受此换算影响。
pub fn display_status(real: i32) -> i32 {
    if real > DISPLAY_STATUS_THRESHOLD {
        (real - DISPLAY_STATUS_THRESHOLD) / 2 + DISPLAY_STATUS_THRESHOLD
    } else {
        real
    }
}

/// 五维数组显示值换算（逐维 [`display_status`]）
pub fn display_status_array(five: Array5) -> Array5 {
    let mut out = five;
    for v in out.iter_mut() {
        *v = display_status(*v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 显示值换算：阈值内原值、超阈值减半、数组分维独立换算
    #[test]
    fn test_display_status() {
        println!("1200 → {}", display_status(1200));
        assert_eq!(display_status(1199), 1199, "阈值内原值");
        assert_eq!(display_status(1200), 1200, "恰好阈值不减半");
        println!("3276 → {}", display_status(3276));
        assert_eq!(display_status(3276), 2238);
        assert_eq!(display_status(2326), 1763);
        assert_eq!(display_status(1702), 1451);
        assert_eq!(display_status(2084), 1642);
        let five: Array5 = [3276, 2326, 1702, 1194, 2084];
        let disp = display_status_array(five);
        println!("array {five:?} → {disp:?}");
        assert_eq!(disp, [2238, 1763, 1451, 1194, 1642], "分维独立换算");
    }

    /// 终局帧口径与快照口径必须同源（同一组数值 → 同一分数）
    #[test]
    fn test_final_score_from_frame_matches_snapshot_口径() -> anyhow::Result<()> {
        let root = umasim::utils::get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        umasim::gamedata::init_global()?;

        let frame: FinalScorePayload = serde_json::from_str(
            r#"{
                "scenarioId": 14, "single_mode_chara_id": 6243, "turn": 77, "state": 2,
                "fiveStatus": [3242, 2222, 1723, 1134, 2398],
                "fiveStatusLimit": [3242, 2444, 2206, 2200, 2506],
                "skillPt": 7987, "totalHints": 21
            }"#
        )?;
        let base = GameStatusBase {
            turn: 77,
            five_status: frame.five_status,
            five_status_limit: frame.five_status_limit,
            skill_pt: frame.skill_pt,
            total_hints: frame.total_hints,
            ..Default::default()
        };
        let from_frame = final_score_from_frame(&frame, false, false);
        let from_snapshot = final_score(&base);
        println!("终局帧评分={from_frame} 快照口径评分={from_snapshot} 等级={}", rank_name(from_frame));
        assert_eq!(from_frame, from_snapshot, "两种来源同口径：同一数值必须同分");
        Ok(())
    }

    /// 切者 / 小切 PT 项加成：×1.1 / ×1.04 只放大 PT 分量（2026-10-01 口径）
    #[test]
    fn test_qiezhe_xiaoqie_pt_factor() -> anyhow::Result<()> {
        let root = umasim::utils::get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        umasim::gamedata::init_global()?;

        let frame: FinalScorePayload = serde_json::from_str(
            r#"{
                "scenarioId": 14, "single_mode_chara_id": 6243, "turn": 77, "state": 2,
                "fiveStatus": [3242, 2222, 1723, 1134, 2398],
                "fiveStatusLimit": [3242, 2444, 2206, 2200, 2506],
                "skillPt": 7987, "totalHints": 21
            }"#
        )?;
        let none = final_score_from_frame(&frame, false, false);
        let qiezhe = final_score_from_frame(&frame, true, false);
        let xiaoqie = final_score_from_frame(&frame, false, true);
        println!("无状态={none} 切者={qiezhe} 小切={xiaoqie}");
        assert!(qiezhe > xiaoqie && xiaoqie > none, "×1.1 > ×1.04 > ×1.0");

        // 交叉验证：增量 = PT 分量 × (factor − 1)，五维分不受影响
        // （口径与 score_parts 一致：total_pt floor 后整体乘 rate × factor 再截断）
        let cons = global!(GAMECONSTANTS);
        let total_pt = (frame.skill_pt as f32 + frame.total_hints as f32 * cons.hint_pt_rate).floor() as i32;
        let pt_f = total_pt as f32 * cons.pt_score_rate;
        let expect_qiezhe = (pt_f * 1.1) as i32 - pt_f as i32;
        let expect_xiaoqie = (pt_f * 1.04) as i32 - pt_f as i32;
        println!("total_pt={total_pt} pt_f={pt_f} 切者增量(期望)={expect_qiezhe} 实际={}", qiezhe - none);
        assert_eq!(qiezhe - none, expect_qiezhe, "切者增量 = PT 分量 × 0.1");
        assert_eq!(xiaoqie - none, expect_xiaoqie, "小切增量 = PT 分量 × 0.04");
        Ok(())
    }

    /// 需要真实 gamedata（GAMECONSTANTS 查表口径），与项目测试同法：
    /// cwd 切到 workspace 根 + init_global
    #[test]
    fn test_final_score_and_rank() -> anyhow::Result<()> {
        let root = umasim::utils::get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        umasim::gamedata::init_global()?;
        let base = GameStatusBase {
            turn: 77,
            five_status: [1200, 1100, 1000, 900, 800],
            five_status_limit: [1500; 5],
            skill_pt: 500,
            skill_score: 0,
            total_hints: 10,
            ..Default::default()
        };
        let score = final_score(&base);
        let rank = rank_name(score);
        println!("final_score={score} rank={rank}");
        assert!(score > 0, "五维 + PT 折算应得正分");
        assert!(!rank.is_empty());
        // score_parts 口径交叉验证：pt 分量 = total_pt × pt_score_rate
        let cons = global!(GAMECONSTANTS);
        let total_pt = (500.0 + 10.0 * cons.hint_pt_rate).floor() as i32;
        let pt_part = (total_pt as f32 * cons.pt_score_rate) as i32;
        let five_part: i32 = (0..5)
            .map(|i| cons.status_final_score(base.five_status[i].min(base.five_status_limit[i])))
            .sum();
        println!("pt_part={pt_part} five_part={five_part} 合计={}", pt_part + five_part);
        assert_eq!(score, pt_part + five_part, "calc_score = pt + five（skill=0）");
        Ok(())
    }
}
