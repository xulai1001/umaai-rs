//! 终局帧（`finalScore.json`）通信结构：育成结束·点技能前的真机终局数据
//!
//! ## 来源与用途
//!
//! SendGameStatusPlugin 在「育成结束、点技能前」那一帧（`chara_info.state ∈ {2,3}`
//! 且无未处理事件）**单独写 `finalScore.json`**，不写 `thisTurn.json`：
//!
//! - 回合快照链路（stage dispatch / 末回合判定 / 复盘 timeline）不受污染；
//! - 该帧已含**全部结局事件**（育成结束 `401407`、通用 `5011`、友人结束），
//!   以及末回合比赛奖励——即 `thisTurn.json` 末尾缺失的那约 2700 分来源。
//!
//! 评分仍由 `Uma::calc_score` 计算（与 AI 评估轴同源）：真机 `/finish` 的
//! `rank_score`（属性与 Hint 已转成技能后的总评分）**不在本通道内**。
//!
//! ## 与 `GameStatusBase` 的关系
//!
//! 本结构是**扁平**的（无 `baseGame` 外壳），字段名沿用 baseGame 段命名，但
//! **不能**直接用 `GameStatusBase` 反序列化——后者要求 `vital` / `cardId` /
//! `persons` / `personDistribution` / `trainLevelCount` 等字段在场，终局帧一个都不带。

use serde::{Deserialize, Serialize};

use umasim::{game::Uma, gamedata::GAMECONSTANTS, global, utils::Array5};

/// 本通道支持的剧本 ID（与 `GameStatusRamen::scenario_id()` 一致，单测守护）
pub const SUPPORTED_SCENARIO_ID: u32 = 14;

/// 终局帧（`finalScore.json`）扁平结构
///
/// 字段命名与 `GameStatusBase` 的 baseGame 段保持同构（个别字段为 snake_case，
/// 与 C# 端字段名一致）；缺失字段容忍，便于插件端逐步补字段。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinalScorePayload {
    /// 剧本 ID（本通道只支持 14 = 拉面）
    pub scenario_id: u32,
    /// 育成 ID（`single_mode_chara_id`，归局键，单调递增）
    #[serde(default, rename = "single_mode_chara_id")]
    pub single_mode_chara_id: Option<u64>,
    /// 马娘 ID
    #[serde(default)]
    pub uma_id: u32,
    /// 马娘星数
    #[serde(default)]
    pub uma_star: u32,
    /// 回合（0 基；终局帧为末回合）
    #[serde(default)]
    pub turn: i32,
    /// `chara_info.state`（2/3 = 育成结束态）
    #[serde(default)]
    pub state: i32,
    /// `chara_info.playing_state`
    #[serde(default, rename = "playing_state")]
    pub playing_state: i32,
    /// 终局五维（**已 `ReviseOver1200`**，与 `thisTurn.json` 同口径）
    ///
    /// **契约必填**：缺字段直接判解析失败（插件字段漂移要炸出声，不能静默按 0 打分）。
    pub five_status: Array5,
    /// 五维上限（同上；**契约必填**——评分查表按 `min(五维, 上限)` 取值，
    /// 缺字段落 0 会把五维分整体清零）
    pub five_status_limit: Array5,
    /// 点技能前的**剩余**技能点（不含已学技能价值；**契约必填**）
    pub skill_pt: i32,
    /// 总 Hint 等级（`skill_tips_array` 各项 `level` 之和，与快照 `baseGame.totalHints` 同口径）
    ///
    /// 插件 2026-10 起下发；旧帧缺字段容忍（落 0，等价于此前「恒 0」口径）。
    #[serde(default)]
    pub total_hints: i32,
    /// 本局累计花掉的技能点（EventLogger 统计；当前实测恒 0）
    #[serde(default)]
    pub skill_pt_spent: i32,
    /// 两次继承的属性增量（按继承顺序；**仅复盘分析用**）
    ///
    /// 后续会从 `thisTurn.json` 的 baseGame 段移除，届时本通道成为唯一来源。
    #[serde(default, rename = "inheritGains")]
    pub inherit_gains: Vec<i32>
}

impl FinalScorePayload {
    /// 局号（`None` = 帧内缺字段，调用方应忽略该帧）
    pub fn game_id(&self) -> Option<u64> {
        self.single_mode_chara_id
    }

    /// 是否为本通道支持的剧本（当前仅拉面）
    pub fn is_supported_scenario(&self) -> bool {
        self.scenario_id == SUPPORTED_SCENARIO_ID
    }

    /// 帧内容是否可用（第二道闸：字段在场但填了全 0 的坏帧）
    ///
    /// 真实终局帧的五维与其上限必然为正（育成结束时五维都是三位数以上、上限几百到几千）；
    /// 全 0 只可能是插件写错字段/漏填。若放行，`min(五维, 上限)` 会把五维分整体清零，
    /// 却仍被标成「真机终局帧」——用户会看到一个小黑板上不存在的低分。故一律判为不可用，
    /// 由调用方告警 + 回落末快照口径。
    pub fn is_usable(&self) -> bool {
        self.five_status.iter().any(|&v| v > 0) && self.five_status_limit.iter().any(|&v| v > 0)
    }

    /// 按 `Uma::calc_score` 同源口径计算终局评分（五维查表 + PT 折算）
    ///
    /// `skill_score` 恒 0（本通道不携带已学技能分）、`pt_score_rate_factor` 恒 1.0
    /// （帧内无切者 / 小切标志——`umaai_review` 复盘侧从末快照代入该标志，此处
    /// 仅做显示不引入快照依赖）——属「点技能前」的口径，略低于小黑板最终分。
    ///
    /// # Panics
    ///
    /// 需已 `gamedata::init_global`（查表依赖 `GAMECONSTANTS`），未初始化会 panic。
    pub fn calc_score(&self) -> i32 {
        Uma {
            five_status: self.five_status,
            five_status_limit: self.five_status_limit,
            skill_pt: self.skill_pt,
            skill_score: 0,
            total_hints: self.total_hints,
            ..Default::default()
        }
        .calc_score()
    }

    /// 终局评分对应的等级名（`GameConstants::get_rank_name`，需已 init）
    pub fn rank_name(&self) -> String {
        global!(GAMECONSTANTS).get_rank_name(self.calc_score())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{GameStatus, ramen::GameStatusRamen};

    /// 扁平 payload 解析：字段映射（含 snake_case 覆盖项）+ 局号 + 剧本判定
    #[test]
    fn test_parse_final_payload() {
        let raw = r#"{
            "scenarioId": 14,
            "single_mode_chara_id": 6243,
            "umaId": 109701,
            "umaStar": 5,
            "turn": 77,
            "state": 2,
            "playing_state": 1,
            "fiveStatus": [3226, 2162, 1678, 1089, 2338],
            "fiveStatusLimit": [3242, 2444, 2206, 2200, 2506],
            "skillPt": 7717,
            "totalHints": 21,
            "skillPtSpent": 0,
            "inheritGains": [11, 22]
        }"#;
        let p: FinalScorePayload = serde_json::from_str(raw).expect("终局帧应可解析");
        println!("终局帧: {p:?}");
        assert_eq!(p.game_id(), Some(6243));
        assert!(p.is_supported_scenario());
        assert_eq!(p.five_status, [3226, 2162, 1678, 1089, 2338]);
        assert_eq!(p.five_status_limit, [3242, 2444, 2206, 2200, 2506]);
        assert_eq!(p.skill_pt, 7717);
        assert_eq!(p.total_hints, 21);
        assert_eq!(p.state, 2);
        assert_eq!(p.inherit_gains, vec![11, 22]);
    }

    /// 缺字段容忍：只有最小集（局号 + 剧本）也能解析，其余落默认值
    ///
    /// 注意：**五维 / 上限 / 技能点是契约必填**，缺任一即解析失败（见下方单测）。
    #[test]
    fn test_parse_final_payload_minimal() {
        let raw = r#"{"scenarioId":14,"single_mode_chara_id":1,"fiveStatus":[1,0,0,0,0],"fiveStatusLimit":[1,0,0,0,0],"skillPt":0}"#;
        let p: FinalScorePayload = serde_json::from_str(raw).expect("最小集应可解析");
        println!("最小终局帧: {p:?}");
        assert_eq!(p.game_id(), Some(1));
        assert_eq!(p.five_status, [1, 0, 0, 0, 0]);
        assert_eq!(p.total_hints, 0, "旧帧缺 totalHints 应落 0");
        assert!(p.inherit_gains.is_empty());
    }

    /// 契约必填字段：缺 `fiveStatus` / `fiveStatusLimit` / `skillPt` 任一 → 解析失败
    ///
    /// 静默按 0 填充会让五维分整体清零，却仍被当作「真机终局帧」报出去（见 `is_usable`）。
    #[test]
    fn test_required_fields_rejected_when_missing() {
        let cases = [
            (r#"{"scenarioId":14,"single_mode_chara_id":1,"fiveStatusLimit":[1200,1200,1200,1200,1200],"skillPt":10}"#, "缺 fiveStatus"),
            (r#"{"scenarioId":14,"single_mode_chara_id":1,"fiveStatus":[100,100,100,100,100],"skillPt":10}"#, "缺 fiveStatusLimit"),
            (r#"{"scenarioId":14,"single_mode_chara_id":1,"fiveStatus":[100,100,100,100,100],"fiveStatusLimit":[1200,1200,1200,1200,1200]}"#, "缺 skillPt"),
        ];
        for (raw, why) in cases {
            let r = serde_json::from_str::<FinalScorePayload>(raw);
            println!("{why} → is_ok={}", r.is_ok());
            assert!(r.is_err(), "{why} 必须解析失败");
        }
    }

    /// 第二道闸：字段在场但全 0（插件写错字段）→ `is_usable` 为 false
    #[test]
    fn test_all_zero_frame_not_usable() {
        let zero = r#"{"scenarioId":14,"single_mode_chara_id":1,"fiveStatus":[0,0,0,0,0],"fiveStatusLimit":[0,0,0,0,0],"skillPt":7000}"#;
        let p: FinalScorePayload = serde_json::from_str(zero).expect("可解析（字段在场）");
        println!("全 0 帧 usable={}", p.is_usable());
        assert!(!p.is_usable(), "全 0 五维/上限必须判为不可用");

        let ok = r#"{"scenarioId":14,"single_mode_chara_id":1,"fiveStatus":[3226,2162,1678,1089,2338],"fiveStatusLimit":[3242,2444,2206,2200,2506],"skillPt":7717}"#;
        let p2: FinalScorePayload = serde_json::from_str(ok).expect("可解析");
        assert!(p2.is_usable(), "正常帧应可用");
    }

    /// 其它剧本（如温泉 12）不进本通道，由调用方忽略
    #[test]
    fn test_final_payload_other_scenario_rejected() {
        let p: FinalScorePayload = serde_json::from_str(
            r#"{"scenarioId":12,"single_mode_chara_id":9,"fiveStatus":[1,0,0,0,0],"fiveStatusLimit":[1,0,0,0,0],"skillPt":0}"#,
        ).expect("可解析");
        println!("温泉终局帧 is_supported={}", p.is_supported_scenario());
        assert!(!p.is_supported_scenario());
    }

    /// 剧本常量与拉面协议一致（防止两处漂移）
    #[test]
    fn test_supported_scenario_matches_ramen() {
        let ramen = GameStatusRamen::scenario_id();
        println!("本通道剧本 = {SUPPORTED_SCENARIO_ID} / GameStatusRamen = {ramen}");
        assert_eq!(SUPPORTED_SCENARIO_ID, ramen);
    }

    /// 终局评分 / 等级：与 `Uma::calc_score` 同源（五维查表 + PT 折算），需 gamedata
    #[test]
    fn test_calc_score_and_rank() -> anyhow::Result<()> {
        let root = umasim::utils::get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = umasim::gamedata::init_global();

        let raw = r#"{
            "scenarioId": 14, "single_mode_chara_id": 6243,
            "fiveStatus": [3226, 2162, 1678, 1089, 2338],
            "fiveStatusLimit": [3242, 2444, 2206, 2200, 2506],
            "skillPt": 7717, "totalHints": 21
        }"#;
        let p: FinalScorePayload = serde_json::from_str(raw).expect("终局帧应可解析");
        let score = p.calc_score();
        let rank = p.rank_name();
        println!("终局评分={score} 等级={rank}");

        // 交叉验证：分数 = 五维查表分之和 + PT 折算分（skill_score 恒 0）
        let cons = global!(GAMECONSTANTS);
        let five_part: i32 = (0..5)
            .map(|i| cons.status_final_score(p.five_status[i].min(p.five_status_limit[i])))
            .sum();
        let total_pt = (p.skill_pt as f32 + p.total_hints as f32 * cons.hint_pt_rate).floor() as i32;
        let pt_part = (total_pt as f32 * cons.pt_score_rate) as i32;
        assert_eq!(score, five_part + pt_part, "calc_score = 五维 + PT（skill=0）");
        assert!(!rank.is_empty(), "等级名不应为空");
        Ok(())
    }
}