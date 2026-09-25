//! AI 决策输出标准格式
//!
//! 多个下游（Android/MCP/WebSocket）共享同一结构。Trainer trait
//! 仅输出 `action_index`；附加的决策上下文（候选评分、耗时、搜索深度等）
//! 通过 [`Trainer::last_decision`](crate::game::Trainer::last_decision)
//! 旁路暴露，便于面向用户的 Trainer（MCTS、手写策略）逐步实现。
//!
//! 设计原则：
//!
//! - 不强制任何字段非空；调用方按需填充
//! - `Serialize`/`Deserialize` 双派生，便于 JSON / bincode 互通
//! - 剧本特有扩展字段用 `serde_json::Value`，避免在此结构内堆叠剧本 enum
//!
//! ## 2026-09 简化
//!
//! 旧版 stub 字段（`reason` / `search_depth` / `visit_count` / `score_breakdown` /
//! `elapsed_ms`）已删除——JSON 输出不再暴露这些字段。**保留** `candidate_scores`
//! / `candidate_n`（按用户拍板"备选选项分复用"，下游需要展示"为什么选 A 不选 B"）。
//! `scenario_extra` 承载 luck_score / action_luck / reason 三类剧本特化信息。

use serde::{Deserialize, Serialize};

/// AI 决策输出标准格式
///
/// 与 `Trainer` trait 分离。Trainer 接口保持只输出 `action_index`，
/// 额外上下文通过 [`Trainer::last_decision`](crate::game::Trainer::last_decision) 提供。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionInfo {
    /// 选中的动作索引（在传入候选列表中的位置）
    pub action_index: usize,

    /// 选中动作的评分（按 Trainer 内部约定的口径，如手写加权均分 / MCTS 搜索均分）
    pub score: f32,

    /// 决策子类型标识（**新增 2026-09**，按用户拍板"区分 partial decision"）
    ///
    /// partial decision 是拉面剧本的常态——三阶段（吃面 + 隐藏诀窍 + 训练）会拆成
    /// 多次 select_action，每次只描述当前子阶段。C# 端必须能区分：
    /// - `ramen_select`：吃哪个面（**不含**隐藏诀窍——三阶段路径下独立）
    /// - `special_select`：吃哪个面 + 隐藏诀窍用法（基于 RamenSelect 选择）
    /// - `train`：训练 / 比赛 / 休息 / 外出 / 治病
    /// - `region_select`：年度地区选择
    /// - `super_ramen_select`：超级拉面选择（71 回合后）
    /// - `event`：事件选择
    ///
    /// main.rs `calc_ramen_training` 在每次 select_action 前 snapshot `game.stage` 并填这个字段。
    /// onsen 路径填 `"train"` / `"event"`。
    pub decision_kind: String,

    /// 所有候选动作的评分，按 `actions` 顺序排列（**完整保留**）
    ///
    /// 与 [`Self::action_index`] 等长，便于下游展示"为什么选 A 不选 B"。
    /// 按用户 2026-09 拍板：备选选项分是下游必要信息，**不**从 struct 删除。
    pub candidate_scores: Vec<f32>,

    /// 各候选动作的可读描述（**新增 2026-09**，按用户拍板"备选选项名下游不可缺"）
    ///
    /// 与 [`Self::candidate_scores`] / [`Self::candidate_n`] **严格同长同序同截断**。
    /// C# 端 `action_index` 拿到的是数字（候选下标），**没有这个字段无法映射动作名**
    /// ——尤其拉面组合动作（吃面配方 + 特殊目标 + 操作三阶段）名字极长，
    /// 数字索引本身没有语义。下游必须按 `candidate_descriptions[i]` 取名。
    ///
    /// 实现路径：onsen 从 `SearchOutput.actions[i].to_string()` 取；
    /// 拉面 MCTS 从 `RamenSearchOutput.actions[i].to_string()` 取并在 `LastSearchSummary`
    /// 缓存；手写策略从 `actions[i].to_string()` 取并在 `LastDecisionSummary` 缓存。
    pub candidate_descriptions: Vec<String>,

    /// 各候选的 rollout 样本数（**完整保留**，与 [`Self::candidate_scores`] 严格同长同序同截断）
    ///
    /// UCB 下各候选跑数可能悬殊，手写策略 / 随机等无局数概念的 trainer 留空
    /// （`Vec::default()` 即 `vec![]`）。下游可用此字段算候选置信度 / luck baseline。
    pub candidate_n: Vec<u32>,

    /// 剧本相关扩展字段（**JSON 顶层唯一额外信息出口**）
    ///
    /// 承载四类信息（按 trainer 是否支持灵活挂载）：
    /// - `luck_score`：本局 + 本回合运气分（[`crate::luck_score::LuckScoreSnapshot`]）
    /// - `action_luck`：每候选"选项后运气分"（按局数加权 T(n,action_i) − T(n)）
    /// - `reason`：human mode reason 输出所需信息（[`crate::output::DecisionReasonData`]）
    /// - `ramen_action`：选中动作的 to_string（仅 ramen 剧本挂——`RamenAction::to_string()`
    ///   已含吃面 + 隐藏诀窍 + 操作三阶段信息，按用户拍板"AIRed 端只显示不解析"）
    ///
    /// 不在本结构内堆叠剧本 enum——以 `serde_json::Value` 形式挂载，调用方按需解析。
    pub scenario_extra: Option<serde_json::Value>
}

/// 决策来源标签：网络推理选出的动作（`ramen_trainer_policy = "nn"`，见 `umaai::ramen_nn`）
pub const SOURCE_RAMEN_NN: &str = "ramen_nn";

/// 决策来源标签：网络模式下自选比赛硬守门命中，直接选「比赛」，没有推理
pub const SOURCE_RAMEN_RACE_GATE: &str = "ramen_race_gate";

/// 决策来源标签：网络模式下该阶段转交手写策略，没有推理
pub const SOURCE_RAMEN_HANDWRITTEN_STAGE: &str = "ramen_handwritten_stage";

/// [`DecisionInfo::scenario_extra`] 里承载网络参考推荐的键名（`mcts_nn_hint` 模式）
///
/// 值为 `{"choice": 网络推荐的动作文本, "same_as_executed": 是否与执行推荐相同}`。
pub const NN_HINT_KEY: &str = "nn_hint";

impl DecisionInfo {
    /// [`Self::scenario_extra`] 里承载决策来源的键名
    pub const SOURCE_KEY: &'static str = "decision_source";

    /// 标注本条决策由谁做出
    ///
    /// 只写 `scenario_extra` 里的来源键，不改任何评分字段。`scenario_extra` 已是
    /// JSON 对象时就地插入，否则新建一个只含该键的对象。
    pub fn with_source(mut self, label: &str) -> Self {
        let value = serde_json::Value::String(label.to_string());
        match self.scenario_extra {
            Some(serde_json::Value::Object(ref mut map)) => {
                map.insert(Self::SOURCE_KEY.to_string(), value);
            }
            _ => {
                let mut map = serde_json::Map::new();
                map.insert(Self::SOURCE_KEY.to_string(), value);
                self.scenario_extra = Some(serde_json::Value::Object(map));
            }
        }
        self
    }

    /// 读取决策来源标签；未标注时为 `None`
    pub fn source_label(&self) -> Option<&str> {
        self.scenario_extra
            .as_ref()?
            .get(Self::SOURCE_KEY)?
            .as_str()
    }

    /// 构造一个最小可用的 `DecisionInfo`（仅含 action_index）
    pub fn from_index(action_index: usize) -> Self {
        Self {
            action_index,
            ..Self::default()
        }
    }

    /// 构造含选中评分的 `DecisionInfo`
    pub fn from_index_and_score(action_index: usize, score: f32) -> Self {
        Self {
            action_index,
            score,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::utils::Checks;

    /// 简化后字段集：7 个（action_index / score / decision_kind / candidate_scores /
    /// candidate_descriptions / candidate_n / scenario_extra）——旧 stub 字段
    /// （reason / search_depth / visit_count / score_breakdown / elapsed_ms）已删
    #[test]
    fn test_default_is_zero_index() {
        let info = DecisionInfo::default();
        assert_eq!(info.action_index, 0);
        assert_eq!(info.score, 0.0);
        assert_eq!(info.decision_kind, "", "默认空字符串（由 caller 填）");
        assert!(info.candidate_scores.is_empty());
        assert!(info.candidate_descriptions.is_empty());
        assert!(info.candidate_n.is_empty(), "默认无局数概念");
        assert!(info.scenario_extra.is_none());
    }

    /// 来源标签：写入 / 读出 / 未标注为 None，且不动评分字段与已有键
    #[test]
    fn test_source_label_roundtrip() -> Result<()> {
        let mut c = Checks::new();
        let bare = DecisionInfo::from_index(3);
        println!("未标注 → {:?}", bare.source_label());
        c.check(bare.source_label().is_none(), "未标注时来源为 None");

        let tagged = DecisionInfo {
            action_index: 3,
            candidate_descriptions: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            ..DecisionInfo::default()
        }
        .with_source(SOURCE_RAMEN_NN);
        println!("标注后 → {:?}", tagged.source_label());
        c.check(tagged.source_label() == Some(SOURCE_RAMEN_NN), "标注后读回同一标签");
        c.check(tagged.candidate_scores.is_empty() && tagged.score == 0.0, "标注来源不改评分字段");

        let merged = DecisionInfo {
            scenario_extra: Some(serde_json::json!({"ramen_action": "吃面/札幌"})),
            ..DecisionInfo::default()
        }
        .with_source(SOURCE_RAMEN_NN);
        let kept = merged
            .scenario_extra
            .as_ref()
            .and_then(|v| v.get("ramen_action"))
            .and_then(|v| v.as_str());
        println!("已有键 → {kept:?}");
        c.check(kept == Some("吃面/札幌"), "已有 scenario_extra 的键不丢");
        c.check(merged.source_label() == Some(SOURCE_RAMEN_NN), "已有对象上也能读回标签");
        c.finish()
    }

    #[test]
    fn test_from_index_minimal() {
        let info = DecisionInfo::from_index(3);
        assert_eq!(info.action_index, 3);
        assert_eq!(info.score, 0.0);
    }

    #[test]
    fn test_from_index_and_score() {
        let info = DecisionInfo::from_index_and_score(2, 1234.5);
        assert_eq!(info.action_index, 2);
        assert!((info.score - 1234.5).abs() < 1e-6);
    }

    #[test]
    fn test_serde_roundtrip_minimal() {
        let info = DecisionInfo::default();
        let json = serde_json::to_string(&info).expect("serialize");
        let back: DecisionInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(info, back);
    }

    #[test]
    fn test_serde_roundtrip_with_scenario_extra() {
        // 验证 scenario_extra（含 luck_score / action_luck / reason / ramen_action）
        // + candidate_descriptions + decision_kind 能正确序列化往返
        let info = DecisionInfo {
            action_index: 2,
            score: 1500.75,
            decision_kind: "ramen_select".to_string(),
            candidate_scores: vec![100.0, 200.0, 1500.75, 300.0],
            candidate_descriptions: vec![
                "不吃面".to_string(),
                "吃面/札幌".to_string(),
                "吃面/中山-全(替换Bx1+Ax2)".to_string(),
                "吃面/千叶".to_string()
            ],
            candidate_n: vec![100, 200, 1500, 300],
            scenario_extra: Some(serde_json::json!({
                "scenario": "ramen",
                "luck_score": {
                    "initial_terminal_baseline": 50078.0,
                    "current_terminal_baseline": 50354.0,
                    "total_luck_score": 276.0,
                    "last_turn_delta": -121.0
                },
                "action_luck": {"0": -50.0, "1": 125.5},
                "ramen_action": "吃面/中山-全(替换Bx1+Ax2)",
                "reason": {
                    "metric": "score",
                    "chosen_desc": "吃面/中山-全(替换Bx1+Ax2)",
                    "chosen_mean": 65000.0,
                    "chosen_n": 1024,
                    "rivals": [
                        {"index": 2, "desc": "吃面/中山-全(替换Bx1+Ax2)", "gap": 2200.0,
                         "confidence": 0.95, "n": 800, "mean": 67200.0, "sd": 200.0,
                         "pros": [{"key":"speed_final","label":"速","unit":"score","delta":30.0}],
                         "cons": []}
                    ]
                }
            }))
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let back: DecisionInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(info, back);
    }

    /// 顶层 JSON 不输出 5 个已删 stub 字段——保留 7 字段（含 decision_kind /
    /// candidate_descriptions）
    #[test]
    fn test_json_top_level_omits_stub_fields() {
        let info = DecisionInfo {
            action_index: 1,
            score: 42.0,
            decision_kind: "ramen_select".to_string(),
            candidate_scores: vec![10.0, 42.0, 30.0],
            candidate_descriptions: vec![
                "不吃面".to_string(),
                "吃面/中山-全(替换Bx1+Ax2)".to_string(),
                "不吃面".to_string()
            ],
            candidate_n: vec![100, 200, 50],
            scenario_extra: None
        };
        let v = serde_json::to_value(&info).expect("to_value");
        assert_eq!(v["action_index"], 1);
        assert_eq!(v["score"], 42.0);
        assert_eq!(v["decision_kind"], "ramen_select", "decision_kind 顶层保留");
        assert!(v["candidate_scores"].is_array(), "candidate_scores 保留");
        assert!(v["candidate_descriptions"].is_array(), "candidate_descriptions 保留");
        assert!(v["candidate_n"].is_array(), "candidate_n 保留");
        // 顶层不应再有这些已删字段
        assert!(v.get("reason").is_none(), "reason 已从 DecisionInfo 删除");
        assert!(v.get("search_depth").is_none(), "search_depth 已删除");
        assert!(v.get("visit_count").is_none(), "visit_count 已删除");
        assert!(v.get("score_breakdown").is_none(), "score_breakdown 已删除");
        assert!(v.get("elapsed_ms").is_none(), "elapsed_ms 已删除");
        // 候选描述数组与 scores / n 严格同长
        assert_eq!(
            v["candidate_descriptions"].as_array().unwrap().len(),
            v["candidate_scores"].as_array().unwrap().len(),
            "candidate_descriptions 与 candidate_scores 严格同长"
        );
    }
}
