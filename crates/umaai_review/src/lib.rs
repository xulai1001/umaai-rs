//! umaai_review：单局复盘分析引擎
//!
//! 架构与口径见 `.trae/documents/replay_review.md`：**Trae Skill（归因层）+
//! 独立 Rust bin（分析引擎）**，计算下沉到 Rust、判断力留给 LLM。
//!
//! 产物：
//! - `digest.json`：结构化指标（喂 LLM 的上下文包，§3.5 schema）
//! - `report.html`：网页版报告（minijinja 模板 + 内联自绘 SVG，零 JS；M5 里程碑）
//!
//! 模块（按文档 §11 实施顺序）：
//! - [`pack`]：局包解包与文件角色识别（§11 步骤 1）
//! - [`gdata`]：gamedata 目录解析与全局初始化（§9.2 bin 侧子集）
//! - [`timeline`]：快照解析 → timeline（§11 步骤 2；只反序列化不调 `into_game`）
//! - [`decisions`]：decisions.csv → decisions / coverage / luck（§11 步骤 2）
//! - [`schedule`]：赛程与自由比赛（§6.3）
//! - [`score`]：终局评分与等级换算（§11 步骤 3、§3.4）
//! - [`digest`]：digest.json 组装与落盘
//! - [`brief`]：brief.md 渲染——六问事实预答，SKILL 层一次 Read 即可动笔
//! - [`execution`]：实际执行动作推断 + findings 偏离清单（§11 步骤 4、§5.4、§6.9）
//! - [`checks`]：检查项引擎——伪波动标记 / 超级拉面期 / 坏手法 findings（§6.1、§6.6、§6.7）
//! - [`inherit`]：继承质量分析（§7）
//! - [`clones`]：分身彩圈观测，A/B 两类分开统计（§6.5）
//! - [`profile`]：训练画像——逐年训练次数 × 五维净增 × 运气归属（增长顺序分析）
//! - [`report`]：report.html 渲染——minijinja 外置模板 + `plot::svg` 四图（§8、§11 步骤 7）
//! - 待实施：SKILL.md（步骤 8）、端到端验证与发布打包（步骤 9-10）
//!
//! 代码常量（§10）：`YEAR_BOUNDARIES` / `INHERIT_TURNS` / `SUPER_RAMEN_START`
//! 定义在 [`checks`]。

pub mod brief;
pub mod checks;
pub mod clones;
pub mod decisions;
pub mod digest;
pub mod execution;
pub mod gdata;
pub mod inherit;
pub mod pack;
pub mod profile;
pub mod report;
pub mod schedule;
pub mod score;
pub mod timeline;
