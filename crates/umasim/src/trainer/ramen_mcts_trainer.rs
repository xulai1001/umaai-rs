//! 拉面杯 MCTS 训练员
//!
//! 用扁平蒙特卡洛搜索（[`FlatSearch<RamenGame>`]）替换手写策略的部分决策点，
//! 其余决策点仍走 [`RecommendedRamenTrainer`]。
//!
//! # 为什么不复用 `MctsTrainer`
//!
//! `MctsTrainer` 只 `impl Trainer<OnsenGame>`，且字段与温泉强耦合
//! （`HandwrittenEvaluator` 只实现了 `Evaluator<OnsenGame>`、`OnsenAction::Dig`
//! 特判）。把它泛型化要连带掀开 `umaai` 的调用签名，代价远大于另写一个薄壳。
//! 搜索核心 [`FlatSearch`] 本身已泛型化，拉面侧缺的只是最外层这一层。
//!
//! # 阶段门控
//!
//! 一局约 171 个决策点（实测单局：Train 69 / RamenSelect 61 / SpecialSelect 25 /
//! Event 15 / RegionSelect 3 / SuperRamenSelect 1），全搜代价高。[`RamenSearchStages`] 允许只搜指定阶段，
//! 未选中的阶段直接转发给推荐策略（`RecommendedRamenTrainer`）。这样既能压预算，也能单独测量
//! 「只搜 Train」/「只搜 RamenSelect」各自的边际收益。
//!
//! # 事件选项不走搜索
//!
//! [`Trainer::select_choice`] / [`Trainer::select_event_choice`] 的候选不来自
//! [`Game::list_actions`]，通用 rollout 入口 `apply_action` 吃不下，一律转发手写策略。
//!
//! # 合并动作搜索（`use_combined_ramen_select`）
//!
//! 打开时 `RamenSelect` 用 `list_combined_ramen_select_actions` 一次搜
//! `(ramen, targets)`，再把最优 `ramen` 映射回三阶段候选下标；紧随其后的
//! `SpecialSelect` 直接返回缓存的 targets，不再搜索。
//!
//! 这会改变对外层 rng 的消耗：`FlatSearch::search` 每次恰好消耗一次
//! `next_u64`。三阶段路径在 RamenSelect + SpecialSelect 各搜一次（2 次），
//! 合并路径只在 RamenSelect 搜一次（1 次）。随机序列整体位移，拉面基线作废。
//! 这是预期行为，不是 bug。关闭本开关即退回改动前的三阶段分别搜。

use std::{
    collections::HashMap,
    sync::{
        Arc,
        Mutex,
        atomic::{AtomicUsize, Ordering}
    },
    time::{Duration, Instant}
};

use anyhow::{Result, anyhow, bail};
use log::{debug, info};
use rand::prelude::StdRng;

use super::RecommendedRamenTrainer;
use crate::{
    game::{
        Game, Trainer,
        ramen::{Operation, RamenGame, RamenStage, policy::FIXED_SUPER_RAMEN_INDEX}
    },
    gamedata::{EventChoice, EventData},
    output::{
        DecisionInfo as DecisionInfoProto,
        reason::{DecisionReasonData, DecisionReasonNoopSink, ReasonMetric, analyze_narrow_win}
    },
    search::{ActionResult, FlatSearch, RamenSearchOutput, SearchConfig, SearchProbe, TerminalStats}
};

/// 搜索哪些阶段的门控开关
///
/// 字段对应 [`RamenStage`] 中会产生多候选的阶段。未列出的阶段
/// （`Begin` / `BeginAfterRegionSelect` / `Distribute` / `AfterTrain` /
/// `NextTurn` / `Settlement`）不产生真正的选择空间，无需门控。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamenSearchStages {
    /// 训练/比赛选择（决策点最多，约占一局的 45%）
    pub train: bool,
    /// 吃哪碗面（约占 27%）
    pub ramen_select: bool,
    /// 隐藏风味用法
    pub special_select: bool,
    /// 年度地区选择（turn 2 / 23 / 47 的 `RegionSelect` 阶段，含第 1 年）
    pub region_select: bool,
    /// 超级拉面选择（回合 71 后、一局一次，3 个候选）
    ///
    /// 默认配置 `ramen_search_stages = "train,ramen"` 不打开本开关。
    /// [`Self::all`] 现在会真正搜这一步；历史 `all` 基线因此作废。
    pub super_ramen_select: bool
}

impl RamenSearchStages {
    /// 全部阶段都搜
    ///
    /// **历史 `all` 基线已作废**：超级拉面接入 trainer 后 `all` 会真的搜
    /// `SuperRamenSelect`；第 1 年地区抬到 `RegionSelect` 阶段边界后，`all`
    /// 也会真的搜 turn 2 的地区选择（以前卡在 `Begin` 内部搜不到）。
    pub fn all() -> Self {
        Self {
            train: true,
            ramen_select: true,
            special_select: true,
            region_select: true,
            super_ramen_select: true
        }
    }

    /// 一个阶段都不搜（等价于纯手写策略，用于对照组）
    pub fn none() -> Self {
        Self {
            train: false,
            ramen_select: false,
            special_select: false,
            region_select: false,
            super_ramen_select: false
        }
    }

    /// 只搜训练阶段
    pub fn train_only() -> Self {
        Self {
            train: true,
            ..Self::none()
        }
    }

    /// 只搜吃面阶段
    pub fn ramen_only() -> Self {
        Self {
            ramen_select: true,
            ..Self::none()
        }
    }

    /// 解析逗号分隔的阶段名（CLI 用）
    ///
    /// 可用名：`all` / `none` / `train` / `ramen` / `special` / `region` / `super`。
    /// 例：`"train,ramen"`。
    ///
    /// # 三条严格性约定
    ///
    /// 这些输入直接决定实验分组，静默接受歧义输入会让对照组悄悄退化成纯手写策略，
    /// 是最难发现的一类错，故一律 `Err`：
    ///
    /// - 未知阶段名
    /// - 空串 / 只有逗号（否则静默得到 `none`）
    /// - `all` / `none` 与其他名混用（否则 `train,none` 与 `none,train` 结果不同）
    pub fn parse(spec: &str) -> Result<Self> {
        let names: Vec<&str> = spec.split(',').map(str::trim).filter(|n| !n.is_empty()).collect();
        if names.is_empty() {
            anyhow::bail!("搜索阶段为空（要表达「不搜索」请显式写 none）");
        }
        if names.iter().any(|n| matches!(*n, "all" | "none")) {
            if names.len() > 1 {
                anyhow::bail!("all / none 必须单独使用，不能与其他阶段名混用: {spec}");
            }
            return Ok(if names[0] == "all" { Self::all() } else { Self::none() });
        }
        let mut stages = Self::none();
        for name in names {
            match name {
                "train" => stages.train = true,
                "ramen" => stages.ramen_select = true,
                "special" => stages.special_select = true,
                "region" => stages.region_select = true,
                "super" => stages.super_ramen_select = true,
                other => {
                    anyhow::bail!("未知搜索阶段: {other}（可用 all/none/train/ramen/special/region/super）")
                }
            }
        }
        Ok(stages)
    }

    /// 该阶段是否应走搜索
    ///
    /// 取引用而非按值：`RamenStage` 未实现 `Copy`（上游类型，不在本次改动范围内）。
    pub fn contains(&self, stage: &RamenStage) -> bool {
        match stage {
            RamenStage::Train => self.train,
            RamenStage::RamenSelect => self.ramen_select,
            RamenStage::SpecialSelect => self.special_select,
            RamenStage::RegionSelect => self.region_select,
            RamenStage::SuperRamenSelect => self.super_ramen_select,
            _ => false
        }
    }
}

impl Default for RamenSearchStages {
    fn default() -> Self {
        Self::all()
    }
}

/// 最优动作的取分口径
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamenSelection {
    /// 结算评分（`calc_score`）
    Score,
    /// 计入 PT 偏好的评分（`calc_score_with_pt_favor`）
    Pt
}

/// 一次 `select_action` 走过的路径（决策探针用）
///
/// 用枚举而非布尔组合：`searched` / `cache_hit` / `single_candidate` 三个布尔
/// 里只有四种组合合法，枚举让非法组合根本表达不出来。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionPath {
    /// 真正跑了一次 [`FlatSearch::search`](crate::search::FlatSearch::search)
    Searched,
    /// `SpecialSelect` 直接复用了合并搜索缓存的 targets
    CombinedCacheHit,
    /// 单候选短路，转发手写策略
    FallbackSingleCandidate,
    /// 阶段门控关闭，转发手写策略
    FallbackGated
}

/// 一次 `select_action` 调用的成本记录（仅测量用）
///
/// 与 [`SearchProbe`](crate::search::SearchProbe) 的区别：后者只量搜索内核，
/// 本结构量的是**整个决策**——含 `list_actions` 之后的合并候选枚举、搜索、
/// 以及搜索之后的 `stash_*` / `emit_decision_reason` 收尾。未搜索的决策
/// （门控关 / 单候选 / 缓存命中）同样会留下记录，故「执行决策次数」= 本记录条数。
#[derive(Debug, Clone)]
pub struct DecisionProbe {
    /// 决策发生时的回合
    pub turn: i32,
    /// 决策发生时的阶段
    pub stage: RamenStage,
    /// `select_action` 收到的候选数（三阶段口径，**不是**合并候选数）
    pub actions_len: usize,
    /// 本次决策走了哪条路径
    pub path: DecisionPath,
    /// 本次决策结束后 `last_decision()` 是否有内容可供上层 emit
    ///
    /// 用来区分「没搜索」与「搜了但不对外暴露」。合并 `RamenSelect` 路径自上游
    /// 按面聚合回三阶段下标后也会暴露摘要，此处为 `true`。
    pub exposes_decision_info: bool,
    /// 决策开始时刻（与外层链路对齐用）
    pub started: Instant,
    /// 决策墙钟耗时（含搜索前后的全部处理）
    pub elapsed: Duration
}

/// 上一次真正走过 MCTS 搜索的最小摘要（供 `last_decision` 读取）
///
/// **只缓存决策协议需要的字段**，不复制整个 [`RamenSearchOutput`]（含每个候选
/// 的 `ActionResult.distribution` 数组，clone 成本过大）。`last_decision` 要的
/// 三件套：分数 / 局数 / 选中下标，加上候选可读描述（`descriptions`）。
///
/// 2026-09 简化：移除 `reason_text` 字段——`DecisionInfo::reason` 已删除，
/// 终局维度差值改走 `LastReasonSink` 缓存的 `DecisionReasonData` 挂到
/// `scenario_extra.reason`（main.rs `emit_with_luck_decision` 接线）。
///
/// 2026-09 扩展：新增 `descriptions` 字段——拉面组合动作（吃面配方 + 特殊目标 +
/// 操作三阶段）名字极长，AIRedirector 仅靠 `action_index` 数字完全无法映射
/// 动作名，必须挂描述才能展示。
#[derive(Debug, Clone)]
struct LastSearchSummary {
    /// 中选者在 `action_results` 中的下标
    chosen_idx: usize,
    /// 全候选分数（按 action_results 顺序，**真实评分口径**——`SearchScore::score`
    /// 即 `calc_score()`，不含 `pt_favor_rate` 缩放）
    ///
    /// 选择动作走 [`RamenMctsTrainer::selection`] 指定的轴（默认 `Pt`），但**展示与运气分口径
    /// 固定为真实评分**：`candidate_scores` 会经 `emit_with_luck_decision` 换算 T(n)
    /// baseline 得出运气分，若在此处取被 `pt_favor_rate` 放大的 `score_pt`，运气分
    /// （以及 AIRed 显示的候选分）会随该系数虚增、不再反映真实评分得失。
    scores: Vec<f64>,
    /// 全候选的 rollout 样本数（按 action_results 顺序）
    counts: Vec<u32>,
    /// 全候选的可读描述（按 action_results 顺序，与 scores / counts 严格同长同序）
    descriptions: Vec<String>
}

/// 从 [`DecisionReasonData`] 抽出"评分最高的未中选候选"的最显著维度差，拼成简短文本
///
/// 输出形如 `vs #2 智+180 PT-33`，对应 `render_reason_lines` 给 rivals[0] 编的 `#2` 号。
/// rivals 为空（单候选/中选者评分最高）时返回 `None`——与 `analyze_narrow_win` 同口径。
///
/// 2026-09 简化：当前**未被使用**——`DecisionInfo::reason` 删除后，简化文本
/// 改由 `scenario_extra.reason.rivals[0].pros[0] + cons[0]` 表达。保留函数
/// 以备 human mode 调试或 AIRed 端简化展示需要。
#[allow(dead_code)]
fn summarize_ramen_reason(data: &DecisionReasonData) -> Option<String> {
    let best_rival = data.rivals.first()?;
    let mut parts: Vec<String> = Vec::new();
    for dim in best_rival.pros.iter().take(1).chain(best_rival.cons.iter().take(1)) {
        if dim.unit == "flag" {
            parts.push(format!("{}{:+.0}%", dim.label, dim.delta * 100.0));
        } else {
            parts.push(format!("{}{:+.0}", dim.label, dim.delta));
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!("vs #2 {}", parts.join(" ")))
}

/// 拉面杯 MCTS 训练员
///
/// 被门控选中的阶段走 [`FlatSearch`]，其余转发给内置的
/// [`RecommendedRamenTrainer`]。搜索的 rollout 基策同样是推荐策略
/// （由 `FlatSearchGame::default_rollout_trainer` 提供），因此本训练员
/// 是「手写策略 + 搜索」的严格叠加：门控全关时行为与纯推荐策略一致。
///
/// rollout 基策可经 [`Self::with_nn_rollout`] 换成网络，此时搜索出的是
/// `Q^NN` 而非 `Q^手写`——那已经是另一个教师，与旧基线不可比。
pub struct RamenMctsTrainer {
    /// 扁平搜索器
    pub search: FlatSearch<RamenGame>,
    /// 未搜索阶段与事件选项的回退策略
    pub fallback: RecommendedRamenTrainer,
    /// 取分口径：默认 `Pt`，保持上游生产策略；研究入口可显式选择 `Score`。
    /// 展示与运气分口径与本字段无关，固定为真实评分。
    pub selection: RamenSelection,
    /// 搜索哪些阶段
    pub stages: RamenSearchStages,
    /// 是否输出每步决策日志
    pub verbose: bool,
    /// `RamenSelect` 是否用合并动作（ramen + targets 一次决策）搜索
    ///
    /// 打开时 `SpecialSelect` 不再是独立决策点：`RamenSelect` 的搜索结果里
    /// 已经含 targets，`SpecialSelect` 直接返回缓存值。
    /// 关闭时退回三阶段分别搜（改动前行为）。
    ///
    /// **RNG 消耗会变**：`FlatSearch::search` 对外层 rng 恰好消耗一次 `next_u64`。
    /// 打开后 SpecialSelect 零消耗，随机序列整体位移，拉面基线作废。这是预期的。
    pub use_combined_ramen_select: bool,
    /// 最近一次搜索决策的候选统计文本（供 `LoggingTrainer` 写入决策日志）
    ///
    /// 用 `Mutex` 而非 `RefCell`：`Trainer` 在搜索/并行场景要求 `Sync`。
    last_breakdown: Mutex<Option<String>>,
    /// 本训练员真正走过搜索的决策次数（转发给手写策略的不计）
    ///
    /// `Trainer::select_action` 只有 `&self`，故用原子量。用途是让「门控是否生效」
    /// 可观测：只看分数无法区分「搜索没提分」与「门控写错、根本没搜」。
    searched: AtomicUsize,
    /// `SpecialSelect` 直接命中合并搜索缓存的次数
    ///
    /// 与 [`Self::searched`] 同理，用原子量是因为 `select_action` 只有 `&self`。
    /// 用途是钉住「缓存检查必须在门控早退之前」：若它被挪到早退之后，
    /// `special_select` 门控关闭时合并搜索选出的 targets 会被**静默丢弃**、
    /// 改由手写策略另选，而分数上看不出来——本计数器归零才看得见。
    combined_cache_hits: AtomicUsize,
    /// `RamenSelect` 合并搜索选出的 targets，供紧随其后的 `SpecialSelect` 复用
    ///
    /// 用 `Mutex` 而非 `RefCell`：`Trainer` 在搜索/并行场景要求 `Sync`
    /// （与既有 `last_breakdown` 同理）。
    pending_combined_targets: Mutex<Option<[i32; 3]>>,
    /// 上一次真正走过搜索的最小摘要（供 [`Trainer::last_decision`](crate::game::Trainer::last_decision) 读取）
    ///
    /// 早退分支（单候选 / 门控关 / 转发 fallback）会清成 `None`——避免把
    /// 上一次搜索的陈旧数据当成本次输出。
    last_search_summary: Mutex<Option<LastSearchSummary>>,
    /// 决策理由原始数据出口（每回合都发出 JSON；umasim 默认接日志）
    ///
    /// 可读文字不走此出口，由 [`Self::emit_decision_reason`] 渲染后上屏。
    pub reason_sink: Arc<dyn crate::output::DecisionReasonSink>,
    /// 决策成本探针出口（**仅测量用**，默认 `None`）
    ///
    /// 关闭时 `select_action` 只多一次 `Option` 判断后直接转
    /// [`Self::select_action_inner`]，不取时钟、不读原子量。
    decision_probe: Option<Arc<Mutex<Vec<DecisionProbe>>>>
}

impl RamenMctsTrainer {
    /// 用指定搜索配置创建（默认搜全部阶段、按 PT 口径取最优、打开合并动作搜索）
    pub fn new(config: SearchConfig) -> Self {
        Self {
            search: FlatSearch::<RamenGame>::new(config),
            fallback: RecommendedRamenTrainer::new(),
            selection: RamenSelection::Pt,
            stages: RamenSearchStages::all(),
            verbose: false,
            use_combined_ramen_select: true,
            last_breakdown: Mutex::new(None),
            searched: AtomicUsize::new(0),
            combined_cache_hits: AtomicUsize::new(0),
            pending_combined_targets: Mutex::new(None),
            last_search_summary: Mutex::new(None),
            reason_sink: Arc::new(DecisionReasonNoopSink),
            decision_probe: None
        }
    }

    /// 挂上决策成本探针出口（**仅测量用**，见 [`DecisionProbe`]）
    pub fn with_decision_probe(mut self, sink: Arc<Mutex<Vec<DecisionProbe>>>) -> Self {
        self.decision_probe = Some(sink);
        self
    }

    /// 本训练员真正走过搜索的决策次数
    pub fn searched_count(&self) -> usize {
        self.searched.load(Ordering::Relaxed)
    }

    /// 把 rollout 基策换成网络（`Q^手写` → `Q^NN`）
    ///
    /// `max_turn` 为网络生效的回合上限（含），`None` = 整局都用网络。
    /// 只影响**搜索内部**的模拟：未被门控选中的阶段仍由 [`Self::fallback`]
    /// 手写策略决定，本方法不改变它。
    ///
    /// ❗成本：网络单步推理约比手写慢一个量级，而 rollout 要跑到终局。
    /// 手写 rollout 下 `search_n=512` 已是约 25 分钟/局，全程网络会再乘上去；
    /// 先用 `max_turn` 做混合 rollout 或调小 `search_n` 定价，再决定预算。
    ///
    /// 同时打开 [`FlatSearch::with_strict_rollout`]：网络的推理失败很可能与局面
    /// 相关，默认那种「丢掉失败的 rollout 继续统计」会把估值变成以推理成功为
    /// 条件、且各候选条件互不相同，排序被污染而分数上看不出来。
    #[cfg(feature = "onnx")]
    pub fn with_nn_rollout(mut self, nn: Arc<super::RamenNnTrainer>, max_turn: Option<i32>) -> Self {
        self.search = self
            .search
            .map_rollout_trainer(|r| r.with_neural_net(nn, max_turn))
            .with_strict_rollout(true);
        self
    }

    /// 给内部搜索器挂上成本探针（**仅测量用**）
    ///
    /// 探针只观测、不参与分配与排序，搜索结果与 RNG 消耗逐位不变，
    /// 契约见 [`FlatSearch::with_probe`](crate::search::FlatSearch::with_probe)。
    /// 单独给一个转发方法，是因为 `search` 字段虽然是 `pub`，但 `with_probe`
    /// 取 `self` 按值返回，外部想挂探针得先把字段搬出来再放回去。
    pub fn with_search_probe(mut self, sink: Arc<Mutex<Vec<SearchProbe>>>) -> Self {
        self.search = self.search.with_probe(sink);
        self
    }

    /// 设置取分口径（`Pt` 为生产默认；研究入口显式选择 `Score`）
    pub fn with_selection(mut self, selection: RamenSelection) -> Self {
        self.selection = selection;
        self
    }

    /// 设置"友人出行必须走完 5 次"的完成硬门限（对应 `game_config.toml` 的
    /// `friend_complete_required`）：作用于 fallback 与 rollout 中的手写分支。
    /// 已装载的 NN 保留；网络动作不额外套用手写友人完成硬门限。
    pub fn with_friend_complete_required(mut self, required: bool) -> Self {
        self.fallback = self.fallback.with_friend_complete_required(required);
        // rollout 基策同步：搜索内部评估的未来必须与正式策略同一口径，
        // 否则 MCTS 会按"可以不走完"的世界线打分，与实际执行不一致。
        // 就地改写而非整体替换：已装载的网络 rollout 不能被这一步冲掉（调用顺序无关）。
        self.search = self
            .search
            .map_rollout_trainer(|r| r.with_friend_complete_required(required));
        self
    }

    /// 设置搜索阶段门控
    pub fn with_stages(mut self, stages: RamenSearchStages) -> Self {
        self.stages = stages;
        self
    }

    /// 设置是否输出每步决策日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// `SpecialSelect` 直接命中合并搜索缓存的次数
    pub fn combined_cache_hits(&self) -> usize {
        self.combined_cache_hits.load(Ordering::Relaxed)
    }

    /// 设置 `RamenSelect` 是否走合并动作搜索
    pub fn with_combined_ramen_select(mut self, on: bool) -> Self {
        self.use_combined_ramen_select = on;
        self
    }

    /// 设置决策理由原始数据出口（默认 [`DecisionReasonNoopSink`] 静默；需要原始 JSON 时传
    /// [`crate::output::LogJsonSink`] 或自定义实现）
    pub fn with_reason_sink(mut self, sink: Arc<dyn crate::output::DecisionReasonSink>) -> Self {
        self.reason_sink = sink;
        self
    }

    /// 输出决策理由：**只**把原始数据经 [`Self::reason_sink`] 发出，可读文字由宿主渲染
///
/// 每回合都调用 [`analyze_narrow_win`]；当前不再用分差门限决定是否输出，
/// 分差仅用于着色档位。比较口径保持上游真实 Score 轴，避免改变客户端理由展示。
/// 终局差异日志之后调用，两段日志可互相印证。
///
/// **2026-10 修改**：删掉 `verbose` 下的 `info!` 文字上屏——可读文字统一由宿主
/// （umaai human 模式）从 [`Self::reason_sink`]（`LastReasonSink`）取回后调
/// `render_reason_lines` 用 `println!` 渲染。此前同一份文字会同时出现在日志与屏幕
/// （链式决策里表现为「每个决策一组 log + 一组 print」），且该开关多次被误设回 `true`。
fn emit_decision_reason(&self, turn: i32, chosen: usize, output: &RamenSearchOutput) {
        let metric = ReasonMetric::Score;
        let Some(data) = analyze_narrow_win(
            turn,
            metric,
            self.search.config().reason_gap_threshold,
            self.search.config().reason_max_display,
            chosen,
            output
        ) else {
            return;
        };
        self.reason_sink.emit(&data);
    }

    /// 获取搜索配置
    pub fn config(&self) -> &SearchConfig {
        self.search.config()
    }

    /// 缓存本次搜索的候选统计（次数 / 均分 / 标准差 / PT 均分）
    fn stash_search_breakdown(&self, output: &RamenSearchOutput) {
        let text = output
            .actions
            .iter()
            .zip(output.action_results.iter())
            .enumerate()
            .map(|(i, (action, (res, res_pt)))| {
                format!(
                    "#{i} {action} n={} mean={:.0} sd={:.0} pt={:.0}",
                    res.count(),
                    res.mean(),
                    res.stdev(),
                    res_pt.mean()
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        if let Ok(mut slot) = self.last_breakdown.lock() {
            *slot = Some(text);
        }
    }

    /// 输出终局多维记录：其余候选相对**实际选中动作**的差值
    ///
    /// **debug 级输出**（2026-08-29 起）：屏幕默认（info）不再显示，需要
    /// 逐候选对照时把 log_level 调到 debug。数据本身不变。
    ///
    /// 只打差值而非绝对值：各候选的绝对面板高度相似，人眼分辨不出；
    /// 「选这个动作，最终智力会多 300」才是可读的因果陈述。
    ///
    /// 锚点取 `chosen`（即 `select_action` 真正返回的下标）。
    ///
    /// 差值只在**均值**层面成立。阈值类维度（`rmj_ok_*`）本身已是每次 rollout
    /// 内部归约出的 0/1，其均值是达成率，差值即达成率之差——不要再拿它与 PT
    /// 均值互推，那正是这套观测要避免的错误。
    fn log_terminal_breakdown(&self, turn: i32, chosen: usize, output: &RamenSearchOutput) {
        if !self.verbose || output.terminal_results.len() != output.actions.len() {
            return;
        }
        let Some(base) = output.terminal_results.get(chosen) else {
            return;
        };

        // 基准按 key 建表：两次 visit 靠键名配对，而不是靠下标。
        // 宏保证同类型的遍历顺序一致，但下标对齐正是 `NamedMetricRef` 要消灭的
        // 那种耦合——增删维度时不该出现静默错位。
        let mut base_dims: HashMap<&'static str, f64> = HashMap::new();
        base.visit(&mut |m| {
            base_dims.insert(m.key, m.result.mean());
        });

        for (i, action) in output.actions.iter().enumerate() {
            if i == chosen {
                continue;
            }
            let Some(stats) = output.terminal_results.get(i) else {
                continue;
            };
            let mut parts: Vec<String> = Vec::new();
            stats.visit(&mut |m| {
                let Some(base_mean) = base_dims.get(m.key) else {
                    return;
                };
                let delta = m.result.mean() - base_mean;
                // 只报可见差异，否则每行都被 20 余维刷屏
                match m.unit {
                    "flag" if delta.abs() >= 0.02 => {
                        parts.push(format!("{}{:+.0}%", m.key, delta * 100.0));
                    }
                    "flag" => {}
                    _ if delta.abs() >= 1.0 => parts.push(format!("{}{delta:+.0}", m.key)),
                    _ => {}
                }
            });
            if !parts.is_empty() {
                // 降级为 debug：屏幕默认（info 级）不再每候选刷一行，需要对照时
                // 把 log_level 调到 debug 即可恢复
                debug!("[回合 {}][终局差异] {action} vs 选中: {}", turn + 1, parts.join(" "));
            }
        }
    }

    /// 清空本次缓存（转发给手写策略时用，避免读到上一条搜索的陈旧文本）
    fn clear_breakdown(&self) {
        if let Ok(mut slot) = self.last_breakdown.lock() {
            *slot = None;
        }
    }

    /// 超级拉面平局回退：分数与选项二完全相同时改选选项二
    ///
    /// `deck_can_split == false`（卡组训练类型数 < 5）时
    /// [`RamenAction::distribute_super_ramen_clones`](crate::game::ramen::RamenAction)
    /// 直接早返回，三个选项对结局**完全等价**：CRN 下各候选逐位同分，
    /// `best_action_idx` 的 `max_by` 取到的是候选 0，于是打开 `super` 门控会把
    /// 手写一直固定的选项二静默换成选项一。**分数不变，变的是状态与日志**——
    /// 属于最难排查的一类差异。
    ///
    /// 因此只在**确实平局**时向手写回退对齐；一旦搜索真的分出高下，
    /// 就完全按搜索结果走，不干预。
    ///
    /// 非 `SuperRamenSelect` 阶段原样返回。
    ///
    /// 平局判定保留上游策略：PT 轴比 `.1.mean()`；研究用 Score 轴比 `.0.mean()`。
    /// 这里不随 rollout 后端变更调整上游的并列裁决规则。
    fn break_super_ramen_tie(
        game: &RamenGame, actions: &[<RamenGame as Game>::Action], output: &RamenSearchOutput,
        selection: RamenSelection, idx: usize
    ) -> usize {
        if game.stage != RamenStage::SuperRamenSelect {
            return idx;
        }
        let Some(fallback_idx) = actions.iter().position(|a| {
            matches!(a.operation, Operation::SuperRamenSelect(i) if i == FIXED_SUPER_RAMEN_INDEX)
        }) else {
            return idx;
        };
        if fallback_idx == idx {
            return idx;
        }
        // 取不到统计就不干预
        let (Some(chosen), Some(fallback)) =
            (output.action_results.get(idx), output.action_results.get(fallback_idx))
        else {
            return idx;
        };
        let metric = |r: &(ActionResult, ActionResult)| match selection {
            RamenSelection::Score => r.0.mean(),
            RamenSelection::Pt => r.1.mean()
        };
        if metric(chosen) == metric(fallback) { fallback_idx } else { idx }
    }

    /// 取出并清空合并搜索缓存的 targets
    fn take_pending_combined_targets(&self) -> Option<[i32; 3]> {
        match self.pending_combined_targets.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take()
        }
    }

    /// 写入合并搜索缓存的 targets（`None` 表示不吃面或不缓存）
    fn store_pending_combined_targets(&self, targets: Option<[i32; 3]>) {
        let mut slot = match self.pending_combined_targets.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner()
        };
        *slot = targets;
    }

    /// 把本次搜索摘要写入 `last_search_summary`（供 [`Trainer::last_decision`](crate::game::Trainer::last_decision) 读取）
    ///
    /// 仅缓存决策协议需要的字段（分数 / 局数 / 选中下标 + 候选描述），不复制整个
    /// [`RamenSearchOutput`]——`ActionResult.distribution` 数组 clone 成本过大。
    ///
    /// **分数口径 = 真实评分**（`SearchScore::score` / `calc_score()`）：本摘要供
    /// `last_decision()` → `candidate_scores` → 运气分 baseline 与 AIRed 展示使用，
    /// 故不采用被 `pt_favor_rate` 缩放的 `score_pt`（选择口径）。历史上此处误取
    /// `score_pt`，导致 `pt_favor_rate ≠ 1` 时运气分与显示候选分同步虚增。
    ///
    /// 2026-09 简化：移除 `reason_text` 计算——`DecisionInfo::reason` 删除后
    /// 该文本不再挂到决策协议。完整 `DecisionReasonData` 改由 `emit_decision_reason`
    /// 通过 `LastReasonSink` 缓存，挂到 `scenario_extra.reason`（main.rs 接线）。
    fn stash_last_summary(&self, output: &RamenSearchOutput, chosen_idx: usize) {
        // **真实评分轴**（`action_results` 是 `(score, score_pt)` 对）：
        // 取 `.0` 即 `calc_score()`，不含 `pt_favor_rate` 缩放。展示与运气分必须用
        // 真实评分——动作选择另走 [`Self::selection`] 指定的轴，两者互不影响。
        let scores: Vec<f64> = output.action_results.iter().map(|(score, _)| score.mean()).collect();
        let counts: Vec<u32> = output.action_results.iter().map(|(s, _)| s.count()).collect();
        // 候选可读描述：与 scores / counts 严格同长同序（按 action_results 顺序）
        let descriptions: Vec<String> = output
            .actions
            .iter()
            .map(|a| a.to_string())
            .collect();
        if let Ok(mut slot) = self.last_search_summary.lock() {
            *slot = Some(LastSearchSummary { chosen_idx, scores, counts, descriptions });
        }
    }

    /// 把**合并搜索**结果聚合为与三阶段 `actions` 对齐的搜索摘要
    ///
    /// 合并路径的候选是 `(ramen, targets)` 组合（最多约 28 个），与三阶段 `RamenSelect`
    /// 候选（`[不吃面] + [每个面一个]`）下标不对应，故不能直接走
    /// [`Self::stash_last_summary`]。此处按 `ramen` 把组合候选折回三阶段下标：同一个面的
    /// 多个 targets 变体按**局数加权均值**合并（`Σ(mean_i × n_i) / Σ n_i`）。
    ///
    /// **口径自洽**：聚合后 `Σ(candidate_scores[j] × candidate_n[j]) / Σ n` 恒等于
    /// 「全部合并候选的局数加权均值」——运气分 T(n) baseline 与合并搜索的动作空间
    /// 口径一致（不吃面只有一个候选，聚合退化为原值）。
    ///
    /// 分数取 `.0` 真实评分轴（`calc_score()`），与 [`Self::stash_last_summary`] 同口径。
    fn stash_combined_summary(
        &self,
        combined: &[crate::game::ramen::RamenAction],
        output: &RamenSearchOutput,
        actions: &[crate::game::ramen::RamenAction],
        chosen_idx: usize
    ) {
        let mut scores = vec![0.0_f64; actions.len()];
        let mut counts = vec![0_u32; actions.len()];
        for (j, act) in actions.iter().enumerate() {
            let mut weighted = 0.0_f64;
            let mut n = 0_u32;
            for (i, cand) in combined.iter().enumerate() {
                if cand.ramen != act.ramen {
                    continue;
                }
                let (res, _) = &output.action_results[i];
                weighted += res.mean() * res.count() as f64;
                n += res.count();
            }
            scores[j] = if n > 0 { weighted / n as f64 } else { 0.0 };
            counts[j] = n;
        }
        let descriptions: Vec<String> = actions.iter().map(|a| a.to_string()).collect();
        if let Ok(mut slot) = self.last_search_summary.lock() {
            *slot = Some(LastSearchSummary { chosen_idx, scores, counts, descriptions });
        }
    }

    /// 清空 `last_search_summary`（早退 / 转发 fallback 时调用）
    ///
    /// 与 [`Self::clear_breakdown`] 一一对应：搜索没真发生过，就不该让
    /// `last_decision` 看到上一次搜索的陈旧数据。
    fn clear_last_summary(&self) {
        if let Ok(mut slot) = self.last_search_summary.lock() {
            *slot = None;
        }
    }
}

impl Default for RamenMctsTrainer {
    fn default() -> Self {
        Self::new(SearchConfig::default())
    }
}

impl RamenMctsTrainer {
    /// [`Trainer::select_action`] 的原始实现
    ///
    /// 决策探针只在外层计时壳里读写，不进本函数——保证「挂不挂探针」不改变
    /// 这里的任何一步。
    fn select_action_inner(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        // (A) SpecialSelect 命中缓存 —— 必须放在早退判断之前。
        // 候选可能只有 1 个，或 stages.special_select 关着，这两种情况都要消费缓存，
        // 否则会污染下一回合的 SpecialSelect。
        if game.stage == RamenStage::SpecialSelect {
            if let Some(t) = self.take_pending_combined_targets() {
                match actions.iter().position(|a| a.special_targets == Some(t)) {
                    Some(idx) => {
                        self.combined_cache_hits.fetch_add(1, Ordering::Relaxed);
                        self.clear_breakdown();
                        self.clear_last_summary();
                        return Ok(idx);
                    }
                    None => {
                        bail!(
                            "SpecialSelect 缓存未命中: 缓存 targets={t:?}，实际候选=[{}]",
                            actions
                                .iter()
                                .map(|a| format!("{:?}", a.special_targets))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
        }

        // (C) RamenSelect 每次做决策时先无条件清一次，防止上一回合遗留
        if game.stage == RamenStage::RamenSelect {
            self.store_pending_combined_targets(None);
        }

        // 单候选无选择空间，跑搜索纯属浪费预算
        // 门控**必须**用未经纠正的 `game.stage`（第 1 年地区已是正规 `RegionSelect`）
        if actions.len() <= 1 || !self.stages.contains(&game.stage) {
            self.clear_breakdown();
            self.clear_last_summary();
            return self.fallback.select_action(game, actions, rng);
        }

        // (B) RamenSelect 走合并搜索（排除 race_turn：那边 list_actions 是比赛动作）
        if self.use_combined_ramen_select && game.stage == RamenStage::RamenSelect && !game.is_race_turn()
        {
            let combined = game.list_combined_ramen_select_actions();
            if combined.len() > 1 {
                self.searched.fetch_add(1, Ordering::Relaxed);
                let output = self.search.search(game, &combined, rng)?;
                let idx = match self.selection {
                    RamenSelection::Score => output.best_action_idx,
                    RamenSelection::Pt => output.best_action_pt_idx()
                };
                let best = combined
                    .get(idx)
                    .ok_or_else(|| anyhow!("合并搜索最优下标 {idx} 超出候选数 {}", combined.len()))?;
                // 不吃面时 next() 会直接推到 Train，不会有 SpecialSelect；留缓存会污染下一回合
                if best.ramen.is_none() {
                    self.store_pending_combined_targets(None);
                } else {
                    self.store_pending_combined_targets(best.special_targets);
                }
                self.stash_search_breakdown(&output);
                self.log_terminal_breakdown(game.turn() as i32, idx, &output);
                self.emit_decision_reason(game.turn() as i32, idx, &output);
                if self.verbose {
                    let (res, _) = &output.action_results[idx];
                    info!(
                        "[MCTS][回合 {}] 阶段 {:?} 合并 {} 候选 -> combined#{idx} {} (mean={:.0} n={})",
                        game.turn(),
                        game.stage,
                        combined.len(),
                        best,
                        res.mean(),
                        res.count()
                    );
                }
                match actions.iter().position(|a| a.ramen == best.ramen) {
                    Some(three_idx) => {
                        // 合并搜索的 candidates 是 (ramen, targets) 组合，与三阶段
                        // actions 列表的下标不对应（同一 ramen 跨多个 targets 候选）。
                        // 2026-09 修复：不再 `clear_last_summary()`——改为按 `ramen`
                        // 聚合回三阶段下标后暴露 DecisionInfo。原实现让 `last_decision`
                        // 返回 None，导致「只吃面」回合（吃面后不链式接训练决策）整回合
                        // 没有任何运气分更新（luck 只挂在带搜索评分的末决策上）。
                        self.stash_combined_summary(&combined, &output, actions, three_idx);
                        return Ok(three_idx);
                    }
                    None => {
                        bail!(
                            "RamenSelect 合并搜索结果在三阶段候选中找不到: best.ramen={:?}，实际候选=[{}]",
                            best.ramen,
                            actions
                                .iter()
                                .map(|a| format!("{:?}", a.ramen))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
            // combined.len() <= 1：不走合并，落回原逻辑
        }

        // 地区候选预过滤（可选，默认关）：第 3 年 `C(10,3)=120` 个候选是本局最大集合，
        // 仅初组 bootstrap 就有 `120 × search_group_size` 条 rollout。先用手写地区先验
        // （与 rollout 基策同源）排序取 top-K，再对子集跑常规 MCTS。
        // `ramen_region_prune_topk == 0` 时不进入本分支，生产行为逐位不变。
        let region_topk = self.search.config().ramen_region_prune_topk;
        if region_topk > 0 && game.stage == RamenStage::RegionSelect && actions.len() > region_topk {
            let (hand_idx, prior) = self.fallback.region_prior(game, actions)?;
            let mut keep: Vec<usize> = (0..actions.len()).collect();
            keep.sort_by(|&a, &b| prior[b].score.total_cmp(&prior[a].score));
            keep.truncate(region_topk);
            // 强制并入手写 argmax：保证剪枝结果不劣于「本点走纯手写」
            if !keep.contains(&hand_idx) {
                keep[region_topk - 1] = hand_idx;
            }
            // 子集按原始下标升序，便于与全量候选逐位对照
            keep.sort_unstable();
            let subset: Vec<<RamenGame as Game>::Action> =
                keep.iter().map(|&i| actions[i].clone()).collect();
            self.searched.fetch_add(1, Ordering::Relaxed);
            let output = self.search.search(game, &subset, rng)?;
            let idx = match self.selection {
                RamenSelection::Score => output.best_action_idx,
                RamenSelection::Pt => output.best_action_pt_idx()
            };
            self.stash_search_breakdown(&output);
            self.log_terminal_breakdown(game.turn() as i32, idx, &output);
            self.emit_decision_reason(game.turn() as i32, idx, &output);
            if self.verbose {
                let (res, _) = &output.action_results[idx];
                info!(
                    "[MCTS][回合 {}] 阶段 {:?} 地区预过滤 {}->{} -> #{idx} {} (mean={:.0} n={})",
                    game.turn(),
                    game.stage,
                    actions.len(),
                    subset.len(),
                    subset[idx],
                    res.mean(),
                    res.count()
                );
            }
            self.stash_last_summary(&output, idx);
            return Ok(keep[idx]);
        }

        self.searched.fetch_add(1, Ordering::Relaxed);
        let output = self.search.search(game, actions, rng)?;
        let idx = match self.selection {
            RamenSelection::Score => output.best_action_idx,
            RamenSelection::Pt => output.best_action_pt_idx()
        };
        let idx = Self::break_super_ramen_tie(game, actions, &output, self.selection, idx);
        self.stash_search_breakdown(&output);
        self.log_terminal_breakdown(game.turn() as i32, idx, &output);
        self.emit_decision_reason(game.turn() as i32, idx, &output);
        if self.verbose {
            let (res, _) = &output.action_results[idx];
            info!(
                "[MCTS][回合 {}] 阶段 {:?} {} 候选 -> #{idx} {} (mean={:.0} n={})",
                game.turn(),
                game.stage,
                actions.len(),
                actions[idx],
                res.mean(),
                res.count()
            );
        }
        self.stash_last_summary(&output, idx);
        Ok(idx)
    }
}

impl Trainer<RamenGame> for RamenMctsTrainer {
    /// 挂了决策探针时多一层计时壳，否则直接转 [`Self::select_action_inner`]
    ///
    /// 壳只读 `searched` / `combined_cache_hits` 的前后差值判断走了哪条路径，
    /// 不改变决策语义与 RNG 消耗。
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        let Some(sink) = self.decision_probe.as_ref() else {
            return self.select_action_inner(game, actions, rng);
        };
        let searched_before = self.searched.load(Ordering::Relaxed);
        let cache_before = self.combined_cache_hits.load(Ordering::Relaxed);
        let turn = game.turn();
        let stage = game.stage.clone();
        let started = Instant::now();
        let out = self.select_action_inner(game, actions, rng);
        let elapsed = started.elapsed();
        let path = if self.combined_cache_hits.load(Ordering::Relaxed) > cache_before {
            DecisionPath::CombinedCacheHit
        } else if self.searched.load(Ordering::Relaxed) > searched_before {
            DecisionPath::Searched
        } else if actions.len() <= 1 {
            DecisionPath::FallbackSingleCandidate
        } else {
            DecisionPath::FallbackGated
        };
        let probe = DecisionProbe {
            turn,
            stage,
            actions_len: actions.len(),
            path,
            exposes_decision_info: self
                .last_search_summary
                .lock()
                .map(|slot| slot.is_some())
                .unwrap_or(false),
            started,
            elapsed
        };
        match sink.lock() {
            Ok(mut v) => v.push(probe),
            Err(e) => debug!("[决策][探针] 记录失败（锁中毒）: {e}")
        }
        out
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_breakdown();
        self.clear_last_summary();
        self.fallback.select_choice(game, choices, rng)
    }


    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_breakdown();
        self.clear_last_summary();
        self.fallback.select_event_choice(game, event, choices, rng)
    }

    /// 搜索决策返回候选统计；转发决策返回手写策略自己的分解
    fn last_breakdown(&self) -> Option<String> {
        match self.last_breakdown.lock().ok().and_then(|slot| slot.clone()) {
            Some(text) => Some(text),
            None => self.fallback.last_breakdown()
        }
    }

    /// 上一次真正走过 MCTS 搜索的协议格式
    ///
    /// 普通搜索与合并搜索成功后返回 `Some`；合并候选先聚合回三阶段下标。
    /// 单候选、门控转发与 SpecialSelect 缓存命中清空摘要，避免显示旧搜索。
    ///
    /// 候选评分固定采用真实 Score 均值，与选动作的 PT/Score 轴独立；
    /// 按真实评分排序并截断到 `SearchConfig::reason_max_display`（分数与 `candidate_n` 同步）。
    /// 选中者若被截断在 top-N 之外则插入首位，`action_index` 重定位到截断后下标。
    ///
    /// `reason` 字段来自 [`summarize_ramen_reason`]（评分最高的未中选候选的最显著
    /// 终局维度差）；rivals 为空或维度不可见时为 `None`。
    fn last_decision(&self) -> Option<DecisionInfoProto> {
        let summary = self.last_search_summary.lock().ok()?.clone()?;
        if summary.scores.is_empty() || summary.chosen_idx >= summary.scores.len() {
            return None;
        }

        let max_n = self.search.config().reason_max_display.max(1);
        let mut indexed: Vec<(usize, f64, u32)> = summary
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s, summary.counts[i]))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let ordered: Vec<(usize, f64, u32)> = if indexed.len() <= max_n {
            indexed
        } else {
            indexed.truncate(max_n);
            // 选中者不在 top-N 时插入首位（极少见：MCTS 选中者基本总在前 max_n 内）
            if indexed.iter().any(|(i, _, _)| *i == summary.chosen_idx) {
                indexed
            } else {
                let mut v =
                    vec![(summary.chosen_idx, summary.scores[summary.chosen_idx], summary.counts[summary.chosen_idx])];
                v.extend(indexed);
                v
            }
        };

        let action_index = ordered.iter().position(|(i, _, _)| *i == summary.chosen_idx).unwrap_or(0);
        // 候选描述按 ordered 顺序取（与 candidate_scores / candidate_n 严格同长同序同截断）
        let candidate_descriptions: Vec<String> = ordered
            .iter()
            .map(|(i, _, _)| summary.descriptions[*i].clone())
            .collect();
        Some(DecisionInfoProto {
            action_index,
            score: summary.scores[summary.chosen_idx] as f32,
            // decision_kind 由 main.rs calc_ramen_training 内部 snapshot stage 填——
            // trainer 不感知 stage，按用户拍板"由发起决策的 umaai 从外部保存状态"
            decision_kind: String::new(),
            candidate_scores: ordered.iter().map(|(_, s, _)| *s as f32).collect(),
            candidate_descriptions,
            candidate_n: ordered.iter().map(|(_, _, n)| *n).collect(),
            scenario_extra: None
        })
    }
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;
    use crate::{
        gamedata::{GAMECONSTANTS, init_global},
        global,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 准备一局固定种子的拉面局面
    fn setup(seed: u64) -> Result<(RamenGame, StdRng)> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let (decision_rng, rule_master) = crate::bench::seeded_rngs(seed, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        Ok((game, decision_rng))
    }

    /// 阶段门控字符串解析
    #[test]
    fn test_search_stages_parse() -> Result<()> {
        let mut c = Checks::new();
        let s = RamenSearchStages::parse("train,ramen")?;
        println!("parse(train,ramen) = {s:?}");
        c.check(
            s.train && s.ramen_select && !s.special_select && !s.region_select && !s.super_ramen_select,
            "只有 train / ramen_select 为真"
        );
        c.check(s.contains(&RamenStage::Train), "Train 命中");
        c.check(!s.contains(&RamenStage::SpecialSelect), "SpecialSelect 不命中");
        c.check(!RamenSearchStages::all().contains(&RamenStage::Begin), "Begin 永不命中");
        c.check(
            !RamenSearchStages::all().contains(&RamenStage::BeginAfterRegionSelect),
            "BeginAfterRegionSelect 永不命中"
        );

        // 四类必须报错的输入：静默接受会让实验对照组悄悄退化成纯手写策略
        c.check(RamenSearchStages::parse("train,bogus").is_err(), "未知阶段名报错");
        c.check(RamenSearchStages::parse("").is_err(), "空串报错（不静默当 none）");
        c.check(RamenSearchStages::parse(" , ").is_err(), "只有逗号报错");
        c.check(RamenSearchStages::parse("train,none").is_err(), "none 与其他名混用报错");
        c.check(RamenSearchStages::parse("all,train").is_err(), "all 与其他名混用报错");
        c.check(RamenSearchStages::parse("all")?.train, "单独 all 有效");
        c.check(!RamenSearchStages::parse("none")?.train, "单独 none 有效");
        c.finish()
    }

    /// 第 1 年地区选择已是阶段边界上的正规决策点，门控 `region` 必须进搜索
    #[test]
    fn test_year1_region_is_searched() -> Result<()> {
        use crate::{
            game::ramen::Operation,
            trainer::ramen_handwritten_trainer::ramen_effective_stage
        };

        let mut c = Checks::new();
        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        game.run_stage(&hw, &mut rng)?;
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::RegionSelect && game.turn() == 2 {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        c.check(reached, "真实推进到 turn 2 RegionSelect");
        c.check(game.turn() == 2, "回合仍为 2");

        let actions = game.list_actions()?;
        println!(
            "根: turn={} stage={:?} 候选={}",
            game.turn(),
            game.stage,
            actions.len()
        );
        c.check(actions.len() > 1, "第 1 年有多个地区候选");
        c.check(
            actions
                .iter()
                .all(|a| matches!(a.operation, Operation::RegionSelect(_))),
            "候选全是 RegionSelect（不是训练+吃面回退）"
        );

        let eff = ramen_effective_stage(&game, &actions);
        println!("ramen_effective_stage = {eff:?} raw = {:?}", game.stage);
        c.check(eff == RamenStage::RegionSelect, "有效阶段是 RegionSelect");
        c.check(game.stage == RamenStage::RegionSelect, "raw game.stage 已是 RegionSelect");

        let gate = RamenSearchStages {
            region_select: true,
            ..RamenSearchStages::none()
        };
        c.check(gate.contains(&game.stage), "门控 region 命中第 1 年");

        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(2).with_ucb(false))
            .with_stages(gate);
        let _idx = trainer.select_action(&game, &actions, &mut rng)?;
        c.check(trainer.searched_count() == 1, "第 1 年 RegionSelect 走过搜索");
        c.finish()
    }

    /// 地区候选预过滤：`ramen_region_prune_topk=K` 时只搜手写先验 top-K（并强制并入手写 argmax），
    /// 返回值必落在该集合内；`K=0` 时行为由 [`Self::test_stages_none_matches_recommended`] 等既有对拍覆盖。
    #[test]
    fn test_region_prune_topk_returns_kept_candidate() -> Result<()> {
        let mut c = Checks::new();
        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        game.run_stage(&hw, &mut rng)?;
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::RegionSelect && game.turn() == 2 {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        c.check(reached, "真实推进到 turn 2 RegionSelect");

        let actions = game.list_actions()?;
        c.check(actions.len() > 3, "第 1 年候选数 > K，剪枝确实发生");

        // 与实现同口径重建期望保留集合：按手写地区先验降序取 top-3，并强制并入 argmax
        let (hand_idx, prior) = hw.region_prior(&game, &actions)?;
        let mut expect: Vec<usize> = (0..actions.len()).collect();
        expect.sort_by(|&a, &b| prior[b].score.total_cmp(&prior[a].score));
        expect.truncate(3);
        if !expect.contains(&hand_idx) {
            expect[2] = hand_idx;
        }

        let gate = RamenSearchStages {
            region_select: true,
            ..RamenSearchStages::none()
        };
        let trainer = RamenMctsTrainer::new(
            SearchConfig::default()
                .with_search_n(2)
                .with_ucb(false)
                .with_ramen_region_prune_topk(3)
        )
        .with_stages(gate);
        let idx = trainer.select_action(&game, &actions, &mut rng)?;
        println!("地区剪枝 K=3: 返回 idx={idx}，保留集合={expect:?}");
        c.check(trainer.searched_count() == 1, "地区剪枝分支走过搜索");
        c.check(expect.contains(&idx), "返回值落在预过滤保留集合内");
        c.finish()
    }

    /// 门控全关时必须与正式推荐策略 [`RecommendedRamenTrainer`] **逐位一致**
    ///
    /// 这是实验的对照组正确性前提：若两者不一致，说明 MCTS 壳自己额外消耗了
    /// 随机流或改了决策，后续「搜索提分多少」的差值就无从归因。
    /// 2026-08-27 切换：原对照 `RamenHandwrittenTrainer`（纯 RamenPolicy，缺平衡/联动等
    /// 机制）已不再是生产路径；现在对照正式推荐策略，等同于把搜索壳的"无操作"边界钉死。
    #[test]
    fn test_stages_none_matches_recommended() -> Result<()> {
        let seed = 42;

        let (mut game_rec, mut rng_rec) = setup(seed)?;
        game_rec.run_full_game(&RecommendedRamenTrainer::new(), &mut rng_rec)?;
        let score_rec = game_rec.uma.calc_score();

        let (mut game_mcts, mut rng_mcts) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(8))
            .with_stages(RamenSearchStages::none());
        game_mcts.run_full_game(&trainer, &mut rng_mcts)?;
        let score_mcts = game_mcts.uma.calc_score();

        let mut c = Checks::new();
        println!("推荐={score_rec} / MCTS(stages=none)={score_mcts}");
        println!(
            "  五维 {:?} vs {:?}  PT {} vs {}  super_ramen {:?} vs {:?}",
            game_rec.uma.five_status,
            game_mcts.uma.five_status,
            game_rec.ramen.scenario_pt,
            game_mcts.ramen.scenario_pt,
            game_rec.ramen.super_ramen,
            game_mcts.ramen.super_ramen
        );
        c.check(score_rec == score_mcts, "门控全关 == 推荐策略");
        c.check(game_rec.uma.five_status == game_mcts.uma.five_status, "五维一致");
        c.check(game_rec.uma.skill_pt == game_mcts.uma.skill_pt, "技能点一致");
        c.check(game_rec.ramen.scenario_pt == game_mcts.ramen.scenario_pt, "剧本 PT 一致");
        c.check(game_rec.ramen.super_ramen == game_mcts.ramen.super_ramen, "super_ramen 一致");
        // 2026-09-18：preset 起 super_choice_mode=3（按终盘缺口与卡型数选范围），本局不是平局，
        // 不再固定落在选项二；钉具体值以防选择来源被悄悄换掉。
        // 2026-09-21：配额定档 [0,3,5] 后本局超级拉面落入选项二（sup=1）。
        c.check(game_rec.ramen.super_ramen == Some(1), "门控关时与推荐策略同选（选项二）");
        c.check(trainer.searched_count() == 0, "门控全关时一次搜索都没发生");
        c.finish()
    }

    /// 只搜训练阶段跑通整局（小预算冒烟）
    #[test]
    fn test_mcts_train_only_full_game() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::train_only());
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let score = game.uma.calc_score();
        println!(
            "MCTS(train, search_n=4) 整局: 回合={} 评分={} ({}) 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            global!(GAMECONSTANTS).get_rank_name(score)
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(score > 0, "评分为正");
        // 「末次决策有 breakdown」几乎恒真（末步必是转发、回落到手写分解），
        // 改为统计整局真正走过搜索的次数——这才是门控生效的证据
        println!("  整局走搜索的决策数={}", trainer.searched_count());
        c.check(trainer.searched_count() > 0, "确实走过搜索");
        c.check(trainer.searched_count() <= 80, "只搜 Train（约 69 个点），没有蔓延到其他阶段");
        c.finish()
    }

    /// rollout 的根动作必须走策略流，不能走通用 `apply_action`
    ///
    /// 真实对局中 `run_train` 用 `apply_action_with_strategy`（优先用局面内策略流），
    /// 而旧 `simulate_common` 直接 `apply_action(action, rng)`。本测试扫过整局所有
    /// 多候选 Train 决策点，统计两条路径跑到终局的分数有多少个点不同——
    /// 若一个都不同不了，说明该修复是空操作，需要重新评估。
    #[test]
    fn test_root_action_uses_strategy_stream() -> Result<()> {
        use rand::SeedableRng;

        use crate::search::FlatSearchGame;

        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        let seed = 12345u64;
        let (mut checked, mut differ) = (0usize, 0usize);
        let mut first_diff = None;

        while game.next() {
            if matches!(game.stage, RamenStage::Train) {
                let actions = game.list_actions()?;
                if actions.len() > 1 {
                    // 同一个动作、同一个种子，两条 apply 路径各自跑到终局
                    let mut scores = [0i32; 2];
                    for (k, score) in scores.iter_mut().enumerate() {
                        let mut g = game.fork_for_rollout(seed);
                        let mut r = StdRng::seed_from_u64(seed);
                        if k == 0 {
                            g.apply_action(&actions[0], &mut r)?;
                        } else {
                            g.apply_root_action(&actions[0], &mut r)?;
                        }
                        while g.next() {
                            g.run_stage(&hw, &mut r)?;
                        }
                        *score = g.uma.calc_score();
                    }
                    checked += 1;
                    if scores[0] != scores[1] {
                        differ += 1;
                        first_diff.get_or_insert((game.turn(), scores[0], scores[1]));
                    }
                }
            }
            game.run_stage(&hw, &mut rng)?;
        }

        println!("扫过 {checked} 个多候选 Train 决策点，其中 {differ} 个两条路径终局分数不同");
        if let Some((turn, a, b)) = first_diff {
            println!("  首个差异: 回合 {turn} 通用={a} 策略流={b}");
        }
        let mut c = Checks::new();
        c.check(differ > 0, "修复非空操作（至少一个 Train 决策点两条路径结果不同）");
        c.finish()
    }

    /// 同种子两次整局结果一致（搜索层的 CRN 种子由传入 rng 派生）
    #[test]
    fn test_mcts_reproducible() -> Result<()> {
        let seed = 7;
        let mut scores = Vec::new();
        for _ in 0..2 {
            let (mut game, mut rng) = setup(seed)?;
            let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
                .with_stages(RamenSearchStages::train_only());
            game.run_full_game(&trainer, &mut rng)?;
            scores.push(game.uma.calc_score());
        }
        let mut c = Checks::new();
        println!("两次评分: {scores:?}");
        c.check(scores[0] == scores[1], "可复现");
        c.finish()
    }

    /// 吃面 + 隐藏风味两阶段都搜（P1.2 / P1.3 对照与测量用）
    fn ramen_and_special_stages() -> RamenSearchStages {
        RamenSearchStages {
            ramen_select: true,
            special_select: true,
            ..RamenSearchStages::none()
        }
    }

    /// 按需运行：整局输出终局多维诊断，人工看可读性
    ///
    /// 不是断言测试，是**给合作伙伴看仪表长什么样**的观察壳，故 `#[ignore]`。
    /// 手动跑：
    /// `cargo test -p umasim --lib -- test_terminal_breakdown_demo --ignored --nocapture`
    #[test]
    #[ignore = "整局诊断输出演示，按需手动运行"]
    fn test_terminal_breakdown_demo() -> Result<()> {
        // 必须早于 setup：全局 logger 只初始化一次，setup 里设的是 error 级，
        // 会把诊断用的 info! 整个吞掉
        let _ = init_test_logger("info");
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        // search_n 取小值：本壳看的是输出形态，不是分数
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(16).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .verbose(true);
        game.run_full_game(&trainer, &mut rng)?;

        println!(
            "整局结束: 评分={} 五维={:?} 上限={:?} skill_pt={} 逐年PT={:?} RMJ={:?}",
            game.uma.calc_score(),
            game.uma.five_status,
            game.uma.five_status_limit,
            game.uma.skill_pt,
            game.ramen.yearly_scenario_pt,
            game.ramen.rmj_results
        );
        Ok(())
    }

    /// 硬性验收 1 的对照尺子：`use_combined_ramen_select = false` 必须与改动前逐位相同
    ///
    /// 改动前（字段尚不存在、等价于三阶段分别搜）实测：
    /// 评分=55153 五维=[2958, 1742, 2200, 866, 1112] skill_pt=7390 scenario_pt=0 searched_count=46
    #[test]
    fn test_combined_gate_off_full_game() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(false);
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let score = game.uma.calc_score();
        let searched = trainer.searched_count();
        println!(
            "gate-off 整局: 回合={} 评分={} 五维={:?} skill_pt={} scenario_pt={} searched_count={} 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            game.uma.five_status,
            game.uma.skill_pt,
            game.ramen.scenario_pt,
            searched
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        // 2026-08-25 更新：不在判定与得意率解耦 + 地区分身缺席优先，模拟数值变化，基准重抓
        // 2026-08-27 更新（两次叠加）：
        // (1) 五维上限剧本化，速度上限 2958→3337，整局数值变化；
        // (2) fallback 与 rollout 均切到 RecommendedRamenTrainer。
        //     ⚠ gate-off **不是**纯推荐策略跑局——本测试用 ramen_and_special_stages()，
        //     ramen/special 两阶段仍在搜（searched_count=66），只是不合并成单动作。
        //     纯推荐策略的对照在 test_stages_none_matches_recommended（stages=none，
        //     searched_count=0），同卡组 seed=42 的纯推荐快照见 bench.rs 的 64336。
        //     别拿这里的 62698 当 REC 基线，会误判搜索掉分幅度。
        // 上游 (2) 抓的 66705 / [3258,...] 是在 (1) 之前测的，两者叠加后已在本分支重抓。
        // 2026-09 更新：吃面 PT 增量 / eat_count 延后到 NextTurn，训练阶段用吃面前 PT
        // 算 ramen_pt_effect / region_bonus 档位，整局数值变化（拉面效果变弱导致整局偏低），
        // 基准重抓。
        // 2026-09-18 重抓：上一版数值早于 preset 定稿（本次改动实测逐位不变，仅为同步）。
        // 2026-09-21 重抓：友人出行跨年配额定档 [0,3,5]（原 [0,2,5]），整局路径变化。
        // 2026-10-04 重抓：超级拉面效果修正（只保留 RMJ + finals、接入选中选项的
        // +100 训练上限），URA 训练数值变化，整局路径与终局数值变化。
        // 2026-10-06 重抓：PT 上段上限口径修正——普通回合 `ramen_basic_effect.status_limit`
        // 同时抬属性与 PT 上限（Y2 +20 / Y3 +40），吃面回合 PT 上限下降，整局路径与终局数值变化。
        // 2026-10-06 重抓：五维评分表换用 URA `StatusToPoint`（raw 表 3802 项），
        // 手写策略的 marginal gain 随之下调，决策路径与终局数值整体变化。
        // 2026-10-07 重抓：拉面 PT 口径改为 M2——PT 友情**不**剔除 RMJ（友情对属性与 PT 同时生效），
        // 且吃面回合的效果档位按**吃面前** PT 取。整局路径与终局数值变化。
        c.check(score == 61472, "评分与改动前逐位相同");
        c.check(
            game.uma.five_status == [3337, 1793, 2196, 936, 1213],
            "五维与改动前逐位相同"
        );
        c.check(game.uma.skill_pt == 8002, "技能点与改动前逐位相同");
        c.check(game.ramen.scenario_pt == 0, "剧本 PT 与改动前逐位相同");
        c.check(searched == 61, "searched_count 与改动前逐位相同");
        c.finish()
    }

    /// 默认打开合并搜索；链式 setter 能关掉
    #[test]
    fn test_combined_default_on() -> Result<()> {
        // `RamenMctsTrainer::default()` 会构造 `HandwrittenEvaluator`，后者
        // `load_onsen_order().expect(..)` 依赖工作目录与全局数据；不初始化则本测试
        // 只在别的测试先跑过时才碰巧通过（顺序依赖）。
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let t = RamenMctsTrainer::default();
        println!("default use_combined_ramen_select = {}", t.use_combined_ramen_select);
        let mut c = Checks::new();
        c.check(t.use_combined_ramen_select, "new()/default 默认打开");
        c.check(t.selection == RamenSelection::Pt, "生产默认保持上游 PT 选择轴");
        let t2 = t.with_combined_ramen_select(false);
        println!(
            "after with_combined_ramen_select(false) = {}",
            t2.use_combined_ramen_select
        );
        c.check(!t2.use_combined_ramen_select, "setter 关闭");
        let research = t2.with_selection(RamenSelection::Score);
        c.check(research.selection == RamenSelection::Score, "研究入口可显式选择 Score 轴");
        c.finish()
    }

    /// 只统计不干预的包装训练员：记录各阶段调用次数，以及其中真正走过搜索的次数
    struct CountingTrainer {
        /// 被包装的 MCTS 训练员
        inner: RamenMctsTrainer,
        /// `RamenSelect` 的 `select_action` 调用次数
        ramen_select_calls: AtomicUsize,
        /// `SpecialSelect` 的 `select_action` 调用次数
        special_select_calls: AtomicUsize,
        /// `RamenSelect` 中真正走过搜索的次数
        ramen_select_searches: AtomicUsize,
        /// `SpecialSelect` 中真正走过搜索的次数
        special_select_searches: AtomicUsize
    }

    impl CountingTrainer {
        /// 包装一个已构造好的 `RamenMctsTrainer`
        fn wrap(inner: RamenMctsTrainer) -> Self {
            Self {
                inner,
                ramen_select_calls: AtomicUsize::new(0),
                special_select_calls: AtomicUsize::new(0),
                ramen_select_searches: AtomicUsize::new(0),
                special_select_searches: AtomicUsize::new(0)
            }
        }
    }

    impl Trainer<RamenGame> for CountingTrainer {
        fn select_action(
            &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
        ) -> Result<usize> {
            let before = self.inner.searched_count();
            let idx = self.inner.select_action(game, actions, rng)?;
            let did_search = self.inner.searched_count() > before;
            match game.stage {
                RamenStage::RamenSelect => {
                    self.ramen_select_calls.fetch_add(1, Ordering::Relaxed);
                    if did_search {
                        self.ramen_select_searches.fetch_add(1, Ordering::Relaxed);
                    }
                }
                RamenStage::SpecialSelect => {
                    self.special_select_calls.fetch_add(1, Ordering::Relaxed);
                    if did_search {
                        self.special_select_searches.fetch_add(1, Ordering::Relaxed);
                    }
                }
                _ => {}
            }
            Ok(idx)
        }

        fn select_choice(
            &self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            self.inner.select_choice(game, choices, rng)
        }

        fn select_event_choice(
            &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            self.inner.select_event_choice(game, event, choices, rng)
        }

        fn last_breakdown(&self) -> Option<String> {
            self.inner.last_breakdown()
        }
    }

    /// 硬性验收 2：合并开启时 SpecialSelect 大多数走缓存命中，少数走搜索
    ///
    /// 2026-08-27 修订：原断言 `special_searches == 0` 在 fallback 切到 `RecommendedRamenTrainer`
    /// 后偶发失败——race_turn 时 `RamenSelect` 走非合并搜索路径（缓存写不进去），若 trainer
    /// 在该回合选了某个 ramen，下一阶段 SpecialSelect 出现时缓存 miss 必须重搜一次。这是
    /// REC 决策倾向带来的合法新行为，不是缓存检查逻辑问题。
    ///
    /// 2026-08-28 再修订：上一版把断言改成 `special_calls > special_searches`，实测这局是
    /// 29 次调用、1 次重搜——该条件下搜 28 次也能绿，等于没有守门。现钉逐位快照
    /// `special_calls == 29` 与 `special_searches == 1`（与本文件其余快照同口径），
    /// 占比上界只作第二道网。重抓快照时请一并核对重搜数没有变大。
    #[test]
    fn test_combined_on_skips_special_search() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let inner = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(true);
        let trainer = CountingTrainer::wrap(inner);
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let score = game.uma.calc_score();
        let ramen_calls = trainer.ramen_select_calls.load(Ordering::Relaxed);
        let special_calls = trainer.special_select_calls.load(Ordering::Relaxed);
        let ramen_searches = trainer.ramen_select_searches.load(Ordering::Relaxed);
        let special_searches = trainer.special_select_searches.load(Ordering::Relaxed);
        let searched = trainer.inner.searched_count();
        println!(
            "gate-on 整局: 回合={} 评分={} searched_count={} 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            searched
        );
        println!(
            "  RamenSelect 调用={ramen_calls} 搜索={ramen_searches} / SpecialSelect 调用={special_calls} 搜索={special_searches}"
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(score > 0, "评分为正");
        c.check(ramen_calls > 0, "RamenSelect 被调用过");
        c.check(ramen_searches > 0, "RamenSelect 走过搜索");
        c.check(special_calls > 0, "SpecialSelect 被调用过（缓存命中路径）");
        // 2026-08-28 收紧：原断言 `special_calls > special_searches` 在 29 次调用里
        // 搜 28 次也绿，等于没有守门。合并路径整个失效都抓不住。
        // 改回本文件通行的逐位快照：29 次调用只有 1 次重搜（第 3 年 race_turn 选面，
        // `select_action` 的合并短路 `!game.is_race_turn()` 不成立，见本文件 495-547）。
        // 2026-09 更新：吃面 PT 增量延后到 NextTurn 后，本回合 PT 档位提升延后生效，
        // 整局搜索路径微小变化，SpecialSelect 调用 / 重搜数基线重抓。
        // 2026-09-18 重抓：上一版快照早于 preset 定稿（本次改动实测逐位不变，仅为同步）。
        // 2026-09-21 重抓：友人出行配额定档 [0,3,5]，SpecialSelect 调用 28→29
        // （重搜仍为 0，语义上界断言不变）。
        // 2026-10-04 重抓：超级拉面效果修正（RMJ + finals、选中选项 +100 上限），
        // 整局搜索路径变化，SpecialSelect 调用 29、重搜 0。
        // 2026-10-06 重抓：PT 上段上限口径修正（普通回合 basic.status_limit 同时抬 PT 上限），
        // 吃面回合 PT 上限下降 → 决策倾向变化，SpecialSelect 调用 29→26、重搜 0→2。
        // 2026-10-06 重抓：五维评分表换用 URA `StatusToPoint`，决策倾向再变，
        // SpecialSelect 调用 26→27、重搜 2→1。
        // 2026-10-07 重抓：拉面 PT 口径改为 M2（PT 友情不剔 RMJ + 吃面按吃面前 PT），
        // 决策倾向再变，SpecialSelect 调用 27→30、重搜 1→0。
        c.check(special_calls == 30, "SpecialSelect 调用数与改动前逐位相同");
        c.check(special_searches == 0, "SpecialSelect 重搜数与改动前逐位相同");
        // 再留一条与具体数字解耦的语义上界，防止将来重抓快照时把比例抬上去
        c.check(
            special_searches * 5 < special_calls,
            "SpecialSelect 绝大多数走缓存命中（重搜占比 < 20%）"
        );
        c.check(
            searched == ramen_searches + special_searches,
            "整局 searched_count 等于两阶段搜索合计"
        );
        c.finish()
    }

    /// 只搜 `ramen`、不搜 `special` 时，合并搜索选出的 targets 仍必须被采用
    ///
    /// 钉「缓存检查必须在门控早退之前」：挪到早退之后，targets 会被静默丢弃、
    /// 改由手写策略另选，分数上看不出来，只有 `combined_cache_hits()` 归零才暴露。
    #[test]
    fn test_combined_cache_used_when_special_gate_off() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let stages = RamenSearchStages {
            ramen_select: true,
            ..RamenSearchStages::none()
        };
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(stages)
            .with_combined_ramen_select(true);
        game.run_full_game(&trainer, &mut rng)?;
        let hits = trainer.combined_cache_hits();
        println!(
            "special 门控关: 回合={} 评分={} searched_count={} combined_cache_hits={hits}",
            game.turn(),
            game.uma.calc_score(),
            trainer.searched_count()
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(trainer.searched_count() > 0, "RamenSelect 走过合并搜索");
        c.check(hits > 0, "SpecialSelect 必须命中合并缓存（门控关也要用）");
        c.finish()
    }

    /// 门控 `super`：整局恰好搜索一次；门控关时为 0
    #[test]
    fn test_super_ramen_gate_searches_once() -> Result<()> {
        let seed = 42;

        let (mut game_on, mut rng_on) = setup(seed)?;
        let stages_on = RamenSearchStages {
            super_ramen_select: true,
            ..RamenSearchStages::none()
        };
        let trainer_on = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(stages_on);
        game_on.run_full_game(&trainer_on, &mut rng_on)?;
        let searched_on = trainer_on.searched_count();
        println!(
            "gate=super: 回合={} 评分={} searched={} super_ramen={:?}",
            game_on.turn(),
            game_on.uma.calc_score(),
            searched_on,
            game_on.ramen.super_ramen
        );

        let (mut game_off, mut rng_off) = setup(seed)?;
        let trainer_off = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::none());
        game_off.run_full_game(&trainer_off, &mut rng_off)?;
        let searched_off = trainer_off.searched_count();
        println!(
            "gate=none: 回合={} searched={} super_ramen={:?}",
            game_off.turn(),
            searched_off,
            game_off.ramen.super_ramen
        );

        let mut c = Checks::new();
        c.check(game_on.turn() == 77, "门控开跑满 77 回合");
        c.check(searched_on == 1, "门控 super 整局恰好搜索一次");
        c.check(searched_off == 0, "门控关时一次搜索都没有");
        // 2026-09-18：preset 起 super_choice_mode=3，本局落在选项一（同 test_stages_none_matches_recommended）。
        c.check(game_off.ramen.super_ramen == Some(1), "门控关与推荐策略同选（选项二）");
        c.finish()
    }

    /// 根节点冒烟：真实推进到 SuperRamenSelect，小 search_n 跑通 3 候选，
    /// apply_root_action 后下一阶段是 turn 72 的 Begin
    #[test]
    fn test_super_ramen_search_root_smoke() -> Result<()> {
        use crate::search::{FlatSearch, FlatSearchGame};

        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        game.run_stage(&hw, &mut rng)?;
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::SuperRamenSelect {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        let mut c = Checks::new();
        c.check(reached, "真实推进到 SuperRamenSelect");
        c.check(game.turn() == 71, "超级拉面选择发生在回合 71");

        let actions = game.list_actions()?;
        println!(
            "根: turn={} stage={:?} 候选={} {:?}",
            game.turn(),
            game.stage,
            actions.len(),
            actions.iter().map(|a| a.to_string()).collect::<Vec<_>>()
        );
        c.check(actions.len() == 3, "根上恰好 3 个候选");

        let search = FlatSearch::<RamenGame>::new(SearchConfig::default().with_search_n(4).with_ucb(false));
        let output = search.search(&game, &actions, &mut rng)?;
        println!(
            "search 最优 #{} {} 各候选 n={:?}",
            output.best_action_pt_idx(),
            actions[output.best_action_pt_idx()],
            output.action_results.iter().map(|(r, _)| r.count()).collect::<Vec<_>>()
        );
        c.check(output.action_results.len() == 3, "搜索覆盖 3 个候选");
        c.check(
            output.action_results.iter().all(|(r, _)| r.count() > 0),
            "每个候选都有样本"
        );

        let best = &actions[output.best_action_pt_idx()];
        game.apply_root_action(best, &mut rng)?;
        c.check(game.stage == RamenStage::SuperRamenSelect, "apply_root_action 不切阶段");
        c.check(game.turn() == 71, "apply_root_action 不推进回合");
        c.check(game.ramen.super_ramen.is_some(), "根动作已写入 super_ramen");

        let advanced = game.next();
        println!("next()={} turn={} stage={:?}", advanced, game.turn(), game.stage);
        c.check(advanced, "next() 能推进");
        c.check(game.turn() == 72, "下一回合是 72");
        c.check(game.stage == RamenStage::Begin, "下一阶段是 Begin");
        c.finish()
    }

    /// 门控 `region`：三年 RegionSelect 都是多候选且门控命中（测试 init 为 All）
    ///
    /// 第 3 年 `ramen_region_strategy=fixed` 时 `list_actions` 只有 1 个候选，
    /// 不会进搜索（少一次）。本测试不跑搜索，只数会触发搜索的决策点。
    #[test]
    fn test_region_gate_three_years() -> Result<()> {
        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        let gate = RamenSearchStages {
            region_select: true,
            ..RamenSearchStages::none()
        };
        let mut visits = 0usize;
        let mut searchable = 0usize;
        game.run_stage(&hw, &mut rng)?;
        while game.next() {
            if game.stage == RamenStage::RegionSelect {
                visits += 1;
                let actions = game.list_actions()?;
                let would = actions.len() > 1 && gate.contains(&game.stage);
                println!(
                    "RegionSelect turn={} 候选={} would_search={would}",
                    game.turn(),
                    actions.len()
                );
                if would {
                    searchable += 1;
                }
            }
            game.run_stage(&hw, &mut rng)?;
        }
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(visits == 3, "三年各到一次 RegionSelect");
        c.check(searchable == 3, "All 策略下三年都是多候选，门控各搜一次");
        c.finish()
    }

    /// 第 1 年根交给 FlatSearch：每个候选都能跑到终局且有样本；同根同种子两次逐位一致
    #[test]
    fn test_year1_region_search_root_smoke() -> Result<()> {
        use crate::search::{FlatSearch, FlatSearchGame};

        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        game.run_stage(&hw, &mut rng)?;
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::RegionSelect && game.turn() == 2 {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        let mut c = Checks::new();
        c.check(reached, "真实推进到 turn 2 RegionSelect");
        let actions = game.list_actions()?;
        println!(
            "根: turn={} stage={:?} 候选={}",
            game.turn(),
            game.stage,
            actions.len()
        );
        c.check(actions.len() > 1, "第 1 年多个地区候选");

        let search = FlatSearch::<RamenGame>::new(SearchConfig::default().with_search_n(2).with_ucb(false));
        let output = search.search(&game, &actions, &mut rng)?;
        println!(
            "search 最优 #{} 各候选 n={:?}",
            output.best_action_pt_idx(),
            output.action_results.iter().map(|(r, _)| r.count()).collect::<Vec<_>>()
        );
        c.check(output.action_results.len() == actions.len(), "搜索覆盖全部候选");
        c.check(
            output.action_results.iter().all(|(r, _)| r.count() > 0),
            "每个候选都有样本"
        );

        use rand::SeedableRng;
        let search2 = FlatSearch::<RamenGame>::new(SearchConfig::default().with_search_n(2).with_ucb(false));
        let mut rng_a = StdRng::seed_from_u64(99);
        let mut rng_b = StdRng::seed_from_u64(99);
        let a = search.search(&game, &actions, &mut rng_a)?;
        let b = search2.search(&game, &actions, &mut rng_b)?;
        let same = a.action_results.iter().zip(b.action_results.iter()).all(|((ra, _), (rb, _))| {
            ra.count() == rb.count() && (ra.mean() - rb.mean()).abs() < f64::EPSILON
        });
        println!(
            "同根同种子两次: best {} vs {} same={same}",
            a.best_action_pt_idx(), b.best_action_pt_idx()
        );
        c.check(same, "同根同种子两次逐位一致");
        c.check(a.best_action_pt_idx() == b.best_action_pt_idx(), "最优下标一致");

        let best = &actions[output.best_action_pt_idx()];
        game.apply_root_action(best, &mut rng)?;
        c.check(game.stage == RamenStage::RegionSelect, "apply_root_action 不切阶段");
        c.check(game.turn() == 2, "apply_root_action 不推进回合");
        let advanced = game.next();
        println!("next()={} turn={} stage={:?}", advanced, game.turn(), game.stage);
        c.check(advanced, "next() 能推进");
        c.check(game.turn() == 2, "地区选择后回合仍为 2");
        c.check(
            game.stage == RamenStage::BeginAfterRegionSelect,
            "turn 2 RegionSelect 下一阶段是 BeginAfterRegionSelect"
        );
        c.finish()
    }

    /// last_decision 协议字段：候选评分与局数同步截断到 reason_max_display
    ///
    /// §7 验证策略要求：`last_decision` 跑一局后取所有决策，断言 `score` /
    /// `candidate_scores` 非零、`candidate_n` 与 `candidate_scores` 严格同长。
    /// 合并搜索路径 Step 2 暂不覆盖（输出 None）；只测普通搜索路径（train_only 门控）。
    #[test]
    fn test_last_decision_exposes_search_protocol() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::train_only());

        let mut train_decisions = 0usize;
        let mut emitted = 0usize;
        // 推进到第一个 Train 阶段，触发一次普通搜索路径
        while game.next() {
            if game.stage == RamenStage::Train {
                let actions = game.list_actions()?;
                if actions.len() > 1 {
                    let _idx = trainer.select_action(&game, &actions, &mut rng)?;
                    let info = trainer.last_decision();
                    println!(
                        "Train turn={} candidates={} -> last_decision={}",
                        game.turn(),
                        actions.len(),
                        if info.is_some() { "Some" } else { "None" }
                    );
                    if let Some(info) = info {
                        emitted += 1;
                        train_decisions += 1;
                        let mut c = Checks::new();
                        c.check(info.score != 0.0, "选中评分非零");
                        c.check(!info.candidate_scores.is_empty(), "候选评分非空");
                        c.check(
                            info.candidate_scores.len() == info.candidate_n.len(),
                            "candidate_n 与 candidate_scores 严格同长"
                        );
                        c.check(
                            info.candidate_scores.len() <= actions.len(),
                            "截断后候选数 <= 原始候选数"
                        );
                        c.check(info.action_index < info.candidate_scores.len(), "选中下标在截断后范围内");
                        // reason_max_display 默认 5；选择口径含 PT 加成、这里按 mean 排序，
                        // 选中者掉出 top-5 时按 last_decision 的设计插入首位 → 最多 5+1 项。
                        c.check(info.candidate_scores.len() <= 6, "截断到 reason_max_display=5（含选中者首位插入）");
                        // candidate_n 各元素 > 0（真实 MCTS rollout 数）
                        c.check(info.candidate_n.iter().all(|&n| n > 0), "每个候选都有正样本数");
                        c.finish()?;
                    }
                    break;
                }
            }
            let _ = game.run_stage(&trainer, &mut rng)?;
        }

        let mut c = Checks::new();
        println!("Train 决策 {train_decisions} 次，last_decision 有值 {emitted} 次");
        c.check(train_decisions > 0, "至少跑到一次 Train 决策");
        c.check(emitted > 0, "至少有一次一次 last_decision 有值");
        c.finish()
    }

    /// 合并搜索路径暴露 `last_decision`：候选与三阶段 actions 同长同序、聚合口径自洽
    ///
    /// 回归背景（2026-09）：合并路径原 `clear_last_summary()` → `last_decision()` 恒为
    /// `None`，「只吃面」回合（吃面后不链式接训练）整回合没有任何运气分更新。
    /// 修复后按「面」把 `(ramen, targets)` 组合候选聚合回三阶段下标再 stash，
    /// 聚合采用**局数加权均值**（`Σ(mean_i × n_i) / Σ n_i`），与运气分 T(n) baseline 同口径。
    #[test]
    fn test_combined_path_exposes_last_decision() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(8).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(true);

        // 推进到第一个 RamenSelect 决策点（turn>=2 剧本机制启动后的吃面点）
        for _ in 0..48 {
            if game.stage == RamenStage::RamenSelect {
                break;
            }
            let _ = game.run_stage(&trainer, &mut rng)?;
            if !game.next() {
                break;
            }
        }
        println!("推进到 stage={:?} turn={}", game.stage, game.turn());
        if game.stage != RamenStage::RamenSelect {
            println!("未到达 RamenSelect（开局无面可选？），跳过本回归用例");
            return Ok(());
        }
        let actions = game.list_actions()?;
        let combined = game.list_combined_ramen_select_actions();
        println!("三阶段候选 {} 个，合并候选 {} 个", actions.len(), combined.len());
        if actions.len() <= 1 || combined.len() <= 1 {
            println!("候选不足（单候选不触发合并路径），跳过");
            return Ok(());
        }

        let idx = trainer.select_action(&game, &actions, &mut rng)?;
        let info = trainer.last_decision().expect("合并路径必须暴露 last_decision");
        let mut c = Checks::new();
        c.check(info.candidate_scores.len() == actions.len(), "候选评分与三阶段 actions 同长");
        c.check(info.candidate_descriptions.len() == actions.len(), "候选描述与三阶段 actions 同长");
        c.check(info.candidate_n.len() == actions.len(), "候选局数与三阶段 actions 同长");
        c.check(info.action_index < info.candidate_scores.len(), "action_index 在截断后范围内");
        // 聚合后的描述 = 三阶段候选描述（按 actions 顺序），选中描述与 actions[idx] 一致
        let chosen_desc = info.candidate_descriptions.get(info.action_index).cloned().unwrap_or_default();
        let action_desc = actions[idx].to_string();
        c.check(chosen_desc == action_desc, "选中描述与 actions[选中] 一致");
        // 局数加权总和 > 0（搜索真实跑过）
        let total_n: u32 = info.candidate_n.iter().sum();
        c.check(total_n > 0, "候选局数之和 > 0");
        println!(
            "RamenSelect last_decision: 选中={action_desc} n_actions={} total_n={total_n}",
            actions.len()
        );
        c.finish()
    }

    /// 早退分支（单候选）last_decision 必须返回 None，避免上一次搜索的陈旧数据
    #[test]
    fn test_last_decision_none_on_singleton() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::train_only());

        // 跑一局，每步决策都查 last_decision：单候选路径必须 None
        let mut singlet_seen = 0usize;
        let mut summary_seen = 0usize;
        while game.next() {
            let actions = game.list_actions()?;
            if actions.len() == 1 {
                let _ = trainer.select_action(&game, &actions, &mut rng)?;
                if trainer.last_decision().is_none() {
                    singlet_seen += 1;
                }
                if trainer.searched_count() > 0 {
                    summary_seen += 1;
                }
            } else {
                let _ = trainer.select_action(&game, &actions, &mut rng)?;
                let _ = trainer.last_decision();
            }
            game.run_stage(&trainer, &mut rng)?;
        }
        let mut c = Checks::new();
        println!("单候选决策数 {singlet_seen}, 走过搜索 {summary_seen}");
        c.check(singlet_seen > 0, "整局至少有一个单候选决策");
        c.finish()
    }

    // ========== 决策成本探针（仅测量用）验收 ==========

    /// 挂 / 不挂决策探针必须给出**逐位相同**的整局结果与 RNG 位置
    ///
    /// 探针壳只读原子计数判断路径；一旦它改变决策或随机流，量出来的就不是
    /// 生产成本。用整局（而非单点）比较：决策链上任何一处偏移都会放大到终局分。
    #[test]
    fn test_decision_probe_neutral_full_game() -> Result<()> {
        let seed = 7;
        let cfg = SearchConfig::default().with_search_n(4).with_ucb(false);
        let stages = RamenSearchStages::train_only();

        let (mut game_a, mut rng_a) = setup(seed)?;
        let plain = RamenMctsTrainer::new(cfg.clone()).with_stages(stages);
        game_a.run_full_game(&plain, &mut rng_a)?;
        let after_a = rng_a.next_u64();

        let sink: Arc<Mutex<Vec<DecisionProbe>>> = Arc::new(Mutex::new(Vec::new()));
        let (mut game_b, mut rng_b) = setup(seed)?;
        let probed = RamenMctsTrainer::new(cfg)
            .with_stages(stages)
            .with_decision_probe(sink.clone());
        game_b.run_full_game(&probed, &mut rng_b)?;
        let after_b = rng_b.next_u64();

        let probes = sink.lock().map_err(|e| anyhow!("探针锁中毒: {e}"))?;
        let searched = probes.iter().filter(|p| p.path == DecisionPath::Searched).count();
        let gated = probes.iter().filter(|p| p.path == DecisionPath::FallbackGated).count();
        let single = probes
            .iter()
            .filter(|p| p.path == DecisionPath::FallbackSingleCandidate)
            .count();
        let cache = probes
            .iter()
            .filter(|p| p.path == DecisionPath::CombinedCacheHit)
            .count();
        println!(
            "关探针: 分数={} searched_count={} | 开探针: 分数={} searched_count={}",
            game_a.uma().calc_score(),
            plain.searched_count(),
            game_b.uma().calc_score(),
            probed.searched_count()
        );
        println!(
            "决策记录 {} 条: 搜索={searched} 门控转发={gated} 单候选转发={single} 合并缓存={cache}",
            probes.len()
        );
        println!("rng 后继: 关={after_a:#018x} 开={after_b:#018x}");

        let mut c = Checks::new();
        c.check(game_a.uma().calc_score() == game_b.uma().calc_score(), "整局终局分数一致");
        c.check(plain.searched_count() == probed.searched_count(), "搜索次数一致");
        c.check(after_a == after_b, "整局结束后 rng 位置一致");
        c.check(searched == probed.searched_count(), "标为 Searched 的记录数等于 searched_count");
        c.check(probes.len() > searched, "未搜索的决策同样留下记录（否则量不到执行决策次数）");
        c.check(
            probes.iter().all(|p| p.elapsed.as_nanos() > 0),
            "每条记录都有非零耗时"
        );
        c.finish()
    }

    /// 合并 `RamenSelect` 路径：搜索后按面聚合暴露 `DecisionInfo`
    ///
    /// 钉住探针如实记录「搜了且摘要对外可见」：合并路径曾清空摘要（字段为 false），
    /// 上游按面聚合后改为暴露，期望值随之为 true。
    #[test]
    fn test_decision_probe_marks_combined_decision_info() -> Result<()> {
        let sink: Arc<Mutex<Vec<DecisionProbe>>> = Arc::new(Mutex::new(Vec::new()));
        let (mut game, mut rng) = setup(11)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(true)
            .with_decision_probe(sink.clone());
        // 只跑前若干阶段：拿到至少一次 RamenSelect 搜索即可
        for _ in 0..400 {
            if !game.next() {
                break;
            }
            game.run_stage(&trainer, &mut rng)?;
            let hit = sink
                .lock()
                .map_err(|e| anyhow!("探针锁中毒: {e}"))?
                .iter()
                .any(|p| p.stage == RamenStage::RamenSelect && p.path == DecisionPath::Searched);
            if hit {
                break;
            }
        }
        let probes = sink.lock().map_err(|e| anyhow!("探针锁中毒: {e}"))?;
        let combined: Vec<&DecisionProbe> = probes
            .iter()
            .filter(|p| p.stage == RamenStage::RamenSelect && p.path == DecisionPath::Searched)
            .collect();
        for p in &combined {
            println!(
                "RamenSelect 回合 {} 候选 {} 耗时 {:?} 暴露 DecisionInfo={}",
                p.turn, p.actions_len, p.elapsed, p.exposes_decision_info
            );
        }
        let mut c = Checks::new();
        c.check(!combined.is_empty(), "至少捕获一次 RamenSelect 搜索");
        // 2026-09-18 合并上游：合并搜索路径改为按面聚合回三阶段下标后暴露
        // `last_decision()`（此前清空摘要，导致「只吃面」回合整回合没有运气分更新）。
        // 探针的语义仍是「搜了但摘要是否对外可见」，期望值随之从 false 翻成 true。
        c.check(
            combined.iter().all(|p| p.exposes_decision_info),
            "合并路径搜索后 last_decision 有内容（上游按面聚合后暴露）"
        );
        c.finish()
    }
}
