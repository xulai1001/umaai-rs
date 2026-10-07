//! 扁平蒙特卡洛搜索
//!
//! 对每个合法动作执行多次模拟，统计分数分布，选择最优动作。
//! 支持两种搜索策略：
//! - 均匀分配：每个动作平均分配搜索次数（并行化）
//! - UCB 分配：根据 UCB 公式动态分配搜索资源（C++ UmaAi 风格）

use std::{
    collections::HashMap,
    sync::{
        Arc,
        Mutex,
        atomic::{AtomicUsize, Ordering}
    },
    time::{Duration, Instant}
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use log::{debug, warn};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use super::{
    RamenSearchOutput,
    config::{SearchConfig, TOTAL_TURN},
    ramen_terminal::RamenTerminal,
    result::{ActionResult, OrderedRollouts, SearchOutput},
    searchable::{FlatSearchGame, SearchScore},
    seeds::RolloutSeeds,
    terminal::{NoTerminal, RolloutOutcome, TerminalRecord}
};
#[cfg(feature = "onnx")]
use crate::neural::{ThreadLocalNeuralNetLeafEvaluator, ThreadLocalNeuralNetLeafStatsSnapshot};
use crate::{
    game::{
        Game,
        onsen::{OnsenTurnStage, action::OnsenAction, game::OnsenGame},
        ramen::{RamenAction, RamenGame}
    },
    gamedata::EventChoice,
    neural::{Evaluator, HandwrittenEvaluator, ValueOutput}
};

/// 单个候选在一次搜索中的全部累加结果
///
/// 建新类型而非四元组：两条评分口径量纲相同、极易写反，再加上终局统计与失败
/// 计数后，位置参数已经不可读。
///
/// `D` 未接入观测的剧本为 [`NoTerminalStats`](super::terminal::NoTerminalStats)
/// （ZST）。`ordered` 在开关关闭时为 `None`，不分配有序缓冲的堆内存。
struct CandidateAccum<D> {
    /// 结算评分统计（参与排序）
    score: ActionResult,
    /// 计入 PT 偏好的评分统计（参与排序）
    score_pt: ActionResult,
    /// 终局多维统计（不参与排序）
    terminal: D,
    /// rollout 失败次数
    failed: usize,
    /// 已**计划**的 rollout 次数（成功 + 失败；仅供成本探针读取）
    ///
    /// 与 `failed` 同为观测量，不参与任何排序或终止判据——UCB 的终止判据仍用
    /// `search_ucb` 内部的 `planned` 向量，这里只是把同一个数带出内核。
    planned: usize,
    /// 按 rollout 序号对齐的原始分（`score` 轴）
    ///
    /// `None`：开关关闭，不分配。`Some`：`ordered[k]` 为第 k 次的分数，
    /// 失败则为内层 `None`。Vec 按需 `resize(idx + 1, None)` 增长。
    ordered: Option<Vec<Option<f64>>>
}

impl<D: Default> CandidateAccum<D> {
    /// 创建空累加器
    ///
    /// `record_ordered` 为真时 `ordered` 为 `Some(空 Vec)`，按序号按需增长；
    /// 为假时为 `None`，不分配有序缓冲。
    fn new(record_ordered: bool) -> Self {
        Self {
            score: ActionResult::new(),
            score_pt: ActionResult::new(),
            terminal: D::default(),
            failed: 0,
            planned: 0,
            ordered: if record_ordered {
                Some(Vec::new())
            } else {
                None
            }
        }
    }
}

impl<D> CandidateAccum<D> {
    /// 确保有序槽位覆盖到 `idx`（失败序号保持 `None`）
    fn ensure_ordered_slot(&mut self, idx: usize) {
        if let Some(ordered) = self.ordered.as_mut() {
            if ordered.len() <= idx {
                ordered.resize(idx + 1, None);
            }
        }
    }

    /// 并入一次成功 rollout 的结果
    ///
    /// 三份统计在同一处推进，避免出现「评分记了、终局漏了」的样本集合错位。
    /// `idx` 是本次 rollout 的序号（`offset + k`），失败项不走本方法，由调用方
    /// [`Self::ensure_ordered_slot`] 留空。
    fn push<T>(&mut self, idx: usize, outcome: &RolloutOutcome<T>)
    where
        T: TerminalRecord<Stats = D>
    {
        self.score.add(outcome.score.score);
        self.score_pt.add(outcome.score.score_pt);
        outcome.terminal.accumulate_into(&mut self.terminal);
        // 单次判空即完成「补齐槽位 + 写入」：开关关闭时（生产路径）这里只剩一次
        // 恒假的 Option 判别，与合并前同构；写在两个方法里会白白多一次分支
        if let Some(ordered) = self.ordered.as_mut() {
            if ordered.len() <= idx {
                ordered.resize(idx + 1, None);
            }
            ordered[idx] = Some(outcome.score.score);
        }
    }
}

#[derive(Clone)]
enum LeafEvaluator {
    Handwritten,
    #[cfg(feature = "onnx")]
    NeuralNet(ThreadLocalNeuralNetLeafEvaluator)
}

impl LeafEvaluator {
    fn _name(&self) -> &'static str {
        match self {
            LeafEvaluator::Handwritten => "handwritten",
            #[cfg(feature = "onnx")]
            LeafEvaluator::NeuralNet(_) => "nn"
        }
    }

    fn evaluate(&self, rollout_evaluator: &HandwrittenEvaluator, game: &OnsenGame) -> ValueOutput {
        match self {
            LeafEvaluator::Handwritten => rollout_evaluator.evaluate(game),
            #[cfg(feature = "onnx")]
            LeafEvaluator::NeuralNet(nn) => nn.evaluate(game)
        }
    }
}

/// 扁平蒙特卡洛搜索
///
/// 使用手写逻辑进行模拟，统计各动作的分数分布。
#[derive(Clone)]
pub struct FlatSearch<G: FlatSearchGame = OnsenGame>
where
    G::Action: Send + Sync + Clone
{
    /// 手写评估器（温泉 rollout 与 leaf 估值用）
    ///
    /// 仅温泉路径使用：`HandwrittenEvaluator` 只 impl 了 `Evaluator<OnsenGame>`。
    /// 拉面走 `G::RolloutTrainer`，该字段闲置。Phase 1.4 保留此不对称以免掀开
    /// `umaai` 的 `with_leaf_evaluator_handwritten()` 调用签名。
    rollout_evaluator: HandwrittenEvaluator,

    /// leaf eval 评估器（用于 max_depth>0 截断估值；温泉专用）
    leaf_evaluator: LeafEvaluator,

    /// rollout 决策器（由剧本指定）
    rollout_trainer: G::RolloutTrainer,

    /// 搜索配置
    config: SearchConfig,

    /// E4：leaf eval 微批大小（仅在 max_depth>0 && leaf_eval=nn 时生效）
    ///
    /// **当前未接线**：本字段只被写入、从未被搜索逻辑读取。
    /// 原设计意图是批量推理（NN 评估器批处理），待 rollout 评估器接入后再消费。
    /// 配置链（`MctsConfig::rollout_batch_size` → `with_rollout_batch_size`）同样是空转。
    rollout_batch_size: usize,

    /// rollout 失败是否让整次搜索失败（默认 `false` = 历史行为）
    ///
    /// 在当前模拟组执行并收集结果后返回错误，不立即取消已经调度的模拟。
    ///
    /// 默认路径下单条 rollout 失败只被计入 `failed` 并告警，搜索照常用剩下的样本
    /// 排序——手写基策的失败是零星的，丢掉几条不改变结论。
    ///
    /// ❗**换过 rollout 基策的实验必须打开**：若失败与状态相关（网络在某类局面
    /// 上推理报错），剩下的样本就是「以推理成功为条件」的分布，各候选的条件还
    /// 互不相同，排序会被系统性污染，而分数上完全看不出来。
    strict_rollout: bool,

    /// 批量 rollout 后端（**目前仅拉面使用**）
    ///
    /// 置上之后，[`FlatSearch::<RamenGame>::search`] 不再逐条调用 rollout 闭包，
    /// 而是先让后端一次算完该根上全部 `(候选, rollout 种子)`，闭包退化为查表。
    /// 候选分配、CRN 种子派生、排序与统计**仍然全部走内核**，口径不变。
    ///
    /// 泛型结构体上挂一个非泛型 trait 对象是有意为之：温泉没有批量后端，
    /// 为它引入一层泛型参数只会污染所有调用点。
    batch_rollout: Option<Arc<dyn RamenBatchRollout>>,

    /// 成本探针出口（**仅测量用**，默认 `None`）
    ///
    /// 置上之后每次 [`Self::search_with_terminal`] 结束会追加一条 [`SearchProbe`]。
    ///
    /// 关闭时的开销**不是零**，而是：每次搜索一次 `Option::map`（不取时钟）、
    /// 每个 rollout 组一次 `Option::is_some` 分支（不累加计数）。候选分配、
    /// 种子派生、统计与排序不读本字段，故搜索结果与 RNG 消耗逐位不变。
    probe: Option<Arc<Mutex<Vec<SearchProbe>>>>
}

/// 探针的三份候选计数（`planned` / `succeeded` / `failed` 同序同长）
///
/// 建新类型而非三元组：三个 `Vec<usize>` 位置写反不会编译报错，也不会在数值上
/// 立刻露馅（都是同一量级的计数）。
#[derive(Debug, Clone)]
struct ProbeCounts {
    /// 每候选**计划**的 rollout 次数
    planned: Vec<usize>,
    /// 每候选**成功**的 rollout 次数
    succeeded: Vec<usize>,
    /// 每候选**失败**的 rollout 次数
    failed: Vec<usize>
}

/// 一次搜索内核调用的成本记录（仅测量用）
///
/// 内核不对外暴露 `CandidateAccum`，而「每候选计划 / 成功 / 失败次数」正是
/// 预算核实要看的量。本结构把这些观测量原样带出，**不参与**任何判据。
#[derive(Debug, Clone)]
pub struct SearchProbe {
    /// 根局面回合
    pub turn: usize,
    /// 剧本阶段键（[`FlatSearchGame::crn_stage_key`]）
    pub stage_key: u64,
    /// 真正进入搜索的候选数
    pub candidates: usize,
    /// 每候选**计划**的 rollout 次数（成功 + 失败）
    pub planned: Vec<usize>,
    /// 每候选**成功**的 rollout 次数
    pub succeeded: Vec<usize>,
    /// 每候选**失败**的 rollout 次数
    pub failed: Vec<usize>,
    /// 本次搜索的激进度因子
    pub radical_factor: f64,
    /// `weighted_mean(radical_factor)` 口径（Score 轴）的中选下标
    ///
    /// 只是观测量：生产默认按 PT 轴（`best_action_pt_idx`）选择，二者可能不同，
    /// 不能当作训练员实际执行的动作。
    pub best_action_idx: usize,
    /// 搜索内核开始时刻（与外层链路对齐用）
    pub started: Instant,
    /// 搜索内核墙钟耗时；批量后端的 `precompute` 在内核之前执行，不计入此值。
    /// 批量推理总成本应由调用方单独测量，不能拿本字段代替端到端耗时。
    pub elapsed: Duration
}

impl SearchProbe {
    /// 计划 rollout 总数（= 总续跑数，含失败）
    pub fn total_planned(&self) -> usize {
        self.planned.iter().sum()
    }

    /// 成功 rollout 总数
    pub fn total_succeeded(&self) -> usize {
        self.succeeded.iter().sum()
    }

    /// 失败 rollout 总数
    pub fn total_failed(&self) -> usize {
        self.failed.iter().sum()
    }
}

/// 一次批量 rollout 的结果表
///
/// 键是 `(候选下标, rollout 种子)`；种子由内核的 [`RolloutSeeds`] 派生，
/// **不吃候选下标**，故各候选在同一 `j` 上共享种子（CRN 的载体）。
pub type RamenBatchTable = HashMap<(usize, u64), RolloutOutcome<RamenTerminal>>;

/// 批量 rollout 后端
///
/// 让「一次算完整个根」的执行方式（例如跨 rollout 攒批送 GPU）能接进生产教师，
/// 而不必另写一套简化版教师。实现者只负责**算结果**，不参与排序与统计。
pub trait RamenBatchRollout: Send + Sync {
    /// 算完 `game` 上每个候选的 `n` 条 rollout
    ///
    /// 仅支持均匀分配；搜索入口在调用后端前拒绝 UCB 与重复候选。
    /// 必须恰好为每个 `(候选下标, seeds.seed_at(j))`（`j` 取 `0..n`）给出结果，
    /// 不得缺项或带多余键。搜索入口独立校验此结构契约，不受 strict 开关影响。
    ///
    /// # 错误
    ///
    /// 后端自身执行失败时报错。
    fn precompute(
        &self, game: &RamenGame, actions: &[RamenAction], seeds: &RolloutSeeds, n: usize
    ) -> Result<RamenBatchTable>;
}

impl<G: FlatSearchGame> FlatSearch<G>
where
    G::Action: Send + Sync + Clone
{
    /// 创建搜索器
    pub fn new(config: SearchConfig) -> Self {
        Self {
            rollout_evaluator: HandwrittenEvaluator::new(),
            leaf_evaluator: LeafEvaluator::Handwritten,
            rollout_trainer: G::default_rollout_trainer(),
            config,
            rollout_batch_size: 1,
            strict_rollout: false,
            batch_rollout: None,
            probe: None
        }
    }

    /// 挂上成本探针出口（**仅测量用**，见 [`SearchProbe`]）
    ///
    /// 默认不挂。**不挂 ≠ 零开销**：结构体多一个 `Option` 字段，每次搜索多一次
    /// `Option::map`（不取时钟）、每个 rollout 组多一次 `Option::is_some` 分支
    /// （不累加计数）。它保证的是**不改变决策语义**——候选分配、CRN 种子派生、
    /// 统计与排序都不读本字段，故搜索结果与 RNG 消耗逐位不变。
    /// 挂上后每次搜索内核结束追加一条记录。
    pub fn with_probe(mut self, sink: Arc<Mutex<Vec<SearchProbe>>>) -> Self {
        self.probe = Some(sink);
        self
    }

    /// 挂上批量 rollout 后端（仅拉面路径生效）
    ///
    /// 见 [`RamenBatchRollout`]：只替换「rollout 怎么跑」，排序与统计仍走内核。
    pub fn with_batch_rollout(mut self, backend: Arc<dyn RamenBatchRollout>) -> Self {
        self.batch_rollout = Some(backend);
        self
    }

    /// 创建默认搜索器
    pub fn default_search() -> Self {
        Self::new(SearchConfig::default())
    }

    /// 设置 leaf eval 为神经网络（用于 max_depth>0 截断估值）
    ///
    /// 仅在 `onnx` feature 下可用；core-only 构建调用会编译错误（编译器提示）。
    #[cfg(feature = "onnx")]
    pub fn with_leaf_evaluator_nn(mut self, model_path: impl Into<String>) -> Self {
        self.leaf_evaluator = LeafEvaluator::NeuralNet(ThreadLocalNeuralNetLeafEvaluator::new(model_path));
        self
    }

    /// 替换 rollout 基策
    ///
    /// 搜索出的 `Q` 是**对 rollout 基策的**动作价值，换掉它就是换掉整个教师：
    /// 拉面用 [`RamenRolloutTrainer`](crate::trainer::RamenRolloutTrainer) 装载
    /// 网络后，`Q^手写` 变成 `Q^NN`，`greedy(Q)` 即一次策略迭代。
    ///
    /// ❗**换了基策的搜索与旧基线不可比**：同一个 `search_n` 下排序依据已经不同，
    /// 此前记录的教师闭环分只对手写 rollout 成立。
    pub fn with_rollout_trainer(mut self, trainer: G::RolloutTrainer) -> Self {
        self.rollout_trainer = trainer;
        self
    }

    /// 设置 rollout 失败时是否让整次搜索失败
    ///
    /// 见 [`Self::strict_rollout`] 字段文档：换过 rollout 基策的实验一律打开。
    /// 本开关阻止用部分成功样本返回结果，不负责立即取消正在执行的模拟组。
    pub fn with_strict_rollout(mut self, strict: bool) -> Self {
        self.strict_rollout = strict;
        self
    }

    /// 强制 leaf eval 回退为 handwritten（默认）
    pub fn with_leaf_evaluator_handwritten(mut self) -> Self {
        self.leaf_evaluator = LeafEvaluator::Handwritten;
        self
    }

    /// 就地改写当前 rollout 基策（不整体替换）
    ///
    /// 多个配置项都要作用到 rollout 基策上（友人出行完成硬门限、网络 rollout），
    /// 用 [`Self::with_rollout_trainer`] 整体替换会把先设的那一项冲掉；本方法把现有
    /// 基策交给 `f` 改写后放回，调用顺序因此无关。
    pub fn map_rollout_trainer(mut self, f: impl FnOnce(G::RolloutTrainer) -> G::RolloutTrainer) -> Self {
        let old = std::mem::replace(&mut self.rollout_trainer, G::default_rollout_trainer());
        self.rollout_trainer = f(old);
        self
    }

    /// 设置 leaf eval 微批大小（仅 nn leaf 生效）
    pub fn with_rollout_batch_size(mut self, batch_size: usize) -> Self {
        self.rollout_batch_size = batch_size.max(1).min(1024);
        self
    }

    /// 获取配置
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// E4 调试：获取 leaf NN 推理统计（仅当 leaf evaluator 为 nn 时存在）
    #[cfg(feature = "onnx")]
    pub fn leaf_nn_stats(&self) -> Option<ThreadLocalNeuralNetLeafStatsSnapshot> {
        match &self.leaf_evaluator {
            LeafEvaluator::NeuralNet(nn) => Some(nn.stats()),
            _ => None
        }
    }

    fn use_parallel_simulation(&self) -> bool {
        // E4.3：leaf eval 使用 thread_local 模型后，可安全恢复 Rayon 并行
        true
    }

    #[cfg(feature = "onnx")]
    fn leaf_nn(&self) -> Option<&ThreadLocalNeuralNetLeafEvaluator> {
        match &self.leaf_evaluator {
            LeafEvaluator::NeuralNet(nn) => Some(nn),
            _ => None
        }
    }

    /// 通用搜索内核
    ///
    /// 负责候选分配（均匀 / UCB）、CRN 种子、统计与并行调度；
    /// 「一次 rollout 怎么跑」由调用方以闭包注入，使剧本特判（如温泉
    /// `Dig`/`Upgrade`）留在各自的具体 impl 里，不进公共 trait。
    ///
    /// # 为什么用闭包而非 trait 钩子
    ///
    /// 泛型 `impl<G> FlatSearch<G>` 内部调用 `self.simulate()` 时，方法解析只会
    /// 找到泛型版本，**永远不会**落到更具体的 `impl FlatSearch<OnsenGame>`。
    /// 若把特判留作具体 impl 的同名方法，它会静默变成死代码、温泉 `Dig`/`Upgrade`
    /// 改走通用路径——编译通过但行为改变。闭包注入从根上避免这个陷阱。
    pub fn search_with<F>(
        &self, game: &G, actions: &[G::Action], rng: &mut StdRng, rollout: F
    ) -> Result<SearchOutput<G::Action>>
    where
        F: Fn(&G, &G::Action, u64) -> Result<SearchScore> + Sync
    {
        self.search_with_terminal(game, actions, rng, |g, a, seed| {
            Ok(RolloutOutcome {
                score: rollout(g, a, seed)?,
                terminal: NoTerminal
            })
        })
    }

    /// 带终局多维记录的搜索内核
    ///
    /// 与 [`Self::search_with`] 的唯一区别是 rollout 闭包多返回一个
    /// [`TerminalRecord`]，内核按候选把它累加进 `T::Stats`。候选分配、CRN 种子、
    /// 失败处理与排序口径**完全共用同一份实现**——多维记录若另起一套内核，
    /// 迟早与生产路径在预算修复或失败计数上分叉。
    ///
    /// 终局记录不参与 `best_action_idx` 的选择，只作为观测量输出。
    pub fn search_with_terminal<F, T>(
        &self, game: &G, actions: &[G::Action], rng: &mut StdRng, rollout: F
    ) -> Result<SearchOutput<G::Action, T::Stats>>
    where
        T: TerminalRecord,
        F: Fn(&G, &G::Action, u64) -> Result<RolloutOutcome<T>> + Sync
    {
        if actions.is_empty() {
            bail!("没有可用动作");
        }
        ensure!(
            G::SUPPORTS_TRUNCATED_LEAF || self.config.max_depth == 0,
            "本剧本没有 leaf 估值器，max_depth 必须为 0（当前 {}）",
            self.config.max_depth
        );

        // 计算激进度因子（C++ 风格，无随机性）
        let radical_factor = self.compute_radical_factor(game.turn() as usize);

        // 本次搜索的 rollout 种子表：所有候选共享，由传入 rng 派生（可复现性入口）
        let seeds = RolloutSeeds::from_rng(rng);
        debug!(
            "[回合 {}] 开始搜索: {} 个动作, search_n={}, max_depth={}, radical_factor={:.1}, ucb={}, 根种子={:#018x}",
            game.turn(),
            actions.len(),
            self.config.search_n,
            self.config.max_depth,
            radical_factor,
            self.config.use_ucb,
            seeds.root()
        );

        // rollout 期间全局静默诊断输出（RAII，Err 提前返回同样恢复）：
        // 覆盖均匀分配 / UCB 两条路径的全部 rayon 并行段，规则层 diag! 与
        // cfg explain 块（`if diagnostic::enabled()` 门控）一并跳过。每回合仅
        // 一条的搜索根 debug! 在 guard 之前；失败汇总 warn_failures 虽在 guard
        // 作用域内，但 warn! 不走 diag 开关，照常输出。多局并行
        // （simulation_count > 1）时任一局的搜索会顺带抑制其他局真实回合的
        // diag——已知局限，见 diagnostic.rs 模块文档。
        let _diag_guard = crate::output::diagnostic::DiagGuard::suppress();
        // 成本探针的计时起点：探针关闭时**不取**时钟，避免为不用的数字付钟表开销
        let probe_started = self.probe.as_ref().map(|_| Instant::now());
        let collected = if self.config.use_ucb {
            self.search_ucb(game, actions, radical_factor, &seeds, &rollout)?
        } else {
            self.search_uniform(game, actions, &seeds, &rollout)?
        };

        // 某候选一次都没跑成功时其统计全是空的，继续用下去等于拿垃圾数据排序
        for (i, acc) in collected.iter().enumerate() {
            if acc.score.count() == 0 {
                bail!("候选动作 {i} 的全部 rollout 均失败，搜索结果不可用");
            }
        }

        let record = self.config.record_ordered_rollouts;
        // 成本探针快照：必须在 `collected` 被消费前取
        let probe_counts: Option<ProbeCounts> = self.probe.as_ref().map(|_| ProbeCounts {
            planned: collected.iter().map(|acc| acc.planned).collect(),
            succeeded: collected.iter().map(|acc| acc.score.count() as usize).collect(),
            failed: collected.iter().map(|acc| acc.failed).collect()
        });
        let mut action_results = Vec::with_capacity(collected.len());
        let mut terminal_results = Vec::with_capacity(collected.len());
        let mut per_candidate = if record {
            Some(Vec::with_capacity(collected.len()))
        } else {
            None
        };
        for acc in collected {
            action_results.push((acc.score, acc.score_pt));
            terminal_results.push(acc.terminal);
            if let Some(ref mut rows) = per_candidate {
                rows.push(acc.ordered.unwrap_or_default());
            }
        }

        let mut out = SearchOutput::with_terminals(
            actions.to_vec(),
            action_results,
            terminal_results,
            radical_factor
        );
        if let Some(per_candidate) = per_candidate {
            out.ordered_rollouts = Some(OrderedRollouts {
                root_seed: seeds.root(),
                per_candidate
            });
        }
        if let (Some(sink), Some(counts), Some(started)) = (self.probe.as_ref(), probe_counts, probe_started) {
            let probe = SearchProbe {
                turn: game.turn() as usize,
                stage_key: game.crn_stage_key(),
                candidates: actions.len(),
                planned: counts.planned,
                succeeded: counts.succeeded,
                failed: counts.failed,
                radical_factor,
                best_action_idx: out.best_action_idx,
                started,
                elapsed: started.elapsed()
            };
            match sink.lock() {
                Ok(mut v) => v.push(probe),
                Err(e) => warn!("[搜索][探针] 记录失败（锁中毒）: {e}")
            }
        }
        Ok(out)
    }

    /// 计算激进度因子
    ///
    /// 使用 C++ UmaAi 的固定公式，不使用随机性：
    /// radical_factor = (剩余回合 / 总回合)^0.5 * 最大激进度
    fn compute_radical_factor(&self, turn: usize) -> f64 {
        let remain_turns = (TOTAL_TURN.saturating_sub(turn)) as f64;
        let factor = (remain_turns / TOTAL_TURN as f64).powf(0.5);
        factor * self.config.radical_factor_max
    }

    /// 按当前 `(回合, 阶段)` 重新播种 rollout 随机流（外挂 CRN）
    ///
    /// **仅规则层未改造的剧本（onsen）使用**：onsen 的规则随机仍走传入的
    /// `&mut StdRng`，靠本方法按 `(rollout 种子, 回合, 阶段)` 重播种对齐各候选；
    /// 拉面规则层已由无状态流接管（RNG Refactor Plan v2 §5.2，`fork_for_rollout`
    /// 注入 rule_master），其 rollout 路径不再调用本方法。
    pub fn reseed_for_stage(&self, rng: &mut StdRng, rollout_seed: u64, game: &G) {
        if !self.config.crn_stage_reseed {
            return;
        }
        *rng = StdRng::seed_from_u64(RolloutSeeds::stage_seed(
            rollout_seed,
            game.turn(),
            game.crn_stage_key()
        ));
    }

    /// rollout 决策器
    pub fn rollout_trainer(&self) -> &G::RolloutTrainer {
        &self.rollout_trainer
    }

    /// 双种子 rollout：决策 RNG 与规则层 `rule_master` 可分开
    ///
    /// 生产路径 [`Self::simulate_common`] 传入相同的两个值，保持既有 CRN
    /// （各候选第 j 次 rollout 共享 `seed_at(j)`）。测量对照把 `rule_master`
    /// 按候选派生，从而拆开「共享规则层随机未来」与「独立抽样」。
    ///
    /// **不改变** [`RolloutSeeds::seed_at`] 与 [`FlatSearchGame::fork_for_rollout`]
    /// 的生产语义。
    pub fn simulate_common_with_seeds(
        &self, game: &G, action: &G::Action, decision_seed: u64, rule_master: u64
    ) -> Result<SearchScore> {
        Ok(self
            .simulate_common_extract(game, action, decision_seed, rule_master, |_| NoTerminal)?
            .score)
    }

    /// 单次 rollout，并在终局就地提取观测记录
    ///
    /// `extract` 在**终局局面尚存活时**调用：`sim_game` 跑完即弃，多维事实无法
    /// 事后补取，必须在这里随评分一起取出。
    ///
    /// 与 [`Self::simulate_common_with_seeds`] 共用同一份 rollout 主体，
    /// 保证观测路径与生产路径逐位同源。
    pub fn simulate_common_extract<T, E>(
        &self, game: &G, action: &G::Action, decision_seed: u64, rule_master: u64, extract: E
    ) -> Result<RolloutOutcome<T>>
    where
        T: TerminalRecord,
        E: Fn(&G) -> T
    {
        let rng = &mut StdRng::seed_from_u64(decision_seed);
        let mut sim_game = game.fork_for_rollout(rule_master);
        // 必须走剧本的真实对局路径（拉面 = 策略流），不能用通用 apply_action
        sim_game.apply_root_action(action, rng)?;
        while sim_game.next() {
            sim_game.run_stage(&self.rollout_trainer, rng)?;
        }
        sim_game.on_simulation_end(&self.rollout_trainer, rng)?;
        Ok(RolloutOutcome {
            score: sim_game.search_score(),
            terminal: extract(&sim_game)
        })
    }

    /// 通用单次 rollout：执行动作后跑到终局
    ///
    /// 只处理 `max_depth == 0`；截断估值需要 leaf 估值器，属剧本专属能力。
    /// 分支一律经 [`FlatSearchGame::fork_for_rollout`] 建立，不得直接 `clone()`
    /// ——那会漏掉剧本内部随机流的重置。
    ///
    /// 决策 RNG 与规则层 `rule_master` 使用同一 `seed`，这是生产 CRN：
    /// 各候选第 j 次 rollout 共享 `seed_at(j)`。对照实验请走
    /// [`Self::simulate_common_with_seeds`]。
    pub fn simulate_common(&self, game: &G, action: &G::Action, seed: u64) -> Result<SearchScore> {
        self.simulate_common_with_seeds(game, action, seed, seed)
    }

    /// 均匀分配搜索（并行化）
    ///
    /// 每个动作平均分配 `search_n` 次搜索。所有候选的第 j 次 rollout 共用
    /// `seeds.seed_at(j)`（CRN 载体），故并行粒度不影响结果。
    ///
    /// 候选之间和候选内部共用 Rayon 线程池；每个候选的结果按 rollout 序号累加。
    fn search_uniform<F, T>(
        &self, game: &G, actions: &[G::Action], seeds: &RolloutSeeds, rollout: &F
    ) -> Result<Vec<CandidateAccum<T::Stats>>>
    where
        T: TerminalRecord,
        F: Fn(&G, &G::Action, u64) -> Result<RolloutOutcome<T>> + Sync
    {
        let n = self.config.search_n;
        let record = self.config.record_ordered_rollouts;
        let run = |action: &G::Action| -> Result<CandidateAccum<T::Stats>> {
            let mut acc = CandidateAccum::<T::Stats>::new(record);
            // offset=0：均匀分配下每个候选都从 rollout 0 开始，天然完全配对
            self.simulate_many(game, action, n, seeds, 0, &mut acc, rollout, "")?;
            Ok(acc)
        };

        let collected: Vec<CandidateAccum<T::Stats>> = if self.use_parallel_simulation() {
            actions.par_iter().map(run).collect::<Result<Vec<_>>>()?
        } else {
            actions.iter().map(run).collect::<Result<Vec<_>>>()?
        };

        Self::warn_failures(&collected, "均匀分配");
        Ok(collected)
    }

    /// 汇总 rollout 失败次数并告警
    ///
    /// rollout 失败会让该候选的样本数少于计划值，静默丢弃会把「跑失败」
    /// 混同于「跑出来分低」。此处不中断搜索（避免偶发失败拖垮实时通道层），
    /// 但必须在日志里留下痕迹。
    fn warn_failures<D>(collected: &[CandidateAccum<D>], stage: &str) {
        let total_failed: usize = collected.iter().map(|acc| acc.failed).sum();
        if total_failed > 0 {
            warn!("[搜索][{stage}] {total_failed} 次 rollout 失败，对应候选的样本数少于计划值");
        }
    }

    /// 对同一候选执行 `n` 次 rollout，保序收集后累加进 `acc`
    ///
    /// 第 k 次取 `seeds.seed_at(offset + k)` 播种，`offset` 为该候选**已计划**的次数。
    /// 失败次数累加进 `acc.failed`（不中断搜索，由调用方汇总告警）。
    /// 临时缓冲保留全部 `Result`，避免失败项压缩槽位或中止余下 rollout。
    /// [`FlatSearch::strict_rollout`] 打开时返回**序号最小**的失败，与逐条执行时
    /// 报出的是同一条（其后的 rollout 已经跑完，只多花时间，不进入统计）。
    fn simulate_many<F, T>(
        &self, game: &G, action: &G::Action, n: usize, seeds: &RolloutSeeds, offset: usize,
        acc: &mut CandidateAccum<T::Stats>, rollout: &F, log_context: &str
    ) -> Result<()>
    where
        T: TerminalRecord,
        F: Fn(&G, &G::Action, u64) -> Result<RolloutOutcome<T>> + Sync
    {
        // 只在挂了探针时才记账：关闭时这里是一次 `Option::is_some` 分支，
        // 每组（而非每条 rollout）判一次。
        if self.probe.is_some() {
            acc.planned += n;
        }
        let run_one = |k: usize| rollout(game, action, seeds.seed_at(offset + k));
        let outcomes: Vec<Result<RolloutOutcome<T>>> = if self.use_parallel_simulation() {
            (0..n).into_par_iter().map(run_one).collect()
        } else {
            (0..n).map(run_one).collect()
        };
        // IndexedParallelIterator 的 collect 保序；逐项累加保持浮点运算顺序。
        for (k, outcome) in outcomes.into_iter().enumerate() {
            let idx = offset + k;
            match outcome {
                Ok(v) => acc.push(idx, &v),
                Err(e) => {
                    if self.strict_rollout {
                        return Err(e).with_context(|| format!("rollout {idx} 失败（strict_rollout 已开启）"));
                    }
                    debug!("[搜索]{log_context} rollout {} 失败: {e}", idx);
                    acc.failed += 1;
                    acc.ensure_ordered_slot(idx);
                }
            }
        }
        Ok(())
    }

    /// UCB 动态分配搜索
    ///
    /// 使用 UCB 公式动态分配搜索资源，好的动作获得更多搜索次数。
    /// UCB 决策是串行的，但每组模拟内部使用 Rayon 并行化。
    ///
    /// # UCB 公式
    /// search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn search_ucb<F, T>(
        &self, game: &G, actions: &[G::Action], radical_factor: f64, seeds: &RolloutSeeds, rollout: &F
    ) -> Result<Vec<CandidateAccum<T::Stats>>>
    where
        T: TerminalRecord,
        F: Fn(&G, &G::Action, u64) -> Result<RolloutOutcome<T>> + Sync
    {
        let num_actions = actions.len();
        let record = self.config.record_ordered_rollouts;
        ensure!(
            self.config.search_group_size > 0,
            "search_group_size 不能为 0（UCB 分配会死循环）"
        );
        // 首组不得越过 search_n：否则 group_size > search_n 时每候选先跑满一组
        // 已经超预算，且 max_planned >= search_n 立刻成立，自适应零次。
        // 只收紧本函数的局部步长，不改 SearchConfig 字段——均匀路径不读
        // group_size，配置对象应保持调用方写入的原值。
        let group_size = self.config.search_group_size.min(self.config.search_n).max(1);
        let use_parallel = self.use_parallel_simulation();

        // 各候选**已计划**的 rollout 次数（≠ 已成功次数）
        //
        // 种子偏移必须用计划次数而非 `ActionResult::count()`：后者会因 rollout 失败
        // 而少计，导致同一 rollout 序号在不同候选上错位，破坏配对。
        let mut planned = vec![group_size; num_actions];

        // 第一阶段：每个动作先搜一组（并行）
        let run_initial = |action: &G::Action| -> Result<CandidateAccum<T::Stats>> {
            let mut acc = CandidateAccum::<T::Stats>::new(record);
            self.simulate_many(game, action, group_size, seeds, 0, &mut acc, rollout, "")?;
            Ok(acc)
        };
        let mut collected: Vec<CandidateAccum<T::Stats>> = if use_parallel {
            actions.par_iter().map(run_initial).collect::<Result<Vec<_>>>()?
        } else {
            actions.iter().map(run_initial).collect::<Result<Vec<_>>>()?
        };

        let mut total_n = (group_size * num_actions) as f64;

        // 第二阶段：UCB 动态分配
        loop {
            // 终止判据用**已计划**次数，不能用 ActionResult::count()（成功次数）
            //
            // 用成功次数会在 rollout 稳定失败时死循环：失败只增加 planned、不增加 count，
            // 于是 count 永远达不到 search_n，而末尾的「零样本」检查在本函数返回之后
            // 才执行，根本到不了。
            let max_planned = planned.iter().copied().max().unwrap_or(0);
            if max_planned >= self.config.search_n {
                break;
            }

            // 使用 UCB 公式选择下一个要搜索的动作
            let best_action_idx = self.select_ucb_action(&collected, radical_factor, total_n);
            let action = &actions[best_action_idx];

            // 该候选已计划 offset 次，本组取 seeds[offset..offset+group_size]。
            // 两个候选因而在 0..min(n_a, n_b) 上完全配对，多出的部分为 unpaired，
            // 这是 CRN 在不等样本数下的标准做法。
            let offset = planned[best_action_idx];
            self.simulate_many(
                game, action, group_size, seeds, offset, &mut collected[best_action_idx], rollout, "[UCB]"
            )?;

            planned[best_action_idx] += group_size;
            total_n += group_size as f64;
        }

        Self::warn_failures(&collected, "UCB");
        Ok(collected)
    }

    /// 使用 UCB 公式选择下一个要搜索的动作
    ///
    /// UCB 公式: search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn select_ucb_action<D>(&self, collected: &[CandidateAccum<D>], radical_factor: f64, total_n: f64) -> usize {
        let sqrt_total = total_n.sqrt();
        let cpuct = self.config.search_cpuct;
        let expected_stdev = self.config.expected_search_stdev;

        let mut best_idx = 0;
        let mut best_search_value = f64::NEG_INFINITY;

        for (i, acc) in collected.iter().enumerate() {
            let n = acc.score.count() as f64;
            if n == 0.0 {
                // 未搜索的动作优先级最高
                return i;
            }

            let value = acc.score.weighted_mean(radical_factor);
            // UCB 公式：value 越高或搜索次数越少，search_value 越高
            let delta = cpuct * expected_stdev * sqrt_total / n;
            let search_value = value + delta;
            //println!("#{i} score: {value:.0}, ucb: {delta:.0}, sqrt_total: {sqrt_total:.0}, n: {n}");
            if search_value > best_search_value {
                best_search_value = search_value;
                best_idx = i;
            }
        }
        // println!("best: #{best_idx}");
        // println!("--------------------");
        best_idx
    }
}

impl FlatSearch<OnsenGame> {
    /// 温泉根节点搜索
    ///
    /// 走通用内核 [`FlatSearch::search_with`]，但注入温泉专属的 rollout 分发：
    /// `Dig`/`Upgrade` 不是回合阶段，而是嵌套的 `pending_selection` 选择，
    /// 必须走各自的特判路径，不能落到 `apply_action -> while next` 的通用流程。
    ///
    /// 保持既有对外签名不变，`MctsTrainer` 与 `umaai` 通道层无需改动。
    pub fn search(
        &self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng
    ) -> Result<SearchOutput<OnsenAction>> {
        self.search_with(game, actions, rng, |game, action, seed| {
            let (score, score_pt) = self.simulate(game, action, seed)?;
            Ok(SearchScore { score, score_pt })
        })
    }

    /// 模拟单个动作到终局
    ///
    /// 从当前状态开始，执行指定动作，然后用手写逻辑走到游戏结束。
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `action`: 要模拟的动作
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// 最终分数
    /// 单次 rollout
    ///
    /// `seed` 为该次 rollout 的种子（由 [`RolloutSeeds::seed_at`] 给出，所有候选共享）。
    /// 开启 [`SearchConfig::crn_stage_reseed`] 时，每进入一个阶段会按
    /// `(seed, 回合, 阶段)` 重新播种，使各候选在同一阶段抽到同一份随机性。
    fn simulate(&self, game: &OnsenGame, action: &OnsenAction, seed: u64) -> Result<(f64, f64)> {
        let rng = &mut StdRng::seed_from_u64(seed);
        if matches!(action, OnsenAction::Dig(_)) {
            self.simulate_onsen_select(game, action, rng)
        } else if matches!(action, OnsenAction::Upgrade(_)) {
            self.simulate_dig_upgrade(game, action, rng)
        } else {
            // 克隆游戏状态
            let mut sim_game = game.clone();
            let trainer_hw = SimulationTrainer {
                evaluator: &self.rollout_evaluator
            };

            // 执行初始动作
            sim_game.apply_action(action, rng)?;

            // max_depth==0：保持旧行为，rollout 跑到终局
            if self.config.max_depth == 0 {
                while sim_game.next() {
                    self.reseed_for_stage(rng, seed, &sim_game);
                    sim_game.run_stage(&trainer_hw, rng)?;
                }
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64
                ));
            }

            // max_depth>0：按 turn 截断；未终局则 leaf eval 估值
            let start_turn = sim_game.turn;
            let max_depth = self.config.max_depth as i32;
            let mut finished = false;

            loop {
                if !sim_game.next() {
                    finished = true;
                    break;
                }
                self.reseed_for_stage(rng, seed, &sim_game);
                sim_game.run_stage(&trainer_hw, rng)?;
                if (sim_game.turn - start_turn) >= max_depth {
                    break;
                }
            }

            if finished {
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64
                ));
            }
            // 有些情况下（例如在达到 max_depth 的同一轮刚好走到终局），可能还未通过 next() 触发 finished。
            // 用 turn>=max_turn 兜底判定终局，并确保 on_simulation_end 被触发，避免漏算最终奖励。
            if sim_game.turn >= sim_game.max_turn() {
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64
                ));
            }

            // 未终局：leaf eval（scoreMean）；PT 口径用“当前 pt_bias”近似对齐
            let v = self.leaf_evaluator.evaluate(&self.rollout_evaluator, &sim_game);
            let score_mean = v.score_mean;
            let current_score = sim_game.uma().calc_score() as f64;
            let current_pt_score = sim_game.uma().calc_score_with_pt_favor() as f64;
            let pt_bias = current_pt_score - current_score;
            Ok((score_mean, score_mean + pt_bias))
        }
    }

    #[cfg(feature = "onnx")]
    /// 单次 rollout，跑到终局或 `max_depth` 截断处（NN leaf 微批路径用）
    ///
    /// `seed` 语义与 [`Self::simulate`] 一致：同一 rollout 序号在所有候选上共享，
    /// 且按 `(回合, 阶段)` 重播种，使本路径与 [`Self::simulate`] 的 CRN 行为一致。
    fn simulate_until_terminal_or_leaf(&self, game: &OnsenGame, action: &OnsenAction, seed: u64) -> Result<SimOutcome> {
        let rng = &mut StdRng::seed_from_u64(seed);
        // Dig/Upgrade 目前仍走完整模拟（未对齐 max_depth）；这里直接复用现有路径，视为 Terminal
        if matches!(action, OnsenAction::Dig(_)) {
            let (s, pt) = self.simulate_onsen_select(game, action, rng)?;
            return Ok(SimOutcome::Terminal { score: s, score_pt: pt });
        }
        if matches!(action, OnsenAction::Upgrade(_)) {
            let (s, pt) = self.simulate_dig_upgrade(game, action, rng)?;
            return Ok(SimOutcome::Terminal { score: s, score_pt: pt });
        }

        // 克隆游戏状态
        let mut sim_game = game.clone();
        let trainer_hw = SimulationTrainer {
            evaluator: &self.rollout_evaluator
        };

        // 执行初始动作
        sim_game.apply_action(action, rng)?;

        // max_depth==0：保持旧行为，rollout 跑到终局
        if self.config.max_depth == 0 {
            while sim_game.next() {
                self.reseed_for_stage(rng, seed, &sim_game);
                sim_game.run_stage(&trainer_hw, rng)?;
            }
            sim_game.on_simulation_end(&trainer_hw, rng)?;
            return Ok(SimOutcome::Terminal {
                score: sim_game.uma().calc_score() as f64,
                score_pt: sim_game.uma().calc_score_with_pt_favor() as f64
            });
        }

        // max_depth>0：按 turn 截断；未终局则返回 leaf features（不在这里做推理）
        let start_turn = sim_game.turn;
        let max_depth = self.config.max_depth as i32;
        let mut finished = false;

        loop {
            if !sim_game.next() {
                finished = true;
                break;
            }
            self.reseed_for_stage(rng, seed, &sim_game);
            sim_game.run_stage(&trainer_hw, rng)?;
            if (sim_game.turn - start_turn) >= max_depth {
                break;
            }
        }

        if finished || sim_game.turn >= sim_game.max_turn() {
            sim_game.on_simulation_end(&trainer_hw, rng)?;
            return Ok(SimOutcome::Terminal {
                score: sim_game.uma().calc_score() as f64,
                score_pt: sim_game.uma().calc_score_with_pt_favor() as f64
            });
        }

        let current_score = sim_game.uma().calc_score() as f64;
        let current_pt_score = sim_game.uma().calc_score_with_pt_favor() as f64;
        let pt_bias = current_pt_score - current_score;
        let features = sim_game.extract_nn_features(None);

        Ok(SimOutcome::Leaf { features, pt_bias })
    }

    /// 模拟选择温泉. 因为没有做成单独的阶段，所以单独处理
    pub fn simulate_onsen_select(
        &self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng
    ) -> Result<(f64, f64)> {
        let mut sim_game = game.clone();
        let mut best_score = (0.0, 0.0);

        sim_game.apply_action(action, rng)?;
        for i in sim_game.get_upgradeable_equipment() {
            let score = self.simulate_dig_upgrade(&sim_game, &OnsenAction::Upgrade(i as i32), rng)?;
            if score.0 > best_score.0 {
                best_score = score;
            }
        }
        Ok(best_score)
    }

    /// 模拟升级挖掘装备
    pub fn simulate_dig_upgrade(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> Result<(f64, f64)> {
        let mut sim_game = game.clone();
        sim_game.apply_action(action, rng)?;
        sim_game.pending_selection = false;
        // 去除pending_selection状态后就可以正常模拟了。
        let trainer_hw = SimulationTrainer {
            evaluator: &self.rollout_evaluator
        };
        while sim_game.next() {
            sim_game.run_stage(&trainer_hw, rng)?;
        }
        sim_game.on_simulation_end(&trainer_hw, rng)?;
        Ok((
            sim_game.uma().calc_score() as f64,
            sim_game.uma().calc_score_with_pt_favor() as f64
        ))
    }
}

impl FlatSearch<RamenGame> {
    /// 拉面根节点搜索
    ///
    /// 全部候选走通用 rollout：拉面的 `RamenSelect` / `SpecialSelect` / `Train`
    /// 已是正规回合阶段（`run_stage` -> `list_actions` -> `select_action`），
    /// 不像温泉 `Dig`/`Upgrade` 那样需要特判。
    ///
    /// # 动作空间
    ///
    /// 同时接受三阶段动作与合并动作。判别式：
    /// `RamenSelect` 阶段 + `StageOnly` + `special_targets.is_some()` → 合并动作，
    /// 由 [`FlatSearchGame::apply_root_action`] 转交 `apply_combined_ramen_decision`。
    /// 其余动作仍走 `apply_action_with_strategy`。
    ///
    /// **不要在同一次搜索的候选表里混用两种动作**：混用会让搜索在语义不一致的
    /// 动作空间上比较，虽然技术上不会崩。
    pub fn search(
        &self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng
    ) -> Result<RamenSearchOutput> {
        let Some(backend) = self.batch_rollout.as_ref() else {
            return self.search_with_terminal(game, actions, rng, |game, action, seed| {
                self.simulate_common_extract(game, action, seed, seed, RamenTerminal::from_game)
            });
        };
        // 整根预计算只实现均匀分配。必须在后端工作前拒绝 UCB，不能先跑满全部
        // 候选，再用 UCB 消费其中一部分或在事后才报告消费次数不符。
        ensure!(!self.config.use_ucb, "批量 rollout 后端仅支持均匀分配，use_ucb 必须为 false");
        ensure!(!actions.is_empty(), "没有可用动作");
        ensure!(self.config.max_depth == 0, "拉面批量 rollout 要求 max_depth 为 0");
        ensure!(self.config.search_n > 0, "批量 rollout 的 search_n 必须为正");
        // 内核回调传入动作引用，查表时按动作还原下标。重复动作会被 position
        // 归到第一项，因此显式拒绝，不改变任何合法候选的顺序或评分。
        for (idx, action) in actions.iter().enumerate() {
            ensure!(!actions[..idx].contains(action), "批量 rollout 候选 {idx} 与前面的候选重复");
        }
        let expected = actions
            .len()
            .checked_mul(self.config.search_n)
            .ok_or_else(|| anyhow!("批量 rollout 的候选数与样本数乘积溢出"))?;
        // 用 rng 的克隆体预派生根种子：内核随后会从**原 rng** 派生出同一个根，
        // 故 RNG 消耗与不接后端时逐位一致。
        let seeds = RolloutSeeds::from_rng(&mut rng.clone());
        let table = backend.precompute(game, actions, &seeds, self.config.search_n)?;
        // 结构错误不能走 simulate_many 的宽松失败路径，否则缺键会被当成一条
        // 失败 rollout 丢弃。先查总数，再逐项查键，连同「等量替换了错误键」也拒绝。
        ensure!(
            table.len() == expected,
            "批量结果条目数 {} 与期望 {expected} 不符（不得缺项或包含多余项）",
            table.len()
        );
        for idx in 0..actions.len() {
            for j in 0..self.config.search_n {
                let seed = seeds.seed_at(j);
                ensure!(
                    table.contains_key(&(idx, seed)),
                    "批量结果缺 (候选 {idx}, rollout {j}, 种子 {seed:#018x})"
                );
            }
        }
        // 守门：`use_ucb=false` 下内核应给每个候选恰好 `search_n` 条。
        // 数的是**内核实际消费次数**——后端产出多少条，与内核用掉多少条是两件事。
        let consumed: Vec<AtomicUsize> = (0..actions.len()).map(|_| AtomicUsize::new(0)).collect();
        let out = self.search_with_terminal(game, actions, rng, |_g, action, seed| {
            let idx = actions
                .iter()
                .position(|x| x == action)
                .ok_or_else(|| anyhow!("批量结果查表时找不到候选下标"))?;
            let outcome = table
                .get(&(idx, seed))
                .cloned()
                .ok_or_else(|| anyhow!("批量结果缺 (候选 {idx}, 种子 {seed:#018x})：后端与内核的种子派生不一致"))?;
            consumed[idx].fetch_add(1, Ordering::Relaxed);
            Ok(outcome)
        })?;
        for (idx, got) in consumed.iter().enumerate() {
            let got = got.load(Ordering::Relaxed);
            ensure!(
                got == self.config.search_n,
                "候选 {idx} 实际消费 {got} 条与 search_n={} 不符：汇总口径不是均匀分配",
                self.config.search_n
            );
        }
        Ok(out)
    }
}

/// 回合阶段编号（种子派生用）
///
/// 显式 match 而非依赖枚举判别值：`OnsenTurnStage` 的变体顺序若调整，
/// 这里会编译报错提醒同步，而不是静默改变所有历史种子。
fn _stage_id(stage: &OnsenTurnStage) -> u64 {
    match stage {
        OnsenTurnStage::Begin => 0,
        OnsenTurnStage::Distribute => 1,
        OnsenTurnStage::Bathing => 2,
        OnsenTurnStage::Train => 3,
        OnsenTurnStage::AfterTrain => 4
    }
}

#[cfg(feature = "onnx")]
enum SimOutcome {
    Terminal { score: f64, score_pt: f64 },
    Leaf { features: Vec<f32>, pt_bias: f64 }
}

/// 模拟用训练员
///
/// 包装 HandwrittenEvaluator，实现 Trainer trait。
struct SimulationTrainer<'a> {
    evaluator: &'a HandwrittenEvaluator
}

impl<'a> crate::game::Trainer<OnsenGame> for SimulationTrainer<'a> {
    fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng) -> Result<usize> {
        // 只有一个动作时直接返回
        if actions.len() <= 1 {
            return Ok(0);
        }

        // 检查是否是温泉选择场景（所有动作都是 Dig）
        let all_dig = actions.iter().all(|a| matches!(a, OnsenAction::Dig(_)));
        if all_dig {
            return Ok(self.evaluator.select_onsen_index(game, actions));
        }

        // 检查是否是装备升级场景
        let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
        if all_upgrade {
            return Ok(self.evaluator.select_upgrade_action(game, actions));
        }

        // 使用 HandwrittenEvaluator 的 select_action 逻辑
        let selected_action = self.evaluator.select_action(game, rng);
        let idx = match &selected_action {
            Some(action) => actions.iter().position(|a| *a == action.selection).unwrap_or(0),
            None => 0
        };

        Ok(idx)
    }

    fn select_choice(&self, game: &OnsenGame, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        // 使用 HandwrittenEvaluator 的 evaluate_choice 逻辑
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (i, _choice) in choices.iter().enumerate() {
            let value = self.evaluator.evaluate_choice(game, i);
            if value > best_value {
                best_value = value;
                best_idx = i;
            }
        }

        Ok(best_idx)
    }
}

// 说明：E6 的“rollout 动作走 NN”已回退；rollout 全程固定使用 SimulationTrainer(HandwrittenEvaluator)。

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rand::{RngCore, SeedableRng};
    use anyhow::anyhow;
    use super::*;
    use crate::{
        game::{
            InheritInfo, Trainer,
            ramen::{RamenAction, RamenGame, RamenStage}
        },
        gamedata::init_global,
        rng::derive_seed,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    /// 单个候选的统计摘要（回归比对的最小单元）
    ///
    /// 只取样本数与均值：直方图整体比对过于笨重，而这两项已足以暴露
    /// 随机流错位——种子一变，均值必然漂移。
    #[derive(Debug, Clone, PartialEq)]
    struct ActionDigest {
        /// 成功样本数
        n: u32,
        /// 分数均值
        mean: f64
    }

    /// 在首个多候选决策点捕获搜索结果，然后固定选 0 号动作
    ///
    /// 直接从外部构造搜索根局面会得到不合法状态（`next()` 空推进会跳过阶段初始化），
    /// 故通过真实的 `run_stage` → `Trainer::select_action` 路径取根节点。
    struct CapturingTrainer {
        /// 被测搜索器
        search: FlatSearch,
        /// 搜索入口种子
        seed: u64,
        /// 捕获到的各候选统计（`None` 表示尚未捕获）
        captured: RefCell<Option<Vec<ActionDigest>>>,
        /// 是否反转候选顺序后再搜索（用于顺序无关性回归）
        reverse: bool
    }

    impl CapturingTrainer {
        /// 构造捕获用 trainer
        fn new(config: SearchConfig, seed: u64, reverse: bool) -> Self {
            Self {
                search: FlatSearch::new(config),
                seed,
                captured: RefCell::new(None),
                reverse
            }
        }

        /// 取出捕获结果
        fn take(&self) -> Result<Vec<ActionDigest>> {
            self.captured
                .borrow_mut()
                .take()
                .ok_or_else(|| anyhow!("整局结束仍未遇到多候选决策点"))
        }
    }

    impl Trainer<OnsenGame> for CapturingTrainer {
        fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], _rng: &mut StdRng) -> Result<usize> {
            if self.captured.borrow().is_none() && actions.len() >= 2 {
                let mut owned = actions.to_vec();
                if self.reverse {
                    owned.reverse();
                }
                let mut rng = StdRng::seed_from_u64(self.seed);
                let out = self.search.search(game, &owned, &mut rng)?;
                let mut digest: Vec<ActionDigest> = out
                    .action_results
                    .iter()
                    .map(|r| ActionDigest {
                        n: r.0.count(),
                        mean: r.0.mean()
                    })
                    .collect();
                // 统一回正序，使正/逆序两次运行的结果可直接逐项比对
                if self.reverse {
                    digest.reverse();
                }
                *self.captured.borrow_mut() = Some(digest);
            }
            Ok(0)
        }

        fn select_choice(&self, _game: &OnsenGame, _choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
            Ok(0)
        }
    }

    /// 捕获首个多候选决策点的**局面本身**（CRN 测量需要直接调 `simulate`）
    struct RootCapture {
        /// 捕获到的 (根局面, 候选表)
        got: RefCell<Option<(OnsenGame, Vec<OnsenAction>)>>
    }

    impl Trainer<OnsenGame> for RootCapture {
        fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], _rng: &mut StdRng) -> Result<usize> {
            if self.got.borrow().is_none() && actions.len() >= 2 {
                *self.got.borrow_mut() = Some((game.clone(), actions.to_vec()));
            }
            Ok(0)
        }

        fn select_choice(&self, _game: &OnsenGame, _choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
            Ok(0)
        }
    }

    /// 取首个多候选决策点的根局面与候选表
    fn root_state() -> Result<(OnsenGame, Vec<OnsenAction>)> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let inherit = InheritInfo {
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = OnsenGame::newgame(102601, &deck, inherit)?;
        let cap = RootCapture { got: RefCell::new(None) };
        let mut rng = StdRng::seed_from_u64(20260822);
        while game.next() {
            game.run_stage(&cap, &mut rng)?;
            if cap.got.borrow().is_some() {
                break;
            }
        }
        cap.got
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("整局结束仍未遇到多候选决策点"))
    }

    /// 样本均值
    fn mean_of(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len().max(1) as f64
    }

    /// 样本方差（无偏）
    fn var_of(xs: &[f64]) -> f64 {
        let m = mean_of(xs);
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len().saturating_sub(1)).max(1) as f64
    }

    /// Pearson 相关系数
    fn corr_of(a: &[f64], b: &[f64]) -> f64 {
        let (ma, mb) = (mean_of(a), mean_of(b));
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let da: f64 = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
        let db: f64 = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();
        if da <= 0.0 || db <= 0.0 { 0.0 } else { cov / (da * db) }
    }

    /// 回归基准专用配置
    ///
    /// 强制 `use_ucb=false`：UCB 的样本分配依赖分数，代码一改样本数就变，
    /// 无法作为「改动前后输出一致」的尺子。均匀分配下每个候选固定 `search_n` 次。
    fn regression_config() -> SearchConfig {
        SearchConfig::default().with_search_n(16).with_ucb(false)
    }

    /// 回归基准配置（UCB 路径）
    ///
    /// `use_ucb` 默认为 `true`，是活跃入口实际走的路径，必须单独覆盖。
    /// `group_size` 取小值以免单次搜索过久。
    fn regression_config_ucb() -> SearchConfig {
        SearchConfig::default()
            .with_search_n(16)
            .with_ucb(true)
            .with_search_group_size(4)
    }

    /// 跑一局到首个多候选决策点，返回该点的搜索统计（均匀分配）
    fn capture(seed: u64, reverse: bool) -> Result<Vec<ActionDigest>> {
        capture_with(regression_config(), seed, reverse)
    }

    /// 同 [`capture`]，但可指定搜索配置
    fn capture_with(config: SearchConfig, seed: u64, reverse: bool) -> Result<Vec<ActionDigest>> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let inherit = InheritInfo {
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = OnsenGame::newgame(102601, &deck, inherit)?;

        let trainer = CapturingTrainer::new(config, seed, reverse);
        // 局面推进本身用固定种子，保证根局面在各次运行间一致
        let mut rng = StdRng::seed_from_u64(20260822);
        while game.next() {
            game.run_stage(&trainer, &mut rng)?;
            if trainer.captured.borrow().is_some() {
                break;
            }
        }
        trainer.take()
    }

    /// 回归 1：同一 seed 两次搜索必须完全一致
    ///
    /// 这是泛型化改造的护栏——没有它，`FlatSearch<G>` 改坏了也无从发现。
    /// 注：仓库测试规范一般要求用 `println` 而非 `assert`，回归基准是刻意的例外：
    /// 只打印不断言，回归就形同虚设。
    #[test]
    fn test_search_reproducible_same_seed() -> Result<()> {
        let a = capture(42, false)?;
        let b = capture(42, false)?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            println!("动作 {i}: n={} mean={:.6} | n={} mean={:.6}", x.n, x.mean, y.n, y.mean);
        }
        assert_eq!(a, b, "同一 seed 两次搜索结果必须完全一致");
        Ok(())
    }

    /// 回归 2：不同 seed 必须给出不同结果（否则种子根本没接进 rollout）
    #[test]
    fn test_search_seed_actually_used() -> Result<()> {
        let a = capture(42, false)?;
        let b = capture(4242, false)?;
        println!("seed=42   : {a:?}");
        println!("seed=4242 : {b:?}");
        assert_ne!(a, b, "换 seed 必须改变搜索结果，否则种子未接入 rollout");
        Ok(())
    }

    /// 回归 3：候选顺序重排后，各动作统计量按动作对齐后不变
    ///
    /// 专抓「候选索引混进种子派生」——一旦 `seed_at` 吃了候选下标，
    /// 重排 actions 就会让同一动作拿到不同随机流，本测试立刻失败。
    #[test]
    fn test_search_invariant_to_action_order() -> Result<()> {
        let normal = capture(42, false)?;
        let reversed = capture(42, true)?;
        for (i, (a, b)) in normal.iter().zip(reversed.iter()).enumerate() {
            println!(
                "动作 {i}: 正序 n={} mean={:.6} | 逆序 n={} mean={:.6}",
                a.n, a.mean, b.n, b.mean
            );
        }
        assert_eq!(normal, reversed, "各动作统计量不应随候选顺序变化");
        Ok(())
    }

    /// 回归 4：UCB 路径下同一 seed 两次搜索必须一致
    ///
    /// `use_ucb` 默认为 `true`，是 `umaai` / `umasim` 活跃入口实际走的路径。
    /// 回归 1~3 写死 `use_ucb=false`（UCB 分配依赖分数，不适合当泛型化的尺子），
    /// 但可复现性本身在 UCB 下同样必须成立，故单独覆盖。
    #[test]
    fn test_search_ucb_reproducible() -> Result<()> {
        let a = capture_with(regression_config_ucb(), 42, false)?;
        let b = capture_with(regression_config_ucb(), 42, false)?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            println!("动作 {i}: n={} mean={:.6} | n={} mean={:.6}", x.n, x.mean, y.n, y.mean);
        }
        assert_eq!(a, b, "UCB 路径下同一 seed 两次搜索结果必须一致");
        Ok(())
    }

    /// UCB 路径的候选顺序敏感性（诊断用，不断言）
    ///
    /// UCB 按分数动态分配预算且平局时取索引最小者，故候选重排**可能**改变各动作
    /// 拿到的样本数——这是算法固有性质，不是可复现性缺陷。本测试只打印差异供观察；
    /// 泛型化的顺序无关性护栏由均匀分配的回归 3 承担。
    #[test]
    fn test_search_ucb_order_sensitivity() -> Result<()> {
        let normal = capture_with(regression_config_ucb(), 42, false)?;
        let reversed = capture_with(regression_config_ucb(), 42, true)?;
        let mut diff = 0usize;
        for (i, (a, b)) in normal.iter().zip(reversed.iter()).enumerate() {
            let same = a == b;
            if !same {
                diff += 1;
            }
            println!(
                "动作 {i}: 正序 n={} mean={:.6} | 逆序 n={} mean={:.6} | {}",
                a.n,
                a.mean,
                b.n,
                b.mean,
                if same { "一致" } else { "不同" }
            );
        }
        println!("UCB 下候选重排导致 {diff}/{} 个动作统计不同", normal.len());
        Ok(())
    }

    // ========== 拉面根节点冒烟（Phase 1.4 验收） ==========

    /// 捕获拉面首个多候选决策点的局面与候选表
    struct RamenRootCapture {
        /// 捕获到的 (根局面, 候选表)
        got: RefCell<Option<(RamenGame, Vec<RamenAction>)>>
    }

    impl Trainer<RamenGame> for RamenRootCapture {
        fn select_action(&self, game: &RamenGame, actions: &[RamenAction], _rng: &mut StdRng) -> Result<usize> {
            // 避开第 3 年地区选择：该阶段最多 120 个候选，冒烟不必付这个代价
            let skip = matches!(game.stage, crate::game::ramen::RamenStage::RegionSelect);
            if self.got.borrow().is_none() && actions.len() >= 2 && !skip {
                *self.got.borrow_mut() = Some((game.clone(), actions.to_vec()));
            }
            Ok(0)
        }

        fn select_choice(&self, _game: &RamenGame, _choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
            Ok(0)
        }
    }

    /// 取拉面首个多候选决策点
    fn ramen_root() -> Result<(RamenGame, Vec<RamenAction>)> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let inherit = InheritInfo {
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = RamenGame::newgame(102601, &deck, inherit)?;
        let cap = RamenRootCapture { got: RefCell::new(None) };
        let mut rng = StdRng::seed_from_u64(20260822);
        while game.next() {
            game.run_stage(&cap, &mut rng)?;
            if cap.got.borrow().is_some() {
                break;
            }
        }
        cap.got
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("整局结束仍未遇到多候选决策点"))
    }

    /// 跑一次拉面根节点搜索，返回各候选统计
    fn ramen_search(seed: u64) -> Result<Vec<ActionDigest>> {
        let (game, actions) = ramen_root()?;
        let search: FlatSearch<RamenGame> = FlatSearch::new(regression_config());
        let mut rng = StdRng::seed_from_u64(seed);
        let out = search.search(&game, &actions, &mut rng)?;
        Ok(out
            .action_results
            .iter()
            .map(|r| ActionDigest {
                n: r.0.count(),
                mean: r.0.mean()
            })
            .collect())
    }

    /// Phase 1.4 验收 1：拉面能跑通根节点搜索，且固定种子可复现
    #[test]
    fn test_ramen_root_search_reproducible() -> Result<()> {
        let (game, actions) = ramen_root()?;
        println!(
            "拉面根局面: 回合 {} 阶段 {:?}，候选 {} 个",
            game.turn(),
            game.stage,
            actions.len()
        );

        let a = ramen_search(42)?;
        let b = ramen_search(42)?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            println!("动作 {i}: n={} mean={:.6} | n={} mean={:.6}", x.n, x.mean, y.n, y.mean);
        }
        assert_eq!(a, b, "拉面根节点搜索必须可复现");
        Ok(())
    }

    /// Phase 1.4 验收 2：换种子必须改变结果
    ///
    /// 拉面的关键在于规则层第二条随机流 `internal_rng`——若 `fork_for_rollout`
    /// 没有注入它，`take_internal_rng()` 会回退 `from_os_rng()`，
    /// 届时验收 1 会失败（同种子两次不一致）。两条测试合起来才能证明两条流都受控。
    #[test]
    fn test_ramen_root_search_seed_used() -> Result<()> {
        let a = ramen_search(42)?;
        let b = ramen_search(4242)?;
        println!("seed=42   : {a:?}");
        println!("seed=4242 : {b:?}");
        assert_ne!(a, b, "换 seed 必须改变拉面搜索结果");
        Ok(())
    }

    /// 构造处于 `RamenSelect`、库存富余、三面可选的局面（P1.1 单测用）
    fn ramen_select_ready() -> Result<RamenGame> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let inherit = InheritInfo {
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = RamenGame::newgame(102601, &deck, inherit)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;
        Ok(game)
    }

    /// P1.1 测试 1：合并动作经 `apply_root_action` 真正写入 targets，不会被清零
    #[test]
    fn test_ramen_combined_action_preserves_targets() -> Result<()> {
        let mut combined_game = ramen_select_ready()?;
        let mut three_game = combined_game.clone();
        let mut rng = StdRng::seed_from_u64(0);

        let actions = combined_game.list_combined_ramen_select_actions();
        let action = actions
            .iter()
            .copied()
            .find(|a| a.ramen.is_some() && matches!(a.special_targets, Some(t) if t != [0, 0, 0]))
            .ok_or_else(|| anyhow!("库存富余下应存在非全零 targets 的合并动作"))?;
        println!(
            "选用合并动作: ramen={:?} special_targets={:?}",
            action.ramen, action.special_targets
        );

        combined_game.apply_root_action(&action, &mut rng)?;
        println!(
            "合并路径: pending_ramen={:?} pending_special_targets={:?} combined_decision={}",
            combined_game.ramen.pending_ramen,
            combined_game.ramen.pending_special_targets,
            combined_game.ramen.combined_decision
        );
        assert_eq!(combined_game.ramen.pending_ramen, action.ramen);
        assert_eq!(
            combined_game.ramen.pending_special_targets,
            action.special_targets.unwrap_or([0, 0, 0])
        );
        assert_ne!(
            combined_game.ramen.pending_special_targets,
            [0, 0, 0],
            "合并路径 targets 必须非全零"
        );
        assert!(
            combined_game.ramen.combined_decision,
            "合并路径应设 combined_decision"
        );

        let three_action = RamenAction::ramen_select(action.ramen);
        three_game.apply_root_action(&three_action, &mut rng)?;
        println!(
            "三阶段路径: pending_ramen={:?} pending_special_targets={:?} combined_decision={}",
            three_game.ramen.pending_ramen,
            three_game.ramen.pending_special_targets,
            three_game.ramen.combined_decision
        );
        assert_eq!(three_game.ramen.pending_ramen, action.ramen);
        assert_eq!(three_game.ramen.pending_special_targets, [0, 0, 0]);
        assert!(
            !three_game.ramen.combined_decision,
            "三阶段路径 combined_decision 应为 false"
        );
        Ok(())
    }

    /// P1.1 测试 2：三阶段动作的根搜索统计与改动前逐位一致
    #[test]
    fn test_ramen_three_stage_action_unchanged() -> Result<()> {
        let (game, actions) = ramen_root()?;
        println!(
            "拉面根局面: 回合 {} 阶段 {:?}，候选 {} 个",
            game.turn(),
            game.stage,
            actions.len()
        );

        let a = ramen_search(42)?;
        let b = ramen_search(42)?;
        // 2026-08-25 更新：不在判定与得意率解耦 + 地区分身缺席优先，rollout 数值变化，基准重抓
        // 2026-08-27 更新（两次叠加）：
        // (1) 五维上限剧本化，速度上限 2958→3337，rollout 终局分数整体抬升；
        // (2) searchable.rs RolloutTrainer 切到 RecommendedRamenTrainer，均值再上移 ~10k。
        // 上游 (2) 的基准是在 (1) 之前测的，两者叠加后已在本分支重抓。
        // 2026-09 更新：吃面 PT 增量 / eat_count 延后到 NextTurn，训练阶段用吃面前 PT
        // 算 ramen_pt_effect / region_bonus 档位，rollout 数值整体下移，基准重抓。
        // 2026-09-17 重抓：合宿训练诀窍全 MAX 修复（d9374e8）后 rollout 数值上移，基准重抓。
        // 2026-09-17 二次重抓：GA 方向 9 旋钮组合档进入 preset 后 rollout 数值再移，基准重抓。
        // 2026-09-18 重抓：第十二轮组合档进入 preset，rollout 数值再移，基准重抓。
        // 2026-09-21 重抓：友人出行跨年配额定档 [0,3,5]（原 [0,2,5]），rollout 数值再移。
        // 2026-10-01 重抓：切者/小切按 1.1 / 1.04 缩放终局评分的 PT 项
        // （`UmaFlags::pt_score_rate_factor`）。rollout 内抽到切者的事件会让终局分
        // 整体上移，均值随之 +118 左右，基准重抓。
        // 2026-10-04 重抓：超级拉面效果修正——只保留 RMJ + finals（去掉误叠加的
        // pt_effect / basic 试食会效果），并接入选中选项的 +100 训练上限，
        // rollout 的 URA 训练数值变化，均值整体下移约 750。
        // 2026-10-06 重抓：PT 上段上限口径修正——普通回合 `ramen_basic_effect.status_limit`
        // 同时抬属性与 PT 上限（「獲得上限アップ」Y2 +20 / Y3 +40），吃面回合 PT 上限下降，
        // rollout 终局 PT 变少，均值整体再下移约 120~190，基准重抓。
        // 2026-10-06 重抓：五维评分表换用 URA `StatusToPoint`（raw 表 3802 项），
        // 终局五维分（显示值 ≥2001 区间）下调，均值整体再下移约 180~300，基准重抓。
        // 2026-10-07 重抓：拉面 PT 口径改为 M2——PT 友情**不**剔除 RMJ（友情对属性与 PT
        // 同时生效），且吃面回合的效果档位按**吃面前** PT 取。PT 收益相对 M1 回升，
        // rollout 终局分上移约 150~640，基准重抓。
        let expected: [(u32, f64); 7] = [
            (16, 64099.375000),
            (16, 63692.062500),
            (16, 63600.812500),
            (16, 63552.062500),
            (16, 63645.062500),
            (16, 63352.812500),
            (16, 63613.375000)
        ];
        for (i, ((x, y), (en, em))) in a.iter().zip(b.iter()).zip(expected).enumerate() {
            println!(
                "动作 {i}: n={} mean={:.6} | n={} mean={:.6} | 改动前 n={} mean={:.6}",
                x.n, x.mean, y.n, y.mean, en, em
            );
        }
        assert_eq!(a, b, "固定种子下拉面根节点搜索必须可复现");
        assert_eq!(a.len(), expected.len(), "候选数应与改动前一致");
        for (i, (got, (en, em))) in a.iter().zip(expected).enumerate() {
            assert_eq!(got.n, en, "动作 {i} 的 n 必须与改动前逐位相同");
            assert_eq!(got.mean, em, "动作 {i} 的 mean 必须与改动前逐位相同");
        }
        Ok(())
    }

    /// P1.1 测试 3：合并动作整局冒烟，全程跳过 SpecialSelect
    #[test]
    fn test_ramen_combined_action_full_game_smoke() -> Result<()> {
        use crate::trainer::RecommendedRamenTrainer;

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let inherit = InheritInfo {
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let (mut rng, rule_master) = crate::bench::seeded_rngs(42, 0);
        let mut game = RamenGame::newgame(102601, &deck, inherit)?;
        game.set_rule_master(rule_master);

        let trainer = RecommendedRamenTrainer::new();
        let mut ramen_select_n = 0usize;
        let mut special_select_n = 0usize;
        let mut combined_n = 0usize;

        {
            let mut step = |game: &mut RamenGame, rng: &mut StdRng| -> Result<()> {
                match game.stage {
                    RamenStage::RamenSelect if !game.is_race_turn() => {
                        ramen_select_n += 1;
                        let actions = game.list_combined_ramen_select_actions();
                        let action = actions
                            .iter()
                            .copied()
                            .find(|a| {
                                a.ramen.is_some()
                                    && matches!(a.special_targets, Some(t) if t != [0, 0, 0])
                            })
                            .or_else(|| actions.iter().copied().find(|a| a.ramen.is_some()))
                            .or_else(|| actions.first().copied())
                            .ok_or_else(|| anyhow!("合并候选为空"))?;
                        game.apply_root_action(&action, rng)?;
                        combined_n += 1;
                        Ok(())
                    }
                    RamenStage::RamenSelect => {
                        ramen_select_n += 1;
                        game.run_stage(&trainer, rng)
                    }
                    RamenStage::SpecialSelect => {
                        special_select_n += 1;
                        game.run_stage(&trainer, rng)
                    }
                    _ => game.run_stage(&trainer, rng)
                }
            };

            step(&mut game, &mut rng)?;
            while game.next() {
                step(&mut game, &mut rng)?;
            }
        }
        game.on_simulation_end(&trainer, &mut rng)?;

        let score = game.uma().calc_score();
        println!(
            "合并动作整局: 回合={} 评分={} RamenSelect={} SpecialSelect={} 合并apply={}",
            game.turn(),
            score,
            ramen_select_n,
            special_select_n,
            combined_n
        );
        assert!(score > 0, "终局评分应 > 0");
        assert!((score as f64).is_finite(), "终局评分应为有限值");
        assert_eq!(special_select_n, 0, "合并路径不应进入 SpecialSelect");
        assert!(combined_n > 0, "至少应用过一次合并动作");
        Ok(())
    }

    /// P1.1 测试 4：非法 targets 的合并动作必须被 `apply_root_action` 拒绝
    #[test]
    fn test_ramen_combined_action_rejects_illegal_targets() -> Result<()> {
        use crate::game::ramen::rules::list_special_targets_for;

        let mut game = ramen_select_ready()?;
        let mut rng = StdRng::seed_from_u64(0);
        let legal = list_special_targets_for(&game.ramen, 0)?;
        println!("面 0 合法 targets: {legal:?}");

        let illegal = [3, 0, 0];
        assert!(
            !legal.contains(&illegal),
            "对照用的 {illegal:?} 不应出现在合法列表里"
        );
        let action = RamenAction::combined_select(Some(0), illegal);
        let result = game.apply_root_action(&action, &mut rng);
        println!("非法 targets 错误: {result:?}");
        assert!(result.is_err(), "非法 targets 必须返回 Err，不能静默接受");
        Ok(())
    }

    /// 拉面 CRN 测量的双种子
    ///
    /// `shared == true`：决策 RNG 与 `rule_master` 都是 `seed_at(j)`（生产行为）。
    /// `shared == false`：决策 RNG 仍共享，`rule_master` 带上候选身份。
    fn ramen_crn_seeds(seeds: &RolloutSeeds, candidate: usize, j: usize, shared: bool) -> (u64, u64) {
        let decision = seeds.seed_at(j);
        if shared {
            (decision, decision)
        } else {
            (decision, derive_seed(decision, &[candidate as u64]))
        }
    }

    /// 只保留双方在同一原始下标 `j` 都成功的样本
    ///
    /// 返回 `(xa, xb, 原始下标)`。先 `flatten` 再按压缩下标配对会把
    /// 「A 的第 5 个成功样本」配到「B 的第 6 个」，本函数禁止那种做法。
    fn aligned_pairs(a: &[Option<f64>], b: &[Option<f64>]) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut xa = Vec::new();
        let mut xb = Vec::new();
        let mut idx = Vec::new();
        for (j, (oa, ob)) in a.iter().zip(b).enumerate() {
            if let (Some(va), Some(vb)) = (*oa, *ob) {
                xa.push(va);
                xb.push(vb);
                idx.push(j);
            }
        }
        (xa, xb, idx)
    }

    /// 一次 CRN 臂的测量摘要
    struct CrnArmStats {
        /// 臂标签
        label: &'static str,
        /// 候选两两的相关系数
        corrs: Vec<f64>,
        /// 候选两两的等效倍率
        gains: Vec<f64>,
        /// 失败的 rollout 次数（所有候选合计）
        failed: usize
    }

    impl CrnArmStats {
        /// 平均相关系数
        fn mean_corr(&self) -> f64 {
            mean_of(&self.corrs)
        }

        /// 平均等效倍率
        fn mean_gain(&self) -> f64 {
            mean_of(&self.gains)
        }
    }

    /// 跑一臂拉面 CRN 测量：共享或独立 `rule_master`
    fn measure_ramen_crn_arm(shared: bool, rollouts: usize) -> Result<CrnArmStats> {
        let (game, actions) = ramen_root()?;
        let search: FlatSearch<RamenGame> = FlatSearch::new(SearchConfig::default());
        let seeds = RolloutSeeds::from_root(20260822);
        let mut cols: Vec<Vec<Option<f64>>> = Vec::with_capacity(actions.len());
        let mut failed = 0usize;
        for (i, action) in actions.iter().enumerate() {
            let col: Vec<Option<f64>> = (0..rollouts)
                .into_par_iter()
                .map(|j| {
                    let (decision, rule_master) = ramen_crn_seeds(&seeds, i, j, shared);
                    search
                        .simulate_common_with_seeds(&game, action, decision, rule_master)
                        .ok()
                        .map(|v| v.score)
                })
                .collect();
            failed += col.iter().filter(|x| x.is_none()).count();
            cols.push(col);
        }

        let mut corrs = Vec::new();
        let mut gains = Vec::new();
        for a in 0..cols.len() {
            for b in (a + 1)..cols.len() {
                let (xa, xb, idx) = aligned_pairs(&cols[a], &cols[b]);
                if idx.len() < 2 {
                    continue;
                }
                let diff: Vec<f64> = xa.iter().zip(&xb).map(|(x, y)| x - y).collect();
                let indep = var_of(&xa) + var_of(&xb);
                let paired = var_of(&diff);
                if paired > 0.0 {
                    gains.push(indep / paired);
                }
                corrs.push(corr_of(&xa, &xb));
            }
        }
        Ok(CrnArmStats {
            label: if shared {
                "共享 rule_master（生产 CRN）"
            } else {
                "独立 rule_master"
            },
            corrs,
            gains,
            failed
        })
    }

    /// 打印一臂摘要并返回平均倍率
    fn print_crn_arm(arm: &CrnArmStats) -> f64 {
        let gmin = arm.gains.iter().copied().fold(f64::INFINITY, f64::min);
        let gmax = arm.gains.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_gain = arm.mean_gain();
        println!(
            "===== {} =====\n候选对数 {} | 失败 {} | 平均 corr = {:.4} | 平均等效倍率 = {:.2}x | 区间 [{:.2}, {:.2}]",
            arm.label,
            arm.corrs.len(),
            arm.failed,
            arm.mean_corr(),
            mean_gain,
            gmin,
            gmax
        );
        mean_gain
    }

    /// 共享臂平均增益下限
    ///
    /// 2026-08-24 实测（ROLLOUTS=24，21 对）：共享均 7.13x（对内区间 3.17–27.34），
    /// 独立均 1.08x。下限取 3.0：低于实测下限对、高于独立臂噪声上沿。
    const RAMEN_SHARED_CRN_GAIN_FLOOR: f64 = 3.0;

    /// 共享 / 独立臂的 `rule_master` 拓扑
    ///
    /// 把对应修复回退：独立臂改回 `seed_at(j)` → 「独立臂 A/B 不等」红；
    /// 共享臂带上候选索引 → 「共享臂 A/B 相等」红。
    #[test]
    fn test_ramen_crn_seed_topology() -> Result<()> {
        let seeds = RolloutSeeds::from_root(20260822);
        let mut c = Checks::new();

        let (_, rm_a0) = ramen_crn_seeds(&seeds, 0, 0, true);
        let (_, rm_b0) = ramen_crn_seeds(&seeds, 1, 0, true);
        println!("共享 j=0: A={rm_a0:#018x} B={rm_b0:#018x}");
        c.check(rm_a0 == rm_b0, "共享臂：同一 j 上候选 A/B 的 rule_master 相等");
        c.check(rm_a0 == seeds.seed_at(0), "共享臂 rule_master == seed_at(j)（生产语义）");

        let (_, rm_a0i) = ramen_crn_seeds(&seeds, 0, 0, false);
        let (_, rm_b0i) = ramen_crn_seeds(&seeds, 1, 0, false);
        println!("独立 j=0: A={rm_a0i:#018x} B={rm_b0i:#018x}");
        c.check(rm_a0i != rm_b0i, "独立臂：同一 j 上候选 A/B 的 rule_master 不等");

        let (_, rm_a1) = ramen_crn_seeds(&seeds, 0, 1, true);
        let (_, rm_a1i) = ramen_crn_seeds(&seeds, 0, 1, false);
        println!("共享 j=1 A={rm_a1:#018x} / 独立 j=1 A={rm_a1i:#018x}");
        c.check(rm_a0 != rm_a1, "不同 j 的 seed 必须不等（共享臂）");
        c.check(rm_a0i != rm_a1i, "不同 j 的 seed 必须不等（独立臂）");

        let (d_a, _) = ramen_crn_seeds(&seeds, 0, 3, false);
        let (d_b, _) = ramen_crn_seeds(&seeds, 1, 3, false);
        c.check(d_a == d_b, "独立臂仍共享决策 RNG（只拆 rule_master）");
        c.check(d_a == seeds.seed_at(3), "决策 RNG 仍是 seed_at(j)");
        c.finish()
    }

    /// 配对必须按原始 `j` 取交集，不能先 `flatten`
    ///
    /// 把对应修复回退（`aligned_pairs` 改成先 flatten 再 zip）→ 本测试红。
    #[test]
    fn test_crn_pair_alignment_keeps_original_j() -> Result<()> {
        // 候选 A 在 j=1 失败，B 全成功：交集应丢掉 j=1 这一对，其余下标保持
        let a = [Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)];
        let b = [Some(10.0), Some(20.0), Some(30.0), Some(40.0), Some(50.0)];
        let (xa, xb, idx) = aligned_pairs(&a, &b);
        println!("交集下标={idx:?} xa={xa:?} xb={xb:?}");

        let mut c = Checks::new();
        c.check(idx.as_slice() == [0, 2, 3, 4], "只丢弃双方未同时成功的 j=1");
        c.check(xa.as_slice() == [1.0, 3.0, 4.0, 5.0], "A 侧按原始 j 取值");
        c.check(
            xb.as_slice() == [10.0, 30.0, 40.0, 50.0],
            "B 侧按原始 j 取值（不是压缩后的 10,20,30,40）"
        );

        let flat_a: Vec<f64> = a.iter().copied().flatten().collect();
        let flat_b: Vec<f64> = b.iter().copied().flatten().collect();
        let n = flat_a.len().min(flat_b.len());
        let flatten_b = &flat_b[..n];
        println!("flatten 后再配对的 B 侧={flatten_b:?}");
        c.check(xb.as_slice() != flatten_b, "交集配对结果必须不同于 flatten 后再配对");
        c.finish()
    }

    /// UCB 首组把 `group_size` 收进 `search_n`
    ///
    /// 把对应修复回退（删掉 `min(group_size, search_n)`）→ 「每候选 count==8」红（会变成 32）。
    #[test]
    fn test_ucb_first_group_clamps_to_search_n() -> Result<()> {
        let (game, actions) = root_state()?;
        // 分数必须拉开：全相等时 UCB 会轮询每个候选，谁都停不住在首组。
        let best = actions[0].clone();
        let dummy = |_: &OnsenGame, action: &OnsenAction, _: u64| -> Result<SearchScore> {
            let score = if action == &best { 80000.0 } else { 10000.0 };
            Ok(SearchScore {
                score,
                score_pt: 0.0
            })
        };

        let cfg_over = SearchConfig::default()
            .with_search_n(8)
            .with_ucb(true)
            .with_search_group_size(32);
        let search = FlatSearch::new(cfg_over);
        let mut rng = StdRng::seed_from_u64(1);
        let out = search.search_with(&game, &actions, &mut rng, dummy)?;
        let over_counts: Vec<u32> = out.action_results.iter().map(|r| r.0.count()).collect();
        println!("search_n=8 group=32 各候选 count: {over_counts:?}");

        let mut c = Checks::new();
        c.check(!over_counts.is_empty(), "至少有一个候选");
        for (i, &n) in over_counts.iter().enumerate() {
            let msg = format!("候选 {i} count==8（不是 32）");
            c.check(n == 8, &msg);
        }

        let cfg_adapt = SearchConfig::default()
            .with_search_n(32)
            .with_ucb(true)
            .with_search_group_size(8);
        let search = FlatSearch::new(cfg_adapt);
        let mut rng = StdRng::seed_from_u64(1);
        let out = search.search_with(&game, &actions, &mut rng, dummy)?;
        let adapt_counts: Vec<u32> = out.action_results.iter().map(|r| r.0.count()).collect();
        println!("search_n=32 group=8 各候选 count: {adapt_counts:?}");
        c.check(adapt_counts.contains(&8), "存在候选 count==8（停在首组）");
        c.check(adapt_counts.iter().any(|&n| n > 8), "存在候选 count>8（自适应发生）");
        c.finish()
    }

    /// 生产 `simulate_common` 与双种子入口传入相同两值必须逐位一致
    #[test]
    fn test_simulate_common_matches_dual_seed_wrapper() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let search: FlatSearch<RamenGame> = FlatSearch::new(SearchConfig::default());
        let seed = RolloutSeeds::from_root(1).seed_at(0);
        let a = search.simulate_common(&game, &actions[0], seed)?;
        let b = search.simulate_common_with_seeds(&game, &actions[0], seed, seed)?;
        println!("common={a:?} dual={b:?}");
        let mut c = Checks::new();
        c.check(a == b, "simulate_common 与同种子双入口逐位一致");
        c.finish()
    }

    /// P0.1A：温泉 `simulate()` 在 `crn_stage_reseed` 开 / 关时分数向量必须不同
    ///
    /// 必须用 [`root_state`]（回合 0 Train），不能用 Dig / Upgrade：
    /// 那两条路径本来就不调用 `reseed_for_stage`，开关对它们无差别。
    #[test]
    fn test_onsen_crn_reseed_changes_result() -> Result<()> {
        let (game, actions) = root_state()?;
        println!(
            "温泉根局面: 回合 {} 阶段 {:?}，候选 {} 个",
            game.turn, game.stage, actions.len()
        );
        ensure!(!actions.is_empty(), "根局面至少要有一个候选");
        let action = &actions[0];
        println!("测的候选: {action:?}");
        ensure!(
            !matches!(action, OnsenAction::Dig(_) | OnsenAction::Upgrade(_)),
            "root_state 必须落在 Train 决策点；Dig/Upgrade 不走 reseed，测了会假红"
        );

        let n = 8;
        let seeds = RolloutSeeds::from_root(20260822);
        let search_on = FlatSearch::new(SearchConfig::default().with_crn_stage_reseed(true));
        let search_off = FlatSearch::new(SearchConfig::default().with_crn_stage_reseed(false));

        let mut vec_on = Vec::with_capacity(n);
        let mut vec_off = Vec::with_capacity(n);
        for j in 0..n {
            let seed = seeds.seed_at(j);
            let (on, _) = search_on.simulate(&game, action, seed)?;
            let (off, _) = search_off.simulate(&game, action, seed)?;
            vec_on.push(on);
            vec_off.push(off);
        }
        println!("crn_stage_reseed=on  : {vec_on:?}");
        println!("crn_stage_reseed=off : {vec_off:?}");
        assert_ne!(vec_on, vec_off, "温泉 simulate 开关必须改变分数向量");
        Ok(())
    }

    /// P0.1B：拉面走 `simulate_common()`，`crn_stage_reseed` 开 / 关结果必须完全相同
    ///
    /// 温泉生产路径走特化 `simulate()`，`simulate_common()` 从不被调用；
    /// 只有拉面才能验证「泛型路径不受该开关影响」。
    #[test]
    fn test_ramen_simulate_common_ignores_reseed() -> Result<()> {
        let (game, actions) = ramen_root()?;
        println!(
            "拉面根局面: 回合 {} 阶段 {:?}，候选 {} 个",
            game.turn(),
            game.stage,
            actions.len()
        );
        ensure!(!actions.is_empty(), "根局面至少要有一个候选");
        let action = &actions[0];
        println!("测的候选: {action:?}");

        let n = 8;
        let seeds = RolloutSeeds::from_root(20260822);
        let search_on: FlatSearch<RamenGame> =
            FlatSearch::new(SearchConfig::default().with_crn_stage_reseed(true));
        let search_off: FlatSearch<RamenGame> =
            FlatSearch::new(SearchConfig::default().with_crn_stage_reseed(false));

        let mut vec_on = Vec::with_capacity(n);
        let mut vec_off = Vec::with_capacity(n);
        for j in 0..n {
            let seed = seeds.seed_at(j);
            let on = search_on.simulate_common(&game, action, seed)?;
            let off = search_off.simulate_common(&game, action, seed)?;
            vec_on.push(on.score);
            vec_off.push(off.score);
        }
        println!("crn_stage_reseed=on  : {vec_on:?}");
        println!("crn_stage_reseed=off : {vec_off:?}");
        assert_eq!(
            vec_on, vec_off,
            "拉面 simulate_common 不得受 crn_stage_reseed 影响"
        );
        Ok(())
    }

    /// 拉面 CRN 收益小样本（常规测试，给增益断言一条非 ignore 防线）
    ///
    /// 独立臂增益 ≈ 1 是「尺子没坏」的锚点；共享臂必须严格大于独立臂，
    /// 且大于事先写入的下限。失败次数必须为 0，否则配对会悄悄丢样本。
    #[test]
    fn test_crn_pairing_gain_ramen_small() -> Result<()> {
        const ROLLOUTS: usize = 24;
        let indep = measure_ramen_crn_arm(false, ROLLOUTS)?;
        let shared = measure_ramen_crn_arm(true, ROLLOUTS)?;
        let g_indep = print_crn_arm(&indep);
        let g_shared = print_crn_arm(&shared);

        let mut c = Checks::new();
        c.check(indep.failed == 0, "独立臂失败次数为 0");
        c.check(shared.failed == 0, "共享臂失败次数为 0");
        c.check(
            (0.8..=1.2).contains(&g_indep),
            "独立臂平均增益 ≈ 1（落在 0.8..=1.2）"
        );
        c.check(g_shared > g_indep, "共享臂增益严格大于独立臂");
        let floor_msg = format!(
            "共享臂增益 {g_shared:.3} > 下限 {RAMEN_SHARED_CRN_GAIN_FLOOR}"
        );
        c.check(g_shared > RAMEN_SHARED_CRN_GAIN_FLOOR, &floor_msg);
        c.finish()
    }

    /// 有序 rollout：开关开时按序号对齐；关时不分配；开/关不扰动搜索本身
    #[test]
    fn test_ordered_rollouts_records_and_neutral() -> Result<()> {
        let (game, actions) = ramen_root()?;
        println!(
            "拉面根局面: 回合 {} 阶段 {:?}，候选 {} 个",
            game.turn(),
            game.stage,
            actions.len()
        );
        let search_n = 8;
        let cfg_off = SearchConfig::default().with_search_n(search_n).with_ucb(false);
        let cfg_on = cfg_off.clone().with_record_ordered_rollouts(true);
        let search_off: FlatSearch<RamenGame> = FlatSearch::new(cfg_off);
        let search_on: FlatSearch<RamenGame> = FlatSearch::new(cfg_on);

        let mut rng_off = StdRng::seed_from_u64(42);
        let mut rng_on = StdRng::seed_from_u64(42);
        let out_off = search_off.search(&game, &actions, &mut rng_off)?;
        let out_on = search_on.search(&game, &actions, &mut rng_on)?;

        println!(
            "off ordered_rollouts is_none={} best={} | on is_some={} best={}",
            out_off.ordered_rollouts.is_none(),
            out_off.best_action_idx,
            out_on.ordered_rollouts.is_some(),
            out_on.best_action_idx
        );

        let mut c = Checks::new();
        c.check(
            out_off.ordered_rollouts.is_none(),
            "开关关时 ordered_rollouts 为 None（不分配）"
        );
        c.check(
            out_on.ordered_rollouts.is_some(),
            "开关开时 ordered_rollouts 为 Some"
        );

        if let Some(ord) = out_on.ordered_rollouts.as_ref() {
            println!("root_seed={:#018x} per_candidate={}", ord.root_seed, ord.per_candidate.len());
            c.check(
                ord.per_candidate.len() == actions.len(),
                "per_candidate 长度等于候选数"
            );
            for (i, (row, (ar, _))) in ord.per_candidate.iter().zip(out_on.action_results.iter()).enumerate()
            {
                let n_some = row.iter().filter(|x| x.is_some()).count() as u32;
                let sum_some: f64 = row.iter().filter_map(|x| *x).sum();
                let sum_ok = (sum_some - ar.sum).abs() < 1e-9;
                println!(
                    "候选 {i}: len={} some={} count={} sum_ordered={:.6} sum_hist={:.6} Δ={:.3e}",
                    row.len(),
                    n_some,
                    ar.count(),
                    sum_some,
                    ar.sum,
                    (sum_some - ar.sum).abs()
                );
                let len_msg = format!("候选 {i} 有序长度 == search_n");
                c.check(row.len() == search_n, &len_msg);
                let n_msg = format!("候选 {i} 非 None 个数 == count");
                c.check(n_some == ar.count(), &n_msg);
                let sum_msg = format!("候选 {i} 非 None 之和与 sum 在 1e-9 内一致");
                c.check(sum_ok, &sum_msg);
            }
        }

        c.check(
            out_off.best_action_idx == out_on.best_action_idx,
            "同种子下开关不改变 best_action_idx"
        );
        for (i, (a, b)) in out_off.action_results.iter().zip(out_on.action_results.iter()).enumerate()
        {
            let mean_msg = format!("候选 {i} mean 逐位相同");
            c.check(a.0.mean() == b.0.mean(), &mean_msg);
        }
        c.finish()
    }

    /// 有序槽位必须**逐序号**对上 CRN 种子，且各候选在同一序号上共享同一种子
    ///
    /// 这是保留顺序的**全部意义**所在：离线 cross-fitting 靠的是「各候选第 k 次
    /// rollout 是同一个随机世界」。只验「非 None 个数 == count」「和一致」不够——
    /// 把整行倒序、或让某个候选整体错开一位，那两条都仍然成立。
    ///
    /// 做法是让 rollout 闭包直接把种子当分数返回（取模保证 `f64` 精确表示），
    /// 于是有序行本身就是该候选实际用过的种子序列，可以逐位对照 `seed_at(k)`。
    #[test]
    fn test_ordered_rollouts_align_with_crn_seeds() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let search_n = 8;
        let cfg = SearchConfig::default()
            .with_search_n(search_n)
            .with_ucb(false)
            .with_record_ordered_rollouts(true);
        let search: FlatSearch<RamenGame> = FlatSearch::new(cfg);

        // 种子原样当分数会超出 f64 的整数精确区间，取模后仍是单射到本次搜索的 8 个种子
        let seed_score = |seed: u64| (seed % 1_000_000) as f64;
        let echo = |_: &RamenGame, _: &RamenAction, seed: u64| -> Result<SearchScore> {
            Ok(SearchScore {
                score: seed_score(seed),
                score_pt: 0.0
            })
        };

        let mut probe = StdRng::seed_from_u64(20260830);
        let root = probe.next_u64();
        let seeds = RolloutSeeds::from_root(root);
        let want: Vec<f64> = (0..search_n).map(|k| seed_score(seeds.seed_at(k))).collect();
        println!("root={root:#018x}");
        println!("期望序列（各候选应逐位相同）: {want:?}");

        let mut rng = StdRng::seed_from_u64(20260830);
        let out = search.search_with(&game, &actions, &mut rng, echo)?;
        let ord = out
            .ordered_rollouts
            .as_ref()
            .ok_or_else(|| anyhow!("开关开时 ordered_rollouts 不应为 None"))?;

        let mut c = Checks::new();
        c.check(ord.root_seed == root, "root_seed 就是本次搜索的 CRN 起点");
        let mut mismatched = 0usize;
        for (i, row) in ord.per_candidate.iter().enumerate() {
            let got: Vec<f64> = row.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            if got != want {
                mismatched += 1;
                println!("  ⚠ 候选 {i} 序列不符: {got:?}");
            }
        }
        println!("{} 个候选，序列不符 {mismatched} 个", ord.per_candidate.len());
        c.check(
            ord.per_candidate.len() == actions.len(),
            "每个候选都有一行有序记录"
        );
        c.check(mismatched == 0, "所有候选的有序槽位逐位等于 seed_at(k)（跨候选完全配对）");
        // 反面：若把某行整体左移一位，上面的逐位比较必须能发现
        let shifted: Vec<f64> = want.iter().skip(1).chain(want.first()).copied().collect();
        c.check(shifted != want, "错开一位的序列与期望不同（说明本测试确有分辨力）");
        c.finish()
    }

    /// 单候选的均匀分配与 UCB 首组、追加组在不同线程数下保持统计、种子序号及失败槽。
    #[test]
    fn test_ordered_rollouts_ucb_keeps_failed_slots() -> Result<()> {
        use rayon::ThreadPoolBuilder;

        let (game, actions) = ramen_root()?;
        let actions = &actions[..1];
        let terminal = RamenTerminal::from_game(&game);
        let search_n = 8;
        let group_size = 4;
        let mut probe = StdRng::seed_from_u64(42);
        let root = probe.next_u64();
        let seeds = RolloutSeeds::from_root(root);
        let fail_indices = [1, search_n - 1];
        let fail_seeds = fail_indices.map(|idx| seeds.seed_at(idx));
        let seed_score = |seed: u64| (seed % 1_000_000) as f64;
        println!("注入失败: root={root:#018x} rollout序号={fail_indices:?}");

        let dummy = |_: &RamenGame, _: &RamenAction, seed: u64| -> Result<RolloutOutcome<RamenTerminal>> {
            if fail_seeds.contains(&seed) {
                bail!("injected failure for seed {seed:#018x}");
            }
            Ok(RolloutOutcome {
                score: SearchScore {
                    score: seed_score(seed),
                    score_pt: seed_score(seed) * 0.37
                },
                terminal
            })
        };

        let mut c = Checks::new();
        for use_ucb in [false, true] {
            let cfg = SearchConfig::default()
                .with_search_n(search_n)
                .with_ucb(use_ucb)
                .with_search_group_size(group_size)
                .with_record_ordered_rollouts(true);
            let search: FlatSearch<RamenGame> = FlatSearch::new(cfg);
            let mut baseline: Option<RamenSearchOutput> = None;
            for threads in [1, 4] {
                let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
                let out = pool.install(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    search.search_with_terminal(&game, actions, &mut rng, dummy)
                })?;
                let ord = out
                    .ordered_rollouts
                    .as_ref()
                    .ok_or_else(|| anyhow!("开关开时 ordered_rollouts 不应为 None"))?;

                c.check(ord.root_seed == root, "root_seed 与本次搜索 CRN 起点一致");
                c.check(ord.per_candidate.len() == actions.len(), "每个候选都有有序记录");
                c.check(
                    ord.per_candidate.iter().all(|row| row.len() == search_n),
                    "单候选执行到计划末尾，覆盖追加组的末位失败"
                );
                for (i, (row, (ar, _))) in ord.per_candidate.iter().zip(out.action_results.iter()).enumerate() {
                    let expected: Vec<Option<f64>> = (0..row.len())
                        .map(|idx| (!fail_indices.contains(&idx)).then(|| seed_score(seeds.seed_at(idx))))
                        .collect();
                    let n_some = row.iter().filter(|x| x.is_some()).count() as u32;
                    println!("ucb={use_ucb} threads={threads} 候选{i}: slots={row:?}, count={}", ar.count());
                    c.check(row == &expected, "失败槽留空，成功槽逐位对应 seed_at(idx)");
                    c.check(n_some == ar.count(), "非空槽数量等于成功样本数");
                    c.check(n_some < row.len() as u32, "失败仍占据计划序号");
                }
                if let Some(base) = &baseline {
                    c.check(out.actions == base.actions && out.best_action_idx == base.best_action_idx, "候选与决策一致");
                    c.check(out.radical_factor.to_bits() == base.radical_factor.to_bits(), "激进度逐位一致");
                    c.check(out.terminal_results == base.terminal_results, "全部终局维度统计一致");
                    let stats = |ar: &ActionResult| {
                        [
                            ar.count() as f64, ar.sum, ar.sum_sq, ar.min(), ar.max(), ar.mean(), ar.stdev(),
                            ar.weighted_mean(out.radical_factor)
                        ].map(f64::to_bits)
                    };
                    for (now, old) in out.action_results.iter().zip(base.action_results.iter()) {
                        c.check(stats(&now.0) == stats(&old.0), "评分轴完整统计逐位一致");
                        c.check(stats(&now.1) == stats(&old.1), "PT轴完整统计逐位一致");
                    }
                } else {
                    baseline = Some(out);
                }
            }
        }
        c.finish()
    }

    /// 候选内部并行：均匀路径在 1 / 4 线程下统计逐位一致、失败槽不压缩，
    /// strict_rollout 报出序号最小的失败；UCB 追加组的失败同样受 strict_rollout 约束。
    #[test]
    fn test_uniform_parallel_threads_and_strict() -> Result<()> {
        use rayon::ThreadPoolBuilder;

        let (game, actions) = ramen_root()?;
        let terminal = RamenTerminal::from_game(&game);
        let search_n = 16;
        let mut probe = StdRng::seed_from_u64(42);
        let root = probe.next_u64();
        let seeds = RolloutSeeds::from_root(root);
        let fail_indices = [1usize, 9, search_n - 1];
        let fail_seeds = fail_indices.map(|idx| seeds.seed_at(idx));
        let seed_score = |seed: u64| (seed % 1_000_000) as f64;
        let dummy = |_: &RamenGame, _: &RamenAction, seed: u64| -> Result<RolloutOutcome<RamenTerminal>> {
            if fail_seeds.contains(&seed) {
                bail!("injected failure for seed {seed:#018x}");
            }
            Ok(RolloutOutcome {
                score: SearchScore { score: seed_score(seed), score_pt: seed_score(seed) * 0.37 },
                terminal
            })
        };
        // 只在 UCB 追加组（序号 ≥ group_size）注入失败
        let append_fail_idx = 5usize;
        let append_fail_seed = seeds.seed_at(append_fail_idx);
        let append_dummy = |_: &RamenGame, _: &RamenAction, seed: u64| -> Result<RolloutOutcome<RamenTerminal>> {
            if seed == append_fail_seed {
                bail!("injected append failure for seed {seed:#018x}");
            }
            Ok(RolloutOutcome {
                score: SearchScore { score: seed_score(seed), score_pt: seed_score(seed) * 0.37 },
                terminal
            })
        };

        let mut c = Checks::new();
        let cfg = SearchConfig::default().with_search_n(search_n).with_ucb(false).with_record_ordered_rollouts(true);
        let ucb_cfg = SearchConfig::default()
            .with_search_n(8)
            .with_ucb(true)
            .with_search_group_size(4)
            .with_record_ordered_rollouts(true);
        let mut baseline: Option<RamenSearchOutput> = None;
        for threads in [1, 4] {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            let search: FlatSearch<RamenGame> = FlatSearch::new(cfg.clone());
            let out = pool.install(|| {
                let mut rng = StdRng::seed_from_u64(42);
                search.search_with_terminal(&game, &actions, &mut rng, dummy)
            })?;
            let ord = out.ordered_rollouts.as_ref().ok_or_else(|| anyhow!("ordered_rollouts 不应为 None"))?;
            for (i, (row, (ar, _))) in ord.per_candidate.iter().zip(out.action_results.iter()).enumerate() {
                let expected: Vec<Option<f64>> = (0..search_n)
                    .map(|idx| (!fail_indices.contains(&idx)).then(|| seed_score(seeds.seed_at(idx))))
                    .collect();
                println!("threads={threads} 候选{i}: count={} len={}", ar.count(), row.len());
                c.check(row == &expected, "失败槽留空，成功槽逐位对应 seed_at(idx)");
                c.check(ar.count() as usize == search_n - fail_indices.len(), "成功样本数 = n - 失败数");
            }
            if let Some(base) = &baseline {
                let stats = |ar: &ActionResult| {
                    [ar.count() as f64, ar.sum, ar.sum_sq, ar.min(), ar.max(), ar.mean(), ar.stdev(),
                        ar.weighted_mean(out.radical_factor)]
                    .map(f64::to_bits)
                };
                c.check(out.best_action_idx == base.best_action_idx, "最优下标一致");
                c.check(out.terminal_results == base.terminal_results, "终局统计一致");
                for (now, old) in out.action_results.iter().zip(base.action_results.iter()) {
                    c.check(stats(&now.0) == stats(&old.0), "评分轴统计逐位一致");
                    c.check(stats(&now.1) == stats(&old.1), "PT 轴统计逐位一致");
                }
            } else {
                baseline = Some(out);
            }

            let strict: FlatSearch<RamenGame> = FlatSearch::new(cfg.clone()).with_strict_rollout(true);
            let err = pool.install(|| {
                let mut rng = StdRng::seed_from_u64(42);
                strict.search_with_terminal(&game, &actions, &mut rng, dummy)
            });
            match err {
                Ok(_) => c.check(false, "strict_rollout 遇到失败必须报错"),
                Err(e) => {
                    let msg = format!("{e:#}");
                    println!("threads={threads} strict 错误: {msg}");
                    c.check(msg.contains("rollout 1 失败（strict_rollout 已开启）"), "报出序号最小的失败（rollout 1）");
                }
            }

            // UCB 追加组：非 strict 时失败只记槽位，strict 时报错
            let loose: FlatSearch<RamenGame> = FlatSearch::new(ucb_cfg.clone());
            let loose_out = pool.install(|| {
                let mut rng = StdRng::seed_from_u64(42);
                loose.search_with_terminal(&game, &actions[..1], &mut rng, append_dummy)
            })?;
            let loose_row = loose_out
                .ordered_rollouts
                .as_ref()
                .and_then(|o| o.per_candidate.first().cloned())
                .unwrap_or_default();
            println!("threads={threads} UCB 非 strict 追加组: slots={loose_row:?}");
            c.check(
                loose_row.len() == 8 && loose_row[append_fail_idx].is_none(),
                "非 strict：追加组失败保留空槽"
            );
            let ucb_strict: FlatSearch<RamenGame> = FlatSearch::new(ucb_cfg.clone()).with_strict_rollout(true);
            let ucb_err = pool.install(|| {
                let mut rng = StdRng::seed_from_u64(42);
                ucb_strict.search_with_terminal(&game, &actions[..1], &mut rng, append_dummy)
            });
            match ucb_err {
                Ok(_) => c.check(false, "strict_rollout 下 UCB 追加组失败必须报错"),
                Err(e) => {
                    let msg = format!("{e:#}");
                    println!("threads={threads} UCB strict 错误: {msg}");
                    c.check(msg.contains("rollout 5 失败（strict_rollout 已开启）"), "UCB 追加组报出 rollout 5");
                }
            }
        }
        c.finish()
    }

    /// CRN 收益实测（拉面 / 目标剧本，大样本）
    ///
    /// A/B 轴是「候选间是否共享 `rule_master`」，不是 `crn_stage_reseed`
    /// （该开关只在温泉 `reseed_for_stage` 路径生效，拉面 `simulate_common` 不读它）。
    ///
    /// `cargo test --release -p umasim --lib test_crn_pairing_gain_ramen -- --ignored --nocapture`
    #[test]
    #[ignore = "CRN 收益测量，耗时较长，按需手动运行"]
    fn test_crn_pairing_gain_ramen() -> Result<()> {
        const ROLLOUTS: usize = 200;
        let (game, actions) = ramen_root()?;
        println!(
            "拉面根局面: 回合 {} 阶段 {:?}，候选 {} 个\n",
            game.turn(),
            game.stage,
            actions.len()
        );

        let indep = measure_ramen_crn_arm(false, ROLLOUTS)?;
        let shared = measure_ramen_crn_arm(true, ROLLOUTS)?;
        let g_indep = print_crn_arm(&indep);
        let g_shared = print_crn_arm(&shared);

        let mut c = Checks::new();
        c.check(indep.failed == 0, "独立臂失败次数为 0");
        c.check(shared.failed == 0, "共享臂失败次数为 0");
        c.check(
            (0.8..=1.2).contains(&g_indep),
            "独立臂平均增益 ≈ 1（落在 0.8..=1.2）"
        );
        c.check(g_shared > g_indep, "共享臂增益严格大于独立臂");
        let floor_msg = format!(
            "共享臂增益 {g_shared:.3} > 下限 {RAMEN_SHARED_CRN_GAIN_FLOOR}"
        );
        c.check(g_shared > RAMEN_SHARED_CRN_GAIN_FLOOR, &floor_msg);
        c.finish()
    }

    /// CRN 收益实测：配对相关系数与等效样本倍率
    ///
    /// 对同一根局面，各候选在 rollout 序号 j 上共享种子，逐 j 收集分数，
    /// 再按候选两两计算：
    ///
    /// - `corr`：配对相关系数。越高说明「同一份未来随机性」共享得越充分。
    /// - `倍率`：`(Var_a + Var_b) / Var(X_a - X_b)`。独立抽样时分母等于分子，
    ///   倍率为 1；配对生效则分母变小、倍率 > 1，等价于把 `search_n` 放大同样倍数。
    ///
    /// 关闭 / 开启 `crn_stage_reseed` 各测一次，差值即按阶段重播种的真实收益。
    /// 计划中「等效 4–10 倍」为无实测支撑的预期值，以本测试输出为准。
    ///
    /// 耗时较长（每配置 候选数 × ROLLOUTS 次整局 rollout），故标记 ignore：
    /// `cargo test --release -p umasim --lib test_crn_pairing_gain -- --ignored --nocapture`
    #[test]
    #[ignore = "CRN 收益测量，耗时较长，按需手动运行"]
    fn test_crn_pairing_gain() -> Result<()> {
        /// 每候选 rollout 次数
        const ROLLOUTS: usize = 200;

        let (game, actions) = root_state()?;
        println!(
            "根局面: 回合 {} 阶段 {:?}，候选 {} 个
",
            game.turn,
            game.stage,
            actions.len()
        );

        for reseed in [false, true] {
            let cfg = SearchConfig::default().with_crn_stage_reseed(reseed);
            let search = FlatSearch::new(cfg);
            let seeds = RolloutSeeds::from_root(20260822);

            // scores[候选][rollout 序号]
            let mut scores: Vec<Vec<f64>> = Vec::with_capacity(actions.len());
            for (i, action) in actions.iter().enumerate() {
                let col: Vec<(usize, Result<(f64, f64)>)> = (0..ROLLOUTS)
                    .into_par_iter()
                    .map(|j| (j, search.simulate(&game, action, seeds.seed_at(j))))
                    .collect();
                let mut ok_scores = Vec::with_capacity(col.len());
                let mut first_err: Option<String> = None;
                for (_, r) in col {
                    match r {
                        Ok(v) => ok_scores.push(v.0),
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e.to_string());
                            }
                        }
                    }
                }
                // 失败会让配对错位（不同候选丢掉的 j 不同），必须先确认没有失败
                println!(
                    "  候选 {i}: 成功 {}/{ROLLOUTS}{}",
                    ok_scores.len(),
                    first_err.map(|e| format!("，首个失败: {e}")).unwrap_or_default()
                );
                scores.push(ok_scores);
            }

            let label = if reseed {
                "开启按阶段重播种"
            } else {
                "仅共享起始种子"
            };
            println!("===== {label} =====");

            let mut corrs = Vec::new();
            let mut gains = Vec::new();
            for a in 0..scores.len() {
                for b in (a + 1)..scores.len() {
                    let (xa, xb) = (&scores[a], &scores[b]);
                    let n = xa.len().min(xb.len());
                    if n < 2 {
                        continue;
                    }
                    let diff: Vec<f64> = (0..n).map(|j| xa[j] - xb[j]).collect();
                    let indep = var_of(&xa[..n]) + var_of(&xb[..n]);
                    let paired = var_of(&diff);
                    let gain = if paired > 0.0 { indep / paired } else { f64::NAN };
                    corrs.push(corr_of(&xa[..n], &xb[..n]));
                    gains.push(gain);
                }
            }
            println!(
                "候选对数 {} | 平均 corr = {:.4} | 平均等效倍率 = {:.2}x | 倍率区间 [{:.2}, {:.2}]",
                corrs.len(),
                mean_of(&corrs),
                mean_of(&gains),
                gains.iter().cloned().fold(f64::INFINITY, f64::min),
                gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
            println!();
        }
        Ok(())
    }

    // ========== 批量后端结构契约（无需模型）验收 ==========

    /// 模拟批量后端返回的完整表与三种结构错误。
    #[derive(Debug, Clone, Copy)]
    enum MockBatchMode {
        /// 返回全部要求的候选与种子。
        Complete,
        /// 遗漏一条结果，但同一候选仍保留其他成功结果。
        Missing,
        /// 额外返回一个不存在的候选下标。
        Extra,
        /// 总条目数正确，但把一条结果换成错误种子。
        WrongSeed
    }

    /// 不运行模拟与推理的批量后端，记录调用次数以验证前置守卫。
    struct MockBatchBackend {
        /// 要构造的结果表形态。
        mode: MockBatchMode,
        /// 实际进入预计算的次数。
        calls: AtomicUsize
    }

    impl MockBatchBackend {
        /// 构造尚未调用的模拟后端。
        fn new(mode: MockBatchMode) -> Self {
            Self { mode, calls: AtomicUsize::new(0) }
        }
    }

    /// 根据候选与 rollout 序号生成可直接逐字段比较的固定结果。
    fn mock_batch_outcome(game: &RamenGame, candidate: usize, j: usize) -> RolloutOutcome<RamenTerminal> {
        RolloutOutcome {
            score: SearchScore {
                score: 60_000.0 + candidate as f64 * 100.0 + j as f64,
                score_pt: 61_000.0 + candidate as f64 * 50.0 + j as f64 * 2.0
            },
            terminal: RamenTerminal::from_game(game)
        }
    }

    impl RamenBatchRollout for MockBatchBackend {
        /// 按测试模式返回固定结果；不加载模型、不运行游戏续跑。
        fn precompute(
            &self, game: &RamenGame, actions: &[RamenAction], seeds: &RolloutSeeds, n: usize
        ) -> Result<RamenBatchTable> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut table = RamenBatchTable::new();
            for candidate in 0..actions.len() {
                for j in 0..n {
                    if candidate == 0 && j == 0 && matches!(self.mode, MockBatchMode::Missing) {
                        continue;
                    }
                    let seed = if candidate == 0 && j == 0 && matches!(self.mode, MockBatchMode::WrongSeed) {
                        seeds.seed_at(n)
                    } else {
                        seeds.seed_at(j)
                    };
                    table.insert((candidate, seed), mock_batch_outcome(game, candidate, j));
                }
            }
            if matches!(self.mode, MockBatchMode::Extra) {
                table.insert((actions.len(), seeds.seed_at(0)), mock_batch_outcome(game, 0, 0));
            }
            Ok(table)
        }
    }

    /// 比较搜索结果的实际字段、两条评分轴统计与有序分数，不生成摘要或哈希。
    fn check_batch_search_results(c: &mut Checks, actual: &RamenSearchOutput, expected: &RamenSearchOutput) {
        c.check(actual.actions == expected.actions, "候选及顺序相同");
        c.check(actual.best_action_idx == expected.best_action_idx, "中选下标相同");
        c.check(actual.radical_factor.to_bits() == expected.radical_factor.to_bits(), "激进度逐位相同");
        c.check(actual.terminal_results == expected.terminal_results, "终局统计相同");
        c.check(actual.action_results.len() == expected.action_results.len(), "候选统计条目数相同");
        for ((a, a_pt), (b, b_pt)) in actual.action_results.iter().zip(&expected.action_results) {
            for (left, right) in [(a, b), (a_pt, b_pt)] {
                c.check(left.count() == right.count(), "评分轴样本数相同");
                let left_values = [
                    left.sum, left.sum_sq, left.min(), left.max(), left.mean(), left.stdev(),
                    left.weighted_mean(actual.radical_factor)
                ];
                let right_values = [
                    right.sum, right.sum_sq, right.min(), right.max(), right.mean(), right.stdev(),
                    right.weighted_mean(expected.radical_factor)
                ];
                c.check(left_values.map(f64::to_bits) == right_values.map(f64::to_bits), "评分轴统计逐位相同");
            }
        }
        match (&actual.ordered_rollouts, &expected.ordered_rollouts) {
            (Some(a), Some(b)) => {
                c.check(a.root_seed == b.root_seed, "有序记录根种子相同");
                c.check(a.per_candidate == b.per_candidate, "每候选有序分数相同");
            }
            (None, None) => {}
            _ => c.check(false, "有序记录的启用状态相同")
        }
    }

    /// UCB 必须在预计算之前拒绝，不得算完整根之后才发现消费次数不符。
    #[test]
    fn test_batch_rollout_rejects_ucb_before_precompute() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let backend = Arc::new(MockBatchBackend::new(MockBatchMode::Complete));
        let config = SearchConfig::default().with_search_n(4).with_ucb(true).with_search_group_size(2);
        let search: FlatSearch<RamenGame> = FlatSearch::new(config).with_batch_rollout(backend.clone());
        let mut rng = StdRng::seed_from_u64(42);
        let mut untouched = rng.clone();
        let error = search.search(&game, &actions, &mut rng).err().ok_or_else(|| anyhow!("批量 UCB 应被拒绝"))?;
        let message = format!("{error:#}");
        println!("批量 UCB 前置检查: {message}");
        let mut c = Checks::new();
        c.check(message.contains("use_ucb 必须为 false"), "错误明确指出不支持 UCB");
        c.check(backend.calls.load(Ordering::Relaxed) == 0, "UCB 被拒绝时后端未调用");
        c.check(rng.next_u64() == untouched.next_u64(), "前置拒绝不推进外层 RNG");
        c.finish()
    }

    /// 重复候选不得通过按动作还原下标的入口，避免结果全部归到第一项。
    #[test]
    fn test_batch_rollout_rejects_duplicate_candidates() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let first = actions.first().ok_or_else(|| anyhow!("测试根缺少候选"))?;
        let duplicated = [first.clone(), first.clone()];
        let backend = Arc::new(MockBatchBackend::new(MockBatchMode::Complete));
        let config = SearchConfig::default().with_search_n(4).with_ucb(false);
        let search: FlatSearch<RamenGame> = FlatSearch::new(config).with_batch_rollout(backend.clone());
        let mut rng = StdRng::seed_from_u64(42);
        let error = search.search(&game, &duplicated, &mut rng).err().ok_or_else(|| anyhow!("重复候选应被拒绝"))?;
        println!("重复候选前置检查: {error:#}");
        let mut c = Checks::new();
        c.check(error.to_string().contains("候选重复"), "错误明确指出重复候选");
        c.check(backend.calls.load(Ordering::Relaxed) == 0, "重复候选被拒绝时后端未调用");
        c.finish()
    }

    /// 缺项、额外项与等长错键在 strict 关闭时也必须让整根失败。
    #[test]
    fn test_batch_rollout_rejects_malformed_table_without_strict() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let mut c = Checks::new();
        for (mode, expected_error) in [
            (MockBatchMode::Missing, "条目数"),
            (MockBatchMode::Extra, "条目数"),
            (MockBatchMode::WrongSeed, "批量结果缺")
        ] {
            let backend = Arc::new(MockBatchBackend::new(mode));
            let config = SearchConfig::default().with_search_n(4).with_ucb(false);
            let search: FlatSearch<RamenGame> = FlatSearch::new(config).with_batch_rollout(backend.clone());
            let mut rng = StdRng::seed_from_u64(42);
            let mut untouched = rng.clone();
            c.check(!search.strict_rollout, "结构检查测试确实关闭 strict");
            let result = search.search(&game, &actions, &mut rng);
            match result {
                Ok(_) => c.check(false, "畸形结果表不得返回部分成功的搜索结果"),
                Err(error) => {
                    let message = format!("{error:#}");
                    println!("mode={mode:?}: {message}");
                    c.check(message.contains(expected_error), "结构错误在汇总前被识别");
                }
            }
            c.check(backend.calls.load(Ordering::Relaxed) == 1, "结果表来自一次后端调用");
            c.check(rng.next_u64() == untouched.next_u64(), "结构拒绝不推进外层 RNG");
        }
        c.finish()
    }

    /// 完整均匀结果表与直接调用原内核逐字段一致，strict 不改变成功路径。
    #[test]
    fn test_batch_rollout_uniform_matches_direct_kernel() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let n = 4;
        let config = SearchConfig::default().with_search_n(n).with_ucb(false).with_record_ordered_rollouts(true);
        let direct: FlatSearch<RamenGame> = FlatSearch::new(config.clone());
        let mut rng = StdRng::seed_from_u64(42);
        let seeds = RolloutSeeds::from_rng(&mut rng.clone());
        let expected = direct.search_with_terminal(&game, &actions, &mut rng, |g, action, seed| {
            let candidate = actions.iter().position(|a| a == action).ok_or_else(|| anyhow!("直接内核候选不存在"))?;
            let j = (0..n).find(|j| seeds.seed_at(*j) == seed).ok_or_else(|| anyhow!("直接内核种子不存在"))?;
            Ok(mock_batch_outcome(g, candidate, j))
        })?;
        let next = rng.next_u64();
        let mut c = Checks::new();
        for strict in [false, true] {
            let backend = Arc::new(MockBatchBackend::new(MockBatchMode::Complete));
            let search: FlatSearch<RamenGame> = FlatSearch::new(config.clone())
                .with_batch_rollout(backend.clone())
                .with_strict_rollout(strict);
            let mut rng = StdRng::seed_from_u64(42);
            let actual = search.search(&game, &actions, &mut rng)?;
            println!("完整批量表 strict={strict} 候选数={} 中选={}", actual.actions.len(), actual.best_action_idx);
            check_batch_search_results(&mut c, &actual, &expected);
            c.check(backend.calls.load(Ordering::Relaxed) == 1, "完整根仅预计算一次");
            c.check(rng.next_u64() == next, "批量与直接内核消耗相同外层 RNG");
        }
        c.finish()
    }

    /// 未装批量后端时仍走原续跑闭包，均匀与默认 UCB 都不受批量守卫影响。
    #[test]
    fn test_no_batch_backend_keeps_default_search_path() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let mut c = Checks::new();
        for use_ucb in [false, true] {
            let config = SearchConfig::default()
                .with_search_n(2)
                .with_search_group_size(1)
                .with_ucb(use_ucb)
                .with_record_ordered_rollouts(true);
            let search: FlatSearch<RamenGame> = FlatSearch::new(config);
            c.check(search.batch_rollout.is_none(), "默认未装载批量后端");
            let mut rng = StdRng::seed_from_u64(42);
            let actual = search.search(&game, &actions, &mut rng)?;
            let next = rng.next_u64();
            let mut direct_rng = StdRng::seed_from_u64(42);
            let expected = search.search_with_terminal(&game, &actions, &mut direct_rng, |g, action, seed| {
                search.simulate_common_extract(g, action, seed, seed, RamenTerminal::from_game)
            })?;
            println!("无批量后端 ucb={use_ucb} 中选={}", actual.best_action_idx);
            check_batch_search_results(&mut c, &actual, &expected);
            c.check(next == direct_rng.next_u64(), "默认入口与原续跑闭包的外层 RNG 相同");
        }
        c.finish()
    }

    // ========== 成本探针（仅测量用）验收 ==========

    /// 探针开 / 关必须给出**逐位相同**的搜索结果与 RNG 位置
    ///
    /// 探针一旦影响候选分配、CRN 种子或排序，用它量出来的成本就不是生产成本。
    /// 这里逐字段比较（不算哈希）：中选下标、激进度、每候选的 n / sum / sum_sq /
    /// min-max 派生量，以及搜索后外层 rng 的下一个输出。
    #[test]
    fn test_probe_neutral_to_search_result() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let mut c = Checks::new();
        for use_ucb in [false, true] {
            let cfg = SearchConfig::default()
                .with_search_n(8)
                .with_ucb(use_ucb)
                .with_search_group_size(4);
            let plain: FlatSearch<RamenGame> = FlatSearch::new(cfg.clone());
            let sink: Arc<Mutex<Vec<SearchProbe>>> = Arc::new(Mutex::new(Vec::new()));
            let probed: FlatSearch<RamenGame> = FlatSearch::new(cfg).with_probe(sink.clone());

            let mut rng_a = StdRng::seed_from_u64(20260913);
            let out_a = plain.search(&game, &actions, &mut rng_a)?;
            let after_a = rng_a.next_u64();

            let mut rng_b = StdRng::seed_from_u64(20260913);
            let out_b = probed.search(&game, &actions, &mut rng_b)?;
            let after_b = rng_b.next_u64();

            println!(
                "ucb={use_ucb} 中选: 关={} 开={} | radical: 关={:.9} 开={:.9} | rng 后继: 关={after_a:#018x} 开={after_b:#018x}",
                out_a.best_action_idx, out_b.best_action_idx, out_a.radical_factor, out_b.radical_factor
            );
            c.check(out_a.best_action_idx == out_b.best_action_idx, "中选下标一致");
            c.check(out_a.radical_factor == out_b.radical_factor, "激进度一致");
            c.check(after_a == after_b, "搜索后外层 rng 位置一致");
            c.check(
                out_a.action_results.len() == out_b.action_results.len(),
                "候选数一致"
            );
            for (i, ((a, a_pt), (b, b_pt))) in out_a
                .action_results
                .iter()
                .zip(out_b.action_results.iter())
                .enumerate()
            {
                println!(
                    "  候选{i}: score 关 n={} sum={:.6} sum_sq={:.6} | 开 n={} sum={:.6} sum_sq={:.6}",
                    a.count(), a.sum, a.sum_sq, b.count(), b.sum, b.sum_sq
                );
                println!(
                    "          score_pt 关 n={} sum={:.6} sum_sq={:.6} | 开 n={} sum={:.6} sum_sq={:.6}",
                    a_pt.count(), a_pt.sum, a_pt.sum_sq, b_pt.count(), b_pt.sum, b_pt.sum_sq
                );
                c.check(a.count() == b.count(), "score 样本数一致");
                c.check(a.sum == b.sum, "score 分数和一致");
                c.check(a.sum_sq == b.sum_sq, "score 平方和一致");
                c.check(a_pt.count() == b_pt.count(), "score_pt 样本数一致");
                c.check(a_pt.sum == b_pt.sum, "score_pt 分数和一致");
                c.check(a_pt.sum_sq == b_pt.sum_sq, "score_pt 平方和一致");
            }

            let probes = sink.lock().map_err(|e| anyhow!("探针锁中毒: {e}"))?;
            c.check(probes.len() == 1, "一次搜索恰好留下一条探针记录");
        }
        c.finish()
    }

    /// 探针计数必须与注入的失败逐条对齐，且 `planned = 成功 + 失败`
    ///
    /// 失败 rollout 只增加计划次数、不增加成功次数；探针若把两者混同，
    /// 「预算花在哪」就会被系统性高估或低估。
    #[test]
    fn test_probe_counts_match_injected_failures() -> Result<()> {
        let (game, actions) = ramen_root()?;
        let actions = &actions[..2];
        let terminal = RamenTerminal::from_game(&game);
        let search_n = 8;
        let group_size = 4;
        let mut root_rng = StdRng::seed_from_u64(4242);
        let seeds = RolloutSeeds::from_root(root_rng.next_u64());
        let fail_indices = [1usize, 5];
        let fail_seeds = fail_indices.map(|idx| seeds.seed_at(idx));
        let dummy = |_: &RamenGame, _: &RamenAction, seed: u64| -> Result<RolloutOutcome<RamenTerminal>> {
            if fail_seeds.contains(&seed) {
                bail!("injected failure for seed {seed:#018x}");
            }
            Ok(RolloutOutcome {
                score: SearchScore {
                    score: (seed % 1_000_000) as f64,
                    score_pt: (seed % 1_000_000) as f64 * 0.37
                },
                terminal
            })
        };

        let mut c = Checks::new();
        for use_ucb in [false, true] {
            let cfg = SearchConfig::default()
                .with_search_n(search_n)
                .with_ucb(use_ucb)
                .with_search_group_size(group_size);
            let sink: Arc<Mutex<Vec<SearchProbe>>> = Arc::new(Mutex::new(Vec::new()));
            let search: FlatSearch<RamenGame> = FlatSearch::new(cfg).with_probe(sink.clone());
            let mut rng = StdRng::seed_from_u64(4242);
            let out = search.search_with_terminal(&game, actions, &mut rng, dummy)?;

            let probes = sink.lock().map_err(|e| anyhow!("探针锁中毒: {e}"))?;
            let p = probes.first().ok_or_else(|| anyhow!("探针没有留下记录"))?;
            println!(
                "ucb={use_ucb} 候选={} 计划={:?} 成功={:?} 失败={:?} 总续跑={}",
                p.candidates, p.planned, p.succeeded, p.failed, p.total_planned()
            );
            c.check(p.candidates == actions.len(), "探针候选数与入参一致");
            c.check(p.planned.len() == actions.len(), "planned 与候选同长");
            for i in 0..actions.len() {
                c.check(
                    p.planned[i] == p.succeeded[i] + p.failed[i],
                    "planned = 成功 + 失败"
                );
                c.check(
                    p.succeeded[i] as u32 == out.action_results[i].0.count(),
                    "探针成功数等于 ActionResult 样本数"
                );
            }
            // 预算语义：均匀分配下每候选都恰好 `search_n`；UCB 下只有**某一个**
            // 候选计划次数达到 `search_n` 即停，其余停在更低处——`search_n` 是
            // 停止阈值，不是「每候选都会跑满」。
            let max_planned = p.planned.iter().copied().max().unwrap_or(0);
            if use_ucb {
                c.check(max_planned >= search_n, "UCB: 最大计划次数达到停止阈值 search_n");
                c.check(
                    p.planned.iter().any(|n| *n < search_n),
                    "UCB: 存在未跑满 search_n 的候选（预算是动态分配的）"
                );
            } else {
                c.check(
                    p.planned.iter().all(|n| *n == search_n),
                    "均匀分配: 每候选恰好 search_n"
                );
            }
            c.check(
                p.total_failed() > 0,
                "注入的失败被记到 failed 而不是被静默丢弃"
            );
            c.check(p.total_planned() == p.total_succeeded() + p.total_failed(), "总数自洽");
        }
        c.finish()
    }
}
