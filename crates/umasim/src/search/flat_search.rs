//! 扁平蒙特卡洛搜索
//!
//! 对每个合法动作执行多次模拟，统计分数分布，选择最优动作。
//! 支持两种搜索策略：
//! - 均匀分配：每个动作平均分配搜索次数（并行化）
//! - UCB 分配：根据 UCB 公式动态分配搜索资源（C++ UmaAi 风格）

use anyhow::Result;
use log::debug;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use super::{
    config::{SearchConfig, TOTAL_TURN},
    result::{ActionResult, SearchOutput}
};
use crate::{
    game::{
        Game,
        onsen::{action::OnsenAction, game::OnsenGame}
    },
    neural::{
        Evaluator,
        HandwrittenEvaluator,
        ThreadLocalNeuralNetLeafEvaluator,
        ThreadLocalNeuralNetLeafStatsSnapshot,
        ValueOutput,
    }
};

#[derive(Clone)]
enum LeafEvaluator {
    Handwritten,
    NeuralNet(ThreadLocalNeuralNetLeafEvaluator),
}

impl LeafEvaluator {
    fn name(&self) -> &'static str {
        match self {
            LeafEvaluator::Handwritten => "handwritten",
            LeafEvaluator::NeuralNet(_) => "nn",
        }
    }

    fn evaluate(&self, rollout_evaluator: &HandwrittenEvaluator, game: &OnsenGame) -> ValueOutput {
        match self {
            LeafEvaluator::Handwritten => rollout_evaluator.evaluate(game),
            LeafEvaluator::NeuralNet(nn) => nn.evaluate(game),
        }
    }
}

/// 扁平蒙特卡洛搜索
///
/// 使用手写逻辑进行模拟，统计各动作的分数分布。
#[derive(Clone)]
pub struct FlatSearch {
    /// 手写评估器（用于模拟）
    rollout_evaluator: HandwrittenEvaluator,

    /// leaf eval 评估器（用于 max_depth>0 截断估值）
    leaf_evaluator: LeafEvaluator,

    /// 搜索配置
    config: SearchConfig,

    /// E4：leaf eval 微批大小（仅在 max_depth>0 && leaf_eval=nn 时生效）
    rollout_batch_size: usize,
}

impl FlatSearch {
    /// 创建搜索器
    pub fn new(config: SearchConfig) -> Self {
        Self {
            rollout_evaluator: HandwrittenEvaluator::new(),
            leaf_evaluator: LeafEvaluator::Handwritten,
            config,
            rollout_batch_size: 1,
        }
    }

    /// 创建默认搜索器
    pub fn default_search() -> Self {
        Self::new(SearchConfig::default())
    }

    /// 设置 leaf eval 为神经网络（用于 max_depth>0 截断估值）
    pub fn with_leaf_evaluator_nn(mut self, model_path: impl Into<String>) -> Self {
        self.leaf_evaluator = LeafEvaluator::NeuralNet(ThreadLocalNeuralNetLeafEvaluator::new(model_path));
        self
    }

    /// 强制 leaf eval 回退为 handwritten（默认）
    pub fn with_leaf_evaluator_handwritten(mut self) -> Self {
        self.leaf_evaluator = LeafEvaluator::Handwritten;
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
    pub fn leaf_nn_stats(&self) -> Option<ThreadLocalNeuralNetLeafStatsSnapshot> {
        match &self.leaf_evaluator {
            LeafEvaluator::NeuralNet(nn) => Some(nn.stats()),
            _ => None,
        }
    }

    fn use_parallel_simulation(&self) -> bool {
        // E4.3：leaf eval 使用 thread_local 模型后，可安全恢复 Rayon 并行
        true
    }

    fn leaf_nn(&self) -> Option<&ThreadLocalNeuralNetLeafEvaluator> {
        match &self.leaf_evaluator {
            LeafEvaluator::NeuralNet(nn) => Some(nn),
            _ => None,
        }
    }

    /// 执行搜索
    ///
    /// 根据配置选择搜索策略：
    /// - use_ucb = true: UCB 动态分配
    /// - use_ucb = false: 均匀分配（并行化）
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// 搜索输出，包含各动作的分数分布和最优动作
    pub fn search(&self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng) -> Result<SearchOutput> {
        if actions.is_empty() {
            anyhow::bail!("没有可用动作");
        }

        // 计算激进度因子（C++ 风格，无随机性）
        let radical_factor = self.compute_radical_factor(game.turn as usize);

        debug!(
            "[回合 {}] 开始搜索: {} 个动作, search_n={}, max_depth={}, leaf_eval={}, radical_factor={:.1}, ucb={}",
            game.turn,
            actions.len(),
            self.config.search_n,
            self.config.max_depth,
            self.leaf_evaluator.name(),
            radical_factor,
            self.config.use_ucb
        );

        // 根据配置选择搜索策略
        let action_results = if self.config.use_ucb {
            self.search_ucb(game, &actions, radical_factor, rng)?
        } else {
            self.search_uniform(game, &actions)?
        };

        Ok(SearchOutput::new(actions.to_vec(), action_results, radical_factor))
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

    /// 均匀分配搜索（并行化）
    ///
    /// 每个动作平均分配 search_n 次搜索，使用 Rayon 并行化。
    fn search_uniform(&self, game: &OnsenGame, actions: &[OnsenAction]) -> Result<Vec<(ActionResult, ActionResult)>> {
        let use_parallel = self.use_parallel_simulation();
        if use_parallel {
            let ret = actions
                .par_iter()
                .map(|action| {
                    let mut result = ActionResult::new();
                    let mut result_pt = ActionResult::new();
                    // 每个线程初始化一次 RNG
                    let mut thread_rng = StdRng::from_os_rng();
                    let _ = self.simulate_many(game, action, self.config.search_n, &mut thread_rng, &mut result, &mut result_pt);
                    (result, result_pt)
                })
                .collect();
            Ok(ret)
        } else {
            let ret = actions
                .iter()
                .map(|action| {
                    let mut result = ActionResult::new();
                    let mut result_pt = ActionResult::new();
                    let mut thread_rng = StdRng::from_os_rng();
                    let _ = self.simulate_many(game, action, self.config.search_n, &mut thread_rng, &mut result, &mut result_pt);
                    (result, result_pt)
                })
                .collect();
            Ok(ret)
        }
    }

    /// UCB 动态分配搜索
    ///
    /// 使用 UCB 公式动态分配搜索资源，好的动作获得更多搜索次数。
    /// UCB 决策是串行的，但每组模拟内部使用 Rayon 并行化。
    ///
    /// # UCB 公式
    /// search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn search_ucb(
        &self, game: &OnsenGame, actions: &[OnsenAction], radical_factor: f64, _rng: &mut StdRng
    ) -> Result<Vec<(ActionResult, ActionResult)>> {
        let num_actions = actions.len();
        let mut action_results: Vec<(ActionResult, ActionResult)> = vec![Default::default(); num_actions];
        let group_size = self.config.search_group_size;
        let use_parallel = self.use_parallel_simulation();

        // 第一阶段：每个动作先搜一组（并行）
        let initial_results: Vec<_> = if use_parallel {
            actions
                .par_iter()
                .map(|action| {
                    let mut result = ActionResult::new();
                    let mut result_pt = ActionResult::new();
                    let mut thread_rng = StdRng::from_os_rng();
                    let _ = self.simulate_many(game, action, group_size, &mut thread_rng, &mut result, &mut result_pt);
                    (result, result_pt)
                })
                .collect()
        } else {
            actions
                .iter()
                .map(|action| {
                    let mut result = ActionResult::new();
                    let mut result_pt = ActionResult::new();
                    let mut thread_rng = StdRng::from_os_rng();
                    let _ = self.simulate_many(game, action, group_size, &mut thread_rng, &mut result, &mut result_pt);
                    (result, result_pt)
                })
                .collect()
        };

        // 合并初始结果
        for (i, result) in initial_results.into_iter().enumerate() {
            action_results[i] = result;
        }

        let mut total_n = (group_size * num_actions) as f64;

        // 第二阶段：UCB 动态分配
        loop {
            // 检查是否有动作达到 search_n
            let max_count = action_results.iter().map(|r| r.0.count()).max().unwrap_or(0);
            if max_count >= self.config.search_n as u32 {
                break;
            }

            // 使用 UCB 公式选择下一个要搜索的动作
            let best_action_idx = self.select_ucb_action(&action_results, radical_factor, total_n);

            // 对选中的动作搜索一组（并行）
            let action = &actions[best_action_idx];
            // E4：nn leaf 时，rollout 收集 leaf features -> infer_batch -> 写入结果
            if self.config.max_depth > 0 && self.leaf_nn().is_some() && self.rollout_batch_size > 1 {
                let nn = self.leaf_nn().expect("nn");

                let outcomes: Vec<_> = if use_parallel {
                    (0..group_size)
                        .into_par_iter()
                        // E4.3-7)：每个 worker 只初始化一次 RNG，避免 tight loop 里反复 from_os_rng()
                        .map_init(|| StdRng::from_os_rng(), |rng, _| self.simulate_until_terminal_or_leaf(game, action, rng).ok())
                        .filter_map(|x| x)
                        .collect()
                } else {
                    let mut out = Vec::with_capacity(group_size);
                    let mut thread_rng = StdRng::from_os_rng();
                    for _ in 0..group_size {
                        if let Ok(v) = self.simulate_until_terminal_or_leaf(game, action, &mut thread_rng) {
                            out.push(v);
                        }
                    }
                    out
                };

                let mut leaf_features: Vec<f32> = Vec::new();
                let mut leaf_pt_bias: Vec<f64> = Vec::new();

                for o in outcomes {
                    match o {
                        SimOutcome::Terminal { score, score_pt } => {
                            action_results[best_action_idx].0.add(score);
                            action_results[best_action_idx].1.add(score_pt);
                        }
                        SimOutcome::Leaf { features, pt_bias } => {
                            leaf_features.extend_from_slice(&features);
                            leaf_pt_bias.push(pt_bias);
                        }
                    }
                }

                if !leaf_pt_bias.is_empty() {
                    let leaf_n = leaf_pt_bias.len();
                    match nn.evaluate_features_batch(&leaf_features, leaf_n) {
                        Ok(values) => {
                            for (i, v) in values.into_iter().enumerate() {
                                let score_mean = v.score_mean;
                                action_results[best_action_idx].0.add(score_mean);
                                action_results[best_action_idx].1.add(score_mean + leaf_pt_bias[i]);
                            }
                        }
                        Err(e) => {
                            log::warn!("[NN][leaf] infer_batch 失败，回退逐样本（性能受限）: {e}");
                            for i in 0..leaf_n {
                                let start = i * 1121;
                                let end = start + 1121;
                                if let Ok(v) = nn.evaluate_features_batch(&leaf_features[start..end], 1) {
                                    let score_mean = v[0].score_mean;
                                    action_results[best_action_idx].0.add(score_mean);
                                    action_results[best_action_idx].1.add(score_mean + leaf_pt_bias[i]);
                                }
                            }
                        }
                    }
                }
            } else {
            let scores: Vec<_> = if use_parallel {
                (0..group_size)
                    .into_par_iter()
                        // E4.3-7)：每个 worker 只初始化一次 RNG，避免 tight loop 里反复 from_os_rng()
                        .map_init(|| StdRng::from_os_rng(), |rng, _| self.simulate(game, action, rng).ok())
                        .filter_map(|x| x)
                    .collect()
            } else {
                    // 单线程分支：复用同一个 RNG，避免每次 rollout 都 from_os_rng() 的高开销
                let mut out = Vec::with_capacity(group_size);
                    let mut thread_rng = StdRng::from_os_rng();
                    for _ in 0..group_size {
                    if let Ok(v) = self.simulate(game, action, &mut thread_rng) {
                        out.push(v);
                    }
                }
                out
            };

            for score in scores {
                action_results[best_action_idx].0.add(score.0);
                action_results[best_action_idx].1.add(score.1);
                }
            }

            total_n += group_size as f64;
        }

        Ok(action_results)
    }

    /// 使用 UCB 公式选择下一个要搜索的动作
    ///
    /// UCB 公式: search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn select_ucb_action(
        &self, action_results: &[(ActionResult, ActionResult)], radical_factor: f64, total_n: f64
    ) -> usize {
        let sqrt_total = total_n.sqrt();
        let cpuct = self.config.search_cpuct;
        let expected_stdev = self.config.expected_search_stdev;

        let mut best_idx = 0;
        let mut best_search_value = f64::NEG_INFINITY;

        for (i, result) in action_results.iter().enumerate() {
            let n = result.0.count() as f64;
            if n == 0.0 {
                // 未搜索的动作优先级最高
                return i;
            }

            let value = result.0.weighted_mean(radical_factor);
            // UCB 公式：value 越高或搜索次数越少，search_value 越高
            let search_value = value + cpuct * expected_stdev * sqrt_total / n;

            if search_value > best_search_value {
                best_search_value = search_value;
                best_idx = i;
            }
        }

        best_idx
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
    fn simulate(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> Result<(f64, f64)> {
        if matches!(action, OnsenAction::Dig(_)) {
            self.simulate_onsen_select(game, action, rng)
        } else if matches!(action, OnsenAction::Upgrade(_)) {
            self.simulate_dig_upgrade(game, action, rng)
        } else {
            // 克隆游戏状态
            let mut sim_game = game.clone();
            let trainer_hw = SimulationTrainer { evaluator: &self.rollout_evaluator };

            // 执行初始动作
            sim_game.apply_action(action, rng)?;

            // max_depth==0：保持旧行为，rollout 跑到终局
            if self.config.max_depth == 0 {
                while sim_game.next() {
                    sim_game.run_stage(&trainer_hw, rng)?;
                }
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64,
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
                sim_game.run_stage(&trainer_hw, rng)?;
                if (sim_game.turn - start_turn) >= max_depth {
                    break;
                }
            }

            if finished {
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64,
                ));
            }
            // 有些情况下（例如在达到 max_depth 的同一轮刚好走到终局），可能还未通过 next() 触发 finished。
            // 用 turn>=max_turn 兜底判定终局，并确保 on_simulation_end 被触发，避免漏算最终奖励。
            if sim_game.turn >= sim_game.max_turn() {
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64,
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

    fn simulate_many(
        &self,
        game: &OnsenGame,
        action: &OnsenAction,
        n: usize,
        rng: &mut StdRng,
        result: &mut ActionResult,
        result_pt: &mut ActionResult,
    ) -> Result<()> {
        // 仅 nn leaf + max_depth>0 才走微批；否则保持旧行为
        if self.config.max_depth > 0 && self.leaf_nn().is_some() && self.rollout_batch_size > 1 {
            let nn = self.leaf_nn().expect("nn");
            let mut pending_features: Vec<f32> = Vec::with_capacity(self.rollout_batch_size * 1121);
            let mut pending_pt_bias: Vec<f64> = Vec::with_capacity(self.rollout_batch_size);

            for _ in 0..n {
                match self.simulate_until_terminal_or_leaf(game, action, rng)? {
                    SimOutcome::Terminal { score, score_pt } => {
                        result.add(score);
                        result_pt.add(score_pt);
                    }
                    SimOutcome::Leaf { features, pt_bias } => {
                        pending_features.extend_from_slice(&features);
                        pending_pt_bias.push(pt_bias);
                        if pending_pt_bias.len() >= self.rollout_batch_size {
                            let leaf_n = pending_pt_bias.len();
                            let values = nn.evaluate_features_batch(&pending_features, leaf_n)?;
                            for (i, v) in values.into_iter().enumerate() {
                                let score_mean = v.score_mean;
                                result.add(score_mean);
                                result_pt.add(score_mean + pending_pt_bias[i]);
                            }
                            pending_features.clear();
                            pending_pt_bias.clear();
                        }
                    }
                }
            }

            if !pending_pt_bias.is_empty() {
                let leaf_n = pending_pt_bias.len();
                let values = nn.evaluate_features_batch(&pending_features, leaf_n)?;
                for (i, v) in values.into_iter().enumerate() {
                    let score_mean = v.score_mean;
                    result.add(score_mean);
                    result_pt.add(score_mean + pending_pt_bias[i]);
                }
                pending_features.clear();
                pending_pt_bias.clear();
            }
            Ok(())
        } else {
            for _ in 0..n {
                if let Ok(score) = self.simulate(game, action, rng) {
                    result.add(score.0);
                    result_pt.add(score.1);
                }
            }
            Ok(())
        }
    }

    fn simulate_until_terminal_or_leaf(
        &self,
        game: &OnsenGame,
        action: &OnsenAction,
        rng: &mut StdRng,
    ) -> Result<SimOutcome> {
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
        let trainer_hw = SimulationTrainer { evaluator: &self.rollout_evaluator };

        // 执行初始动作
        sim_game.apply_action(action, rng)?;

        // max_depth==0：保持旧行为，rollout 跑到终局
        if self.config.max_depth == 0 {
            while sim_game.next() {
                sim_game.run_stage(&trainer_hw, rng)?;
            }
            sim_game.on_simulation_end(&trainer_hw, rng)?;
            return Ok(SimOutcome::Terminal {
                score: sim_game.uma().calc_score() as f64,
                score_pt: sim_game.uma().calc_score_with_pt_favor() as f64,
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
            sim_game.run_stage(&trainer_hw, rng)?;
            if (sim_game.turn - start_turn) >= max_depth {
                break;
            }
        }

        if finished || sim_game.turn >= sim_game.max_turn() {
            sim_game.on_simulation_end(&trainer_hw, rng)?;
            return Ok(SimOutcome::Terminal {
                score: sim_game.uma().calc_score() as f64,
                score_pt: sim_game.uma().calc_score_with_pt_favor() as f64,
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
        //let trainer = SimulationTrainer { evaluator: &self.evaluator };
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
        let trainer_hw = SimulationTrainer { evaluator: &self.rollout_evaluator };
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

enum SimOutcome {
    Terminal { score: f64, score_pt: f64 },
    Leaf { features: Vec<f32>, pt_bias: f64 },
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
            Some(action) => actions.iter().position(|a| a == action).unwrap_or(0),
            None => 0
        };

        Ok(idx)
    }

    fn select_choice(
        &self, game: &OnsenGame, choices: &[crate::gamedata::ActionValue], _rng: &mut StdRng
    ) -> Result<usize> {
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
