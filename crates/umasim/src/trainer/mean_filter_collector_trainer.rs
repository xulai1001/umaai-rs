//! Mean-Filter 样本收集训练员（P0/P1）
//!
//! 目标：
//! - 每次动作决策都用 FlatSearch 选择 best action
//! - 仅在采样回合范围内导出样本（SearchOutput::export_sample）
//! - 按样本 scoreMean（value_target[0]）阈值筛选
//! - P1：支持“精确到 target_samples 停止接收样本”的策略（达到目标后不再接收样本）

use std::cell::RefCell;

use anyhow::Result;
use log::debug;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{
    game::{
        Game,
        Trainer,
        onsen::{action::OnsenAction, game::OnsenGame},
    },
    gamedata::{ActionValue, EventData},
    neural::{Evaluator, HandwrittenEvaluator},
    search::FlatSearch,
    trainer::HandwrittenTrainer,
    training_sample::{CHOICE_DIM, POLICY_DIM, TrainingSample},
};

#[derive(Debug, Clone)]
pub struct MeanFilterCollectorStats {
    /// action 样本候选数（只统计在采样回合范围内的 action 决策）
    pub candidates: u64,
    /// 总 accepted 样本数（action + choice），用于精确停止与 resume 对齐
    pub accepted: u64,
    /// action 样本 accepted 数（仅用于统计展示）
    pub action_accepted: u64,
    pub dropped: u64,
    pub dropped_zero_mean: u64,
    pub search_errors: u64,
    pub policy_sum_not_one: u64,
    pub accepted_score_mean_sum: f64,
    pub accepted_score_mean_min: Option<f64>,
    pub accepted_score_mean_max: Option<f64>,

    // ========== choice（P2）==========
    /// decision event 候选数（choices.len()>1 且 random_choice_prob.is_none）
    pub choice_candidates: u64,
    pub choice_accepted: u64,
    pub choice_dropped: u64,
    pub choice_sum_not_one: u64,
    pub choice_policy_not_zero: u64,
    pub choice_skipped_too_many_options: u64,
    pub choice_skipped_chance_event: u64,

    /// per-turn（长度 78；idx = human_turn - 1，其中 human_turn = game.turn + 1）
    pub turn_candidates: Vec<u64>,
    pub turn_accepted: Vec<u64>,
    pub turn_dropped: Vec<u64>,

    /// per-turn choice（长度 78；idx = human_turn - 1）
    pub turn_choice_candidates: Vec<u64>,
    pub turn_choice_accepted: Vec<u64>,
    pub turn_choice_dropped: Vec<u64>,
}

impl MeanFilterCollectorStats {
    pub fn new() -> Self {
        Self {
            candidates: 0,
            accepted: 0,
            action_accepted: 0,
            dropped: 0,
            dropped_zero_mean: 0,
            search_errors: 0,
            policy_sum_not_one: 0,
            accepted_score_mean_sum: 0.0,
            accepted_score_mean_min: None,
            accepted_score_mean_max: None,
            choice_candidates: 0,
            choice_accepted: 0,
            choice_dropped: 0,
            choice_sum_not_one: 0,
            choice_policy_not_zero: 0,
            choice_skipped_too_many_options: 0,
            choice_skipped_chance_event: 0,
            turn_candidates: vec![0; 78],
            turn_accepted: vec![0; 78],
            turn_dropped: vec![0; 78],
            turn_choice_candidates: vec![0; 78],
            turn_choice_accepted: vec![0; 78],
            turn_choice_dropped: vec![0; 78],
        }
    }

    pub fn accepted_score_mean_avg(&self) -> f64 {
        if self.action_accepted == 0 {
            0.0
        } else {
            self.accepted_score_mean_sum / self.action_accepted as f64
        }
    }

    fn record_accepted_score_mean(&mut self, score_mean: f64) {
        self.accepted_score_mean_sum += score_mean;
        self.accepted_score_mean_min = Some(match self.accepted_score_mean_min {
            Some(v) => v.min(score_mean),
            None => score_mean,
        });
        self.accepted_score_mean_max = Some(match self.accepted_score_mean_max {
            Some(v) => v.max(score_mean),
            None => score_mean,
        });
    }
}

impl Default for MeanFilterCollectorStats {
    fn default() -> Self {
        Self::new()
    }
}

/// choice 评估用的轻量统计（rollouts 次数很小，不需要 ActionResult 的 100k 直方图）
#[derive(Debug, Clone, Default)]
struct ChoiceEvalResult {
    num: u32,
    sum: f64,
    sum_sq: f64,
    scores: Vec<f64>,
}

impl ChoiceEvalResult {
    fn add(&mut self, score: f64) {
        self.num += 1;
        self.sum += score;
        self.sum_sq += score * score;
        self.scores.push(score);
    }

    fn mean(&self) -> f64 {
        if self.num == 0 { 0.0 } else { self.sum / self.num as f64 }
    }

    fn stdev(&self) -> f64 {
        if self.num <= 1 {
            return 0.0;
        }
        let n = self.num as f64;
        let variance = (self.sum_sq - self.sum * self.sum / n) / (n - 1.0);
        variance.max(0.0).sqrt()
    }

    fn weighted_mean(&self, radical_factor: f64) -> f64 {
        if self.num == 0 {
            return 0.0;
        }
        if radical_factor.abs() < 1e-6 {
            return self.mean();
        }

        // 对齐 ActionResult::compute_weighted_mean 的 rank_ratio 算法（按分位数加权）
        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len() as f64;
        let n_inv = 1.0 / n;
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for (i, &score) in sorted.iter().enumerate() {
            let rank_ratio = ((i as f64) + 0.5) * n_inv;
            let weight = rank_ratio.powf(radical_factor);
            weighted_sum += weight * score;
            weight_total += weight;
        }

        if weight_total > 0.0 { weighted_sum / weight_total } else { self.mean() }
    }
}

/// Mean-Filter 样本收集训练员（P0）
///
/// - “动作决策”一律用 FlatSearch 的 best action（稳定）
/// - “数据采样”只在指定 turn 范围内进行，并按 scoreMean 阈值过滤
pub struct MeanFilterCollectorTrainer {
    search: FlatSearch,
    evaluator: HandwrittenEvaluator,

    score_mean_threshold: f64,
    drop_zero_mean: bool,

    // ========== Choice（P2）==========
    collect_choice: bool,
    choice_rollouts_per_option: usize,
    choice_policy_delta: f64,
    choice_score_mean_threshold: Option<f64>,
    choice_skip_if_too_many: bool,
    /// choice 样本是否跟随 action 的采样回合范围（turn_min/turn_max/turn_stride）
    choice_follow_action_turn_range: bool,
    /// 当 choice_follow_action_turn_range=true 且当前回合不采样时：是否仍使用 rollout 决策（否则回退 select_choice）
    choice_rollout_on_uncollected_turns: bool,

    /// 样本上限（P1：用于精确停止；P0 可保持为 u64::MAX）
    target_samples: u64,
    /// 达到 target_samples 后是否切换为“快速完成”（不再跑 FlatSearch，直接用 HandwrittenEvaluator 决策）
    fast_after_target: bool,

    // 注意：OnsenGame.turn 为 0-based（0..=77）
    // 采样范围建议按人类 1..=78 配置，因此这里用 human_turn（turn+1）做判断
    turn_min: i32,
    turn_max: i32,
    turn_stride: i32,

    samples: RefCell<Vec<TrainingSample>>,
    stats: RefCell<MeanFilterCollectorStats>,

    verbose: bool,
}

impl MeanFilterCollectorTrainer {
    pub fn new(search: FlatSearch, score_mean_threshold: f64) -> Self {
        Self {
            search,
            evaluator: HandwrittenEvaluator::new(),
            score_mean_threshold,
            drop_zero_mean: true,
            collect_choice: true,
            choice_rollouts_per_option: 8,
            choice_policy_delta: 50.0,
            choice_score_mean_threshold: None,
            choice_skip_if_too_many: true,
            choice_follow_action_turn_range: true,
            choice_rollout_on_uncollected_turns: false,
            target_samples: u64::MAX,
            fast_after_target: true,
            turn_min: 1,
            turn_max: 78,
            turn_stride: 1,
            samples: RefCell::new(Vec::new()),
            stats: RefCell::new(MeanFilterCollectorStats::new()),
            verbose: false,
        }
    }

    pub fn with_drop_zero_mean(mut self, enabled: bool) -> Self {
        self.drop_zero_mean = enabled;
        self
    }

    pub fn with_collect_choice(mut self, enabled: bool) -> Self {
        self.collect_choice = enabled;
        self
    }

    pub fn with_choice_rollouts_per_option(mut self, rollouts: usize) -> Self {
        self.choice_rollouts_per_option = rollouts.max(1);
        self
    }

    pub fn with_choice_policy_delta(mut self, delta: f64) -> Self {
        self.choice_policy_delta = if delta <= 0.0 { 1e-6 } else { delta };
        self
    }

    pub fn with_choice_score_mean_threshold(mut self, threshold: Option<f64>) -> Self {
        self.choice_score_mean_threshold = threshold;
        self
    }

    pub fn with_choice_skip_if_too_many(mut self, enabled: bool) -> Self {
        self.choice_skip_if_too_many = enabled;
        self
    }

    pub fn with_choice_follow_action_turn_range(mut self, enabled: bool) -> Self {
        self.choice_follow_action_turn_range = enabled;
        self
    }

    pub fn with_choice_rollout_on_uncollected_turns(mut self, enabled: bool) -> Self {
        self.choice_rollout_on_uncollected_turns = enabled;
        self
    }

    pub fn with_target_samples(mut self, target_samples: u64) -> Self {
        self.target_samples = target_samples.max(1);
        self
    }

    pub fn fast_after_target(mut self, enabled: bool) -> Self {
        self.fast_after_target = enabled;
        self
    }

    pub fn with_turn_range(mut self, turn_min: i32, turn_max: i32) -> Self {
        let min = turn_min.max(1);
        let max = turn_max.max(min);
        self.turn_min = min;
        self.turn_max = max;
        self
    }

    pub fn with_turn_stride(mut self, stride: i32) -> Self {
        self.turn_stride = stride.max(1);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn stats_snapshot(&self) -> MeanFilterCollectorStats {
        self.stats.borrow().clone()
    }

    pub fn num_samples(&self) -> usize {
        self.samples.borrow().len()
    }

    /// 取走本局（或当前缓冲）的 accepted 样本
    ///
    /// P1 推荐：主循环每局结束后调用一次，把样本交给 ShardWriter 统一写盘。
    pub fn drain_samples(&self) -> Vec<TrainingSample> {
        std::mem::take(&mut *self.samples.borrow_mut())
    }

    /// resume 时设置 accepted 基础值（仅用于“精确停止”）
    pub fn set_accepted_base(&self, accepted_written: u64) {
        let mut stats = self.stats.borrow_mut();
        stats.accepted = accepted_written;
    }

    fn should_collect_turn(&self, human_turn: i32) -> bool {
        if human_turn < self.turn_min || human_turn > self.turn_max {
            return false;
        }
        let stride = self.turn_stride.max(1);
        (human_turn - self.turn_min) % stride == 0
    }

    fn select_action_fast(&self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng) -> usize {
        // 复用 SimulationTrainer 的选择逻辑（无需 FlatSearch）
        if actions.len() <= 1 {
            return 0;
        }

        let all_dig = actions.iter().all(|a| matches!(a, OnsenAction::Dig(_)));
        if all_dig {
            return self.evaluator.select_onsen_index(game, actions);
        }

        let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
        if all_upgrade {
            return self.evaluator.select_upgrade_action(game, actions);
        }

        let selected_action = self.evaluator.select_action(game, rng);
        match &selected_action {
            Some(action) => actions.iter().position(|a| a == action).unwrap_or(0),
            None => 0,
        }
    }

    fn effective_choice_score_mean_threshold(&self) -> f64 {
        self.choice_score_mean_threshold
            .unwrap_or(self.score_mean_threshold)
    }

    fn compute_radical_factor(&self, turn: i32) -> f64 {
        // 对齐 FlatSearch::compute_radical_factor（固定公式、无随机性）
        // radical_factor = sqrt(remain_turns / TOTAL_TURN) * radical_factor_max
        let total_turn: usize = 78;
        let turn = turn.max(0) as usize;
        let remain_turns = (total_turn.saturating_sub(turn)) as f64;
        let factor = (remain_turns / total_turn as f64).powf(0.5);
        factor * self.search.config().radical_factor_max
    }

    fn eval_decision_event_choices_crn(
        &self,
        game: &OnsenGame,
        event: &EventData,
        num_choices: usize,
        rng: &StdRng,
    ) -> (Vec<f64>, usize, f64, ChoiceEvalResult) {
        // A. 降低 label 噪声：CRN（Common Random Numbers）
        // - 同一个 decision event 内：所有 choice 共享同一组 rollout_seeds（减少差分噪声）
        let radical_factor = self.compute_radical_factor(game.turn);
        let mut eval_rng = rng.clone(); // 不污染主 RNG
        let rollouts = self.choice_rollouts_per_option.max(1);
        let rollout_seeds: Vec<u64> = (0..rollouts).map(|_| eval_rng.random()).collect();

        let mut values: Vec<f64> = Vec::with_capacity(num_choices);
        let mut best_idx = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        let mut best_result = ChoiceEvalResult::default();

        for i in 0..num_choices {
            let r = self.eval_choice_result_with_seeds(game, event, i, &rollout_seeds);
            let v = r.weighted_mean(radical_factor);
            values.push(v);
            if v > best_value {
                best_value = v;
                best_idx = i;
                best_result = r;
            }
        }

        (values, best_idx, best_value, best_result)
    }

    fn eval_choice_result_with_seeds(
        &self,
        game: &OnsenGame,
        event: &EventData,
        choice_idx: usize,
        rollout_seeds: &[u64],
    ) -> ChoiceEvalResult {
        let mut result = ChoiceEvalResult::default();
        let sim_trainer = HandwrittenTrainer::new();
        let seeds = if rollout_seeds.is_empty() {
            // 理论上不会为空：外部会对 rollouts 做 max(1)；这里兜底避免无样本导致 mean=0
            &[0u64][..]
        } else {
            rollout_seeds
        };

        for &seed in seeds {
            let mut rollout_rng = StdRng::seed_from_u64(seed);

            let mut sim_game = game.clone();
            if sim_game.apply_event(event, choice_idx, &mut rollout_rng).is_err() {
                continue;
            }

            // 从下一阶段开始推进到终局（与 FlatSearch::simulate 的处理一致）
            while sim_game.next() {
                if sim_game.run_stage(&sim_trainer, &mut rollout_rng).is_err() {
                    break;
                }
            }
            let _ = sim_game.on_simulation_end(&sim_trainer, &mut rollout_rng);

            result.add(sim_game.uma().calc_score() as f64);
        }

        result
    }

    fn calc_choice_target(&self, values: &[f64]) -> Vec<f32> {
        let mut target = vec![0.0_f32; CHOICE_DIM];
        if values.is_empty() {
            return target;
        }

        let delta = self.choice_policy_delta.max(1e-6);
        let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = values.iter().map(|v| ((v - max_v) / delta).exp()).collect();
        let sum: f64 = exp_values.iter().sum();

        if sum.is_finite() && sum > 0.0 {
            for i in 0..values.len().min(CHOICE_DIM) {
                target[i] = (exp_values[i] / sum) as f32;
            }
            return target;
        }

        // 数值异常时回退为 one-hot（选 max）
        let best_idx = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if best_idx < CHOICE_DIM {
            target[best_idx] = 1.0;
        }
        target
    }
}

impl Trainer<OnsenGame> for MeanFilterCollectorTrainer {
    fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng) -> Result<usize> {
        if actions.len() <= 1 {
            return Ok(0);
        }

        let human_turn = game.turn + 1;
        let turn_idx = (human_turn - 1).clamp(0, 77) as usize;

        // 达到目标后：不再接收样本（且可选快速完成）
        {
            let stats = self.stats.borrow();
            if stats.accepted >= self.target_samples && self.fast_after_target {
                return Ok(self.select_action_fast(game, actions, rng));
            }
        }

        // P0 策略：无论是否采样本回合，都用 search 来做动作决策（避免分布变化）
        let output = match self.search.search(game, actions, rng) {
            Ok(v) => v,
            Err(e) => {
                let mut stats = self.stats.borrow_mut();
                stats.search_errors += 1;
                if self.verbose {
                    debug!("[回合 {human_turn}] FlatSearch 失败，回退选择第一个动作：{e}");
                }
                return Ok(0);
            }
        };

        // 只有“本回合属于采样范围”时才计为 candidate，并尝试 gate
        if self.should_collect_turn(human_turn) {
            let score_mean = output.best_result().mean();

            let mut stats = self.stats.borrow_mut();
            stats.candidates += 1;
            stats.turn_candidates[turn_idx] += 1;

            // P1 精确停止：达到 target_samples 后继续决策推进游戏，但不再接收样本
            let reached_target = stats.accepted >= self.target_samples;
            if reached_target {
                // 不接收样本，但仍然记录“被丢弃”（方便对齐 candidates）
                stats.dropped += 1;
                stats.turn_dropped[turn_idx] += 1;
            } else if self.drop_zero_mean && score_mean == 0.0 {
                stats.dropped += 1;
                stats.dropped_zero_mean += 1;
                stats.turn_dropped[turn_idx] += 1;
            } else if score_mean >= self.score_mean_threshold {
                stats.accepted += 1;
                stats.action_accepted += 1;
                stats.record_accepted_score_mean(score_mean);
                drop(stats);

                let sample = output.export_sample(game, self.search.config());
                let policy_sum: f32 = sample.policy_target.iter().sum();
                if (policy_sum - 1.0).abs() > 1e-3 {
                    let mut stats = self.stats.borrow_mut();
                    stats.policy_sum_not_one += 1;
                }
                if self.verbose && (policy_sum - 1.0).abs() > 1e-3 {
                    debug!(
                        "[回合 {human_turn}] policy_sum 异常：{policy_sum:.6}（可能是 action index 覆盖/越界）"
                    );
                }
                self.samples.borrow_mut().push(sample);
                let mut stats = self.stats.borrow_mut();
                stats.turn_accepted[turn_idx] += 1;
            } else {
                stats.dropped += 1;
                if score_mean == 0.0 {
                    stats.dropped_zero_mean += 1;
                }
                stats.turn_dropped[turn_idx] += 1;
            }
        }

        Ok(output.best_action_idx)
    }

    fn select_event_choice(
        &self,
        game: &OnsenGame,
        event: &EventData,
        choices: &[ActionValue],
        rng: &mut StdRng,
    ) -> Result<usize> {
        if choices.is_empty() {
            return Ok(0);
        }

        let human_turn = game.turn + 1;
        let turn_idx = (human_turn - 1).clamp(0, 77) as usize;

        // 达到目标后：不再接收样本（且可选快速完成，避免额外 rollout 成本）
        {
            let stats = self.stats.borrow();
            if stats.accepted >= self.target_samples && self.fast_after_target {
                return self.select_choice(game, choices, rng);
            }
        }

        // chance 事件：按 random_choice_prob 采样；不采集 choice 样本
        if let Some(probs) = &event.random_choice_prob {
            {
                let mut stats = self.stats.borrow_mut();
                stats.choice_skipped_chance_event += 1;
            }

            // 长度校验：不匹配时回退为均匀随机，保持 chance 语义
            if probs.len() != choices.len() {
                if self.verbose {
                    debug!(
                        "[回合 {human_turn}] 事件#{} {} random_choice_prob.len()={} != choices.len()={}，回退为均匀随机",
                        event.id,
                        event.name,
                        probs.len(),
                        choices.len()
                    );
                }
                return Ok(rng.random_range(0..choices.len()));
            }

            // chance 事件：按权重采样；权重非法时回退为均匀随机
            return match WeightedIndex::new(probs) {
                Ok(weights) => Ok(weights.sample(rng)),
                Err(e) => {
                    if self.verbose {
                        debug!("[回合 {human_turn}] 事件#{} {} random_choice_prob 非法（{}），回退为均匀随机", event.id, event.name, e);
                    }
                    Ok(rng.random_range(0..choices.len()))
                }
            };
        }

        // 决策事件：若未开启 collect_choice，则回退到旧接口（select_choice）
        if !self.collect_choice || choices.len() <= 1 {
            return self.select_choice(game, choices, rng);
        }

        // 维度不匹配：choices.len() > CHOICE_DIM 时无法对齐 label（CHOICE_DIM=8），默认跳过采样并回退选择
        if choices.len() > CHOICE_DIM {
            if self.choice_skip_if_too_many {
                let mut stats = self.stats.borrow_mut();
                stats.choice_skipped_too_many_options += 1;
            }
            return self.select_choice(game, choices, rng);
        }

        // B. choice 与 turn_stride 的关系：默认跟随 action 的采样范围
        // - 不采样回合仍需做出 choice 决策以推进游戏
        // - 可选：不采样回合是否仍使用 rollout 决策（成本高但轨迹分布更稳定）
        let should_collect_turn = self.should_collect_turn(human_turn);
        if self.choice_follow_action_turn_range && !should_collect_turn {
            if self.choice_rollout_on_uncollected_turns {
                let (_values, best_idx, _best_value, _best_result) =
                    self.eval_decision_event_choices_crn(game, event, choices.len(), rng);
                return Ok(best_idx);
            }
            return self.select_choice(game, choices, rng);
        }

        // 评估每个 choice（方案 A：rollouts + Handwritten 推进到终局）
        let (values, best_idx, best_value, best_result) =
            self.eval_decision_event_choices_crn(game, event, choices.len(), rng);

        // 生成 choice 样本（features 必须带 pending_choices）
        let nn_input = game.extract_nn_features(Some(choices));
        let choice_target = self.calc_choice_target(&values);
        let policy_target = vec![0.0_f32; POLICY_DIM];
        let value_target = vec![
            best_result.mean() as f32,
            best_result.stdev() as f32,
            if best_value.is_finite() {
                best_value.max(0.0) as f32
            } else {
                0.0
            },
        ];

        let choice_sum: f32 = choice_target.iter().sum();
        let policy_non_zero = policy_target.iter().any(|&x| x != 0.0);

        // gate + 统计
        {
            let mut stats = self.stats.borrow_mut();
            stats.choice_candidates += 1;
            stats.turn_choice_candidates[turn_idx] += 1;

            let reached_target = stats.accepted >= self.target_samples;
            let score_mean = best_result.mean();
            let threshold = self.effective_choice_score_mean_threshold();

            if reached_target {
                stats.choice_dropped += 1;
                stats.turn_choice_dropped[turn_idx] += 1;
            } else if self.drop_zero_mean && score_mean == 0.0 {
                stats.choice_dropped += 1;
                stats.turn_choice_dropped[turn_idx] += 1;
            } else if score_mean >= threshold {
                stats.choice_accepted += 1;
                stats.accepted += 1;
                stats.turn_choice_accepted[turn_idx] += 1;

                if (choice_sum - 1.0).abs() > 1e-3 {
                    stats.choice_sum_not_one += 1;
                }
                if policy_non_zero {
                    stats.choice_policy_not_zero += 1;
                }

                drop(stats);

                let sample = TrainingSample::new(nn_input, policy_target, choice_target, value_target);
                self.samples.borrow_mut().push(sample);
            } else {
                stats.choice_dropped += 1;
                stats.turn_choice_dropped[turn_idx] += 1;
            }
        }

        Ok(best_idx)
    }

    fn select_choice(&self, game: &OnsenGame, choices: &[ActionValue], _rng: &mut StdRng) -> Result<usize> {
        // P0 不采集 choice 样本，但必须做出 choice 决策以推进游戏。
        // 复用 HandwrittenEvaluator 的 evaluate_choice 逻辑，避免完全随机导致整体质量/通过率下降。
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
