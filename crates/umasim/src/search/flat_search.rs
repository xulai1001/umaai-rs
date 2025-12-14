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
    neural::{Evaluator, HandwrittenEvaluator}
};

/// 扁平蒙特卡洛搜索
///
/// 使用手写逻辑进行模拟，统计各动作的分数分布。
#[derive(Clone)]
pub struct FlatSearch {
    /// 手写评估器（用于模拟）
    evaluator: HandwrittenEvaluator,

    /// 搜索配置
    config: SearchConfig
}

impl FlatSearch {
    /// 创建搜索器
    pub fn new(config: SearchConfig) -> Self {
        Self {
            evaluator: HandwrittenEvaluator::new(),
            config
        }
    }

    /// 创建默认搜索器
    pub fn default_search() -> Self {
        Self::new(SearchConfig::default())
    }

    /// 获取配置
    pub fn config(&self) -> &SearchConfig {
        &self.config
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
            "[回合 {}] 开始搜索: {} 个动作, search_n={}, radical_factor={:.1}, ucb={}",
            game.turn,
            actions.len(),
            self.config.search_n,
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
    fn search_uniform(&self, game: &OnsenGame, actions: &[OnsenAction]) -> Result<Vec<ActionResult>> {
        let action_results: Vec<ActionResult> = actions
            .par_iter()
            .map(|action| {
                let mut result = ActionResult::new();

                // 每个线程初始化一次 RNG
                let mut thread_rng = StdRng::from_os_rng();
                for _ in 0..self.config.search_n {
                    if let Ok(score) = self.simulate(game, action, &mut thread_rng) {
                        result.add(score);
                    }
                }

                result
            })
            .collect();

        Ok(action_results)
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
    ) -> Result<Vec<ActionResult>> {
        let num_actions = actions.len();
        let mut action_results: Vec<ActionResult> = vec![ActionResult::new(); num_actions];
        let group_size = self.config.search_group_size;

        // 第一阶段：每个动作先搜一组（并行）
        let initial_results: Vec<ActionResult> = actions
            .par_iter()
            .map(|action| {
                let mut result = ActionResult::new();
                let mut thread_rng = StdRng::from_os_rng();
                for _ in 0..group_size {
                    if let Ok(score) = self.simulate(game, action, &mut thread_rng) {
                        result.add(score);
                    }
                }
                result
            })
            .collect();

        // 合并初始结果
        for (i, result) in initial_results.into_iter().enumerate() {
            action_results[i] = result;
        }

        let mut total_n = (group_size * num_actions) as f64;

        // 第二阶段：UCB 动态分配
        loop {
            // 检查是否有动作达到 search_n
            let max_count = action_results.iter().map(|r| r.count()).max().unwrap_or(0);
            if max_count >= self.config.search_n as u32 {
                break;
            }

            // 使用 UCB 公式选择下一个要搜索的动作
            let best_action_idx = self.select_ucb_action(&action_results, radical_factor, total_n);

            // 对选中的动作搜索一组（并行）
            let action = &actions[best_action_idx];
            let scores: Vec<f64> = (0..group_size)
                .into_par_iter()
                .filter_map(|_| {
                    let mut thread_rng = StdRng::from_os_rng();
                    self.simulate(game, action, &mut thread_rng).ok()
                })
                .collect();

            for score in scores {
                action_results[best_action_idx].add(score);
            }

            total_n += group_size as f64;
        }

        Ok(action_results)
    }

    /// 使用 UCB 公式选择下一个要搜索的动作
    ///
    /// UCB 公式: search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn select_ucb_action(&self, action_results: &[ActionResult], radical_factor: f64, total_n: f64) -> usize {
        let sqrt_total = total_n.sqrt();
        let cpuct = self.config.search_cpuct;
        let expected_stdev = self.config.expected_search_stdev;

        let mut best_idx = 0;
        let mut best_search_value = f64::NEG_INFINITY;

        for (i, result) in action_results.iter().enumerate() {
            let n = result.count() as f64;
            if n == 0.0 {
                // 未搜索的动作优先级最高
                return i;
            }

            let value = result.weighted_mean(radical_factor);
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
    fn simulate(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> Result<f64> {
        if matches!(action, OnsenAction::Dig(_)) {
            self.simulate_onsen_select(game, action, rng)
        } else if matches!(action, OnsenAction::Upgrade(_)) {
            self.simulate_dig_upgrade(game, action, rng)
        } else {
            // 克隆游戏状态
            let mut sim_game = game.clone();
            let trainer = SimulationTrainer { evaluator: &self.evaluator };

            // 执行初始动作
            sim_game.apply_action(action, rng)?;

            // 推进到下一阶段，继续运行直到游戏结束
            while sim_game.next() {
                sim_game.run_stage(&trainer, rng)?;
            }

            // 触发育成结束奖励
            sim_game.on_simulation_end(&trainer, rng)?;

            // 返回最终分数
            Ok(sim_game.uma().calc_score() as f64)
        }
    }

    /// 模拟选择温泉. 因为没有做成单独的阶段，所以单独处理
    pub fn simulate_onsen_select(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> Result<f64> {
        let mut sim_game = game.clone();
        let mut best_score = 0.0;
        //let trainer = SimulationTrainer { evaluator: &self.evaluator };
        sim_game.apply_action(action, rng)?;
        for i in sim_game.get_upgradeable_equipment() {
            let score = self.simulate_dig_upgrade(&sim_game, &OnsenAction::Upgrade(i as i32), rng)?;
            if score > best_score {
                best_score = score;
            }
        }
        Ok(best_score)
    }

    /// 模拟升级挖掘装备
    pub fn simulate_dig_upgrade(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> Result<f64> {
        let mut sim_game = game.clone();
        sim_game.apply_action(action, rng)?;
        sim_game.pending_selection = false;
        // 去除pending_selection状态后就可以正常模拟了。
        let trainer = SimulationTrainer { evaluator: &self.evaluator };
        while sim_game.next() {
            sim_game.run_stage(&trainer, rng)?;
        }
        sim_game.on_simulation_end(&trainer, rng)?;
        Ok(sim_game.uma().calc_score() as f64)
    }
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
