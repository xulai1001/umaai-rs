//! MCTS 训练员
//!
//! 使用扁平蒙特卡洛搜索进行决策，通过多次模拟评估各决策的价值。
//!
//! # 用途
//! - 高质量决策（比手写策略更优）
//! - 生成高质量训练数据（每个状态有准确的价值估计）
//! - 自对弈训练

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use anyhow::{Result, anyhow};
use colored::Colorize;
use flexi_logger::LogSpecification;
use log::{info, warn};
use rand::prelude::StdRng;

use crate::{
    game::{
        Trainer,
        onsen::{action::OnsenAction, game::OnsenGame}
    },
    gamedata::{ActionValue, GAMECONSTANTS, LOGGER},
    global,
    neural::{Evaluator, HandwrittenEvaluator},
    search::{ActionResult, FlatSearch, SearchConfig, SearchOutput},
    utils::format_luck
};

/// MCTS 训练员
///
/// 使用扁平蒙特卡洛搜索进行动作选择。
/// 对于温泉选择和装备升级使用手写逻辑（这些场景有固定最优策略），
/// 其他动作使用 MCTS 搜索评估。
pub struct MctsTrainer {
    /// 扁平搜索器
    pub search: FlatSearch,
    /// 手写评估器（用于温泉/装备等特殊场景）
    pub evaluator: HandwrittenEvaluator,
    /// 是否输出详细日志
    pub verbose: bool,
    /// 是否搜索温泉
    pub mcts_onsen: bool,
    /// 优先输出哪种结果
    pub mcts_selection: String,
    /// 保存上一回合游戏，用于判断
    pub last_game: Option<OnsenGame>,
    /// 上一回合最好的选择分数. 使用Atomic以实现内部可变
    pub last_score: (AtomicU64, AtomicU64),
    /// 第一回合分数
    pub initial_score: (AtomicU64, AtomicU64),
    /// 保存当前的搜索结果用于输出
    pub search_output: Arc<Mutex<SearchOutput>>
}

impl MctsTrainer {
    /// 创建 MCTS 训练员
    pub fn new(config: SearchConfig) -> Self {
        Self {
            search: FlatSearch::new(config),
            evaluator: HandwrittenEvaluator::new(),
            verbose: false,
            mcts_onsen: false,
            mcts_selection: "pt".to_string(),
            last_game: None,
            last_score: (AtomicU64::new(0), AtomicU64::new(0)),
            initial_score: (AtomicU64::new(0), AtomicU64::new(0)),
            search_output: Arc::new(Mutex::new(SearchOutput::default()))
        }
    }

    /// 创建默认 MCTS 训练员
    pub fn default_trainer() -> Self {
        Self::new(SearchConfig::default())
    }

    /// 设置是否输出详细日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 获取搜索配置
    pub fn config(&self) -> &SearchConfig {
        self.search.config()
    }

    /// 和上一回合比较，看是否同一局
    pub fn is_same_game(&self, game: &OnsenGame) -> bool {
        if let Some(last) = &self.last_game {
            // 马娘ID相同且回合数相同或差1，则再检查卡组
            if last.uma.uma_id == game.uma.uma_id && (last.turn == game.turn || last.turn + 1 == game.turn) {
                for i in 0..6 {
                    if last.deck[i].card_id != game.deck[i].card_id {
                        return false;
                    }
                }
                return true;
            }
        }
        false
    }

    pub fn format_action_result(
        &self, action: &OnsenAction, _result: &ActionResult, score: f64, best_score: f64
    ) -> String {
        let text = format!("{action}: {score:.0}");
        let delta = best_score - score;
        if delta <= 0.0 {
            format!("{}", text.bright_yellow().on_red())
        } else if delta <= 50.0 {
            format!("{}", text.green())
        } else if delta <= 300.0 {
            text
        } else {
            format!("{}", text.bright_black())
        }
    }
    // 计算本回合均分
    fn update_score(&self, game: &OnsenGame, actions: &[OnsenAction], search_output: &SearchOutput) {
        let mut sum = 0.0;
        let mut mean_weighted = 0.0;
        let mut count = 0;
        // 蒙特卡洛比手写逻辑增加的分数，随回合数递减. 补正在估分上
        let mcts_bonus = (78 - game.turn) * global!(GAMECONSTANTS).mcts_turn_bonus;
        let best_action = search_output.best_action();
        for r in &search_output.action_results {
            sum += r.0.sum;
            count += r.0.count();
            mean_weighted += r.0.weighted_mean(search_output.radical_factor) * r.0.count() as f64;
        }
        mean_weighted = (mean_weighted / count as f64) + mcts_bonus as f64;
        let turn_score = sum / count as f64 + mcts_bonus as f64;
        let initial_score = self.initial_score.0.load(Ordering::SeqCst);
        let last_score = self.last_score.0.load(Ordering::SeqCst);
        let luck_overall = turn_score - initial_score as f64;
        let luck_turn = turn_score - last_score as f64;
        let weighted_bonus = mean_weighted - turn_score;
        //let mut race_loss = 0.0;

        // 找到最优动作在原列表中的索引
        let idx = actions.iter().position(|a| a == best_action).unwrap_or(0);
        let mut best_score = search_output.action_results[idx].0.mean();
        for (i, _action) in search_output.actions.iter().enumerate() {
            let weighted_mean = search_output.action_results[i].0.weighted_mean(search_output.radical_factor);
            if weighted_mean > best_score {
                best_score = weighted_mean;
            }
        }
        best_score += mcts_bonus as f64;
        let is_dig_action = actions.iter().any(|a| matches!(a, OnsenAction::Dig(_)));
        if self.verbose {
            // 输出搜索结果
            let mut line = vec![];
            if !is_dig_action {
                info!(
                    "[回合 {}] 均分 {}, 运气: {}(乐观 + {weighted_bonus:.0}), {}",
                    game.turn + 1,
                    format!("{turn_score:.0}").cyan(),
                    format_luck("本局", luck_overall),
                    format_luck("本回合", luck_turn)
                );
            }
            // 输出各动作的分数
            for (i, action) in search_output.actions.iter().enumerate() {
                let result = &search_output.action_results[i];
                let weighted = result.0.weighted_mean(search_output.radical_factor);
                line.push(self.format_action_result(
                    action,
                    &result.0,
                    weighted + mcts_bonus as f64 - mean_weighted,
                    best_score - mean_weighted
                ));
            }
            info!("[回合 {} 重视评分] {}", game.turn + 1, line.join(" "));
        }

        // 保存分数
        if !is_dig_action {
            self.last_score.0.store(turn_score as u64, Ordering::SeqCst);
        }
        if initial_score == 0 {
            self.initial_score.0.store(turn_score as u64, Ordering::SeqCst);
        }
    }

    // 计算本回合PT加成均分
    fn update_score_2(&self, game: &OnsenGame, actions: &[OnsenAction], search_output: &SearchOutput) {
        let mut sum = 0.0;
        let mut mean_weighted = 0.0;
        let mut count = 0;
        let best_action = search_output.best_action_2();

        for r in &search_output.action_results {
            sum += r.1.sum;
            count += r.1.count();
            mean_weighted += r.1.weighted_mean(search_output.radical_factor) * r.1.count() as f64;
        }
        mean_weighted = mean_weighted / count as f64;
        let turn_score = sum / count as f64;
        let initial_score = self.initial_score.1.load(Ordering::SeqCst);

        // 找到最优动作在原列表中的索引
        let idx = actions.iter().position(|a| a == best_action).unwrap_or(0);
        let mut best_score = search_output.action_results[idx].1.mean();
        for (i, _action) in search_output.actions.iter().enumerate() {
            let weighted_mean = search_output.action_results[i].1.weighted_mean(search_output.radical_factor);
            if weighted_mean > best_score {
                best_score = weighted_mean;
            }
        }
        if self.verbose {
            // 输出搜索结果
            let mut line = vec![];
            // 输出各动作的分数
            for (i, action) in search_output.actions.iter().enumerate() {
                let result = &search_output.action_results[i];
                let weighted = result.1.weighted_mean(search_output.radical_factor);
                line.push(self.format_action_result(
                    action,
                    &result.1,
                    weighted - mean_weighted,
                    best_score - mean_weighted
                ));
            }
            info!("[回合 {} 重视 PT ] {}", game.turn + 1, line.join(" "));
        }

        // 保存分数
        self.last_score.1.store(turn_score as u64, Ordering::SeqCst);
        if initial_score == 0 {
            self.initial_score.1.store(turn_score as u64, Ordering::SeqCst);
        }
    }
}

impl Default for MctsTrainer {
    fn default() -> Self {
        Self::default_trainer()
    }
}

impl Trainer<OnsenGame> for MctsTrainer {
    fn select_action(
        &self, game: &OnsenGame, actions: &[<OnsenGame as crate::game::Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        use crate::game::onsen::action::OnsenAction;

        // 只有一个动作时直接返回
        if actions.len() <= 1 {
            return Ok(0);
        }
        //println!("{game:#?}");

        // 检查是否是温泉选择场景（动作是 Dig）
        let is_dig = actions.iter().any(|a| matches!(a, OnsenAction::Dig(_)));
        if is_dig && !self.mcts_onsen {
            // mcts_onsen=false时 温泉选择使用手写逻辑（固定最优顺序）
            let idx = self.evaluator.select_onsen_index(game, actions);
            if self.verbose {
                info!("[回合 {}] 选择温泉（手写逻辑）: {}", game.turn + 1, actions[idx]);
            }
            return Ok(idx);
        }
        /*
            // 检查是否是装备升级场景（所有动作都是 Upgrade）
                let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
                if all_upgrade {
                    // 装备升级：使用手写逻辑
                    let idx = self.evaluator.select_upgrade_action(game, actions);
                    if self.verbose {
                        info!(
                            "[回合 {}] MCTS 选择装备升级（手写逻辑）: {}",
                            game.turn,
                            actions[idx]
                        );
                    }
                    return Ok(idx);
                }
        */
        global!(LOGGER)
            .lock()
            .expect("logger lock")
            .push_temp_spec(LogSpecification::off());

        // 使用 MCTS 搜索
        let search_output = self.search.search(game, actions, rng)?;
        {
            // 保存搜索结果
            let mut s = self.search_output
                .lock()
                .map_err(|_| anyhow!("lock failed"))?;
            *s = search_output.clone();
        }
        global!(LOGGER).lock().expect("logger lock").pop_temp_spec();

        let best_action = search_output.best_action();
        let best_action_2 = search_output.best_action_2();
        let selection = match self.mcts_selection.as_str() {
            "pt" => best_action_2,
            _ => best_action
        };

        // 找到最优动作在原列表中的索引
        //let idx = actions.iter().position(|a| a == best_action).unwrap_or(0);
        let idx = actions.iter().position(|a| a == selection).unwrap_or(0);
        self.update_score(game, actions, &search_output);
        self.update_score_2(game, actions, &search_output);

        Ok(idx)
    }

    fn select_choice(&self, game: &OnsenGame, choices: &[ActionValue], _rng: &mut StdRng) -> Result<usize> {
        // 事件选择：使用手写逻辑
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (i, _choice) in choices.iter().enumerate() {
            let value = self.evaluator.evaluate_choice(game, i);
            if value > best_value {
                best_value = value;
                best_idx = i;
            }
        }

        if self.verbose {
            warn!(
                "[回合 {}] MCTS 选择事件选项（手写逻辑）: {} (索引 {})",
                game.turn + 1,
                choices[best_idx],
                best_idx
            );
        }

        Ok(best_idx)
    }
}
