//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use umasim::{
    game::{Game, InheritInfo, Trainer, basic::BasicGame, onsen::game::OnsenGame},
    gamedata::{GAMECONSTANTS, GameConfig, init_global},
    global,
    sample_collector::GameSample,
    trainer::*,
    training_sample::TrainingSampleBatch,
    utils::init_logger
};

/// 单次模拟结果
struct SimulationResult {
    score: i32,
    pt: i32,
    explain: String
}

/// 运行 OnsenGame（单次），返回模拟结果
fn run_onsen_once<T: Trainer<OnsenGame>>(
    trainer: &T, uma: u32, cards: &[u32; 6], inherit: InheritInfo, rng: &mut StdRng
) -> Result<SimulationResult> {
    let mut game = OnsenGame::newgame(uma, cards, inherit)?;
    game.run_full_game(trainer, rng)?;
    info!("育成结束！");

    let score = game.uma.calc_score();
    let pt = game.uma.total_pt();
    let explain = game.explain()?;

    Ok(SimulationResult { score, pt, explain })
}

/// 运行 BasicGame（单次），返回模拟结果
fn run_basic_once<T: Trainer<BasicGame>>(
    trainer: &T, uma: u32, cards: &[u32; 6], inherit: InheritInfo, rng: &mut StdRng
) -> Result<SimulationResult> {
    let mut game = BasicGame::newgame(uma, cards, inherit)?;
    game.run_full_game(trainer, rng)?;
    info!("育成结束！");

    let score = game.uma.calc_score();
    let pt = game.uma.total_pt();
    let explain = game.explain()?;

    Ok(SimulationResult { score, pt, explain })
}

/// 运行样本收集模式
///
/// 收集训练数据并保存到文件
///
/// # 探索性策略
/// 使用 40% 探索率在温泉选择时引入随机性，让神经网络能够
/// 从不同的挖掘顺序中学习，而不是只记住固定的顺序。
fn run_collector_mode(config: &GameConfig, num_games: usize, rng: &mut StdRng) -> Result<()> {
    // 创建带探索率的训练员
    let trainer = CollectorTrainer::new(); // 默认 40% 探索率

    println!("=== 样本收集模式 ===");
    println!("模拟次数: {}", num_games);
    println!("温泉选择探索率: {:.0}%", trainer.exploration_rate() * 100.0);
    println!("精英比例: 1%");
    println!("输出文件: training_data.bin");
    println!();

    let mut game_samples: Vec<GameSample> = Vec::with_capacity(num_games);

    // 创建进度条
    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-")
    );

    let start = Instant::now();

    // 主循环
    for i in 0..num_games {
        trainer.reset();

        let inherit = InheritInfo {
            blue_count: config.blue_count.clone(),
            extra_count: config.extra_count.clone()
        };

        let mut game = OnsenGame::newgame(config.uma, &config.cards, inherit)?;
        game.run_full_game(&trainer, rng)?;

        let score = game.uma.calc_score();
        game_samples.push(trainer.finalize(score));

        if (i + 1) % 1000 == 0 {
            pb.set_position((i + 1) as u64);
        }
    }

    pb.finish_with_message("模拟完成");
    println!("\n模拟耗时: {:?}", start.elapsed());
    println!("成功收集: {} 局", game_samples.len());

    // 按分数排序（降序）
    game_samples.sort_by(|a, b| b.final_score.cmp(&a.final_score));

    // 统计信息
    if !game_samples.is_empty() {
        let scores: Vec<i32> = game_samples.iter().map(|s| s.final_score).collect();
        let avg = scores.iter().sum::<i32>() as f64 / scores.len() as f64;
        let max_score = scores[0];
        let min_score = scores[scores.len() - 1];
        let avg_score = avg as i32;

        println!("\n分数统计:");
        println!(
            "  最高分: {} ({})",
            max_score,
            global!(GAMECONSTANTS).get_rank_name(max_score)
        );
        println!(
            "  最低分: {} ({})",
            min_score,
            global!(GAMECONSTANTS).get_rank_name(min_score)
        );
        println!(
            "  平均分: {:.0} ({})",
            avg,
            global!(GAMECONSTANTS).get_rank_name(avg_score)
        );
    }

    // 筛选精英样本（Top 5%） 可以根据需求设置 最开始1%太少了太难跑了
    let elite_count = (game_samples.len() / 20).max(1);
    let elite_games: Vec<GameSample> = game_samples.into_iter().take(elite_count).collect();

    println!("\n精英筛选:");
    println!("  筛选数量: {} 局 (Top 5%)", elite_count);
    if let Some(last) = elite_games.last() {
        println!(
            "  最低精英分数: {} ({})",
            last.final_score,
            global!(GAMECONSTANTS).get_rank_name(last.final_score)
        );
    }

    // 合并样本并保存
    let mut all_samples = Vec::new();
    for game in elite_games {
        all_samples.extend(game.samples);
    }

    println!("\n样本统计:");
    println!("  总样本数: {}", all_samples.len());
    println!("  温泉选择探索率: {:.0}%", trainer.exploration_rate() * 100.0);

    let batch = TrainingSampleBatch { samples: all_samples };
    batch.save_binary("training_data.bin")?;

    let file_size = std::fs::metadata("training_data.bin")?.len();
    println!(
        "保存完成: training_data.bin ({:.2} MB)",
        file_size as f64 / 1024.0 / 1024.0
    );
    println!("\n总耗时: {:?}", start.elapsed());

    Ok(())
}

/// 打印多次模拟的统计结果
fn print_simulation_stats(results: &[SimulationResult], elapsed: std::time::Duration) {
    if results.is_empty() {
        return;
    }

    // 找到最高分和最低分
    let (best_idx, best) = results.iter().enumerate().max_by_key(|(_, r)| r.score).unwrap();
    let (worst_idx, worst) = results.iter().enumerate().min_by_key(|(_, r)| r.score).unwrap();

    // 计算平均分
    let avg_score = results.iter().map(|r| r.score as f64).sum::<f64>() / results.len() as f64;
    let avg_pt = results.iter().map(|r| r.pt as f64).sum::<f64>() / results.len() as f64;

    // 打印分隔线
    println!("\n{}", "=".repeat(60));
    println!("多次模拟统计 (共 {} 次)", results.len());
    println!("{}", "=".repeat(60));

    // 打印最高分结算面板
    println!("\n【最高分】第 {} 次模拟:", best_idx + 1);
    println!("{}", best.explain);
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(best.score),
        best.score,
        best.pt
    );

    // 打印最低分结算面板
    println!("\n【最低分】第 {} 次模拟:", worst_idx + 1);
    println!("{}", worst.explain);
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(worst.score),
        worst.score,
        worst.pt
    );

    // 打印统计摘要
    println!("\n{}", "-".repeat(60));
    println!(
        "平均评分: {} {:.0}, 平均PT+Hint: {:.0}",
        global!(GAMECONSTANTS).get_rank_name(avg_score as i32),
        avg_score,
        avg_pt
    );
    println!("总耗时: {:?}, 平均单次: {:?}", elapsed, elapsed / results.len() as u32);
    println!("{}", "=".repeat(60));
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. 先读取配置文件
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let game_config: GameConfig = toml::from_str(&config_file)?;

    // 2. 根据配置初始化日志
    init_logger(&game_config.log_level)?;

    // 3. 再初始化全局数据
    init_global()?;

    let simulation_count = game_config.simulation_count.max(1);

    // 开始计时
    let start = Instant::now();

    let inherit = InheritInfo {
        blue_count: game_config.blue_count.clone(),
        extra_count: game_config.extra_count.clone()
    };

    // 收集模拟结果
    let sim_results: Vec<Result<SimulationResult>> = (0..simulation_count)
        .into_par_iter()
        .map(|_| {
            // 执行具体的模拟过程
            let mut rng = StdRng::from_os_rng();
            match game_config.trainer.as_str() {
                "random" => {
                    let trainer = RandomTrainer;
                    match game_config.scenario.as_str() {
                        "onsen" => {
                            run_onsen_once(&trainer, game_config.uma, &game_config.cards, inherit.clone(), &mut rng)
                        }
                        _ => run_basic_once(&trainer, game_config.uma, &game_config.cards, inherit.clone(), &mut rng)
                    }
                }
                "handwritten" => {
                    let trainer = HandwrittenTrainer::new().verbose(simulation_count == 1);
                    match game_config.scenario.as_str() {
                        "onsen" => {
                            run_onsen_once(&trainer, game_config.uma, &game_config.cards, inherit.clone(), &mut rng)
                        }
                        _ => {
                            println!("警告: 手写策略训练员仅支持 onsen 剧本，使用 random 训练员");
                            let trainer = RandomTrainer;
                            run_basic_once(&trainer, game_config.uma, &game_config.cards, inherit.clone(), &mut rng)
                        }
                    }
                }
                "collector" => {
                    // 样本收集模式：收集训练数据
                    unimplemented!()
                    //run_collector_mode(&game_config, simulation_count, &mut rng)
                }
                "neuralnet" | "nn" => {
                    // 神经网络训练员
                    let model_path = "saved_models/onsen_v1/model.onnx";
                    match NeuralNetTrainer::load(model_path) {
                        Ok(trainer) => {
                            let trainer = trainer.verbose(simulation_count == 1);
                            match game_config.scenario.as_str() {
                                "onsen" => run_onsen_once(
                                    &trainer,
                                    game_config.uma,
                                    &game_config.cards,
                                    inherit.clone(),
                                    &mut rng
                                ),
                                _ => {
                                    println!("警告: 神经网络训练员仅支持 onsen 剧本，使用 random 训练员");
                                    let trainer = RandomTrainer;
                                    run_basic_once(
                                        &trainer,
                                        game_config.uma,
                                        &game_config.cards,
                                        inherit.clone(),
                                        &mut rng
                                    )
                                }
                            }
                        }
                        Err(e) => {
                            println!("错误: 无法加载神经网络模型 '{}': {}", model_path, e);
                            println!("请确保模型文件存在，或使用其他训练员");
                            Err(e)
                        }
                    }
                }
                "mcts" => {
                    // MCTS 训练员
                    use umasim::search::SearchConfig;
                    let search_config = SearchConfig::default()
                        .with_search_n(game_config.mcts.search_n)
                        .with_radical_factor_max(game_config.mcts.radical_factor_max)
                        .with_max_depth(game_config.mcts.max_depth)
                        .with_policy_delta(game_config.mcts.policy_delta)
                        // UCB 参数
                        .with_ucb(game_config.mcts.use_ucb)
                        .with_search_group_size(game_config.mcts.search_group_size)
                        .with_search_cpuct(game_config.mcts.search_cpuct)
                        .with_expected_search_stdev(game_config.mcts.expected_search_stdev)
                        .with_adjust_radical_by_turn(game_config.mcts.adjust_radical_by_turn);
                    let trainer = MctsTrainer::new(search_config).verbose(simulation_count == 1);
                    match game_config.scenario.as_str() {
                        "onsen" => run_onsen_once(
                            &trainer,
                            game_config.uma,
                            &game_config.cards,
                            inherit.clone(),
                            &mut rng
                        ),
                        _ => {
                            println!("警告: MCTS 训练员仅支持 onsen 剧本，使用 random 训练员");
                            let trainer = RandomTrainer;
                            run_basic_once(
                                &trainer,
                                game_config.uma,
                                &game_config.cards,
                                inherit.clone(),
                                &mut rng
                            )
                        }
                    }
                }
                _ => {
                    // 默认使用手动训练员（不支持多次模拟）
                    if simulation_count > 1 {
                        println!("警告: 手动训练员不支持多次模拟，仅运行1次");
                    }
                    let trainer = ManualTrainer;
                    let result = match game_config.scenario.as_str() {
                        "onsen" => {
                            run_onsen_once(&trainer, game_config.uma, &game_config.cards, inherit.clone(), &mut rng)
                        }
                        _ => run_basic_once(&trainer, game_config.uma, &game_config.cards, inherit.clone(), &mut rng)
                    }?;
                    // 单次模拟直接打印结果
                    println!("{}", result.explain);
                    println!(
                        "评分: {} {}, PT: {}",
                        global!(GAMECONSTANTS).get_rank_name(result.score),
                        result.score,
                        result.pt
                    );
                    println!("耗时: {:?}", start.elapsed());
                    Ok(result)
                }
            }
        })
        .collect();

    let mut results = vec![];
    for result in sim_results {
        // 根据 trainer 和 scenario 配置选择训练员和剧本
        let r = result?;

        // 单次模拟时打印每次结果
        if simulation_count < 100 {
            println!("{}", r.explain);
            println!(
                "评分: {} {}, PT: {}",
                global!(GAMECONSTANTS).get_rank_name(r.score),
                r.score,
                r.pt
            );
        }
        results.push(r);
    }

    // 多次模拟时打印统计结果
    if simulation_count > 1 {
        print_simulation_stats(&results, start.elapsed());
    } else {
        println!("耗时: {:?}", start.elapsed());
    }

    Ok(())
}
