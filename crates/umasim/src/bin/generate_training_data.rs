//! 训练数据生成器
//!
//! 运行大规模模拟，收集精英样本用于神经网络训练。
//!
//! # 用法
//! ```bash
//! cargo run --release --bin generate_training_data -- \
//!     --num-games 1000000 \
//!     --top-percent 1 \
//!     --output training_data.bin
//! ```

use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    game::{Game, InheritInfo, onsen::game::OnsenGame},
    gamedata::{GameConfig, init_global},
    sample_collector::GameSample,
    trainer::CollectorTrainer,
    training_sample::TrainingSampleBatch,
    utils::init_logger
};

/// 训练数据生成器命令行参数
#[derive(Parser, Debug)]
#[command(name = "generate_training_data")]
#[command(about = "生成神经网络训练数据")]
struct Args {
    /// 模拟次数
    #[arg(long, default_value = "1000000")]
    num_games: usize,

    /// 精英百分比（取分数最高的 N%）
    #[arg(long, default_value = "1")]
    top_percent: usize,

    /// 输出文件路径
    #[arg(long, default_value = "training_data.bin")]
    output: String,

    /// 进度显示间隔
    #[arg(long, default_value = "1000")]
    progress_interval: usize,

    /// 随机种子（可选）
    #[arg(long)]
    seed: Option<u64>
}

/// 运行单局游戏并收集样本
fn run_single_game(trainer: &CollectorTrainer, config: &GameConfig, rng: &mut StdRng) -> Result<GameSample> {
    // 重置收集器
    trainer.reset();

    // 创建游戏
    let inherit = InheritInfo {
        blue_count: config.blue_count.clone(),
        extra_count: config.extra_count.clone()
    };
    let mut game = OnsenGame::newgame(config.uma, &config.cards, inherit)?;

    // 运行完整游戏
    game.run_full_game(trainer, rng)?;

    // 获取最终分数并完成收集
    let score = game.uma.calc_score();
    Ok(trainer.finalize(score))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 读取配置文件
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let config: GameConfig = toml::from_str(&config_file)?;

    // 初始化日志（使用 warn 级别减少输出）
    init_logger("generate", "info")?;
    init_global()?;

    println!("=== 训练数据生成器 ===");
    println!("模拟次数: {}", args.num_games);
    println!("精英比例: {}%", args.top_percent);
    println!("输出文件: {}", args.output);
    println!();

    // 创建随机数生成器
    let mut rng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng()
    };

    // 创建训练员
    let trainer = CollectorTrainer::new();

    // 收集所有游戏样本
    let mut game_samples: Vec<GameSample> = Vec::with_capacity(args.num_games);

    // 创建进度条
    let pb = ProgressBar::new(args.num_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-")
    );

    let start = Instant::now();

    // 主循环：运行模拟
    for i in 0..args.num_games {
        match run_single_game(&trainer, &config, &mut rng) {
            Ok(sample) => game_samples.push(sample),
            Err(e) => {
                eprintln!("警告: 第 {} 局模拟失败: {}", i + 1, e);
            }
        }

        // 更新进度条
        if (i + 1) % args.progress_interval == 0 {
            pb.set_position((i + 1) as u64);
        }
    }

    pb.finish_with_message("模拟完成");
    println!("\n模拟耗时: {:?}", start.elapsed());
    println!("成功收集: {} 局", game_samples.len());

    // 按分数排序（降序）
    game_samples.sort_by(|a, b| b.final_score.cmp(&a.final_score));

    // 计算统计信息
    if !game_samples.is_empty() {
        let scores: Vec<i32> = game_samples.iter().map(|s| s.final_score).collect();
        let avg = scores.iter().sum::<i32>() as f64 / scores.len() as f64;
        let max = scores[0];
        let min = scores[scores.len() - 1];

        println!("\n分数统计:");
        println!("  最高分: {}", max);
        println!("  最低分: {}", min);
        println!("  平均分: {:.0}", avg);
    }

    // 筛选精英样本
    let elite_count = (game_samples.len() * args.top_percent / 100).max(1);
    let elite_games: Vec<GameSample> = game_samples.into_iter().take(elite_count).collect();

    println!("\n精英筛选:");
    println!("  筛选数量: {} 局 (Top {}%)", elite_count, args.top_percent);
    if let Some(last) = elite_games.last() {
        println!("  最低精英分数: {}", last.final_score);
    }

    // 合并所有精英样本
    let mut all_samples = Vec::new();
    let mut total_turns = 0;
    for game in elite_games {
        total_turns += game.samples.len();
        all_samples.extend(game.samples);
    }

    println!("\n样本统计:");
    println!("  总样本数: {}", all_samples.len());
    println!("  平均每局回合: {:.1}", total_turns as f64 / elite_count as f64);

    // 保存到文件
    println!("\n保存数据...");
    let batch = TrainingSampleBatch { samples: all_samples };
    batch.save_binary(&args.output)?;

    let file_size = std::fs::metadata(&args.output)?.len();
    println!(
        "保存完成: {} ({:.2} MB)",
        args.output,
        file_size as f64 / 1024.0 / 1024.0
    );
    println!("\n总耗时: {:?}", start.elapsed());

    Ok(())
}
