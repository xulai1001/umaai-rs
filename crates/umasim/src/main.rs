//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran
use std::time::Instant;

use anyhow::Result;
use log::info;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    game::{Game, InheritInfo, basic::BasicGame, onsen::game::OnsenGame},
    gamedata::{GAMECONSTANTS, GameConfig, init_global},
    trainer::*,
    utils::init_logger
};

pub mod explain;
pub mod game;
pub mod gamedata;
pub mod trainer;
pub mod utils;

/// 运行 OnsenGame
fn run_onsen<T: crate::game::Trainer<OnsenGame>>(
    trainer: &T,
    uma: u32,
    cards: &[u32; 6],
    inherit: InheritInfo,
    rng: &mut StdRng
) -> Result<()> {
    let mut game = OnsenGame::newgame(uma, cards, inherit)?;
    println!("{}", game.explain()?);
    let score = game.uma.calc_score();
    println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);
    game.run_full_game(trainer, rng)?;
    info!("育成结束！");
    // 最终结果用 println! 输出，log_level=off 时也可见
    println!("{}", game.explain()?);
    let score = game.uma.calc_score();
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(score),
        score,
        game.uma.total_pt()
    );
    Ok(())
}

/// 运行 BasicGame
fn run_basic<T: crate::game::Trainer<BasicGame>>(
    trainer: &T,
    uma: u32,
    cards: &[u32; 6],
    inherit: InheritInfo,
    rng: &mut StdRng
) -> Result<()> {
    let mut game = BasicGame::newgame(uma, cards, inherit)?;
    println!("{}", game.explain()?);
    let score = game.uma.calc_score();
    println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);
    game.run_full_game(trainer, rng)?;
    info!("育成结束！");
    // 最终结果用 println! 输出，log_level=off 时也可见
    println!("{}", game.explain()?);
    let score = game.uma.calc_score();
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(score),
        score,
        game.uma.total_pt()
    );
    Ok(())
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

    let inherit = InheritInfo {
        blue_count: game_config.blue_count.clone(),
        extra_count: game_config.extra_count.clone()
    };
    let mut rng = StdRng::from_os_rng();

    // 开始计时
    let start = Instant::now();

    // 根据 trainer 和 scenario 配置选择训练员和剧本
    match game_config.trainer.as_str() {
        "random" => {
            let trainer = RandomTrainer;
            match game_config.scenario.as_str() {
                "onsen" => run_onsen(&trainer, game_config.uma, &game_config.cards, inherit, &mut rng)?,
                _ => run_basic(&trainer, game_config.uma, &game_config.cards, inherit, &mut rng)?
            }
        }
        _ => {
            // 默认使用手动训练员
            let trainer = ManualTrainer;
            match game_config.scenario.as_str() {
                "onsen" => run_onsen(&trainer, game_config.uma, &game_config.cards, inherit, &mut rng)?,
                _ => run_basic(&trainer, game_config.uma, &game_config.cards, inherit, &mut rng)?
            }
        }
    }

    // 输出耗时（log_level=off 时也可见）
    println!("耗时: {:?}", start.elapsed());

    Ok(())
}
