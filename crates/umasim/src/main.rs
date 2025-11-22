//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran
use anyhow::Result;
use log::info;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    game::{Game, InheritInfo, basic::BasicGame},
    gamedata::{GAMECONSTANTS, GameConfig, init_global},
    trainer::*,
    utils::init_logger
};

pub mod explain;
pub mod game;
pub mod gamedata;
pub mod trainer;
pub mod utils;

#[tokio::main]
async fn main() -> Result<()> {
    init_logger()?;
    init_global()?;
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let game_config: GameConfig = toml::from_str(&config_file)?;
    let mut game = BasicGame::newgame(game_config.uma as u32, &game_config.cards, InheritInfo {
        blue_count: game_config.blue_count.clone(),
        extra_count: game_config.extra_count.clone()
    })?;
    println!("{}", game.explain()?);
    let score = game.uma.calc_score();
    println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);
    let trainer = ManualTrainer {};
    let mut rng = StdRng::from_os_rng();
    game.run_full_game(&trainer, &mut rng)?;
    info!("育成结束！");
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
