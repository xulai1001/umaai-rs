//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran
use std::time::Instant;

use anyhow::{Result, anyhow};
use colored::Colorize;
use log::info;
use rand::{SeedableRng, rngs::StdRng};
use text_to_ascii_art::to_art;
use umasim::{
    game::{
        Game,
        onsen::{OnsenTurnStage, game::OnsenGame}
    },
    gamedata::{GameConfig, init_global},
    neural::{Evaluator, NeuralNetEvaluator},
    trainer::NeuralNetTrainer,
    utils::init_logger
};

use crate::protocol::{
    GameStatusOnsen,
    urafile::{UraFileWatcher, parse_game}
};

pub mod protocol;

pub fn run_evaluate<G, E>(game: &G, evaluator: &E, rng: &mut StdRng) -> Result<()>
where
    G: Game,
    E: Evaluator<G>
{
    let t = Instant::now();
    let score = evaluator.evaluate(&game);
    if let Some(action) = evaluator.select_action(&game, rng) {
        info!(
            "{}",
            format!(
                "AI选择: {action}, 均分: {}, 标准差: {}, Time: {:?}",
                score.score_mean as i64,
                score.score_stdev as i64,
                t.elapsed()
            )
            .bright_green()
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("{}", to_art("UMAAI 0.1".to_string(), "small", 0, 1, 0).expect("here"));
    // 1. 先读取配置文件
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let game_config: GameConfig = toml::from_str(&config_file)?;

    // 2. 根据配置初始化日志
    init_logger(&game_config.log_level)?;

    // 3. 再初始化全局数据
    init_global()?;

    let mut rng = StdRng::from_os_rng();

    // 神经网络训练员
    let model_path = "saved_models/onsen_v1/model.onnx";
    let evaluator =
        NeuralNetEvaluator::load(model_path).map_err(|e| anyhow!("错误: 无法加载神经网络模型 {model_path}: {e:?}"))?;

    // 开始检测文件
    let mut watcher = UraFileWatcher::init()?;
    loop {
        let contents = watcher.watch("thisTurn.json")?;
        match parse_game::<GameStatusOnsen>(&contents) {
            Ok(game) => {
                info!("{}", game.explain_distribution()?);
                run_evaluate(&game, &evaluator, &mut rng)?;
            }
            Err(e) => {
                println!("{}", format!("解析回合信息出错: {e}").red());
                println!("----------");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{env, path::Path, sync::mpsc};

    use anyhow::Result;
    use colored::Colorize;
    use log::info;
    use notify::{Event, RecursiveMode, Watcher};
    use umasim::{
        game::Game,
        gamedata::{GameConfig, init_global},
        utils::init_logger
    };

    use crate::protocol::{
        GameStatusOnsen,
        urafile::{UraFileWatcher, parse_game}
    };

    #[tokio::test]
    async fn test_watch() -> Result<()> {
        let local_app_path = env::var("LOCALAPPDATA")?;
        let urafile_path = format!("{local_app_path}/UmamusumeResponseAnalyzer/PluginData/SendGameStatusPlugin/");

        let (tx, rx) = mpsc::channel::<notify::Result<Event>>();
        let mut watcher = notify::recommended_watcher(tx)?;
        println!("{urafile_path}");
        watcher.watch(Path::new(&urafile_path), RecursiveMode::NonRecursive)?;
        loop {
            let event = rx.recv()??;
            println!("{event:?}");
        }
    }

    #[test]
    fn test_urafile() -> Result<()> {
        // 2. 根据配置初始化日志
        init_logger("info")?;

        // 3. 再初始化全局数据
        init_global()?;
        let mut watcher = UraFileWatcher::init()?;
        loop {
            let contents = watcher.watch("thisTurn.json")?;
            match parse_game::<GameStatusOnsen>(&contents) {
                Ok(game) => {
                    info!("{}", game.explain_distribution()?);
                    println!("----------");
                }
                Err(e) => {
                    println!("{}", format!("解析回合信息出错: {e}").red());
                    println!("----------");
                }
            }
        }
    }
}
