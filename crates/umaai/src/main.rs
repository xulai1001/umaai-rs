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
        Game, InheritInfo, Trainer, onsen::{OnsenTurnStage, action::OnsenAction, game::OnsenGame}
    },
    gamedata::{GameConfig, MctsConfig, init_global},
    neural::{Evaluator, NeuralNetEvaluator},
    search::SearchConfig,
    trainer::{MctsTrainer, NeuralNetTrainer},
    utils::{check_windows_terminal, check_working_dir, init_logger, pause}
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

/// 实际的主函数
async fn main_guard() -> Result<()> {
    println!("{}", to_art("UMAAI 0.1".to_string(), "small", 0, 1, 0).expect("here"));
    // 0. 运行前检查
    check_windows_terminal()?;
    if !fs_err::exists("game_config.toml")? {
        check_working_dir()?;
    }
    // 1. 先读取配置文件
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let game_config: GameConfig = toml::from_str(&config_file)?;
    let mcts_config = SearchConfig::new_game_config(&game_config);
    // 2. 根据配置初始化日志
    init_logger("umaai", &game_config.log_level)?;
    //info!("search_config = {mcts_config:?}");

    // 3. 再初始化全局数据
    init_global()?;

    let mut rng = StdRng::from_os_rng();

    // 神经网络训练员
    //let model_path = "saved_models/onsen_v1/model.onnx";
    //let evaluator =
    //NeuralNetEvaluator::load(model_path).map_err(|e| anyhow!("错误: 无法加载神经网络模型 {model_path}: {e:?}"))?;

    // MCTS训练员
    let mut trainer = MctsTrainer::new(mcts_config).verbose(true);
    trainer.mcts_onsen = game_config.mcts_selected_onsen;

    // 开始检测文件
    let mut watcher = UraFileWatcher::init()?;
    loop {
        let contents = watcher.watch("thisTurn.json")?;
        match parse_game::<GameStatusOnsen>(&contents) {
            Ok(mut game) => {
                //println!("{game:#?}");
                if game.turn <= 1 {
                    // 直接模拟一局看得分，或者输出模拟参数
                    let deck = game.deck.iter()
                        .map(|card| card.card_id * 10 + card.rank)
                        .collect::<Vec<_>>();
                    let inherit = InheritInfo {
                        blue_count: game.inherit.blue_count.clone(),
                        extra_count: game.inherit.extra_count.clone()
                    };
                    info!("- sim 模拟参数: {} {deck:?}, {inherit:?}", game.uma.uma_id);
                }
                //-------------------------------
                println!("{}", game.explain_distribution()?);
                println!("正在计算...");
                if game.pending_selection {
                    // 是温泉选择状态
                    let actions = game.list_actions_onsen_select();
                    let onsen = trainer.select_action(&game, &actions, &mut rng)?;
                    println!("{}", format!("蒙特卡洛：{}", actions[onsen]).magenta());
                    // 前进一步选择升级
                    game.apply_action(&actions[onsen], &mut rng)?;
                    let upgradeable = game.get_upgradeable_equipment();
                    if !upgradeable.is_empty() {
                        let actions = upgradeable
                            .iter()
                            .map(|x| OnsenAction::Upgrade(*x as i32))
                            .collect::<Vec<_>>();
                        let upgrade = trainer.select_action(&game, &actions, &mut rng)?;
                        println!("{}", format!("蒙特卡洛：{}", actions[upgrade]).magenta());
                    }
                } else {
                    // 如果被解析成 Bathing 但没有温泉券合buff，就直接跳过到 Train
                    if game.stage == OnsenTurnStage::Bathing
                        && game.bathing.ticket_num == 0
                        && game.bathing.buff_remain_turn == 0
                    {
                        game.next();
                    }

                    let actions = game.list_actions()?;
                    if actions.is_empty() {
                        continue;
                    }

                    let action_idx = trainer.select_action(&game, &actions, &mut rng)?;
                    let action = actions[action_idx].clone();
                    println!("{}", format!("蒙特卡洛: {action}").bright_green());

                    // 当 mcts 建议 UseTicket(false) 时，直接跳过 Bathing 阶段，继续给出训练推荐。
                    if action == OnsenAction::UseTicket(false) && game.stage == OnsenTurnStage::Bathing {
                        game.next();
                        let actions = game.list_actions()?;
                        if !actions.is_empty() {
                            let action_idx = trainer.select_action(&game, &actions, &mut rng)?;
                            let action = actions[action_idx].clone();
                            println!("{}", format!("蒙特卡洛: {action}").bright_green());
                        }
                    }
                }
            }
            Err(e) => {
                println!("{}", format!("解析回合信息出错: {e}").red());
                println!("----------");
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    match main_guard().await {
        Ok(_) => {}
        Err(e) => {
            println!("{}", "UmaAI 出现错误，即将退出:".red());
            println!("{}", "-----------------------------------".red());
            println!("{}", format!("{e:?}").red());
            pause().expect("pause");
        }
    }
    Ok(())
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
        init_logger("test", "info")?;

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
