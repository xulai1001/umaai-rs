//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran

use anyhow::{Result, anyhow};
use inquire::Select;
use log::{info, warn};
use rand::{SeedableRng, rngs::StdRng};
use umaai::protocol::{GameStatus, GameStatusOnsen, onsen::serialize_game};
use umasim::{
    game::{Game, InheritInfo, onsen::game::OnsenGame},
    gamedata::{GAMECONSTANTS, GameConfig, init_global},
    global,
    search::SearchConfig,
    trainer::*,
    utils::init_logger
};
/// 单次模拟结果
struct SimulationResult {
    score: i32,
    pt: i32,
    explain: String
}

/// 运行 OnsenGame（单次），返回模拟结果
fn run_onsen_once(
    trainer_mcts: MctsTrainer, uma: u32, cards: &[u32; 6], inherit: InheritInfo, load_game: Option<OnsenGame>
) -> Result<SimulationResult> {
    let trainer_hand = HandwrittenTrainer::new().verbose(true);
    let mut rng = StdRng::from_os_rng();
    if let Some(g) = &load_game {
        info!("---- 载入游戏状态 ----");
        println!("{}", g.explain_distribution()?);
        info!("---------------------");
    }
    let mut game = load_game.unwrap_or(OnsenGame::newgame(uma, cards, inherit)?);
    // 使用2个trainer完成一局游戏
    loop {
        let game_for_save = game.clone();
        let mut selected = false; // 表示这阶段是否有选择
        if game.pending_selection || !game.list_actions().unwrap_or_default().is_empty() {
            let mut game2 = game.clone();
            game2.run_stage(&trainer_hand, &mut rng)?;
            selected = true;
        }
        game.run_stage(&trainer_mcts, &mut rng)?;
        if selected {
            let select = Select::new("选择一个选项", vec!["继续", "保存本回合", "退出"]).prompt()?;
            match select {
                "保存本回合" => {
                    let path = format!("logs/turn{}.json", game_for_save.turn);
                    let json = serialize_game(&game)?;
                    fs_err::write(&path, &json)?;
                    let mcts_path = format!("logs/search_turn{}.json", game_for_save.turn);
                    let mcts_result = {
                        let output = trainer_mcts.search_output
                            .lock()
                            .map_err(|_| anyhow!("lock failed"))?;
                        output.to_scores()
                    };
                    let json = serde_json::to_string_pretty(&mcts_result)?;
                    fs_err::write(&mcts_path, &json)?;

                    warn!("已保存回合信息 -> {path}, 蒙特卡洛结果 -> {mcts_path}");
                }
                "退出" => {
                    break;
                }
                _ => {}
            }
        }
        if !game.next() {
            break;
        }
    }
    game.on_simulation_end(&trainer_mcts, &mut rng)?;

    info!("育成结束！");

    let score = game.uma.calc_score();
    let pt = game.uma.skill_pt;
    let explain = game.explain()?;

    Ok(SimulationResult { score, pt, explain })
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. 先读取配置文件
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let game_config: GameConfig = toml::from_str(&config_file)?;

    // 2. 根据配置初始化日志
    init_logger("analyzer", &game_config.log_level)?;

    // 3. 再初始化全局数据
    init_global()?;

    // 如果指定了文件名，则载入游戏
    let mut load_game = None;
    if let Some(filename) = std::env::args().nth(1) {
        info!("载入 {filename} ...");
        let contents = fs_err::read_to_string(&filename)?;
        let status: GameStatusOnsen = serde_json::from_str(&contents)?;
        load_game = Some(status.into_game()?);
    }
    //let simulation_count = game_config.simulation_count.max(1);
    //let simulation_count = 1;
    // 开始计时
    //let start = Instant::now();

    let inherit = InheritInfo {
        blue_count: game_config.blue_count.clone(),
        extra_count: game_config.extra_count.clone()
    };

    let search_config = SearchConfig::default()
        .with_search_n(game_config.mcts.search_n)
        .with_radical_factor_max(game_config.mcts.radical_factor_max)
        .with_max_depth(game_config.mcts.max_depth)
        .with_policy_delta(game_config.mcts.policy_delta)
        // UCB 参数
        .with_ucb(game_config.mcts.use_ucb)
        .with_search_group_size(game_config.mcts.search_group_size)
        .with_search_cpuct(game_config.mcts.search_cpuct)
        .with_expected_search_stdev(game_config.mcts.expected_search_stdev);
    info!("search_config = {search_config:?}");
    let mut trainer_mcts = MctsTrainer::new(search_config).verbose(true);
    trainer_mcts.mcts_onsen = game_config.mcts_selected_onsen;
    trainer_mcts.mcts_selection = game_config.mcts_selection.clone();

    let sim_result = run_onsen_once(trainer_mcts, game_config.uma, &game_config.cards, inherit, load_game)?;

    println!("{}", sim_result.explain);
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(sim_result.score),
        sim_result.score,
        sim_result.pt
    );
    Ok(())
}
