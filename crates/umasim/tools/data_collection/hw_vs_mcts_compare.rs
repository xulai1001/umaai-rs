//! 手写 vs MCTS 对比工具：同 seed 跑两种策略，输出评分对比。
//!
//! 用法（环境变量控制）：
//! - `配卡索引`: 81 / 97 / 76
//! - `搜索次数`: MCTS search_n，默认 1024
//! - `每策略局数`: 默认 10

use std::{env, fs::File, io::Write};

use anyhow::{Context, Result, ensure};
use umasim::{
    bench::{self, CardPickOpts, DeckComposition},
    gamedata::{GAMECONSTANTS, init_global_with_config},
    global,
    search::SearchConfig,
    trainer::{LoggingTrainer, RecommendedRamenTrainer, RamenMctsTrainer},
    utils::{get_workspace_root, load_game_config}
};

const BASE_SEED: u64 = 995_100;
const UMA: u32 = 102601;
const FRIEND: u32 = 303054;
const INHERIT: umasim::game::InheritInfo = umasim::game::InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [10, 10, 20, 20, 20, 40]
};

fn composition(index: usize) -> Result<DeckComposition> {
    let mut all = Vec::new();
    for speed in 0..=3 {
        for stamina in 0..=3 {
            for power in 0..=3 {
                for guts in 0..=3 {
                    for wisdom in 0..=3 {
                        let counts = [speed, stamina, power, guts, wisdom];
                        if counts.iter().sum::<usize>() == 5 {
                            all.push(DeckComposition { counts, name: String::new() });
                        }
                    }
                }
            }
        }
    }
    all.get(index).cloned().with_context(|| format!("配卡索引越界: {index}"))
}

fn main() -> Result<()> {
    let composition_index: usize = env::var("COMP_IDX")?.parse()?;
    let search_n: usize = env::var("SEARCH_N").map_or(Ok(1024usize), |v| v.parse())?;
    let runs: u64 = env::var("RUNS").map_or(Ok(10u64), |v| v.parse())?;

    std::env::set_current_dir(get_workspace_root()?)?;
    init_global_with_config(&load_game_config()?)?;
    let composition = composition(composition_index)?;
    let reps = bench::select_representatives(&CardPickOpts::default())?;
    let deck = composition.build_deck(&reps.picked, FRIEND)?;

    let mut file = File::create("hw_vs_mcts.csv")?;
    writeln!(file, "comp_idx,comp_name,run,hw_score,hw_pt,hw_status,hw_spd,hw_sta,hw_pow,hw_gut,hw_wis,hw_eat_y3,hw_rmj,mcts_score,mcts_pt,mcts_status,mcts_spd,mcts_sta,mcts_pow,mcts_gut,mcts_wis,mcts_eat_y3,mcts_rmj,diff,mcts_win")?;

    let mut hw_scores = Vec::new();
    let mut mcts_scores = Vec::new();
    let mut hw_status_sum = [0i64; 5];
    let mut mcts_status_sum = [0i64; 5];
    let mut hw_eat_y3_sum = 0i64;
    let mut mcts_eat_y3_sum = 0i64;
    let mut hw_rmj_sum = 0i64;
    let mut mcts_rmj_sum = 0i64;

    for run_index in 0..runs {
        // 手写策略
        let hw_trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), run_index);
        let hw = bench::run_seeded(UMA, &deck, &INHERIT, BASE_SEED, run_index, &hw_trainer)?;

        // MCTS 策略
        let mcts_trainer = LoggingTrainer::new(
            RamenMctsTrainer::new(SearchConfig::default().with_search_n(search_n)),
            run_index
        );
        let mcts = bench::run_seeded(UMA, &deck, &INHERIT, BASE_SEED, run_index, &mcts_trainer)?;

        let diff = mcts.score - hw.score;
        let mcts_win = mcts.score > hw.score;

        writeln!(
            file,
            "{composition_index},{comp_name},{run_index},{hw_score},{hw_pt},{hw_st},{hw_spd},{hw_sta},{hw_pow},{hw_gut},{hw_wis},{hw_eat},{hw_rmj},{mcts_score},{mcts_pt},{mcts_st},{mcts_spd},{mcts_sta},{mcts_pow},{mcts_gut},{mcts_wis},{mcts_eat},{mcts_rmj},{diff},{mcts_win}",
            comp_name = composition.name(),
            hw_score = hw.score, hw_pt = hw.skill_pt, hw_st = status_score(&hw.five_status),
            hw_spd = hw.five_status[0], hw_sta = hw.five_status[1], hw_pow = hw.five_status[2], hw_gut = hw.five_status[3], hw_wis = hw.five_status[4],
            hw_eat = hw.yearly_eat_count[2], hw_rmj = hw.rmj_ok,
            mcts_score = mcts.score, mcts_pt = mcts.skill_pt, mcts_st = status_score(&mcts.five_status),
            mcts_spd = mcts.five_status[0], mcts_sta = mcts.five_status[1], mcts_pow = mcts.five_status[2], mcts_gut = mcts.five_status[3], mcts_wis = mcts.five_status[4],
            mcts_eat = mcts.yearly_eat_count[2], mcts_rmj = mcts.rmj_ok,
            diff = diff,
            mcts_win = if mcts_win { "yes" } else { "no" }
        )?;

        hw_scores.push(hw.score);
        mcts_scores.push(mcts.score);
        for i in 0..5 {
            hw_status_sum[i] += hw.five_status[i] as i64;
            mcts_status_sum[i] += mcts.five_status[i] as i64;
        }
        hw_eat_y3_sum += hw.yearly_eat_count[2] as i64;
        mcts_eat_y3_sum += mcts.yearly_eat_count[2] as i64;
        hw_rmj_sum += hw.rmj_ok as i64;
        mcts_rmj_sum += mcts.rmj_ok as i64;

        println!(
            "run{run_index}: HW={} MCTS={} diff={:+} {} | HW[spd,sta,pow,gut,wis]={:?} eat_y3={} rmj={} | MCTS[...]={:?} eat_y3={} rmj={}",
            hw.score, mcts.score, diff,
            if mcts_win { "MCTS+" } else if mcts.score < hw.score { "HW+" } else { "=" },
            hw.five_status, hw.yearly_eat_count[2], hw.rmj_ok,
            mcts.five_status, mcts.yearly_eat_count[2], mcts.rmj_ok
        );
    }

    let n = runs as i64;
    let hw_avg = hw_scores.iter().sum::<i32>() as f64 / runs as f64;
    let mcts_avg = mcts_scores.iter().sum::<i32>() as f64 / runs as f64;
    let hw_wins = hw_scores.iter().zip(mcts_scores.iter()).filter(|(h, m)| h > m).count();
    let mcts_wins = hw_scores.iter().zip(mcts_scores.iter()).filter(|(h, m)| m > h).count();

    println!("\n=== Summary (comp={} search_n={} runs={}) ===", composition_index, search_n, runs);
    println!("HW   avg={:.0} (min={} max={})", hw_avg, hw_scores.iter().min().unwrap(), hw_scores.iter().max().unwrap());
    println!("MCTS avg={:.0} (min={} max={})", mcts_avg, mcts_scores.iter().min().unwrap(), mcts_scores.iter().max().unwrap());
    println!("Diff: {:+.0}  MCTS wins={}/{} HW wins={}/{}", mcts_avg - hw_avg, mcts_wins, runs, hw_wins, runs);
    println!("\n=== Five-status avg diff (MCTS - HW) ===");
    let names = ["spd", "sta", "pow", "gut", "wis"];
    for i in 0..5 {
        let hw_avg_s = hw_status_sum[i] as f64 / n as f64;
        let mcts_avg_s = mcts_status_sum[i] as f64 / n as f64;
        println!("  {:>3}: HW={:.0} MCTS={:.0} diff={:+.0}", names[i], hw_avg_s, mcts_avg_s, mcts_avg_s - hw_avg_s);
    }
    println!("\n=== Y3 eat count avg: HW={:.1} MCTS={:.1} diff={:+.1}", hw_eat_y3_sum as f64 / n as f64, mcts_eat_y3_sum as f64 / n as f64, (mcts_eat_y3_sum - hw_eat_y3_sum) as f64 / n as f64);
    println!("=== RMJ ok avg: HW={:.2} MCTS={:.2} diff={:+.2}", hw_rmj_sum as f64 / n as f64, mcts_rmj_sum as f64 / n as f64, (mcts_rmj_sum - hw_rmj_sum) as f64 / n as f64);

    ensure!(runs > 0, "runs must > 0");
    Ok(())
}

fn status_score(status: &[i32; 5]) -> i32 {
    let constants = global!(GAMECONSTANTS);
    status
        .iter()
        .map(|&value| constants.five_status_final_score[(value.max(0) as usize).min(constants.five_status_final_score.len() - 1)])
        .sum()
}
