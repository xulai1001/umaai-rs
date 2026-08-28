//! 目标配卡的正式 preset 精确隔离专项矩阵。

use std::{env, fs::File, io::Write};

use anyhow::{Context, Result, ensure};
use umasim::{
    bench::{self, CardPickOpts, DeckComposition},
    gamedata::{GAMECONSTANTS, init_global_with_config},
    global,
    trainer::{LoggingTrainer, RecommendedRamenTrainer},
    utils::{get_workspace_root, load_game_config}
};

const BASE_SEED: u64 = 995_100;
const UMA: u32 = 102601;
const FRIEND: u32 = 303054;
const INHERIT: umasim::game::InheritInfo = umasim::game::InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [10, 10, 20, 20, 20, 40]
};

fn env_f32(name: &str, default: f32) -> Result<f32> {
    Ok(env::var(name).map_or(Ok(default), |value| value.parse::<f32>().with_context(|| format!("{name} 不是数字")))?)
}

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

fn status_score(status: &[i32; 5]) -> i32 {
    let constants = global!(GAMECONSTANTS);
    status
        .iter()
        .map(|&value| constants.five_status_final_score[(value.max(0) as usize).min(constants.five_status_final_score.len() - 1)])
        .sum()
}

fn main() -> Result<()> {
    let variant = env::var("方案标识")?;
    let composition_index: usize = env::var("配卡索引")?.parse()?;
    let gap = env_f32("属性缺口权重百分比", 0.0)? / 100.0;
    let overflow = env_f32("属性溢出权重百分比", 0.0)? / 100.0;
    let pt = [env_f32("第一年技能点权重", 16.0)?, env_f32("第二年技能点权重", 64.0)?, env_f32("第三年技能点权重", 64.0)?];
    let sacrifice = env_f32("长期结构最大让分", 140.0)?;
    let window = env_f32("吃面对盘权重百分比", 10.0)? / 100.0;
    let reserve = env_f32("属性空间预留", 40.0)?;
    let bond = env_f32("早期羁绊每点价值", 8.0)?;
    let hint = env_f32("诀窍提示价值", 6.0)?;
    let weakboost = env_f32("弱训练加成", 0.0)?;
    let region_weak_cover = env_f32("区域弱覆盖权重", 0.0)?;
    let eat_requires_covered = env::var("吃面要求覆盖训练").map(|v| v == "1" || v == "true").unwrap_or(false);
    let shard: u64 = env::var("分片序号")?.parse()?;
    let runs: u64 = env::var("每分片局数")?.parse()?;

    std::env::set_current_dir(get_workspace_root()?)?;
    init_global_with_config(&load_game_config()?)?;
    let composition = composition(composition_index)?;
    let reps = bench::select_representatives(&CardPickOpts::default())?;
    let deck = composition.build_deck(&reps.picked, FRIEND)?;
    let mut file = File::create("专项矩阵结果.csv")?;
    writeln!(file, "方案标识,配卡索引,配卡,属性缺口权重,属性溢出权重,第一年技能点权重,第二年技能点权重,第三年技能点权重,长期结构最大让分,吃面对盘权重,属性空间预留,早期羁绊每点价值,诀窍提示价值,分片序号,局序号,基线总分,候选总分,基线技能点,候选技能点,基线属性分,候选属性分,完全一致")?;

    for offset in 0..runs {
        let run_index = shard * runs + offset;
        let base_trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), run_index);
        let candidate_trainer = LoggingTrainer::new(RecommendedRamenTrainer::with_experiment_overrides(pt, gap, overflow, sacrifice, window, reserve, bond, hint, weakboost, region_weak_cover, eat_requires_covered), run_index);
        let base = bench::run_seeded(UMA, &deck, &INHERIT, BASE_SEED, run_index, &base_trainer)?;
        let candidate = bench::run_seeded(UMA, &deck, &INHERIT, BASE_SEED, run_index, &candidate_trainer)?;
        let identical = base.score == candidate.score && base.skill_pt == candidate.skill_pt && base.five_status == candidate.five_status;
        writeln!(file, "{variant},{composition_index},{},{gap},{overflow},{},{},{},{sacrifice},{window},{reserve},{bond},{hint},{shard},{run_index},{},{},{},{},{},{},{identical}", composition.name(), pt[0], pt[1], pt[2], base.score, candidate.score, base.skill_pt, candidate.skill_pt, status_score(&base.five_status), status_score(&candidate.five_status))?;
    }
    ensure!(runs > 0, "每分片局数必须大于0");
    Ok(())
}
