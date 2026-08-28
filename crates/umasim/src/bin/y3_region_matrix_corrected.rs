//! 第三年拉面地区组合全局评分矩阵。
//! 每个第三年组合都用生产 MCTS 跑完整育成，只把第三年 RegionSelect 强制为当前组合。

use std::{env, path::Path};
use anyhow::{Result, bail};
use rand::prelude::StdRng;
use serde::Serialize;
use umasim::{
    bench,
    game::{Game, Trainer, InheritInfo, ramen::{Operation, RamenAction, RamenGame, rules::get_region_combinations}},
    gamedata::{EventChoice, EventData, init_global_with_config},
    search::SearchConfig,
    trainer::{RamenMctsTrainer, RamenSearchStages},
    utils::{get_workspace_root, load_game_config},
};

// 与拉面 MCTS 固定种子基线保持一致，不读取当前 game_config.toml 的临时 101901 覆盖。
const UMA: u32 = 102601;
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 3, 0, 0, 0],
    extra_count: [0, 30, 0, 0, 30, 30],
};
const BASE_SEED: u64 = 2_026_082_500;

#[derive(Serialize)]
struct Row {
    combo_index: usize,
    region_ids: String,
    run: usize,
    score: i32,
    skill_pt: i32,
    scenario_pt: i32,
    rmj_success: usize,
    speed: i32,
    stamina: i32,
    power: i32,
    guts: i32,
    wisdom: i32,
}

struct FixedY3 {
    inner: RamenMctsTrainer,
    combo: [usize; 3],
}

impl Trainer<RamenGame> for FixedY3 {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        if game.turn() == 47 && actions.iter().all(|a| matches!(a.operation, Operation::RegionSelect(_))) {
            return actions
                .iter()
                .position(|a| a.operation == Operation::RegionSelect(self.combo))
                .ok_or_else(|| anyhow::anyhow!("第三年组合 {:?} 不在候选集中", self.combo));
        }
        self.inner.select_action(game, actions, rng)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.inner.select_choice(game, choices, rng)
    }

    fn select_event_choice(&self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.inner.select_event_choice(game, event, choices, rng)
    }

    fn last_breakdown(&self) -> Option<String> {
        self.inner.last_breakdown()
    }
}

fn main() -> Result<()> {
    env::set_current_dir(get_workspace_root()?)?;
    let config = load_game_config()?;
    init_global_with_config(&config)?;
    let runs: usize = env::var("每组合局数").unwrap_or_else(|_| "10".into()).parse()?;
    let start: usize = env::var("起始组合").unwrap_or_else(|_| "0".into()).parse()?;
    let end: usize = env::var("结束组合").unwrap_or_else(|_| "120".into()).parse()?;
    let combos = get_region_combinations(2)?;
    if end > combos.len() || start >= end {
        bail!("组合范围无效: {start}..{end}");
    }

    let mut writer = csv::Writer::from_path(Path::new("第三年拉面组合评分.csv"))?;
    for (offset, combo) in combos[start..end].iter().enumerate() {
        let combo_index = start + offset;
        for run in 0..runs {
            let (mut rng, rule_master) = bench::seeded_rngs(BASE_SEED, (combo_index * runs + run) as u64);
            let mut game = RamenGame::newgame(UMA, &DECK, INHERIT)?;
            game.set_rule_master(rule_master);
            let search = SearchConfig::new_game_config(&config);
            let trainer = FixedY3 {
                inner: RamenMctsTrainer::new(search)
                    .with_stages(RamenSearchStages::parse("train,ramen")?),
                combo: *combo,
            };
            game.run_full_game(&trainer, &mut rng)?;
            let status = game.uma.five_status;
            writer.serialize(Row {
                combo_index,
                region_ids: combo.iter().map(usize::to_string).collect::<Vec<_>>().join("/"),
                run,
                score: game.uma.calc_score(),
                skill_pt: game.uma.skill_pt,
                scenario_pt: game.ramen.scenario_pt,
                rmj_success: game.ramen.rmj_results.iter().filter(|&&ok| ok).count(),
                speed: status[0], stamina: status[1], power: status[2], guts: status[3], wisdom: status[4],
            })?;
        }
        writer.flush()?;
        println!("组合 {combo_index:03} {:?} 完成 {runs} 局", combo);
    }
    Ok(())
}
