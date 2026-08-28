//! 第三年“先判断吃面、再判断体力”同种子配对实验。
//!
//! 两个候选只改变第三年体力门：旧基准为 `[30, 30, 0]`；新候选为
//! `[30, 30, 30]`，但第三年已经吃面时跳过体力硬门。地区、训练评分、友人节奏及
//! 其他正式 preset 字段保持一致。

use std::{env, path::Path};

use anyhow::Result;
use umasim::{
    bench::{self, CardPickOpts, DeckComposition},
    game::{Game, InheritInfo, ramen::RamenGame},
    gamedata::init_global_with_config,
    trainer::RecommendedRamenTrainer,
    utils::{get_workspace_root, load_game_config}
};

/// 配对实验基础随机种子。
const BASE_SEED: u64 = 1_066_000;
/// 实验马娘。
const UMA: u32 = 102601;
/// 拉面杯新友人满破 idrank。
const FRIEND: u32 = 303054;
/// 与当前 3速1耐1智专项矩阵一致的继承配置。
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [10, 10, 20, 20, 20, 40]
};

/// 根据方案名构造唯一发生体力门差异的训练员。
fn trainer(name: &str) -> Result<RecommendedRamenTrainer> {
    match name {
        "旧版第三年无门" => Ok(RecommendedRamenTrainer::y3_vital_legacy_baseline()),
        "先吃面不判体力_不吃门30" => Ok(RecommendedRamenTrainer::new()),
        _ => anyhow::bail!("未知第三年体力方案: {name}")
    }
}

/// 构造与既有第三年专项实验一致的 3速1耐1智卡组。
fn deck() -> Result<[u32; 6]> {
    let composition = DeckComposition {
        counts: [3, 1, 0, 0, 1],
        name: String::new()
    };
    let representatives = bench::select_representatives(&CardPickOpts::default())?;
    composition.build_deck(&representatives.picked, FRIEND)
}

/// 执行当前分片并写出逐局 CSV；各方案共享 `BASE_SEED + 局序号`，可直接配对。
fn main() -> Result<()> {
    std::env::set_current_dir(get_workspace_root()?)?;
    init_global_with_config(&load_game_config()?)?;

    let name = env::var("方案")?;
    let shard: u64 = env::var("分片序号")?.parse()?;
    let runs: u64 = env::var("每分片局数")?.parse()?;
    let deck = deck()?;
    let mut rows = Vec::with_capacity(runs as usize);

    for offset in 0..runs {
        let run_index = shard * runs + offset;
        let (mut rng, rule_master) = bench::seeded_rngs(BASE_SEED, run_index);
        let mut game = RamenGame::newgame(UMA, &deck, INHERIT.clone())?;
        game.set_rule_master(rule_master);
        game.run_full_game(&trainer(&name)?, &mut rng)?;
        rows.push(vec![
            name.clone(),
            run_index.to_string(),
            game.uma.calc_score().to_string(),
            game.uma.skill_pt.to_string(),
            game.ramen.scenario_pt.to_string(),
            game.ramen.rmj_results.iter().filter(|&&ok| ok).count().to_string(),
            game.uma.five_status[0].to_string(),
            game.uma.five_status[1].to_string(),
            game.uma.five_status[2].to_string(),
            game.uma.five_status[3].to_string(),
            game.uma.five_status[4].to_string()
        ]);
    }

    bench::write_csv(
        Path::new("第三年先吃面体力实验.csv"),
        &[
            "方案",
            "局序号",
            "总分",
            "技能点",
            "第三年结束PT",
            "RMJ成功年数",
            "速度",
            "耐力",
            "力量",
            "根性",
            "智力"
        ],
        &rows
    )
}
