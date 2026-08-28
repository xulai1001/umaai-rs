//! 剩余矿脉配对矩阵：同 seed A/B 验证 v44 交接文档第 6 节列出的未扫旋钮。
//!
//! 每个旋钮的每个候选值，跑 N 局同 seed 配对（baseline `new()` vs
//! `with_mine_overrides`），记录配对胜/平/负、均分差、属性分差、PT 差。
//!
//! 用法（环境变量控制）：
//! - `旋钮`: cap_discount / cook2_stock / y3_hard_floor / friend_proactive /
//!   friend_starve / y3_pre_vital / y3_shortfall
//! - `候选值`: 逗号分隔的数值列表，如 `0.0,0.5,1.0,1.5`
//! - `配卡索引`: 81 / 97 / 76（2速1耐2智 / 3速1耐1智 / 2速1力1根1智）
//! - `每候选局数`: 默认 30
//!
//! 输出 `挖矿配对结果.csv`，每行一局配对。

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

fn parse_f32_list(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|x| x.trim().parse::<f32>().with_context(|| format!("无法解析浮点数: {x}")))
        .collect()
}

fn parse_i32_list(s: &str) -> Result<Vec<i32>> {
    s.split(',')
        .map(|x| x.trim().parse::<i32>().with_context(|| format!("无法解析整数: {x}")))
        .collect()
}

fn main() -> Result<()> {
    let knob_name = env::var("旋钮")?;
    let candidates_str = env::var("候选值")?;
    let composition_index: usize = env::var("配卡索引")?.parse()?;
    let runs: u64 = env::var("每候选局数").map_or(Ok(30u64), |v| v.parse())?;

    // 第二旋钮（可选）：用于双旋钮联合扫，如 friend_proactive × friend_starve
    let knob2_name = env::var("旋钮2").ok();
    let candidates2_str = env::var("候选值2").ok();

    std::env::set_current_dir(get_workspace_root()?)?;
    init_global_with_config(&load_game_config()?)?;
    let composition = composition(composition_index)?;
    let reps = bench::select_representatives(&CardPickOpts::default())?;
    let deck = composition.build_deck(&reps.picked, FRIEND)?;

    // 判断旋钮类型（浮点 or 整数）
    let is_int_knob = matches!(knob_name.as_str(), "y3_hard_floor" | "y3_pre_vital");
    let float_vals = if is_int_knob { Vec::new() } else { parse_f32_list(&candidates_str)? };
    let int_vals = if is_int_knob { parse_i32_list(&candidates_str)? } else { Vec::new() };

    // 第二旋钮解析
    let is_int_knob2 = knob2_name.as_deref().is_some_and(|k| matches!(k, "y3_hard_floor" | "y3_pre_vital"));
    let float_vals2: Vec<f32> = if let (Some(k2), Some(c2)) = (knob2_name.as_ref(), candidates2_str.as_ref()) {
        if k2 == "y3_hard_floor" || k2 == "y3_pre_vital" { Vec::new() } else { parse_f32_list(c2)? }
    } else { Vec::new() };
    let int_vals2: Vec<i32> = if let (Some(k2), Some(c2)) = (knob2_name.as_ref(), candidates2_str.as_ref()) {
        if k2 == "y3_hard_floor" || k2 == "y3_pre_vital" { parse_i32_list(c2)? } else { Vec::new() }
    } else { Vec::new() };

    let mut file = File::create("挖矿配对结果.csv")?;
    if knob2_name.is_some() {
        writeln!(
            file,
            "旋钮1,候选值1,旋钮2,候选值2,配卡索引,配卡名,局序号,基线总分,候选总分,基线技能点,候选技能点,基线属性分,候选属性分,配对结果"
        )?;
    } else {
        writeln!(
            file,
            "旋钮,候选值,配卡索引,配卡名,局序号,基线总分,候选总分,基线技能点,候选技能点,基线属性分,候选属性分,配对结果"
        )?;
    }

    let candidate_count = if is_int_knob { int_vals.len() } else { float_vals.len() };
    ensure!(candidate_count > 0, "候选值列表为空");

    // 第二旋钮候选数
    let candidate2_count = if knob2_name.is_some() {
        if is_int_knob2 { int_vals2.len() } else { float_vals2.len() }
    } else { 1 };
    if knob2_name.is_some() {
        ensure!(candidate2_count > 0, "第二旋钮候选值列表为空");
    }

    for run_index in 0..runs {
        // baseline 每局只跑一次（同 seed 同 deck 同 trainer → 结果相同）
        let base_trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), run_index);
        let base = bench::run_seeded(UMA, &deck, &INHERIT, BASE_SEED, run_index, &base_trainer)?;

        for ci in 0..candidate_count {
            let val_str = if is_int_knob {
                format!("{}", int_vals[ci])
            } else {
                format!("{}", float_vals[ci])
            };

            for ci2 in 0..candidate2_count {
                let val2_str = if knob2_name.is_some() {
                    if is_int_knob2 { format!("{}", int_vals2[ci2]) } else { format!("{}", float_vals2[ci2]) }
                } else { String::new() };

                // 构造候选 trainer：根据旋钮名设置对应参数
                let (mut cap, mut cook2, mut floor, mut fp, mut fs, mut pv, mut vs,
                     mut ev_vital, mut ev_motiv, mut ev_bad) =
                    (None, None, None, None, None, None, None, None, None, None);

                // 第一旋钮
                match knob_name.as_str() {
                    "cap_discount" => cap = Some(float_vals[ci]),
                    "cook2_stock" => cook2 = Some(float_vals[ci]),
                    "y3_hard_floor" => floor = Some(int_vals[ci]),
                    "friend_proactive" => fp = Some(float_vals[ci]),
                    "friend_starve" => fs = Some(float_vals[ci]),
                    "y3_pre_vital" => pv = Some(int_vals[ci]),
                    "y3_shortfall" => vs = Some(float_vals[ci]),
                    "event_vital" => ev_vital = Some(float_vals[ci]),
                    "event_motiv" => ev_motiv = Some(float_vals[ci]),
                    "event_bad" => ev_bad = Some(float_vals[ci]),
                    _ => anyhow::bail!("未知旋钮: {knob_name}")
                }

                // 第二旋钮（如果存在）
                if let Some(k2) = knob2_name.as_deref() {
                    match k2 {
                        "cap_discount" => cap = Some(float_vals2[ci2]),
                        "cook2_stock" => cook2 = Some(float_vals2[ci2]),
                        "y3_hard_floor" => floor = Some(int_vals2[ci2]),
                        "friend_proactive" => fp = Some(float_vals2[ci2]),
                        "friend_starve" => fs = Some(float_vals2[ci2]),
                        "y3_pre_vital" => pv = Some(int_vals2[ci2]),
                        "y3_shortfall" => vs = Some(float_vals2[ci2]),
                        "event_vital" => ev_vital = Some(float_vals2[ci2]),
                        "event_motiv" => ev_motiv = Some(float_vals2[ci2]),
                        "event_bad" => ev_bad = Some(float_vals2[ci2]),
                        _ => anyhow::bail!("未知第二旋钮: {k2}")
                    }
                }

                let candidate_trainer = LoggingTrainer::new(
                    RecommendedRamenTrainer::with_mine_overrides(cap, cook2, floor, fp, fs, pv, vs, ev_vital, ev_motiv, ev_bad),
                    run_index
                );
                let candidate = bench::run_seeded(UMA, &deck, &INHERIT, BASE_SEED, run_index, &candidate_trainer)?;

                let result = if candidate.score > base.score { "胜" }
                    else if candidate.score < base.score { "负" }
                    else { "平" };

                if knob2_name.is_some() {
                    writeln!(
                        file,
                        "{knob_name},{val_str},{},{val2_str},{composition_index},{},{run_index},{},{},{},{},{},{},{result}",
                        knob2_name.as_deref().unwrap_or(""),
                        composition.name(),
                        base.score, candidate.score, base.skill_pt, candidate.skill_pt,
                        status_score(&base.five_status), status_score(&candidate.five_status)
                    )?;
                } else {
                    writeln!(
                        file,
                        "{knob_name},{val_str},{composition_index},{},{run_index},{},{},{},{},{},{},{result}",
                        composition.name(),
                        base.score, candidate.score, base.skill_pt, candidate.skill_pt,
                        status_score(&base.five_status), status_score(&candidate.five_status)
                    )?;
                }
            }
        }
    }
    ensure!(runs > 0, "每候选局数必须大于0");
    Ok(())
}
