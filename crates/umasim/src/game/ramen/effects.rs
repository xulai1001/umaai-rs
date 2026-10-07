//! 拉面杯效果计算
//!
//! 包含基础效果、地区效果、超级拉面效果的叠加计算。
//! 将所有剧本加成来源合并为统一的训练效果，再应用于训练数值计算。

use super::RamenGame;
use crate::{gamedata::ramen::RAMENDATA, global, utils::Array6};

/// 超级拉面选项覆盖的 4 个训练位获得的训练数值上限加成（游戏固定 +100）
const SUPER_RAMEN_STATUS_LIMIT: i32 = 100;

/// 拉面杯训练效果（合并所有来源的加成）
///
/// 由 `calc_ramen_training_effect` 根据当前游戏状态计算得出，
/// 包含所有生效的剧本加成词条的合并结果。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RamenTrainingEffect {
    /// 训练加成（百分比）
    pub xunlian: i32,
    /// 友情训练加成（百分比，仅友情训练时生效）——对**属性与 PT** 上层均生效
    ///
    /// 口径（2026-10-07 定稿）：友情加成正常情况下同时对属性与 PT 生效，
    /// **不剔除** RMJ 结算的友情（推翻早前 M1「RMJ 友情只作用于属性」的方案）。
    pub youqing: i32,
    /// PT加成（百分比）——仅对 PT 上层生效
    pub pt_bonus: i32,
    /// 属性上限增加
    pub status_limit: i32,
    /// PT上限增加
    pub pt_limit: i32,
    /// 失败率下降（百分比）
    pub fail_rate_drop: i32,
    /// 羁绊增加
    pub friendship: i32,
    /// 得意率加成
    pub deyilv: i32,
    /// hint出现率加成
    pub hint: i32,
    /// hint_special标记
    pub hint_special: bool,
    /// 分身数量
    pub clone_count: i32
}

/// 将支援卡下层属性按 100 截断，再应用拉面上层加成与增量上限。
#[inline(always)]
pub(crate) fn apply_ramen_training_effect(mut status: Array6, effect: &RamenTrainingEffect) -> Array6 {
    for value in &mut status {
        *value = (*value).min(100);
    }
    let xunlian_mult = (100 + effect.xunlian) as f64 / 100.0;
    let youqing_mult = (100 + effect.youqing) as f64 / 100.0;
    let pt_bonus_mult = (100 + effect.pt_bonus) as f64 / 100.0;
    let status_limit = 100 + effect.status_limit;
    let pt_limit = 100 + effect.pt_limit;
    for value in &mut status[..5] {
        if *value > 0 {
            let upper_raw = (*value as f64 * xunlian_mult * youqing_mult) as i32 - *value;
            let upper = upper_raw.min(status_limit).max(0);
            *value += upper;
        }
    }
    let pt_upper_raw = (status[5] as f64 * xunlian_mult * youqing_mult * pt_bonus_mult) as i32 - status[5];
    let pt_upper = pt_upper_raw.min(pt_limit).max(0);
    status[5] += pt_upper;
    status
}

/// 把训练效果格式化为词条列表（非 0 才显示），如 `["训+23", "失败率-50", "上限+20"]`
///
/// 供吃面候选预览（`RamenGame::ramen_candidate_preview`）与吃面后效果展示
/// （`explain_ramen_info`）复用，保证两处口径一致。
pub fn format_ramen_effect_parts(eff: &RamenTrainingEffect) -> Vec<String> {
    let mut parts = vec![];
    if eff.xunlian != 0 {
        parts.push(format!("训+{}", eff.xunlian));
    }
    if eff.youqing != 0 {
        parts.push(format!("友情+{}", eff.youqing));
    }
    if eff.deyilv != 0 {
        parts.push(format!("得意+{}", eff.deyilv));
    }
    if eff.fail_rate_drop != 0 {
        parts.push(format!("失败率-{}", eff.fail_rate_drop));
    }
    if eff.friendship != 0 {
        parts.push(format!("羁绊+{}", eff.friendship));
    }
    if eff.status_limit != 0 {
        parts.push(format!("上限+{}", eff.status_limit));
    }
    if eff.pt_bonus != 0 {
        parts.push(format!("PT+{}", eff.pt_bonus));
    }
    if eff.pt_limit != 0 {
        parts.push(format!("PT上限+{}", eff.pt_limit));
    }
    if eff.hint != 0 {
        parts.push(format!("hint+{}", eff.hint));
    }
    if eff.clone_count != 0 {
        parts.push(format!("分身+{}", eff.clone_count));
    }
    if eff.hint_special {
        parts.push("hint全卡".to_string());
    }
    parts
}

/// 根据当前剧本PT查找对应的 `ramen_pt_effect` 档位
///
/// 从高到低查找第一个 `pt_min <= scenario_pt` 的档位。
pub fn find_pt_effect_tier(scenario_pt: i32) -> usize {
    let ramen_data = global!(RAMENDATA);
    let mut tier = 0;
    for (i, pe) in ramen_data.ramen_pt_effect.iter().enumerate() {
        if scenario_pt >= pe.pt_min {
            tier = i;
        }
    }
    tier
}

/// 计算地区词条加成档位
///
/// 每获得 **1000** 点剧本PT提升一档，最高5档（`region_bonus` 长度 6：0/3/5/7/9/10）。
///
/// 实测校准（game6261 第3年 `active_effect_array` 反推）：
/// - turn48_3 `scenario_pt=500`  → bonus 0（tier 0）
/// - turn54_2 `scenario_pt=1650` → bonus 3（tier 1）
/// - turn55_3 `scenario_pt=2300` → bonus 5（tier 2）
/// - turn57_2 `scenario_pt=3000` → bonus 7（tier 3）
///
/// （旧实现按 `/300`，会在同一批数据上算出 3/10/10/10，明显偏大。）
fn calc_region_bonus_tier(year_scenario_pt: i32) -> usize {
    (year_scenario_pt / 1000).min(5) as usize
}

/// 计算超级拉面回合的效果（回合 72-77 自动生效）
///
/// 超级拉面期间：
/// - 第3年RMJ结算效果（`rmj_results[2]`）常驻生效
/// - `finals_effect.base` 效果生效
/// - `finals_effect.extra` 效果仅在支援卡种类 >= 4 时生效
/// - 选中超级拉面选项（`training_limit_options[super_ramen]`）覆盖的 4 个训练位
///   额外获得**属性**上限 +100（「獲得上限」；PT 上限不由选项提供）
/// - **不生效**：`ramen_pt_effect` / `ramen_basic_effect` / `ramen_region_effect`
///   —— 这三者是「试食会（吃面）」专属效果；超级拉面期间不能吃面，故不叠加
///
/// # 参数
/// - `game`: 拉面杯游戏状态
/// - `train`: 训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）
fn calc_finals_effect(game: &RamenGame, train: usize) -> RamenTrainingEffect {
    let ramen_data = global!(RAMENDATA);
    let mut effect = RamenTrainingEffect::default();

    // 1. 第3年RMJ结算效果（rmj_results[2]）在URA期间生效
    if let Some(&success) = game.ramen.rmj_results.get(2) {
        let rmj_effect = if success {
            &ramen_data.ramen_success_effect[2]
        } else {
            &ramen_data.ramen_fail_effect[2]
        };
        effect.youqing += rmj_effect.youqing;
        effect.deyilv += rmj_effect.deyilv;
        effect.hint += rmj_effect.hint;
    }

    // 2. finals_effect.base 效果
    let finals = &ramen_data.finals_effect;
    effect.youqing += finals.base.youqing;

    // 3. finals_effect.extra 效果：支援卡种类 >= 4 时额外生效
    if game.deck_can_split {
        effect.pt_bonus += finals.extra.pt_bonus;
        effect.pt_limit += finals.extra.pt_limit;
        effect.clone_count += finals.extra.clone_count;
    }

    // 4. 超级拉面选项：选中选项覆盖的 4 个训练位 +100 **属性**上限
    //    （「獲得上限」，不抬 PT 上限；未选择选项 / 选项越界 / 该训练位不在覆盖范围时不加）
    if let Some(opt) = game.ramen.super_ramen {
        if let Some(limit_trains) = finals.training_limit_options.get(opt) {
            if limit_trains.contains(&(train as i32)) {
                effect.status_limit += SUPER_RAMEN_STATUS_LIMIT;
            }
        }
    }

    effect
}

/// 计算普通回合的效果（非超级拉面回合）
///
/// 普通回合效果来源：
/// - `ramen_pt_effect`：常驻生效（根据当前剧本PT决定档次）
/// - `ramen_success_effect` / `ramen_fail_effect`：RMJ结算后常驻生效
/// - `ramen_basic_effect`：仅吃面后生效
/// - `ramen_region_effect`：仅吃面后且在 `at_trains` 标注的训练位置生效
///
/// # 参数
/// - `game`: 拉面杯游戏状态
/// - `train`: 训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）
/// - `year_idx`: 年份索引（0-2）
/// - `ramen`: 本次评估的候选面，None 表示不吃面
fn calc_normal_effect(game: &RamenGame, train: usize, year_idx: usize, ramen: Option<usize>) -> RamenTrainingEffect {
    let ramen_data = global!(RAMENDATA);
    let mut effect = RamenTrainingEffect::default();

    // 1. ramen_pt_effect（常驻生效）
    let pt_tier = find_pt_effect_tier(game.ramen.scenario_pt);
    let pt_effect = &ramen_data.ramen_pt_effect[pt_tier];
    effect.xunlian += pt_effect.xunlian;
    effect.deyilv += pt_effect.deyilv;
    effect.hint += pt_effect.hint;

    // 2. ramen_success_effect / ramen_fail_effect（RMJ结算后常驻生效）
    if year_idx >= 1 {
        // year_idx 1 使用 rmj_results[0]，year_idx 2 使用 rmj_results[1]
        let prev_idx = year_idx - 1;
        if let Some(&success) = game.ramen.rmj_results.get(prev_idx) {
            let rmj_effect = if success {
                &ramen_data.ramen_success_effect[prev_idx]
            } else {
                &ramen_data.ramen_fail_effect[prev_idx]
            };
            effect.youqing += rmj_effect.youqing;
            effect.deyilv += rmj_effect.deyilv;
            effect.hint += rmj_effect.hint;
        }
    }

    // 3. ramen_basic_effect（仅吃面后生效）
    //    `status_limit` 是「獲得上限アップ」的统一口径：对属性与 PT 的上段上限同值生效
    //    （Y2 +20 / Y3 +40），因此这里同时累加到 status_limit 与 pt_limit。
    let eating = ramen.is_some();
    if eating && year_idx < ramen_data.ramen_basic_effect.len() {
        let basic = &ramen_data.ramen_basic_effect[year_idx];
        effect.xunlian += basic.xunlian;
        effect.youqing += basic.youqing;
        effect.fail_rate_drop += basic.fail_rate_drop;
        effect.friendship += basic.friendship;
        effect.status_limit += basic.status_limit;
        effect.pt_limit += basic.status_limit;
        effect.hint_special |= basic.hint_special;
    }

    // 4. ramen_region_effect（仅吃面后且在 at_trains 标注位置生效）
    if eating {
        if let Some(ramen_idx) = ramen {
            let region = &ramen_data.ramen_region_effect[ramen_idx];
            if region.at_trains.contains(&(train as i32)) {
                // 地区词条加成随当年剧本PT增加
                //
                // 实测：`region_bonus` **只加到友情**（进属性与 PT 的友情乘子），
                // **不加到 PT加成**。游戏 `active_effect_array` 里 id52 虽显示为
                // `pt_bonus + region_bonus`，但 PT 上层公式实际只用 region 基础 pt_bonus
                // （turn57_2: pt_bonus 用 50 得 133，用 57 得 139，实测为 133）。
                let bonus_tier = calc_region_bonus_tier(game.ramen.scenario_pt);
                let region_bonus = ramen_data.region_bonus.get(bonus_tier).copied().unwrap_or(0);
                effect.xunlian += region.xunlian;
                effect.youqing += region.youqing + region_bonus;
                effect.pt_bonus += region.pt_bonus;
            }
        }
    }

    effect
}

/// 计算拉面杯的训练效果
///
/// 根据当前游戏状态，合并所有生效的加成来源：
/// - 超级拉面回合（72-77）：调用 `calc_finals_effect`
/// - 普通回合：调用 `calc_normal_effect`
///
/// **重要**：非友情训练时 youqing 会被强制归零，调用方无需额外判断。
///
/// # 参数
/// - `game`: 拉面杯游戏状态
/// - `train`: 训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）
/// - `is_shining`: 是否友情训练（非友情训练时 youqing 视为 0）
pub fn calc_ramen_training_effect(game: &RamenGame, train: usize, is_shining: bool) -> RamenTrainingEffect {
    calc_ramen_training_effect_with_ramen(game, train, is_shining, game.ramen.current_ramen)
}

/// 按指定候选面计算训练效果；基础局面保持借用，超级拉面回合仍使用决赛效果。
pub fn calc_ramen_training_effect_with_ramen(
    game: &RamenGame, train: usize, is_shining: bool, ramen: Option<usize>
) -> RamenTrainingEffect {
    let super_ramen = game.is_super_ramen_turn();
    let year_idx = (game.current_year() - 1) as usize;

    let mut effect = if super_ramen {
        // 超级拉面回合：RMJ + finals_effect（含选中选项的 +100 上限），不享受地区效果
        calc_finals_effect(game, train)
    } else {
        // 普通回合：PT常驻 + RMJ常驻 + 吃面基础 + 地区效果
        calc_normal_effect(game, train, year_idx, ramen)
    };

    // 非友情训练时 youqing 不生效（强制归零），属性与 PT 上层同步归零。
    if !is_shining {
        effect.youqing = 0;
    }

    effect
}

/// 计算当前回合生效的剧本得意率总加成
///
/// 按剧本原始规则：剧本得意率只和支援卡的得意率相加（参见 `ramen_memo_cn.md`）。
/// 本函数仅汇总**对训练分布生效**的剧本得意率来源：
/// - `ramen_pt_effect`：常驻生效
/// - `ramen_success_effect` / `ramen_fail_effect`：RMJ 结算后常驻
///
/// **不包含** `ramen_basic_effect`（全部为 0）和 `ramen_region_effect`（无 deyilv 字段）。
///
/// 用于 `RamenGame::deyilv`，与 `calc_finals_effect` / `calc_normal_effect` 中的
/// deyilv 计算保持一致（**超级拉面直接复用 `calc_finals_effect`**）。
///
/// # 参数
/// - `game`: 拉面杯游戏状态
///
/// # 返回
/// 当前回合的剧本得意率总加成（i32，可直接 + 到支援卡 deyilv 上）
pub fn calc_scenario_deyilv(game: &RamenGame) -> i32 {
    let ramen_data = global!(RAMENDATA);
    let year_idx = (game.current_year() - 1) as usize;

    if game.is_super_ramen_turn() {
        // 超级拉面：复用 calc_finals_effect（只含 rmj_results[2] + finals）
        // 只读 deyilv，与训练位置无关，`train` 传 0 即可
        calc_finals_effect(game, 0).deyilv
    } else {
        // 普通回合：pt_effect(当前档) + rmj_results[year-1]
        // 这里 calc_normal_effect 是训练位置相关的，单独算 deyilv 更直接
        let mut deyilv = 0;
        let pt_tier = find_pt_effect_tier(game.ramen.scenario_pt);
        deyilv += ramen_data.ramen_pt_effect[pt_tier].deyilv;
        if year_idx >= 1 {
            let prev_idx = year_idx - 1;
            if let Some(&success) = game.ramen.rmj_results.get(prev_idx) {
                let rmj_effect = if success {
                    &ramen_data.ramen_success_effect[prev_idx]
                } else {
                    &ramen_data.ramen_fail_effect[prev_idx]
                };
                deyilv += rmj_effect.deyilv;
            }
        }
        deyilv
    }
}

/// 应用拉面杯训练效果计算最终训练数值
///
/// 计算公式：
/// - 属性增加值 = lower_value * (100 + xunlian) / 100 * (100 + youqing) / 100
/// - PT增加值 = lower_value * (100 + xunlian) / 100 * (100 + youqing) / 100
///   * (100 + pt_bonus) / 100
///
///   属性与 PT 的友情口径一致（均不剔除 RMJ 友情）；PT 额外乘 `pt_bonus`。
///
/// 上层数值上限（两者口径独立）：
/// - 属性上限 = 100 + status_limit
///   （普通回合来自 `ramen_basic_effect.status_limit`；超级拉面来自选项的「獲得上限+100」）
/// - PT上限 = 100 + pt_limit
///   （普通回合来自 `ramen_basic_effect.status_limit`——「獲得上限アップ」对属性/PT 同值生效；
///   超级拉面来自 `finals_effect.extra.pt_limit` 的「SP獲得上限+100」，与选项的 +100 无关）
///
/// # 参数
/// - `lower_value`: 下层数值（不计算剧本加成的基础训练数值，上限100）
/// - `effect`: 合并后的拉面杯训练效果
/// - `train`: 训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）
///
/// # 返回
/// `(属性增加值, PT增加值)` - 包含下层数值和受上限约束的上层数值的最终训练数值
pub fn apply_ramen_training_value(lower_value: i32, effect: &RamenTrainingEffect, _train: usize) -> (i32, i32) {
    let lower = lower_value.min(100);

    // 计算上层数值
    let xunlian_mult = (100 + effect.xunlian) as f64 / 100.0;
    let youqing_mult = (100 + effect.youqing) as f64 / 100.0;
    let pt_bonus_mult = (100 + effect.pt_bonus) as f64 / 100.0;

    // 属性训练上层数值
    let status_upper_raw = (lower as f64 * xunlian_mult * youqing_mult) as i32 - lower;
    // PT训练上层数值
    let pt_upper_raw = (lower as f64 * xunlian_mult * youqing_mult * pt_bonus_mult) as i32 - lower;

    // 上层数值上限约束
    let status_limit = 100 + effect.status_limit;
    let pt_limit = 100 + effect.pt_limit;

    let status_upper = status_upper_raw.min(status_limit);
    let pt_upper = pt_upper_raw.min(pt_limit);

    // 调试日志：打印约束前后的 upper/lower 值（排查训练数值不对时使用）
    crate::diag!(
        "  apply_ramen_training_value: lower={} (raw={}) \
         xunlian={} youqing={} pt_bonus={} status_limit={} pt_limit={}\n    \
         属性: status_upper_raw={} -> status_limit={} -> status_upper={} (最终={})\n    \
         PT:   pt_upper_raw={} -> pt_limit={} -> pt_upper={} (最终={})",
        lower,
        lower_value,
        effect.xunlian,
        effect.youqing,
        effect.pt_bonus,
        effect.status_limit,
        effect.pt_limit,
        status_upper_raw,
        status_limit,
        status_upper,
        lower + status_upper,
        pt_upper_raw,
        pt_limit,
        pt_upper,
        lower + pt_upper,
    );

    (lower + status_upper, lower + pt_upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::ramen::RamenState,
        gamedata::init_global,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    /// 创建一个用于测试的 RamenGame 实例
    fn make_test_game() -> RamenGame {
        RamenGame {
            ramen: RamenState {
                scenario_pt: 1000,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_calc_effect_pt_only() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 不吃面、非超级拉面回合、无RMJ结果
        // 仅有 ramen_pt_effect 常驻生效
        let mut game = make_test_game();
        game.base.turn = 5; // year 1

        let effect = calc_ramen_training_effect(&game, 0, false);
        println!("PT=1000, 不吃面, 非友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  deyilv={} hint={} fail_rate_drop={}",
            effect.deyilv, effect.hint, effect.fail_rate_drop
        );
        // pt_min=1000 的档位: xunlian=8, deyilv=63, hint=50
        println!("  => 期望: xunlian=8, deyilv=63, hint=50");

        Ok(())
    }

    #[test]
    fn test_calc_effect_with_eating() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let ramen_data = global!(RAMENDATA);

        // 吃面 + year 1 + 友情训练
        let mut game = make_test_game();
        game.base.turn = 5;
        game.ramen.current_ramen = Some(0); // 吃第一种地区拉面
        game.ramen.scenario_pt = 500;

        // 查看 region 0 的 at_trains
        let region0 = &ramen_data.ramen_region_effect[0];
        println!("region 0: name={} at_trains={:?}", region0.name, region0.at_trains);

        // 在 at_trains 包含的位置上测试
        let train_in_region = region0.at_trains[0] as usize;
        let effect = calc_ramen_training_effect(&game, train_in_region, true);
        println!("PT=500, 吃面region0, train={train_in_region}, 友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  fail_rate_drop={} friendship={} status_limit={}",
            effect.fail_rate_drop, effect.friendship, effect.status_limit
        );

        // 在 at_trains 不包含的位置上测试
        let effect2 = calc_ramen_training_effect(&game, 4, true);
        println!("PT=500, 吃面region0, train=4(智), 友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect2.xunlian, effect2.youqing, effect2.pt_bonus
        );

        Ok(())
    }

    #[test]
    fn test_calc_effect_rmj_success() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // year 2, RMJ year 1 成功
        let mut game = make_test_game();
        game.base.turn = 30;
        game.ramen.scenario_pt = 2000;
        game.ramen.rmj_results = vec![true];

        let effect = calc_ramen_training_effect(&game, 0, true);
        println!("year2, PT=2000, RMJ成功, 友情:");
        println!(
            "  xunlian={} youqing={} deyilv={} hint={}",
            effect.xunlian, effect.youqing, effect.deyilv, effect.hint
        );
        // pt_effect(PT=2000): xunlian=12, deyilv=68, hint=70
        // rmj_success[0]: youqing=5, deyilv=80, hint=30
        println!("  => 期望: xunlian=12, youqing=5, deyilv=148, hint=100");

        Ok(())
    }

    #[test]
    fn test_calc_effect_super_ramen() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 超级拉面回合（第 3 年 RMJ 已结算）
        let mut game = make_test_game();
        game.base.turn = 72;
        game.ramen.scenario_pt = 5000;
        game.ramen.rmj_results = vec![true, true, true];

        let effect = calc_ramen_training_effect(&game, 0, true);
        println!("超级拉面, PT=5000, 友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  status_limit={} pt_limit={} clone_count={}",
            effect.status_limit, effect.pt_limit, effect.clone_count
        );
        // 超级拉面：只保留 RMJ + finals，试食会效果（pt_effect / basic）不生效
        // rmj_success(2): youqing=25, deyilv=250, hint=125
        // finals base: youqing=150
        // finals extra (deck_can_split=false 默认): 不生效
        println!("  => 期望: xunlian=0, youqing=175, pt_bonus=0, status_limit=0");
        let mut checks = Checks::new();
        for ramen in [None, Some(0), Some(5)] {
            checks.check(
                calc_ramen_training_effect_with_ramen(&game, 0, true, ramen) == effect,
                "超级拉面效果不受普通候选面影响"
            );
        }
        checks.finish()
    }

    #[test]
    fn test_calc_effect_super_ramen_with_split() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 超级拉面回合（第 3 年 RMJ 已结算） + deck_can_split = true
        let mut game = make_test_game();
        game.base.turn = 73;
        game.ramen.scenario_pt = 5000;
        game.ramen.rmj_results = vec![true, true, true];
        game.deck_can_split = true;

        let effect = calc_ramen_training_effect(&game, 0, true);
        println!("超级拉面, PT=5000, 友情, deck_can_split=true:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  status_limit={} pt_limit={} clone_count={}",
            effect.status_limit, effect.pt_limit, effect.clone_count
        );
        // 超级拉面：只保留 RMJ + finals，试食会效果（pt_effect / basic）不生效
        // rmj_success(2): youqing=25
        // finals base: youqing=150
        // finals extra: pt_bonus=100, pt_limit=100, clone_count=1
        println!("  => 期望: xunlian=0, youqing=175, pt_bonus=100, pt_limit=100, clone_count=1");

        Ok(())
    }

    /// 超级拉面选项（training_limit_options）给覆盖的 4 个训练位 +100 训练上限
    #[test]
    fn test_calc_finals_status_limit_from_option() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 选项二（index 1）= [0,1,2,4]（速/耐/力/智），根(3) 不在覆盖范围
        let mut game = make_test_game();
        game.base.turn = 72;
        game.ramen.rmj_results = vec![true, true, true];
        game.ramen.super_ramen = Some(1);

        let covered = [0usize, 1, 2, 4];
        let mut checks = Checks::new();
        for train in 0..5 {
            let eff = calc_ramen_training_effect(&game, train, true);
            let want = if covered.contains(&train) { 100 } else { 0 };
            println!("选项二 训练{train}: status_limit={} (期望 {want})", eff.status_limit);
            checks.check(eff.status_limit == want, &format!("训练{train} status_limit 应为 {want}"));
        }

        // 未选择超级拉面选项时（super_ramen=None）全部不加
        let mut game2 = make_test_game();
        game2.base.turn = 72;
        for train in 0..5 {
            let eff = calc_ramen_training_effect(&game2, train, true);
            checks.check(eff.status_limit == 0, &format!("未选选项 训练{train} status_limit 应为 0"));
        }
        checks.finish()
    }

    /// 超级拉面：PT 上段上限 = 100 + `finals.extra.pt_limit`（选项的 +100 是属性专属）
    ///
    /// 超级拉面回合不走 `ramen_basic_effect` 分支，故 PT 上限只吃 extra 的
    /// 「SP獲得上限+100」；属性上限单独吃选项的「獲得上限+100」。
    ///
    /// 回归 `logs/game6260/game6260_turn72_2`（turn72 速训练）：下层 PT=63、
    /// youqing=175、pt_bonus=100 → PT 上段 raw=283，上限 100+100=200 → 合计 263
    /// （修复前上限误为 100+100+100=300，算出 346）。
    #[test]
    fn test_super_ramen_pt_upper_cap_uses_finals_extra_only() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 超级拉面 + 支援卡种类 >= 4 + 选中选项二（覆盖训练位 0/1/2/4）
        let mut game = make_test_game();
        game.base.turn = 72;
        game.ramen.scenario_pt = 8250;
        game.ramen.rmj_results = vec![true, true, true];
        game.ramen.super_ramen = Some(1);
        game.deck_can_split = true;

        let effect = calc_ramen_training_effect(&game, 0, true);
        assert_eq!(effect.status_limit, 100, "选项覆盖的速训练位：属性上限 +100");
        assert_eq!(effect.pt_limit, 100, "finals.extra 的 SP獲得上限+100");

        let (status, pt) = apply_ramen_training_value(63, &effect, 0);
        // 属性：upper_raw = 63*2.75-63 = 110（< 200，未触发上限）→ 63+110 = 173
        // PT：  upper_raw = 63*2.75*2.0-63 = 283 → 上限 100+100=200 → 63+200 = 263
        assert_eq!(status, 173, "属性上段上限 = 100 + status_limit = 200");
        assert_eq!(pt, 263, "PT 上段上限 = 100 + pt_limit = 200，不叠加 status_limit");
        Ok(())
    }

    /// 普通回合：吃面时 `ramen_basic_effect.status_limit` 对属性与 PT 上限同值生效
    ///
    /// 原始资料「獲得上限アップ」（`ramen_memo.md`）是统一口径，因此吃面后的
    /// PT 上段上限 = 100 + `basic.status_limit`（Y2 +20 / Y3 +40），与属性上限同值。
    #[test]
    fn test_normal_ramen_basic_status_limit_also_raises_pt() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.ramen.scenario_pt = 0;
        game.ramen.current_ramen = Some(0); // 吃面
        game.ramen.rmj_results = vec![true, true];

        // 第 2 年（turn 24-47）/ 第 3 年（turn 48-71）
        for (turn, want) in [(30, 20), (60, 40)] {
            game.base.turn = turn;
            let effect = calc_ramen_training_effect(&game, 0, true);
            assert_eq!(effect.status_limit, want, "回合 {turn}：属性上限 +{want}");
            assert_eq!(effect.pt_limit, want, "回合 {turn}：「獲得上限アップ」对 PT 同值 → PT 上限 +{want}");
        }
        Ok(())
    }

    #[test]
    fn test_calc_effect_non_shining() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 非友情训练时 youqing 应为 0
        let mut game = make_test_game();
        game.base.turn = 5;
        game.ramen.scenario_pt = 500;
        game.ramen.current_ramen = Some(5); // 吃面
        game.ramen.rmj_results = vec![true]; // 不影响 year 1

        let effect_shining = calc_ramen_training_effect(&game, 0, true);
        let effect_normal = calc_ramen_training_effect(&game, 0, false);
        println!("友情训练: youqing={}", effect_shining.youqing);
        println!("普通训练: youqing={}", effect_normal.youqing);
        println!("  => 普通训练 youqing 应为 0");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_status() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // 属性训练: lower=50, xunlian=20, youqing=10, pt_bonus=0
        let effect = RamenTrainingEffect {
            xunlian: 20,
            youqing: 10,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(50, &effect, 0);
        // upper = 50 * 1.2 * 1.1 - 50 = 66 - 50 = 16
        // status = 50 + 16 = 66
        // pt = 50 + 16 = 66 (pt_bonus=0 时与属性相同)
        println!("lower=50, xunlian=20, youqing=10, pt_bonus=0:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=66, pt=66");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_pt() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // PT训练: lower=50, xunlian=20, youqing=10, pt_bonus=50
        let effect = RamenTrainingEffect {
            xunlian: 20,
            youqing: 10,
            pt_bonus: 50,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(50, &effect, 0);
        // status upper = 50 * 1.2 * 1.1 - 50 = 16, status = 66
        // pt upper = 50 * 1.2 * 1.1 * 1.5 - 50 = 99 - 50 = 49, pt = 99
        println!("lower=50, xunlian=20, youqing=10, pt_bonus=50:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=66, pt=99");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_upper_limit() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // 上层数值超过上限时应被截断
        let effect = RamenTrainingEffect {
            xunlian: 100,
            youqing: 100,
            pt_bonus: 100,
            status_limit: 50,
            pt_limit: 100,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(80, &effect, 0);
        // status upper raw = 80 * 2.0 * 2.0 - 80 = 240, cap = 100+50=150
        // status = 80 + 150 = 230
        // pt upper raw = 80 * 2.0 * 2.0 * 2.0 - 80 = 560, cap = 100+100=200（不含 status_limit）
        // pt = 80 + 200 = 280
        println!("lower=80, xunlian=100, youqing=100, pt_bonus=100, status_limit=50, pt_limit=100:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=230, pt=280");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_lower_cap() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // lower_value 超过 100 时应被截断到 100
        let effect = RamenTrainingEffect {
            xunlian: 20,
            youqing: 0,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(150, &effect, 0);
        // lower = min(150, 100) = 100
        // upper = 100 * 1.2 - 100 = 20
        // status = pt = 120
        println!("lower=150(截断为100), xunlian=20:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=120, pt=120");

        Ok(())
    }

    // ========== calc_scenario_deyilv 测试 ==========

    /// 普通回合：PT 1000 + 无 RMJ → 仅 pt_effect(PT=1000档) 的 deyilv
    #[test]
    fn test_calc_scenario_deyilv_normal_pt_only() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 5; // year 1
        game.ramen.scenario_pt = 1000;
        // rmj_results 为空（year 1）

        let deyilv = calc_scenario_deyilv(&game);
        // pt_min=1000 的档位: deyilv=63
        println!("year1, PT=1000, 无 RMJ: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 63);
        Ok(())
    }

    /// 普通回合：PT 1000 + RMJ 成功 → pt_effect + rmj_success[0].deyilv
    #[test]
    fn test_calc_scenario_deyilv_normal_with_rmj_success() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 30; // year 2
        game.ramen.scenario_pt = 1000;
        game.ramen.rmj_results = vec![true]; // year 1 RMJ 成功

        let deyilv = calc_scenario_deyilv(&game);
        // pt_effect(PT=1000).deyilv = 63
        // rmj_success[0].deyilv = 80
        // 总计 = 63 + 80 = 143
        println!("year2, PT=1000, RMJ成功: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 143);
        Ok(())
    }

    /// 普通回合：RMJ 失败 → pt_effect + rmj_fail[year-1].deyilv
    #[test]
    fn test_calc_scenario_deyilv_normal_with_rmj_fail() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 30; // year 2
        game.ramen.scenario_pt = 1000;
        game.ramen.rmj_results = vec![false]; // year 1 RMJ 失败

        let deyilv = calc_scenario_deyilv(&game);
        // pt_effect(PT=1000).deyilv = 63
        // rmj_fail[0].deyilv = 30
        println!("year2, PT=1000, RMJ失败: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 93); // 63 + 30
        Ok(())
    }

    /// 超级拉面：RMJ 成功 → 只取 rmj_success[2].deyilv（pt_effect 不生效）
    #[test]
    fn test_calc_scenario_deyilv_super_ramen() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 72; // 超级拉面回合（URA）
        game.ramen.scenario_pt = 5000;
        game.ramen.rmj_results = vec![true, true, true]; // 前三年都成功

        let deyilv = calc_scenario_deyilv(&game);
        // 超级拉面只保留 RMJ：pt_effect 不生效
        // rmj_success[2].deyilv = 250
        println!("超级拉面 turn=72, PT=5000, RMJ成功: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 250);
        Ok(())
    }

    /// 超级拉面：RMJ 失败 → 只取 rmj_fail[2].deyilv（pt_effect 不生效）
    #[test]
    fn test_calc_scenario_deyilv_super_ramen_rmj_fail() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 73;
        game.ramen.scenario_pt = 5000;
        game.ramen.rmj_results = vec![true, true, false]; // year 3 RMJ 失败

        let deyilv = calc_scenario_deyilv(&game);
        // 超级拉面只保留 RMJ：pt_effect 不生效
        // rmj_fail[2].deyilv = 150
        println!("超级拉面 turn=73, PT=5000, RMJ失败: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 150);
        Ok(())
    }
}
