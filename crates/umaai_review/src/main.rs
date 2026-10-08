//! 单局复盘分析引擎 bin（`umaai_review`）
//!
//! 方案与口径见 `.trae/documents/replay_review.md`：对 `logs/game{id}.zip`
//! 产出 `digest.json`（结构化指标，喂 LLM 的上下文包）；`report.html`
//! （minijinja 模板 + 内联自绘 SVG，零 JS）在 M5 里程碑补齐。
//!
//! 当前实现进度（文档 §11 实施顺序）：**步骤 1-3** —— 解包 + 角色识别 +
//! timeline / decisions / coverage / schedule + 终局评分与等级换算 →
//! `digest.json`。
//!
//! ## 用法
//!
//! ```text
//! cargo run --release -p umaai_review -- --zip logs/game6234.zip
//!     [--out logs/game6234] [--gamedata <path>]
//! ```

use std::{fs, path::PathBuf, process::ExitCode};

use anyhow::{Context as _, Result, anyhow};
use lexopt::{Arg, ValueExt};

use umaai_review::{brief, checks, clones, decisions, digest, execution, gdata, inherit, pack, profile, report, schedule, timeline};
use umasim::{game::SupportCard, gamedata::GAMEDATA, utils::load_game_config};

/// CLI 参数（lexopt，与项目主 bin 惯例一致）
#[derive(Debug)]
struct CliArgs {
    /// 局包路径（`logs/game{id}.zip`，必需）
    zip: PathBuf,
    /// 输出目录（默认局包同级 `{zip_stem}/`）
    out: Option<PathBuf>,
    /// gamedata 目录（默认按文档 §9.2 优先级解析）
    gamedata: Option<PathBuf>,
    /// 叙述文件（skill 写的 4 段叙述；给了就注入 report.html，不给则保留占位）
    narrative: Option<PathBuf>,
}

/// 解析 CLI（`--zip` / `--out` / `--gamedata` / `--narrative` / `-h`）
fn parse_cli() -> Result<CliArgs> {
    let mut zip: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut gamedata: Option<PathBuf> = None;
    let mut narrative: Option<PathBuf> = None;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Short('h') | Arg::Long("help") => {
                print_usage();
                std::process::exit(0);
            }
            Arg::Long("zip") => zip = Some(parser.value()?.parse()?),
            Arg::Long("out") => out = Some(parser.value()?.parse()?),
            Arg::Long("gamedata") => gamedata = Some(parser.value()?.parse()?),
            Arg::Long("narrative") => narrative = Some(parser.value()?.parse()?),
            _ => return Err(arg.unexpected().into()),
        }
    }
    Ok(CliArgs {
        zip: zip.ok_or_else(|| anyhow!("缺少 --zip <局包路径>（如 logs/game6234.zip）"))?,
        out,
        gamedata,
        narrative,
    })
}

/// 默认输出目录：局包同级 `{zip_stem}/`（如 `logs/game6234.zip` → `logs/game6234/`）
fn default_out_dir(zip: &std::path::Path) -> PathBuf {
    let stem = zip
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("game");
    zip.parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(stem)
}

/// 用法说明
fn print_usage() {
    println!(
        "用法: umaai_review --zip <logs/game{{id}}.zip> [--out <dir>] [--gamedata <path>] [--narrative <md>]\n\
         \x20 --zip       局包路径（必需）\n\
         \x20 --out       输出目录（默认局包同级 {{game}}/）\n\
         \x20 --gamedata gamedata 目录（默认按文档 §9.2 优先级解析）\n\
         \x20 --narrative 叙述文件（4 段，`<!-- overview -->` 等标记分隔）；给了就注入 report.html，\n\
         \x20            不给则 report.html 保留 NARRATIVE 占位（供 skill 二次回填）"
    );
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("错误: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// 主流程（文档 §11 步骤 1-3）：解包 → gamedata → timeline / decisions →
/// schedule / score → digest.json
fn run() -> Result<()> {
    let args = parse_cli()?;
    // chdir（gdata::init）前把所有路径绝对化
    let zip_abs = gdata::absolutize(&args.zip);
    let out_dir = args.out.as_deref().map_or_else(
        || default_out_dir(&zip_abs),
        |o| gdata::absolutize(o),
    );
    // 叙述文件（skill 写的 4 段）：chdir 前先绝对化并读入
    let narrative = match &args.narrative {
        Some(p) => {
            let abs = gdata::absolutize(p);
            let text = fs::read_to_string(&abs)
                .with_context(|| format!("读取叙述文件失败: {}", abs.display()))?;
            report::parse_narrative(&text)
        }
        None => report::Narrative::default(),
    };

    // ① 解包 + 角色识别（§11 步骤 1）
    let p = pack::open_zip(&zip_abs)?;

    // ② gamedata 解析与初始化（§9.2；命中后 set_current_dir）
    let gd = gdata::resolve(args.gamedata.as_deref(), &zip_abs);
    // 自带标记必须在 init 之前读：init 会 chdir，之后相对路径的 --gamedata 就失效了
    let gamedata_bundled = gd.as_ref().and_then(|d| gdata::bundled_note(d));
    let gamedata_ok = match &gd {
        Some(dir) => {
            gdata::init(dir)?;
            true
        }
        None => false,
    };

    // ③ timeline / decisions（§11 步骤 2）
    let tl = timeline::build(&p.snaps);
    let dec = decisions::parse(p.decisions_csv.as_deref())?;

    // ④ schedule / 终局评分（§11 步骤 3 + §6.3）
    let uma_data = if gamedata_ok {
        p.meta
            .as_ref()
            .map(|m| m.uma_id)
            .or_else(|| tl.first_status.as_ref().map(|s| s.base_game.uma_id))
            .and_then(|id| GAMEDATA.get().and_then(|g| g.get_uma(id).ok()))
    } else {
        None
    };
    let race_history: Vec<i32> = tl
        .last_status
        .as_ref()
        .map(|s| s.base_game.race_history.clone())
        .unwrap_or_default();
    let last_turn = tl
        .last_status
        .as_ref()
        .map(|s| s.base_game.turn)
        .unwrap_or(0);
    let mut sched = schedule::build(uma_data, &race_history, last_turn);
    // 自选比赛期限波动（用户口径）：>8000 的运气波动通常是「不补赛将育成失败」的
    // 低估 + 补赛达标后回升，净变≈0 → 进伪波动标记 + 赛程注记（供叙事点出「极限达标」）
    let swings = checks::free_race_swings(&dec.rows, &sched);
    for s in &swings {
        sched.notes.push(checks::free_race_swing_note(s));
    }
    if !sched.notes.is_empty() {
        println!("赛程注记: {}", sched.notes.join("; "));
    }

    // ⑤ 实际执行推断（§11 步骤 4、§5.4）+ digest 组装落盘
    // 必赛回合 ∪ URA 决赛（73/75/77 剧本固定赛）→ 「比赛」判定兜底
    let mut race_turns = sched.mandatory_turns.clone();
    race_turns.extend([73, 75, 77]);
    let exec = execution::build(&tl.rows, &dec.rows, &race_turns);

    // ⑥ 检查项引擎（§11 步骤 5-6）：伪波动标记 / 超级拉面期 / 坏手法 / 继承 / 分身
    let mut flags = checks::flagged_turns(&tl.rows);
    flags.extend(checks::free_race_swing_flags(&swings));
    flags.extend(checks::skill_learned_flags(&tl.rows));
    flags.sort_by_key(|f| f.turn);
    flags.dedup();
    let mut extra_findings: Vec<execution::Finding> = Vec::new();
    if let Some(stats) = checks::super_ramen_stats(&dec.rows) {
        extra_findings.push(checks::super_ramen_finding(&stats));
    }
    extra_findings.extend(checks::bad_habits(&tl.rows, &exec, &sched, &race_history));
    // 继承参考值 = Σ(extra_count 五维) + 126（game_config [config_override]）
    let inherit_ref = if gamedata_ok {
        load_game_config()
            .ok()
            .map(|c| c.extra_count[..5].iter().sum::<i32>() + 126)
    } else {
        None
    };
    let inherit_block = inherit::build(&tl.rows, &exec, inherit_ref);
    // 分身观测需卡组 cardType（gamedata 缺失 → 整块跳过）
    let card_types: Option<Vec<i32>> = if gamedata_ok {
        tl.first_status.as_ref().map(|s| {
            s.base_game
                .card_id
                .iter()
                .map(|&idrank| SupportCard::new(idrank).map(|c| c.card_type).unwrap_or(-1))
                .collect()
        })
    } else {
        None
    };
    let clones_block = clones::build(&p.snaps, card_types.as_deref(), &exec);

    // 训练画像（逐年训练次数 × 五维净增 × 运气归属；增长顺序分析用）。
    // 传 flags：自选 swing / 学技能回合的 Δ 归 luck_program，不带偏训练归属
    let training = profile::build(&tl.rows, &exec.rows, &dec.rows, &flags);

    let inputs = digest::Inputs {
        pack: &p,
        timeline: &tl,
        decisions: &dec,
        execution: exec.clone(),
        schedule: sched,
        gamedata_ok,
        flags,
        inherit: Some(inherit_block.clone()),
        clones: clones_block.clone(),
        training: training.clone(),
        extra_findings: extra_findings.clone(),
        gamedata_bundled,
    };
    let d = digest::build(&inputs);
    let digest_path = digest::write_json(&d, &out_dir)?;
    // brief.md（LLM 复盘简报：六问事实预答，SKILL 层一次 Read 即可动笔）
    let brief_path = brief::render(&d, &out_dir)?;
    // report.html（§11 步骤 7：minijinja 模板 + plot::svg 四图）
    let report_path = report::render(&d, &out_dir, &narrative)?;

    // ⑥ 摘要输出
    println!("局包: {}", zip_abs.display());
    println!("输出目录: {}", out_dir.display());
    if let Some(dir) = &gd {
        println!("gamedata: {}", dir.display());
    }
    for line in p.self_check() {
        println!("{line}");
    }
    println!("timeline: {} 行（解析失败 {} 份）", tl.rows.len(), tl.parse_errors.len());
    println!(
        "decisions: {} 行 calc / skip {} 行 / no_emit {} 行",
        dec.rows.len(),
        dec.coverage.skip.by_reason.values().sum::<u64>(),
        dec.coverage.no_emit
    );
    println!(
        "运气分: 序列 {} 点，回合合计 top_gain {:?}",
        dec.luck.series.len(),
        dec.luck.top_gain.iter().map(|t| (t.turn, t.delta as i64)).collect::<Vec<_>>()
    );
    match (&d.meta.final_score, &d.meta.rank) {
        (Some(s), Some(r)) => println!(
            "终局评分: {s}（{r}，来源 {}{}）",
            d.meta.final_source,
            if d.meta.is_qiezhe {
                "，切者×1.1"
            } else if d.meta.is_xiao_qie {
                "，小切×1.04"
            } else {
                ""
            }
        ),
        _ => println!("终局评分: 不可用（gamedata 缺失）"),
    }
    if exec.comparable > 0 {
        let rate = exec.matched as f64 / exec.comparable as f64 * 100.0;
        println!(
            "执行推断: {} 回合可比，一致 {}（{rate:.1}%），偏离 {}",
            exec.comparable,
            exec.matched,
            exec.findings.len()
        );
    }
    println!(
        "伪波动标记: {} 回合（年界 / 继承 / RMJ / 开局第1年地区选择）；findings 共 {} 条（偏离 {} + 检查项 {}）",
        d.luck.flagged_turns.len(),
        d.findings.len(),
        exec.findings.len(),
        extra_findings.len()
    );
    {
        let inh = &inherit_block;
        let refs = inh
            .reference_value
            .map(|r| format!("（参考值 {r}）"))
            .unwrap_or_default();
        let detail = inh
            .contributions
            .iter()
            .map(|c| format!("turn{}: {:+}（偏差 {:+?}）", c.turn, c.five_status_sum, c.deviation))
            .collect::<Vec<_>>()
            .join("; ");
        println!("继承质量{refs}: {detail}");
    }
    if let Some(cl) = &clones_block {
        println!(
            "分身观测: 地区分身 新增 {} 彩圈 {}（随机{}/规则{}）训练 {} / 超级拉面分身 新增 {} 彩圈 {}（随机{}/规则{}）",
            cl.region.new_clones,
            cl.region.rainbow_clones,
            cl.region.rainbow_luck,
            cl.region.rainbow_strategy,
            cl.region.trained_clones,
            cl.super_ramen_clones.new_clones,
            cl.super_ramen_clones.rainbow_clones,
            cl.super_ramen_clones.rainbow_luck,
            cl.super_ramen_clones.rainbow_strategy
        );
    }
    // 训练画像摘要：每年「训练次数 / 净增」领先维 + 全局运气归属最大维
    {
        let attr = ["速", "耐", "力", "根", "智"];
        let years: Vec<String> = training
            .years
            .iter()
            .map(|y| {
                let (ci, c) = y
                    .train_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, c)| **c)
                    .map(|(i, c)| (attr[i], *c))
                    .unwrap_or(("-", 0));
                let (gi, g) = y
                    .gains
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, g)| **g)
                    .map(|(i, g)| (attr[i], *g))
                    .unwrap_or(("-", 0));
                format!("{} 训{ci}×{c} 增{gi}{g:+}", y.label)
            })
            .collect();
        let (li, l) = training
            .luck_by_attr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, l)| (attr[i], *l))
            .unwrap_or(("-", 0.0));
        println!(
            "训练画像: {}；运气归属最大 {}{:+.0}（非训练回合 {:+.0}，程序性大波动回合 {:+.0}）",
            years.join(" / "),
            li,
            l,
            training.luck_other,
            training.luck_program
        );
    }
    println!("digest: {}", digest_path.display());
    println!("简报: {}", brief_path.display());
    println!("报告: {}", report_path.display());
    println!("（当前实现：文档 §11 步骤 1-7；SKILL.md / 发布打包于后续里程碑落盘）");
    Ok(())
}
