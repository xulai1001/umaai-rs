//! Mean-Filter 训练数据生成器（P1）
//!
//! P0 已验证“按样本 scoreMean 阈值筛选”链路可行；P1 目标是工程化：
//! - 分片写盘（part_*.bin）+ manifest + resume
//! - 统计可观测（overall / per-turn / scoreMean 分位数）
//! - 精确到 target_samples：达到目标后继续推进游戏，但不再接收样本（避免写盘后 truncate）
//!
//! 用法示例（建议优先在 game_config.toml 配置 [collector]，CLI 仅临时覆盖）：
//! ```bash
//! cargo run --release --bin generate_mean_filtered_data -- \
//!   --config game_config.toml \
//!   --output-dir training_data/mean_filtered_60k \
//!   --target-samples 100000 \
//!   --score-mean-threshold 60000 \
//!   --search-n 256 \
//!   --threads 24
//! ```

use std::{path::{Path, PathBuf}, time::Instant};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use chrono::Local;
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    collector::{
        CollectorManifest,
        ManifestSearchConfig,
        ShardWriter,
        calc_score_mean_summary,
        compute_file_signature,
        compute_text_hash_fnv1a64,
        load_score_mean_values,
        try_get_git_commit,
    },
    game::{Game, InheritInfo, onsen::game::OnsenGame},
    gamedata::{GameConfig, init_global},
    search::{FlatSearch, SearchConfig},
    trainer::MeanFilterCollectorTrainer,
    utils::init_logger,
};

#[derive(Parser, Debug)]
#[command(name = "generate_mean_filtered_data")]
#[command(about = "按每条样本 scoreMean 阈值筛选生成训练数据（P2：action+choice 分片写盘 + manifest + resume）")]
struct Args {
    /// 配置文件路径
    #[arg(long, default_value = "game_config.toml")]
    config: String,

    /// 输出目录（覆盖 [collector].output_dir）
    #[arg(long)]
    output_dir: Option<String>,

    /// 输出名称（覆盖 [collector].output_name）
    #[arg(long)]
    output_name: Option<String>,

    /// 目标 accepted 样本数（覆盖 [collector].target_samples）
    #[arg(long)]
    target_samples: Option<usize>,

    /// 最大模拟局数（覆盖 [collector].max_games）
    #[arg(long)]
    max_games: Option<usize>,

    /// scoreMean 阈值（覆盖 [collector].score_mean_threshold）
    #[arg(long)]
    score_mean_threshold: Option<f64>,

    /// 是否丢弃 scoreMean==0（覆盖 [collector].drop_zero_mean）
    #[arg(long)]
    drop_zero_mean: Option<bool>,

    /// 启用 choice 样本采集（覆盖 [collector].collect_choice=true）
    #[arg(long)]
    collect_choice: bool,

    /// 禁用 choice 样本采集（覆盖 [collector].collect_choice=false）
    #[arg(long)]
    no_collect_choice: bool,

    /// choice 评估：每个选项的 rollout 次数（覆盖 [collector].choice_rollouts_per_option）
    #[arg(long)]
    choice_rollouts_per_option: Option<usize>,

    /// choice softmax 温度（覆盖 [collector].choice_policy_delta）
    #[arg(long)]
    choice_policy_delta: Option<f64>,

    /// choice gate 阈值（覆盖 [collector].choice_score_mean_threshold）
    #[arg(long)]
    choice_score_mean_threshold: Option<f64>,

    /// 搜索次数（覆盖 [collector].search_n；若都未设置，将默认使用 128，避免误用 mcts.search_n=10240）
    #[arg(long)]
    search_n: Option<usize>,

    /// 并行线程数（覆盖 [collector].threads）
    #[arg(long)]
    threads: Option<usize>,

    /// 每片样本数（覆盖 [collector].shard_size）
    #[arg(long)]
    shard_size: Option<usize>,

    /// 采样回合范围（覆盖 [collector].turn_min/turn_max/turn_stride；人类回合 1..=78）
    #[arg(long)]
    turn_min: Option<i32>,
    #[arg(long)]
    turn_max: Option<i32>,
    #[arg(long)]
    turn_stride: Option<i32>,

    /// 进度输出间隔（覆盖 [collector].progress_interval）
    #[arg(long)]
    progress_interval: Option<usize>,

    /// 强制启用 resume（覆盖 [collector].resume=true）
    #[arg(long)]
    resume: bool,

    /// 允许覆盖输出目录（覆盖 [collector].overwrite=true；危险操作）
    #[arg(long)]
    overwrite: bool,

    /// 随机种子（仅尽量控制“游戏过程”的随机性；FlatSearch 并行线程 RNG 不完全可复现）
    #[arg(long)]
    seed: Option<u64>,

    /// 是否输出更详细的调试信息
    #[arg(long)]
    verbose: bool,
}

fn clamp_turn_range(turn_min: i32, turn_max: i32) -> (i32, i32) {
    let min = turn_min.max(1).min(78);
    let max = turn_max.max(min).min(78);
    (min, max)
}

fn sanitize_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        // Windows/跨平台不安全字符统一替换
        let bad = matches!(ch, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*');
        out.push(if bad { '_' } else { ch });
    }
    out.trim().trim_matches('.').to_string()
}

fn append_suffix_to_last_component(dir: &Path, suffix: &str) -> PathBuf {
    let suffix = suffix.trim();
    if suffix.is_empty() {
        return dir.to_path_buf();
    }
    let file_name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("run");
    let new_name = format!("{file_name}_{suffix}");
    match dir.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(new_name),
        _ => PathBuf::from(new_name),
    }
}

fn make_unique_dir(mut dir: PathBuf) -> PathBuf {
    if !dir.exists() {
        return dir;
    }
    let parent = dir.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    let base_name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("run").to_string();
    for i in 1..1000 {
        let candidate = parent.join(format!("{base_name}_{i:02}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    // 极端情况下找不到空位，直接返回原路径（后续会由 open_or_create 报错）
    dir
}

fn build_effective_search_config(
    game_config: &GameConfig,
    search_n_cli: Option<usize>,
) -> SearchConfig {
    // 先从 mcts 段构造（兼容已有配置）
    let mut cfg = SearchConfig::new_game_config(game_config);

    // mean-filter 默认不要沿用 10240；除非明确配置/CLI 覆盖
    let user_specified = game_config.collector.search_n.is_some() || search_n_cli.is_some();
    if let Some(v) = game_config.collector.search_n {
        cfg.search_n = v;
    }
    if let Some(v) = search_n_cli {
        cfg.search_n = v;
    }
    if !user_specified {
        cfg.search_n = 128;
    }

    // 其余字段：collector 覆盖（None 则保持 mcts 的值）
    if let Some(v) = game_config.collector.max_depth {
        cfg.max_depth = v;
    }
    if let Some(v) = game_config.collector.radical_factor_max {
        cfg.radical_factor_max = v;
    }
    if let Some(v) = game_config.collector.policy_delta {
        cfg.policy_delta = v;
    }
    if let Some(v) = game_config.collector.use_ucb {
        cfg.use_ucb = v;
    }
    if let Some(v) = game_config.collector.search_group_size {
        cfg.search_group_size = v;
    }
    if let Some(v) = game_config.collector.search_cpuct {
        cfg.search_cpuct = v;
    }
    if let Some(v) = game_config.collector.expected_search_stdev {
        cfg.expected_search_stdev = v;
    }

    // clamp（避免误配置）
    cfg.search_n = cfg.search_n.max(1);
    if cfg.policy_delta <= 0.0 {
        cfg.policy_delta = 1e-6;
    }
    cfg.search_group_size = cfg.search_group_size.max(1);
    if cfg.use_ucb && cfg.search_group_size > cfg.search_n {
        cfg.search_group_size = cfg.search_n;
    }
    cfg
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 读取配置文件
    let config_file_text = fs_err::read_to_string(&args.config)
        .with_context(|| format!("读取配置文件失败: {}", args.config))?;
    let mut game_config: GameConfig = toml::from_str(&config_file_text)
        .with_context(|| format!("解析配置文件失败: {}", args.config))?;

    // 允许 CLI 覆盖 collector 段（只影响本次运行，且会写入 manifest 的 effective config）
    if let Some(v) = &args.output_dir {
        game_config.collector.output_dir = v.clone();
    }
    if let Some(v) = &args.output_name {
        game_config.collector.output_name = v.clone();
    }
    if let Some(v) = args.target_samples {
        game_config.collector.target_samples = v;
    }
    if let Some(v) = args.max_games {
        game_config.collector.max_games = v;
    }
    if let Some(v) = args.score_mean_threshold {
        game_config.collector.score_mean_threshold = v;
    }
    if let Some(v) = args.drop_zero_mean {
        game_config.collector.drop_zero_mean = v;
    }
    if args.collect_choice && args.no_collect_choice {
        return Err(anyhow!("不能同时指定 --collect-choice 与 --no-collect-choice"));
    }
    if args.collect_choice {
        game_config.collector.collect_choice = true;
    }
    if args.no_collect_choice {
        game_config.collector.collect_choice = false;
    }
    if let Some(v) = args.choice_rollouts_per_option {
        game_config.collector.choice_rollouts_per_option = v;
    }
    if let Some(v) = args.choice_policy_delta {
        game_config.collector.choice_policy_delta = v;
    }
    if let Some(v) = args.choice_score_mean_threshold {
        game_config.collector.choice_score_mean_threshold = Some(v);
    }
    if let Some(v) = args.threads {
        game_config.collector.threads = v;
    }
    if let Some(v) = args.shard_size {
        game_config.collector.shard_size = v;
    }
    if let Some(v) = args.turn_min {
        game_config.collector.turn_min = v;
    }
    if let Some(v) = args.turn_max {
        game_config.collector.turn_max = v;
    }
    if let Some(v) = args.turn_stride {
        game_config.collector.turn_stride = v;
    }
    if let Some(v) = args.progress_interval {
        game_config.collector.progress_interval = v;
    }
    if args.resume {
        game_config.collector.resume = true;
    }
    if args.overwrite {
        game_config.collector.overwrite = true;
    }

    // clamp collector 参数
    game_config.collector.target_samples = game_config.collector.target_samples.max(1);
    game_config.collector.max_games = game_config.collector.max_games.max(1);
    game_config.collector.shard_size = game_config.collector.shard_size.max(1);
    game_config.collector.threads = game_config.collector.threads.max(1);
    game_config.collector.turn_stride = game_config.collector.turn_stride.max(1);
    game_config.collector.progress_interval = game_config.collector.progress_interval.max(1);
    game_config.collector.choice_rollouts_per_option = game_config.collector.choice_rollouts_per_option.max(1);
    if game_config.collector.choice_policy_delta <= 0.0 {
        game_config.collector.choice_policy_delta = 1e-6;
    }
    if let Some(v) = &mut game_config.collector.choice_score_mean_threshold {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let (turn_min, turn_max) = clamp_turn_range(game_config.collector.turn_min, game_config.collector.turn_max);
    game_config.collector.turn_min = turn_min;
    game_config.collector.turn_max = turn_max;
    if game_config.collector.manifest_name.is_empty() {
        game_config.collector.manifest_name = "manifest.json".to_string();
    }
    if game_config.collector.score_mean_values_name.is_empty() {
        game_config.collector.score_mean_values_name = "score_mean_values.bin".to_string();
    }
    if game_config.collector.output_timestamp_format.is_empty() {
        game_config.collector.output_timestamp_format = "%Y%m%d_%H%M%S".to_string();
    }

    // 初始化日志/全局数据
    init_logger("mean_filter", &game_config.log_level)?;
    init_global()?;

    // 初始化 rayon 全局线程池（可能已被其它库初始化；重复初始化要降级提示）
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()
    {
        eprintln!("warn: rayon 全局线程池初始化失败（可能已初始化）：{e}");
    }

    // 构造 SearchConfig（优先 collector.search_* 覆盖；若都未配置则 search_n 默认 128）
    let search_config = build_effective_search_config(&game_config, args.search_n);
    let manifest_search_config = ManifestSearchConfig::from_search_config(&search_config);

    // 输出目录（支持 base_dir + name + timestamp）
    let base_dir = PathBuf::from(&game_config.collector.output_dir);
    let mut output_dir = if game_config.collector.output_name.trim().is_empty() {
        base_dir
    } else {
        base_dir.join(sanitize_path_segment(&game_config.collector.output_name))
    };
    if game_config.collector.output_append_timestamp {
        let ts_raw = Local::now()
            .format(&game_config.collector.output_timestamp_format)
            .to_string();
        let ts = sanitize_path_segment(&ts_raw);
        output_dir = append_suffix_to_last_component(&output_dir, &ts);
        // 避免同一秒重复启动导致目录冲突：在不打算 resume/overwrite 时自动找一个可用目录
        if output_dir.exists() && !game_config.collector.resume && !game_config.collector.overwrite {
            output_dir = make_unique_dir(output_dir);
        }
    }
    let manifest_path = output_dir.join(&game_config.collector.manifest_name);

    // 复现信息：配置 hash + git commit + gamedata/model 签名
    let config_hash = compute_text_hash_fnv1a64(&config_file_text);
    let repo_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // 若 resume 且已有 manifest：禁止与当前配置 hash 不一致（防止混入不同配置的数据）
    if output_dir.exists() && game_config.collector.resume && !game_config.collector.overwrite && manifest_path.exists() {
        let old = CollectorManifest::load(&manifest_path)?;
        if old.config_hash_fnv1a64 != config_hash {
            return Err(anyhow!(
                "检测到 manifest 的 config_hash 与当前配置不一致（会导致数据混杂）。建议使用新的 output_dir，或显式 --overwrite 重新生成。\n- output_dir: {}\n- manifest_hash: {}\n- current_hash : {}",
                output_dir.display(),
                old.config_hash_fnv1a64,
                config_hash
            ));
        }
    }

    let git_commit = try_get_git_commit(&repo_dir);

    let mut gamedata_sig = Vec::new();
    for p in [
        "gamedata/constants.json",
        "gamedata/events.json",
        "gamedata/umaDB.json",
        "gamedata/cardDB.json",
    ] {
        let path = Path::new(p);
        if path.exists() {
            let hash = fs_err::metadata(path).map(|m| m.len() <= 32 * 1024 * 1024).unwrap_or(false);
            gamedata_sig.push(compute_file_signature(path, hash)?);
        }
    }

    let model_sig = {
        let path = Path::new(&game_config.neuralnet_model_path);
        if path.exists() {
            Some(compute_file_signature(path, false)?)
        } else {
            None
        }
    };

    println!("=== Mean-Filter 数据生成器（P2）===");
    println!("config                : {}", args.config);
    println!("output_dir_effective  : {}", output_dir.display());
    println!("output_dir_base       : {}", game_config.collector.output_dir);
    println!("output_name           : {}", game_config.collector.output_name);
    println!("output_append_timestamp: {}", game_config.collector.output_append_timestamp);
    println!("output_timestamp_format: {}", game_config.collector.output_timestamp_format);
    println!("target_samples        : {}", game_config.collector.target_samples);
    println!("max_games             : {}", game_config.collector.max_games);
    println!("score_mean_threshold  : {}", game_config.collector.score_mean_threshold);
    println!("drop_zero_mean        : {}", game_config.collector.drop_zero_mean);
    println!("collect_choice        : {}", game_config.collector.collect_choice);
    println!(
        "choice_rollouts       : {}",
        game_config.collector.choice_rollouts_per_option
    );
    println!(
        "choice_policy_delta   : {}",
        game_config.collector.choice_policy_delta
    );
    println!(
        "choice_threshold      : {:?} (effective={})",
        game_config.collector.choice_score_mean_threshold,
        game_config
            .collector
            .choice_score_mean_threshold
            .unwrap_or(game_config.collector.score_mean_threshold)
    );
    println!(
        "choice_skip_if_too_many: {}",
        game_config.collector.choice_skip_if_too_many
    );
    println!(
        "choice_follow_action_turn_range: {}",
        game_config.collector.choice_follow_action_turn_range
    );
    println!(
        "choice_rollout_on_uncollected_turns: {}",
        game_config.collector.choice_rollout_on_uncollected_turns
    );
    println!("fast_after_target     : {}", game_config.collector.fast_after_target);
    println!(
        "turn_range            : {}..={} stride={}",
        game_config.collector.turn_min,
        game_config.collector.turn_max,
        game_config.collector.turn_stride
    );
    println!("threads               : {}", game_config.collector.threads);
    println!("shard_size            : {}", game_config.collector.shard_size);
    println!("resume/overwrite      : {}/{}", game_config.collector.resume, game_config.collector.overwrite);
    println!("search_n              : {}", search_config.search_n);
    println!("ucb                   : {}", search_config.use_ucb);
    println!("search_group_size     : {}", search_config.search_group_size);
    println!("policy_delta          : {}", search_config.policy_delta);
    println!("radical_factor_max    : {}", search_config.radical_factor_max);
    println!("git_commit            : {:?}", git_commit);
    println!("config_hash_fnv1a64    : {}", config_hash);
    println!("note                  : FlatSearch 并行线程 RNG 来自 from_os_rng()，search 不保证完全可复现");
    println!();

    // 打开/创建 ShardWriter（包含 resume 与 source-of-truth 扫描）
    let (mut writer, scan) = ShardWriter::open_or_create(
        &output_dir,
        &game_config.collector.manifest_name,
        &game_config.collector.score_mean_values_name,
        game_config.collector.shard_size,
        game_config.collector.resume,
        game_config.collector.overwrite,
        || {
            Ok(CollectorManifest::new(
                &output_dir,
                git_commit,
                &args.config,
                config_hash,
                gamedata_sig,
                model_sig,
                game_config.collector.clone(),
                &search_config,
            ))
        },
    )?;

    // 若目录里已有 part 且本次有效配置（尤其 threshold/search_n/turn_range）不一致，直接拒绝继续
    if scan.accepted_written > 0 && !writer.manifest.parts.is_empty() && !game_config.collector.overwrite {
        // 只做“会改变数据分布/质量”的关键字段比对；允许 threads/progress_interval 等运行参数变化
        let old_c = &writer.manifest.collector_config;
        if (old_c.score_mean_threshold - game_config.collector.score_mean_threshold).abs() > 1e-9
            || old_c.drop_zero_mean != game_config.collector.drop_zero_mean
            || old_c.collect_choice != game_config.collector.collect_choice
            || old_c.choice_rollouts_per_option != game_config.collector.choice_rollouts_per_option
            || (old_c.choice_policy_delta - game_config.collector.choice_policy_delta).abs() > 1e-9
            || old_c.choice_score_mean_threshold != game_config.collector.choice_score_mean_threshold
            || old_c.choice_skip_if_too_many != game_config.collector.choice_skip_if_too_many
            || old_c.choice_follow_action_turn_range != game_config.collector.choice_follow_action_turn_range
            || old_c.choice_rollout_on_uncollected_turns != game_config.collector.choice_rollout_on_uncollected_turns
            || old_c.turn_min != game_config.collector.turn_min
            || old_c.turn_max != game_config.collector.turn_max
            || old_c.turn_stride != game_config.collector.turn_stride
        {
            return Err(anyhow!(
                "输出目录已包含历史数据，但 collector 关键参数不一致（会导致数据混杂）。请使用新的 output_dir 或显式 --overwrite。\n- output_dir: {}",
                output_dir.display()
            ));
        }

        let old_s = &writer.manifest.search_config_effective;
        let now_s = ManifestSearchConfig::from_search_config(&search_config);
        if old_s.search_n != now_s.search_n
            || old_s.use_ucb != now_s.use_ucb
            || old_s.search_group_size != now_s.search_group_size
            || (old_s.policy_delta - now_s.policy_delta).abs() > 1e-9
            || (old_s.radical_factor_max - now_s.radical_factor_max).abs() > 1e-9
            || (old_s.search_cpuct - now_s.search_cpuct).abs() > 1e-9
            || (old_s.expected_search_stdev - now_s.expected_search_stdev).abs() > 1e-9
        {
            return Err(anyhow!(
                "输出目录已包含历史数据，但 search_config 关键参数不一致（会导致数据混杂）。请使用新的 output_dir 或显式 --overwrite。\n- output_dir: {}",
                output_dir.display()
            ));
        }
    }

    // RNG（仅尽量控制游戏本体随机性）
    let mut rng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    // 组装 trainer（P2：action + choice，精确到 target_samples）
    let search = FlatSearch::new(search_config);
    let trainer = MeanFilterCollectorTrainer::new(search, game_config.collector.score_mean_threshold)
        .with_drop_zero_mean(game_config.collector.drop_zero_mean)
        .with_collect_choice(game_config.collector.collect_choice)
        .with_choice_rollouts_per_option(game_config.collector.choice_rollouts_per_option)
        .with_choice_policy_delta(game_config.collector.choice_policy_delta)
        .with_choice_score_mean_threshold(game_config.collector.choice_score_mean_threshold)
        .with_choice_skip_if_too_many(game_config.collector.choice_skip_if_too_many)
        .with_choice_follow_action_turn_range(game_config.collector.choice_follow_action_turn_range)
        .with_choice_rollout_on_uncollected_turns(game_config.collector.choice_rollout_on_uncollected_turns)
        .with_target_samples(game_config.collector.target_samples as u64)
        .fast_after_target(game_config.collector.fast_after_target)
        .with_turn_range(game_config.collector.turn_min, game_config.collector.turn_max)
        .with_turn_stride(game_config.collector.turn_stride)
        .verbose(args.verbose);

    // resume：以文件系统为准（accepted_written）
    let accepted_base = scan.accepted_written;
    trainer.set_accepted_base(accepted_base);

    // resume：保留旧统计基线（本次 stats 只记录增量；写 manifest 时做 base + delta）
    let base_progress = writer.manifest.progress.clone();
    let base_per_turn = writer.manifest.per_turn.clone();

    let start = Instant::now();
    let mut games_run_delta: u64 = 0;
    let mut games_failed_delta: u64 = 0;

    while trainer.stats_snapshot().accepted < game_config.collector.target_samples as u64
        && (base_progress.games_run + games_run_delta) < game_config.collector.max_games as u64
    {
        // 创建新局
        let inherit = InheritInfo {
            blue_count: game_config.blue_count.clone(),
            extra_count: game_config.extra_count.clone(),
        };
        let mut game = match OnsenGame::newgame(game_config.uma, &game_config.cards, inherit) {
            Ok(g) => g,
            Err(e) => {
                games_failed_delta += 1;
                eprintln!("warn: newgame 失败：{e}");
                continue;
            }
        };

        // 运行一整局（内部会在达到 target 后不再接收样本，且可切换为快速决策）
        match game.run_full_game(&trainer, &mut rng) {
            Ok(()) => {
                games_run_delta += 1;

                // 每局结束后 drain accepted 样本，统一写盘（避免 IO 混入 select_action 热路径）
                let drained = trainer.drain_samples();
                writer.push_samples(drained)?;

                // 更新 manifest（base + delta）
                let stats = trainer.stats_snapshot();
                writer.manifest.collector_config = game_config.collector.clone();
                writer.manifest.search_config_effective = manifest_search_config.clone();

                writer.manifest.progress.games_run = base_progress.games_run + games_run_delta;
                writer.manifest.progress.games_failed = base_progress.games_failed + games_failed_delta;
                writer.manifest.progress.candidates = base_progress.candidates + stats.candidates;
                writer.manifest.progress.accepted = stats.accepted;
                writer.manifest.progress.dropped = base_progress.dropped + stats.dropped;
                writer.manifest.progress.dropped_zero_mean = base_progress.dropped_zero_mean + stats.dropped_zero_mean;
                writer.manifest.progress.search_errors = base_progress.search_errors + stats.search_errors;
                writer.manifest.progress.policy_sum_not_one = base_progress.policy_sum_not_one + stats.policy_sum_not_one;
                writer.manifest.progress.choice_candidates = base_progress.choice_candidates + stats.choice_candidates;
                writer.manifest.progress.choice_accepted = base_progress.choice_accepted + stats.choice_accepted;
                writer.manifest.progress.choice_dropped = base_progress.choice_dropped + stats.choice_dropped;
                writer.manifest.progress.choice_sum_not_one = base_progress.choice_sum_not_one + stats.choice_sum_not_one;
                writer.manifest.progress.choice_policy_not_zero = base_progress.choice_policy_not_zero + stats.choice_policy_not_zero;
                writer.manifest.progress.choice_skipped_too_many_options =
                    base_progress.choice_skipped_too_many_options + stats.choice_skipped_too_many_options;
                writer.manifest.progress.choice_skipped_chance_event =
                    base_progress.choice_skipped_chance_event + stats.choice_skipped_chance_event;

                // per-turn 合并：idx = human_turn - 1
                for i in 0..78 {
                    writer.manifest.per_turn.candidates[i] = base_per_turn.candidates[i] + stats.turn_candidates[i];
                    writer.manifest.per_turn.accepted[i] = base_per_turn.accepted[i] + stats.turn_accepted[i];
                    writer.manifest.per_turn.dropped[i] = base_per_turn.dropped[i] + stats.turn_dropped[i];
                    writer.manifest.per_turn.choice_candidates[i] =
                        base_per_turn.choice_candidates[i] + stats.turn_choice_candidates[i];
                    writer.manifest.per_turn.choice_accepted[i] =
                        base_per_turn.choice_accepted[i] + stats.turn_choice_accepted[i];
                    writer.manifest.per_turn.choice_dropped[i] =
                        base_per_turn.choice_dropped[i] + stats.turn_choice_dropped[i];
                }

                // 进度输出
                let total_games_run = writer.manifest.progress.games_run;
                if total_games_run % game_config.collector.progress_interval as u64 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
                    let accepted_total = stats.accepted;
                    let accepted_new_total = accepted_total.saturating_sub(accepted_base);
                    let accepted_per_s = if elapsed > 0.0 { accepted_new_total as f64 / elapsed } else { 0.0 };

                    let action_candidates = stats.candidates;
                    let action_accepted_new = stats.action_accepted;
                    let action_accept_rate = if action_candidates > 0 {
                        action_accepted_new as f64 / action_candidates as f64
                    } else {
                        0.0
                    };

                    let choice_candidates = stats.choice_candidates;
                    let choice_accepted_new = stats.choice_accepted;
                    let choice_accept_rate = if choice_candidates > 0 {
                        choice_accepted_new as f64 / choice_candidates as f64
            } else {
                0.0
            };
            println!(
                        "[games={}] accepted_total={} (+{}) action:+{}/{} ({:.2}%) choice:+{}/{} ({:.2}%) dropped={} choice_dropped={} accepted/s={:.2} search_errors={} elapsed={:.1}s",
                        total_games_run,
                        accepted_total,
                        accepted_new_total,
                        action_accepted_new,
                        action_candidates,
                        action_accept_rate * 100.0,
                        choice_accepted_new,
                        choice_candidates,
                        choice_accept_rate * 100.0,
                stats.dropped,
                        stats.choice_dropped,
                accepted_per_s,
                stats.search_errors,
                elapsed,
                    );
                }

                // 每局写一次 manifest，尽量减少“part 写完但 manifest 未更新”的窗口
                writer.save_manifest()?;
        }
            Err(e) => {
                games_failed_delta += 1;
                // 丢弃本局缓冲（避免把异常局的半局数据混入）
                let _ = trainer.drain_samples();
                eprintln!("warn: 模拟失败：{e}");
                continue;
            }
        }
    }

    // flush 最后一片
    writer.finish()?;

    // 更新 scoreMean 分位数（基于 score_mean_values.bin；必要时 open_or_create 已自动对齐/重建）
    let mut score_means = load_score_mean_values(writer.score_mean_values_path())?;
    writer.manifest.score_mean = calc_score_mean_summary(&mut score_means);

    // 最终保存 manifest（补齐 search_config_effective / score_mean）
    writer.manifest.search_config_effective = manifest_search_config;
    writer.save_manifest()?;

    println!("\n=== 完成 ===");
    println!("output_dir_effective  : {}", output_dir.display());
    println!("games_run             : {}", writer.manifest.progress.games_run);
    println!("games_failed          : {}", writer.manifest.progress.games_failed);
    println!("accepted_total        : {}", writer.accepted_written());
    println!("action_candidates     : {}", writer.manifest.progress.candidates);
    println!("action_dropped        : {}", writer.manifest.progress.dropped);
    println!("choice_candidates     : {}", writer.manifest.progress.choice_candidates);
    println!("choice_accepted       : {}", writer.manifest.progress.choice_accepted);
    println!("choice_dropped        : {}", writer.manifest.progress.choice_dropped);
    println!("search_errors         : {}", writer.manifest.progress.search_errors);
    println!(
        "scoreMean             : min={:?} mean={:?} p50={:?} p90={:?} p99={:?}",
        writer.manifest.score_mean.min.map(|v| v.round() as i64),
        writer.manifest.score_mean.mean.map(|v| v.round() as i64),
        writer.manifest.score_mean.p50.map(|v| v.round() as i64),
        writer.manifest.score_mean.p90.map(|v| v.round() as i64),
        writer.manifest.score_mean.p99.map(|v| v.round() as i64),
    );
    println!("elapsed               : {:?}", start.elapsed());

    Ok(())
}
