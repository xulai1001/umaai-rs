# UmaAI-RS 项目特定上下文

最后核对于 2026-09-21（合宿休息审计工具提交后）；与 `crates/umasim` / `crates/umaai` 当前代码对齐。

## 项目结构

### 工作空间
- Cargo 工作空间，成员三个 crate：`crates/umasim`（核心模拟库）+ `crates/umaai`（PC 通道层）+ `crates/umaai_review`（单局复盘分析，见「umaai_review 复盘分析层」节）
- `crates/unused/` 是遗留 bin 脚本（无 Cargo.toml，不参与构建），仅作历史参考
- workspace 根目录即项目根（AGENTS.md 所在目录）；`.portable` 为符号链接（勿跨盘操作）

### umasim（核心层，`crates/umasim`）
- **核心层不含终端 UI / 网络 / 文件监视依赖**（feature 拆分后：cli/diag/onnx/profiler 全部门控）
- **features**：
  - `cli`：终端交互与日志（inquire / colored / comfy-table / env_logger / flexi_logger / clap）
  - `diag`：规则层诊断日志宏体（`diag!` 裁剪）
  - `onnx`：tract-onnx / ndarray（神经网络评估）
  - `profiler`：pprof-rs（sim_profiler / mcts_profiler，仅 Linux/macOS）
  - 默认 `default = ["cli", "diag"]`；`umaai` 依赖 `default-features = false, features = ["cli"]`
- **模块结构**（`src/`）：
  - `game/`：规则与状态（`base` 通用 / `onsen` 温泉 / `ramen` 拉面 / `inherit` / `simulator` / `traits` / `uma`）
  - `gamedata/`：静态数据（uma / support_card / event / config / onsen / ramen）
  - `search/`：MCTS 搜索（flat_search / config / result / seeds / terminal / ramen_terminal）
  - `trainer/`：训练员（手写 / MCTS / NN / collector，见「训练员」节）
  - `output/`：决策 / 视图 / 理由 / sink / 决策日志（见「输出层」节）
  - `rng.rs`：无状态三流随机体系（见「RNG」节）
  - `bench.rs`：批量模拟与基线（`GameOutcome` / `load_player_builds`）
  - `sampler.rs` / `sample_collector.rs` / `training_sample.rs` / `neural/`：数据采集与 NN 管线
  - `explain.rs` / `utils.rs`：可读说明与工具
- **bins**（均按项目惯例用 lexopt 解析参数）：
  - `umasim`（`src/main.rs`，需 `cli`）：命令行单局模拟入口，按配置 trainer/scenario 跑局
  - `bench_base`：固定种子批量跑批（`--runs` / `--seed` / `--log` / `--out` / `--trainer` / `--deck` 覆盖卡组 / `--builds` build 过滤 / `--tokens` 手写变体 / `--region-weak-cover`，mcts 另有 `--search-*` 簇）
  - `bench_compositions`：遍历 101 种卡构成（`--runs` / `--seed` / `--friend` / `--trainer random|handwritten|recommended` / `--min-panel` / `--pool-size` / `--pick`，recommended 加 `--region-*` 覆盖）
  - `ramen_manual`（需 `cli`+`diag`）：玩家手动拉面局（inquire 交互，强制 `scenario="ramen"` + `trainer="manual"`）
  - `ramen_player`（需 `cli`+`diag`）：手动决策记录器，从 `bench_config.toml [player_builds]` 取 build 跑局，产出 GameOutcome CSV + 决策 CSV
  - `mcts_profiler`（需 `profiler`）：MCTS 性能剖析
  - `sim_profiler`（`tools/data_collection/`，需 `profiler`）：模拟 CPU/堆剖析
  - `ramen_teacher_collect` / `ramen_export_npy` / `ramen_space_bench`（需 `cli`）：NN 教师数据采集 / 导出 / 动作空间扫描
  - `trainer_overhead_diagnostic` / `mcts_rollout_switch_verify` / `calc_training_value_microbench`（`tools/data_collection/`）：诊断与微基准
  - `composition_profile_matrix` / `minimal_strategy_ab`：实验用（前者已退役，仅打印结论）
- `ramen_mcts_pair_bench`：**MCTS 训练员 vs 手写逻辑整局配对基准**（同 (build, 种子, 局号) 下两策略各跑整局、共享规则主种子，配对差 Δ = 评分_mcts − 评分_handwritten，逐局 CSV + 均值/SE/95%CI 汇总；MCTS 参数默认取生产实际值，见「MCTS vs 手写评估」节）
- `ramen_region_topk`：**region 决策点 top-K 候选 dump**（手写策略推进到 turn 2/23/47 RegionSelect 决策点后跑一次 FlatSearch，输出 top-K 候选的 mean/stdev/count/weighted_mean/was_chosen CSV + top1-top2 Δmean 与 stdev 中位数 + mean vs radical 排序差异汇总；用于研究 region 门控下 top 选项的均值差与方差分布，见「region 决策点分析」节）

### umaai（通道层，`crates/umaai`）
- **职责**：监听 `thisTurn.json` → 按 `scenarioId` 分发重建游戏 → AI 决策 → sink 输出（屏幕 / stdout JSON）
- **features**：`release-pause`（发布版出错时暂停等待按键）；`onnx`（拉面杯神经网络，见「拉面杯神经网络」节）
- **模块结构**（`src/`）：
  - `main.rs`：薄调度（Args 解析 / 初始化 / watch 循环 + 按 `ParsedGame` 分发 / run_evaluate）
  - `protocol/`：协议层（`mod.rs` 的 `GameStatus` trait + `GameStatusBase` + `extract_scenario_id` / `parse_game_by_scenario` / `ParsedGame`；`onsen.rs` / `ramen.rs` / `story.rs` / `urafile.rs`）
  - `scenario/`：剧本处理块（`onsen.rs` 的 `process_onsen` / `ramen.rs` 的 `process_ramen`，对训练员泛型）
  - `ramen_nn.rs`：客户端拉面决策器装配（按 `ramen_trainer_policy` 选搜索训练员、网络参考壳或整局网络；`ramen_nn/hint.rs` 与 `ramen_nn/whole.rs`，`onnx` feature）
  - `decision/`：决策后处理（`luck_score.rs` 的 `LuckScoreTracker` / `mod.rs` 的 emit 系列 / `record.rs` 在线记录器 / `zip_export.rs` 局末打包）
  - `plot/`：自绘 SVG（`svg.rs` 构建器 + `luck_trend.rs` 单局趋势图）
  - `utils.rs`：终端热键与保存（`SAVED_GAME` / `handle_f2`）
- **bins**（`src/bin/` 自动发现）：
  - `ramen_turn_inspect`：单份拉面 JSON → into_game + MCTS → human 诊断输出
  - `luck_replay`：按局/回合/序号重放快照 → 决策明细 + 波动统计 CSV（`--games` / `--search-n` / `--seed` / `--limit`）
  - `luck_probe`：可覆盖起始 bonus/rmj/pt 的整局模拟探针

### 脚本工具（`scripts/`）
- `export_support_card/`：支援卡数据导出
- 绘图与扫参：`plot_luck_trend.py`（运气分趋势图，与 `plot/` Rust 版样式对齐）、`plot_ptfavor_scan.py` / `summarize_ptfavor_scan.py`（pt_favor_rate 扫参汇总）
- `analyze_rest_picks.py`：合宿/满体力「一选休息」审计——按 `is_xiahesu` 回合与体力分桶统计决策日志休息率（配 bench_base 决策日志 `vital` 列，见 issues.md 对应条目）
- `deck_crn_compare.py`：卡组 CRN 对比
- `friend_pacing_compare.py`：友人出行配额（`fcap` 等 token）配对对比——同 build 同 seed 配对差 + 走完率 / 逐年出行次数 / 风味浪费等结构指标
- `bench_commit_compare.py`：跨 commit CPU 耗时对比编排（配合 `perf_probe` bin，见下「性能监测」节）
- `ramen_nn/`：NN 管线脚本

## 配置文件

### 游戏数据（`gamedata/`）
- `cardDB.json` / `umaDB.json` / `events.json` / `text_data_dict.json`：静态数据
- `scenario_onsen.json` / `scenario_ramen.json`：剧本配置（含拉面杯机制、`five_status_limit_base` 等）
- `constants.json`：公共常量（`no_event_turns` 等）
- `default_config.toml`：开发者默认配置（按 simulation / search / policy / output / dev 五段组织，平铺解析）
- `game_config.toml`（workspace 根）：用户覆盖配置
- `bench_config.toml`（workspace 根）：**仅 bench 系列 bin 专用**（uma / friend / blue_count / extra_count / runs / seed / out_dir / decision_log / trainer / `[player_builds]`），不参与游戏运行时配置

### 配置加载（`umasim/src/utils.rs` + `gamedata/config.rs`）
- `load_game_config()`：统一读 `gamedata/default_config.toml` + `game_config.toml` merge 后 `validate_game_config`
- 路径解析：`resolve_default_config_path()`（环境变量 `UMAI_DATA_DIR` > 工作目录 + `gamedata/`）；`resolve_user_config_path()`
- 用户配置缺失时自动用兜底 `OverrideGameConfig`，不阻塞启动

### 五个子配置 + 聚合壳
- `SimulationConfig`：剧本 / 训练员 / 马娘 / 卡组 / 模拟次数
- `SearchConfig`：`MctsConfig`（search_n / radical_factor_max / max_depth / rollout_evaluator / use_ucb / search_group_size / search_cpuct / expected_search_stdev / crn_stage_reseed / ramen_search_stages 等）+ 用户可调项（`mcts_turn_bonus` / `pt_favor_rate` / `race_grades`）
- `PolicyConfig`：拉面杯第 3 年地区选择策略（`ramen_region_strategy` all/fixed + `ramen_region_fixed`）
- `OutputConfig`：日志级别（统计级别预留）
- `DeveloperConfig`：collector + 线程数
- `GameConfig` 保留聚合壳 + `simulation()` / `search()` / `policy()` / `output()` / `dev()` 访问器；`OverrideGameConfig::merge` 处理覆盖
- 已迁移到 `default_config.toml` 的用户可调项：`mcts_turn_bonus` / `pt_favor_rate`（定档 2.0）/ `race_grades`
- 温泉遗留段（`onsen_order` / `mcts_selected_onsen`）标注「Phase 6 预期删除」，`[config_override]` 段保留（OverrideGameConfig 解析依赖）

### 在线记录开关（`luck_record`）
- `luck_record = true`（默认）：umaai 实时运行按局落盘 `logs/game{id}/`（`id` = `single_mode_chara_id`）
- `game_config.toml [config_override]` 可覆盖为 `false`；离线 sim/bench/重放工具不读该开关

### 拉面杯友人出行完成要求（`friend_complete_required`）
- `friend_complete_required = true`（默认）：5 次友人出行**必须走完**。手写策略施加完成硬门限——隐藏风味闸门不再阻断出行、"剩余出行次数 ≥ 剩余可出行回合数"时强制出行（可出行回合＝本人赛程排除必赛、夏合宿 60-63、超级拉面 72-77）
- `game_config.toml [config_override]` 可覆盖为 `false`（回到纯动态估值口径，允许主动跳过价值不足的第 5 次）
- 该开关经 `main.rs` 传给 `RamenMctsTrainer::with_friend_complete_required`，同时作用于 fallback 手写策略与搜索 rollout 基策（`FlatSearch::with_rollout_trainer`）；`bench_base` 也读该配置（token 里写 `freq` / `freqoff` 时以 token 为准）
- 跨年配额 preset 为 `[0,3,5]`（`friend_outing_cumulative_caps`：第 1 年不启用 / 第 2 年 3 / 第 3 年补满），实验 token `fcap` 可复现其它档位；成因与代价见 issues.md 对应条目

### 拉面杯神经网络（`ramen_trainer_policy`，umaai）
- 只作用于客户端实际对局的动作决策；事件选项始终走手写策略，搜索内部模拟仍是手写 rollout
- `"mcts"`（默认）：既有搜索逻辑
- `"mcts_nn_hint"`：执行与 `mcts` 完全相同（网络在随机流副本上推理），在 `mcts` 会输出决策的多候选步骤上另挂 `scenario_extra.nn_hint`（网络推荐 + 是否与执行一致），human 模式多打印一行「神经网络参考」；网络推理失败只警告、不影响执行
- `"nn"`：全部动作决策由网络 argmax 直接给出，不做搜索；来源标签 `decision_source` 区分网络推理 / 自选比赛守门 / 转交手写三类出口；远快于搜索、分数更低，是以分数换速度的选项；只适用于训练覆盖的卡组构成（清单见 `default_config.toml` 注释）
- 后两项需要 `cargo build --release --features onnx -p umaai`，并设置 `ramen_nn_model_path`（同目录需 `<模型>.onnx.json` 旁车，模型不入库）；未开 feature、缺模型、旁车或图输出契约不符时启动即报错，不回退
- 模型加载走 `RamenNnTrainer::load`：固定 batch 编译 + 图输出契约校验（单输出 / f32 / `[batch, 245]`）

## 开发环境

- 以 Windows 为主；umaai 已支持 Ubuntu/Linux 构建（`winscribe` / `windows` 依赖以 `cfg(windows)` 限定，Linux 加 `libc`）
- Shell：PowerShell（Linux 下 bash）
- Release 配置：`opt-level = 3`、`codegen-units = 1`、`lto = "thin"`、`debug = true`；`.cargo/config.toml` 按目标启用 `target-cpu=native`
- 工具链：Rust 1.98 / edition 2024；cargo fmt 使用 Nightly 格式规则，**只能由用户手动执行**

## 性能监测（跨 commit CPU 耗时对比）

监测不同 commit 之间手写 / MCTS 策略的 **CPU 耗时波动与评分区别**。工具链两级：

- `crates/umasim/src/bin/perf_probe.rs`（Release 构建）——**当前状态耗时探针**
  - `perf_probe root --turn 60 [--search-n N] [--rounds 3]`：手写策略固定种子推进到目标回合
    Train 决策点 → 生产搜索参数（继承 `game_config.toml` [mcts] 段）整根搜索；输出单行
    `ROOT label=.. turn=.. candidates=.. searched=.. best=.. best_mean=.. root_score=.. elapsed_ms=..`。
    - `root_score` = 根局面真实累计评分（`calc_score`，不乘 pt 偏好）——同种子下只随策略 /
      评分公式变化，是两版「评分区别」的确定性标记；两版一致 ⇒ 计时差纯来自代码性能。
    - `searched` = 总 rollout 数（工作量一致性依据）；`best_mean` 为搜索轴（score_pt）均值。
  - `perf_probe whole --runs 100`：手写策略整局耗时批次 + 评分统计（mean/median/min/max/std）。
  - 默认口径：`bench_config.toml` 的 uma/friend/继承因子/种子（61444）、speed preset build
    （`--deck` 覆盖）、搜索种子 61444（每轮 +round）。
- `scripts/bench_commit_compare.py`——**跨 commit 编排**（用法见下）

### bench_commit_compare.py 用法

```bash
# 对比旧 commit 与当前工作树（当前工作树含未提交改动）
python3 scripts/bench_commit_compare.py --base 04c739c --head current
# 对比两个 commit；严格可比建议显式 --search-n（否则各版用自己的 game 配置）
python3 scripts/bench_commit_compare.py --base 86de303 --head 04c739c --rounds 3 --search-n 8192
# base 版本无 perf_probe 时自动回退 bench 模式；附加 MCTS 小预算整局与手写整局
python3 scripts/bench_commit_compare.py --base 86de303 --head current --bench-mcts --whole-runs 100
```

行为要点：

- **量具自动选择**：两版都含 `perf_probe` → probe 模式（固定 Train 根默认 32/60 回合，逐轮
  交替配对，比较耗时中位数、搜索工作量一致性与两版 `root_score` 评分差）；否则 → bench 模式
  （`bench_base` 手写整局同 seed 配对耗时，可选 MCTS `--search-n 64` 小预算整局）。
- **--head current** 直接用当前工作树（不建 worktree）；其余 ref 自动 `git worktree add` 独立
  检出，各自独立 `CARGO_TARGET_DIR` 构建，默认**串行**（`--build-jobs 2` 可开并行）。
- **口径守卫**：两 commit gamedata 内容不同 → 告警（模拟数值口径不一致）；同 turn 两版
  `searched` 不同 → 告警（局面/策略已变，耗时差含漂移分量）；`root_score` 评分差直接展示。
- 产物：`logs/commit-compare/<base>__<head>_<时间戳>/`：`manifest.json`（commit/机器/参数）、
  `rounds.csv`（逐轮明细含 root_score）、`report.txt`、`run.log`（逐步执行日志——后台卡死时
  唯一能从文件系统读到进度的诊断信号）。
- 其他参数：`--dry-run`（只打印计划）、`--keep`（保留 worktree，默认删除）、`--probe-only`
  （拒绝回退 bench）、`--roots 32,60` / `--seed` / `--search-seed` / `--deck` / `--build` 等。

注意事项：

- 结果只作**同机配对对照**（逐轮交替执行抵消环境漂移），不作跨机器承诺；同 commit 耗时本身
  有 ±18% 量级漂移（见 perf_profiling.md B.3 注③），跨次数字不可直接对比。
- probe 模式要求被测 commit 内含 `perf_probe`（工具提交后产生的新 commit 才有）；对更老
  commit 自动回退 bench 模式。
- 后台运行时若 cargo 构建「产物已产出但进程不退、CPU≈0」，多为被杀任务残留的孤儿 cargo 仍
  持有目标目录 `.cargo-lock`（可用 `flock -n <该文件> -c true` 探测），需在宿主侧
  `lsof` / `fuser` 定位后 kill。

## MCTS vs 手写整局配对评估（ramen_mcts_pair_bench）

量化「相同随机局面下 MCTS 训练员比正式推荐手写策略强多少」的闭环基准：

```bash
cargo run --release --bin ramen_mcts_pair_bench -- \
    --seeds 61444,42,7 --runs 3 --out logs/mcts_pair.csv
cargo run --release --bin ramen_mcts_pair_bench -- --help   # 全部参数
```

- **配对口径**：同一 `(build, 种子, 局号)` 下 `RamenMctsTrainer` 与 `RecommendedRamenTrainer` 各跑完整局，两局共用同一规则主种子（`bench::seeded_rngs`），随机未来逐位一致；配对差 `Δ = 评分_mcts − 评分_handwritten`，CI/t 由逐局差聚合。`run_pair` 返回前校验两局 `rule_seed` 相等，配对失效即报错。
- **MCTS 参数默认 = 生产实际值**：`SearchConfig::new_game_config(&game_config)` + `game_config.mcts.ramen_search_stages`，与 umaai 在线运行同款构造（本机：search_n=8192 / stages `train,ramen,region` / UCB 开 / radical 1.4）。`--search-n` 等覆盖后不再代表生产训练员，勿与生产档混比。
- **耗时注意**：生产档 MCTS 整局约 3.4 分钟/局（region 门控第 2/3 年 120 候选占大头），扫测先用 `--runs 1` 探时间；手写侧整局 ≈ 1.3ms。
- **与 bench_base 的差异**：bench_base 各策略独立跑批不配对；本基准强制同种子配对，消除随机世界漂移，差值可归因于策略选择。
- 冒烟测试（bin 内 `#[cfg(test)]`，小预算不读 game_config）：配对守卫 / 同参两次逐位可复现 / CSV 结构。

## region 决策点分析（ramen_region_topk）

研究 region 门控纳入搜索后，**top-K 候选的均值差与方差分布**的工具（不跑整局）：

```bash
cargo run --release --bin ramen_region_topk -- \
    --turns 2,23,47 --builds speed,wisdom --seeds 61444,42,7 \
    --out logs/region_topk.csv
cargo run --release --bin ramen_region_topk -- --help   # 全部参数
```

- **推进 + 搜索**：手写策略（`RecommendedRamenTrainer`，与 MCTS rollout 基策同源）从固定种子推进到指定回合的 **RegionSelect** 决策点（turn 2/23/47 是三年地区选择回合），然后 `FlatSearch::search` 一次拿 SearchOutput。
- **CSV 列**：`seed, run, build, turn, year, candidates_total, searched_total, rank, original_idx, description, count, mean, stdev, weighted_mean, was_chosen`——每行 = 一个 region 点的 top-K 候选。
- **汇总打印**：按 (turn, build) 分组的 `top1.top2 Δmean` 中位数 / `top1.stdev` 与 `top2.stdev` 中位数 / mean 排序 vs radical 加权排序的内部 swap 数。
- **MCTS 参数默认 = 生产实际值**：与 `ramen_mcts_pair_bench` 同款取法；`--search-n` 覆盖仅限对照实验。
- **耗时**：turn=2（10 候选）单点 <2s；turn=23/47（120 候选 × search_n=8192）单点 15-20s。63 region 点（3 turn × 7 build × 3 seed）约 6 分钟。
- **首轮扫测结论（63 region 点）**：top1-top2 Δmean 中位数 16-374 分（占 top1 mean 的 ≤0.6%）；top1 vs top2 stdev 差异中位数 ±5% 且正负不定（**top-K 方差无系统差异**）；21 个 (turn, build) 组合中 5 个出现 mean 排序 vs radical 加权排序的内部 swap（多在第 3 年 power_wisdom/wisdom/speed_wisdom/sta0_wis2），但 top1 与 chosen 通常一致，差异集中在 top2-top3 位置。
- 冒烟测试（bin 内）：推进到 RegionSelect 阶段 + dump top-K + CSV 结构。

## 测试

### 概览（详见 tests_overview.md）
- 2026-09-10 口径：umasim lib 375 个 + umaai lib 12 个 + umaai bin 23 个（`test_watch` / `test_urafile` 历史遗留挂起）
- 运行命令：`cargo test --release -p umasim` / `cargo test --release -p umaai`

### 模拟游戏流程测试
- 位置：`crates/umasim/src/game/ramen/game.rs`
- `test_ramen_silent_loop`：完整拉面剧本 77 回合静默流程（端到端验证）
- `test_manual_trainer_full_game`：ManualTrainer（mock 输入 + PickFirst fallback）完整流程
- `test_manual_trainer_hint_special_path`：第 3 年 hint_special 路径
- 运行：`cargo test -p umasim <测试名>`（release 模式）

### 玩家手动测试程序
- `crates/umasim/src/bin/ramen_manual.rs`：`cargo run --release --bin ramen_manual`（inquire 交互）
- 卡组必须含新友人卡 303051-303054（idrank = cardId×10+突破等级，例如 303174 → cardId 30317 满破 4）

## 拉面杯模块结构（`crates/umasim/src/game/ramen/`）

### 模块入口（mod.rs）
- `RamenStage`：`Begin` → `Distribute` → `RamenSelect` → `SpecialSelect` → `Train` → `AfterTrain` → `NextTurn` → `RegionSelect` → `SuperRamenSelect` → `Settlement`；第 1 年地区选择后经 `BeginAfterRegionSelect`（turn 2 专用，`Begin` 前后半段之间）回 `Distribute`
- `FeelingType`（A/B/C）、`TrainingType`（速/耐/力/根/智）、`Operation`

### 核心类型（state.rs）
- `RamenGame`：通过 Deref 暴露 `BaseGame`；字段：`stage` / `persons` / `ramen: RamenState` / `current_effect: RamenEffect` / `deck_can_split` / `internal_rng`（回退）/ `rule_master` + `turn_fixed` + `strategy` + `event`（RNG 三流，见「RNG」节）
  - `newgame()`：校验卡组必须含新友人卡（idrank 303051-303054）
- `RamenState`：
  - 诀窍：`feeling_stock` / `feeling_slot` / `feeling_queue`；隐藏风味 `special_feeling`
  - 地区拉面：`selected_regions` / `current_ramen` / `super_ramen`
  - 剧本进度：`scenario_pt` / `rmj_results` / `train_level_bonus` / `eat_count`
  - 年度观测：`yearly_scenario_pt` / `yearly_eat_count` / `yearly_selected_regions` / `obs_year` / `yearly_friend_turns` / `yearly_gauge_gain` / `yearly_gauge_overflow`
  - 三阶段决策 pending：`pending_ramen` / `pending_special_targets` / `combined_decision`（`clear_pending()` 一并清空）
  - `train_feeling_type` / `absent_cards`
- `RamenEffect`：效果合并（基础 + 地区 + 超级拉面 + PT 常驻）

### Game trait 实现（game.rs）
- 阶段流转：`RamenStage::next()` 负责回合内；`Game::next()` 跨阶段（合并决策时跳过 SpecialSelect）
- `run_stage()` 分发：`run_begin`（前缀/后缀）/ `run_distribute` / `run_ramen_select` / `run_special_select` / `run_train` / `run_after_train` / `run_region_select` / `run_super_ramen_select`
- `ground_ramen_effects(rng)`：吃面效果立即落地（阶段过渡时自动触发，也可由通信模块直接调用）
- 合并决策接口：`list_combined_ramen_select_actions()` / `apply_combined_ramen_decision()`
- `deyilv()` / `distribute_hint()` / `is_shining_at()` / `calc_training_value*()` override
- `generate_events()`、`manage_persons_on_turn_start()`、`update_refresh_mind()`、`explain()` / `explain_ramen_info()`

### 动作定义（action.rs）
- `RamenAction`：三阶段承载结构（`ramen` + `special_targets` + `operation`），`apply` 按当前 stage 路由
- 构造器：`new(operation)` / `with_ramen` / `combined_select`
- `do_train()`、`distribute_super_ramen_clones()`、`try_add_clone()`、`TrainParams`

### 规则函数（rules.rs）
- 诀窍：`add_gauge` / `add_feeling` / `calc_gauge_base_distribution`
- 做面/吃面：`can_make_ramen` / `consume_for_ramen` / `list_special_targets_for` / `calc_ramen_pt_gain`（PT 增量延后到 NextTurn 结算）
- RMJ：`check_rmj`；地区：`get_region_range` / `get_region_combinations` / `validate_region_selection` / `calc_region_bonus`
- 分身：`get_region_clone_trains` / `get_super_ramen_clone_train_options`；隐藏风味：`get_turn_special_feeling`
- 诀窍槽填充：`fill_gauge_after_train` / `fill_gauge_after_non_train` / `fill_gauge_xiahesu_max`（合宿全 MAX）；训练加成：`calc_train_feeling_bonus` / `apply_friendship_gauge_bonus`

### 效果计算（effects.rs）
- `RamenTrainingEffect` / `calc_ramen_training_effect`（普通/超级拉面总入口）/ `calc_finals_effect`（超级拉面）/ `calc_normal_effect` / `calc_scenario_deyilv` / `apply_ramen_training_value`

### 事件处理（events.rs）
- `FriendEventState` / `assign_train_feeling_type`（每种诀窍至少 1 次）/ `push_hint_event`

### 手写策略（policy.rs）
- `RamenPolicy` + `RamenPolicyConfig`（`default()` / `speed_build()`）+ `RamenPolicyOutput`（`score_breakdown` + 理由）+ `RamenTrainEval`
- `decide_train` / `decide_ramen` / `decide_special` / `decide_region` / `decide_super_ramen` / `decide_event` 各决策点打分（cached 变体复用同回合计算）
- 保留的对外兼容 API：`fixed_region_selection(year_idx)`（每年取前 3 个地区）与 `fixed_super_ramen_selection()`（固定选项二，常量 `FIXED_SUPER_RAMEN_INDEX` 唯一定义处）；生产路径已改走 trainer，这两者是兜底 / 守门

### NN 管线文件
- `features.rs`：定长特征布局（人头按固定下标 + 已登场掩码，维度恒定）
- `policy_schema.rs`：**Policy 头格位冻结**（不吃面 1 + 吃面 200 + … 共 234 格；教师数据按此格位，改动即数据作废）
- `training_sample.rs`：教师样本容器（复用 `PolicySlots`；落盘格式 pilot 期，勿当契约）
- `rng_consistency.rs`：三流跨策略一致性测试（层 2 硬指标 / 层 3 隔离性）
- 测试用 ONNX 模型：workspace 根 `testsupport/onnx_fixture.rs` 当场生成最小模型（umasim / umaai 测试经 `#[path]` 引入，不进正式构建）；依赖真实权重的测试标 `#[ignore]`，显式运行时缺模型即报错

## 通用游戏模块（`crates/umasim/src/game/`）

- `traits.rs`：`Game` trait（`run_full_game` / `list_actions` / `view` 等）+ `Trainer` trait（`select_action` / `last_decision`）
- `base/`：`BasicGame` 无剧本基础逻辑（`basic.rs` / `action.rs` / `person.rs` / `mod.rs`）
- `onsen/`：温泉剧本（**保留未删除**，`scenarioId=12` 仍在线支持；onsen 走外挂 CRN，见「RNG」节）
- `ramen/`：拉面杯（见上节）
- `inherit.rs`：种马继承；`simulator.rs`：单局模拟载体；`uma.rs`：马娘状态；`support_card.rs`：支援卡

## RNG（`rng.rs` 三流体系 + 搜索 CRN）

### 顶层 `rng.rs`
- `splitmix64`（全仓库唯一权威哈希）、`derive_seed(base, parts...)`、`fork_local_stream`、`StreamTag`
- `SplitmixRng`：无状态流（`master + counter * GAMMA` 加法派生，Clone 即独立实例）
- 类型隔离三流：`TurnFixedRng`（人头/角标/hint 触发位，`run_distribute` 独占）/ `EventRng`（回合开始事件链）/ `StrategyRng`（训练/分身/比赛/吃面落地）；接错流编译不过
- `RamenGame` 持有 `rule_master` + 三流（未注入时回退旧行为）；搜索 rollout 注入 `rule_master = rollout 种子`（拉面 CRN 由规则层接管）

### 搜索层 CRN（`search/seeds.rs`）
- `RolloutSeeds::seed_at(j)`：所有候选共享第 j 轮 rollout 种子（候选索引不进种子，`test_search_invariant_to_action_order` 守护）
- onsen 保留 `FlatSearch::reseed_for_stage` / `stage_seed` / `crn_stage_reseed`（外挂重播种）；拉面已退役该调用
- 配对收益实测：onsen/拉面 corr ≈ 0.69，等效样本 ≈ 3.5x

## 搜索层（`crates/umasim/src/search/`）

- `config.rs`：`SearchConfig`（含 `MctsConfig` 明细，见「配置文件」节）
- `flat_search.rs`：`FlatSearch` MCTS 搜索主体（UCB 分配 + 并行 rollout + 结果聚合）
- `result.rs`：`ActionResult` / `SearchOutput`（terminal_stats + histogram）/ `OrderedRollouts` / `ScoreEntry`
- `searchable.rs`：`FlatSearchGame`（`fork_for_rollout` 克隆+注入 master）/ `RolloutHost` / `SearchScore`
- `ramen_terminal.rs`：拉面终局统计（`FROZEN_DIM_KEYS` / `RamenTerminal` / `RamenTerminalStats`）；`terminal.rs`：通用终局
- `seeds.rs`：`RolloutSeeds`（CRN 载体）

## 训练员（`crates/umasim/src/trainer/`）

### 决策粒度分工
- 手写逻辑走三阶段（RamenSelect→SpecialSelect→Train）；MCTS 走合并决策路径（`list_combined_ramen_select_actions` + `apply_combined_ramen_decision`）
- 手写逻辑是 MCTS rollout / 叶估值的基策（bench「handwritten」档 = `RecommendedRamenTrainer`）

### 训练员清单
- `RandomTrainer`（mod.rs）：随机决策器，体力 <45 休息 / 心情 <5 外出 / 否则训练；三阶段优先选有实质候选
- `ManualTrainer`（mod.rs）：inquire 交互（`new()` 真玩家 / `with_mock_inputs` 测试模式）
- `HandwrittenTrainer`（handwritten_trainer.rs）：温泉手写基策（`default()` / `speed_build()` / `.verbose()`）
- `MctsTrainer`（mcts_trainer.rs）：温泉 MCTS（`MctsConfig` 驱动）
- `RamenHandwrittenTrainer`（ramen_handwritten_trainer.rs）：拉面手写（三阶段决策，`last_decision()` 有实现）
- `RamenMctsTrainer`（ramen_mcts_trainer.rs）：拉面 MCTS + `RamenSearchStages`（哪些阶段走搜索：`train,ramen` 实测互补）；`stash_last_summary` 缓存候选评分（运气分走 calc_score 轴真实评分）
- `RecommendedRamenTrainer`（local_ramen_trainer.rs）：**正式推荐拉面策略**（手写基策 + GA 调参 9 旋钮 preset：pt_tradeoff 16→37 / pt_tradeoff_super 0→35 / region_weak_cover_weight 查表→35 / region_youqing_weight 1.5→0.4 / hint_bonus 6→8 / max_base_score_sacrifice 140→200 / ramen_window_weight 0.10→0.15 / checkpoint_scale 0→0.15 / Y1 pt_rate 16→56）；`with_tokens` / `with_region_weights` 覆盖入口
- `LocalRamenTrainer`（local_ramen_trainer.rs）：拉面本地修正策略（实验载体）
- `RamenNnTrainer`（ramen_nn_trainer.rs）：拉面 NN 策略 + `SpecialSelectMode`
- `LoggingTrainer`（logging_trainer.rs）：包一层记录决策
- `CollectorTrainer`（collector_trainer.rs）：训练数据采集（40% 探索率）
- `canonical_ramen_select_root`（ramen_special_root.rs）：拉面吃面候选规范化（搜索根）

## 输出层（`crates/umasim/src/output/`）

- `decision.rs`：`DecisionInfo`（action_index / score / candidate_scores / candidate_n / scenario_extra 等）
- `view.rs`：`GameView`（Game trait 的 `view()` 状态快照）
- `reason.rs`：`DecisionReasonData` + `DecisionReasonSink` trait + `DecisionReasonNoopSink` / `LogJsonSink`；`render_reason_lines()`
  - **触发逻辑**：每回合都输出；`reason_gap_threshold` 保留但不再作触发器（仅决定颜色档位）
  - 行结构：`[回合 N] 首选: <描述>`（亮绿）+ `[回合 N] #K <描述>: ±分差（优势/劣势子项）`
  - 颜色档位：与首选差距 `<30` / `<100` / `<300` / 其余 → 亮绿 / 绿 / 黄 / 灰（`reason_color(gap)`）
  - no-color feature：`--features no-color` 编译通过，输出无 ANSI
- `sink.rs`：`DecisionSink` trait + `EmptySink` / `HumanReadableSink`（屏幕打印）/ `StdoutJsonSink`（`--json` 模式 stdout 严格 JSON）
- `turn_flow.rs`：`RecordingTrainer` / `TurnDecision`（回合决策记录）
- `decision_log.rs`：`DecisionLog` / `DecisionLogRow`（bench 决策轨迹 CSV）
- `diagnostic.rs`：`diag!` 宏（规则层诊断日志，编译期裁剪）

## umaai 通道层细节

### 协议（`protocol/`）
- `mod.rs`：`GameStatus` trait（`scenario_id()` + `into_game()`）；`GameStatusBase`（camelCase 通用段）；`extract_scenario_id(contents)` 窥探分发；`parse_game_by_scenario` → `ParsedGame::Onsen(12)` / `ParsedGame::Ramen(14)`；`BasePersonStatus` / 反向 `From<&BaseGame>`
- `onsen.rs`：`GameStatusOnsen`（scenarioId=12）
- `ramen.rs`：`GameStatusRamen`（scenarioId=14；拉面段 12 字段全覆写 + `single_mode_chara_id` 切局键 + stage dispatch 按 playing_state 1/5/45/46/48 → Train/Event/Settlement/SuperRamen）
- `story.rs`：`StoryStatus`（事件选项信息，`select_event_choice` 用）
- `urafile.rs`：`UraFileWatcher`（notify 监听 `thisTurn.json`；错误事件重读兜底 + 空事件心跳）

### 场景处理（`scenario/`）
- `onsen.rs`：`process_onsen(game, trainer, sink, luck_tracker, rng, json_mode, emit_info, game_config)`（newgame 检测 / 事件训练分发 / emit）
- `ramen.rs`：`process_ramen(game, single_mode_chara_id, trainer, reason_slot, sink, luck_tracker, rng, json_mode, emit_info)`（Begin 早退 / 切局检测 / 链式决策 emit / `compute_next_step`）

### 决策后处理（`decision/`）
- `luck_score.rs`：`LuckScoreTracker`（T(1) / T(n) baseline 按局数加权 / total_luck / last_turn_delta / 跨 chara_id 重置）
- `mod.rs`：`LastReasonSink`（缓存最近一次理由供 human 上屏）+ `emit_with_luck` / `emit_with_luck_decision`
- `record.rs`：`RecordingSink` + `OnlineRecorder`（全局 `RECORDER: OnceLock<Mutex<Option<...>>>`；`init` 前所有入口 no-op，离线工具不会误写）
- `zip_export.rs`：`zip_and_cleanup`（局末把 `logs/game{id}/` 打成 zip 后清理原目录，失败保留）

### 在线决策记录（`decision/record.rs` + `plot/`）
umaai 实时监听时把「接收到的游戏数据」与「策略计算结果」按局落盘，**每局一个目录、产物平铺**：

| 文件 | 内容 |
|---|---|
| `game{id}_turn{turn}[_{seq}].json` | watch 收到的 `thisTurn.json` **原文**（每份一文件；序号与 SendGameStatusPlugin 归档同口径） |
| `decisions.csv` | 逐决策点明细（与离线 `luck_replay` **同 schema**；`step` / `chain_len` 在线留空） |
| `meta.json` | 局元信息：起止时间 / 起始回合 / `mid_entry` / 结束原因 / `snapshots` / `csv_rows` / `decision_rows` / `total_luck_end` |
| `luck_trend.svg` | 该局运气分趋势图（3 子图：期望评分 / 运气分 / 运气波动；局数据完整收尾时**自动生成**） |

- **挂载点**：`main.rs` watch 循环 parse 后调 `record::on_snapshot`；输出 sink 外包 `RecordingSink`
- **切局 / 收尾**：`chara_id` 变化即收尾上一局（`end_reason=switch`）；watch 循环结束（含 Err）调 `finalize_shutdown()`
- **局末自动出图**：触发点 = 末回合第 2 份快照（拉面 `turn77_2`）处理完后立即写 `meta.json`（`end_reason=game_end`）+ 生成 `luck_trend.svg`；切局 / 退出降级为兜底
- **局末自动打包**：仅在 `end_reason=game_end` 时，`zip_and_cleanup` 把 `logs/game{id}/` 打成 `logs/game{id}.zip`（包内条目相对原目录）并清理原目录；切局 / 中途停止不打包（保留目录方便排查）
- **本期范围**：仅拉面（`scenarioId=14`）；温泉无 `single_mode_chara_id` 切局键，未纳入

## 拉面杯在线协议（`ramen_protocol_v2.md`，定稿）

- 顶层 `thisTurn.json` 两段：`ramen`（RamenStatus 镜像）+ `baseGame`（与温泉同构 + 拉面增量字段）；`scenarioId` 14=拉面 / 12=温泉
- `ramen` 段字段：`feeling_gauge_gains[5][3]` / `feeling_gauge` / `feeling_stock` / `special_feeling` / `train_feeling_type` / `active_effect_array` / `super_ramen` / `selected_regions` / `feeling_gauge_gain_base` / `last_ramen`（= region_id）/ `scenario_pt` / `next_scenario_pt`
- 阶段顺序实测：RMJ 结算回合三连拍（ps=1 event → ps=46 → ps=45）；超级拉面选择（ps=1 → ps=5 event 400014017 → ps=46 → ps=48 → 下回合 super_ramen=0/1/2）
- 基线：151 份样本（chara 6204 全 78 回合）驱动 `test_turn_import_v2_full_samples` roundtrip 校验
- 已废止假设清单在文档 §5（last_ramen 每回合更新 / 不需要反推 / playing_state 44/45 不是地区选择 / next_scenario_pt 术语）

## umaai_review 复盘分析层（`crates/umaai_review`）

- **职责**：对在线记录器产出的局包 `logs/game{id}.zip` 做离线复盘——解包解析 →
  `digest.json`（强类型 schema：meta / timeline / decisions / execution / luck /
  schedule / inherit / clones / coverage / findings / context）+ `report.html`
  （minijinja 外置模板 `templates/report.html.j2` + `umaai::plot::svg` 自绘三图
  三表，零 JS；digest 紧凑序列化）。方案与口径见 `replay_review.md`（已实施）
- **bin**：`umaai_review`（`--zip` 必需；`--out` 默认局包同级同名目录
  `logs/game{id}/`；`--gamedata` 多级解析：显式 > `UMAI_DATA_DIR` > 局包向上找
  > cwd，全缺时降级纯 ID 展示并写进 digest 注记）
- **模块**（`src/`）：`pack`（解包 + 文件角色识别，两种布局通吃）、`gdata`、
  `timeline`（快照只反序列化 `GameStatusRamen` 取字段，**不调 `into_game`**）、
  `decisions`（CSV 按表头名取值 + 链推断 + luck 回合合计聚合）、`schedule`、
  `score`（`Uma::calc_score` 同源口径 + `get_rank_name`）、`execution`（实际
  动作推断：必赛回合兜底 + 主增量阈值判训练——智训练不耗体力、体力回升判
  休息）、`checks`（伪波动标记：年界 / 继承 / RMJ / 开局第 1 年地区选择，
  turn 72 双属性；超级拉面期盈亏；坏手法已验证三项 + 训练失败候选）、
  `inherit`（窗口 = 前回合末 → 继承回合首，剥离前回合行动）、`clones`
  （彩圈 = 分身新增落位，A/B 分开统计，B 只统计训练卡，luck/strategy 来源
  二分）、`digest`、`report`
- **依赖**：`umaai` + `umasim`（复用协议结构与口径函数）；`minijinja`（含
  `json` feature）；其余走 workspace 依赖
- **skill**：`.trae/skills/umaai_review/`（SKILL.md 六问叙事框架 + 通俗玩家
  术语 + 归因口径 + 语气基调；`reference/metrics_glossary.md` 口径速查；
  `reference/persona.md` 可选秋川理事长人设，删除即回退默认口吻）
- 测试 25 个：`cargo test --release -p umaai_review`（模板端到端、分身口径、
  继承校准等含 game6234 实测钉死值）

## 相关文档导航

- `.trae/documents/` 现行目录：changelog / issues / glossary / project_context / ramen_memo(×2) / ramen_story_flow / ramen_protocol_v2 / ramen_pt_tradeoff_tuning / replay_review / perf_profiling / tests_overview / master_mdb_data
- `.trae/documents/archive/`：已完成的方案与草案（config_refactor_plan / rng_refactor_plan(×2) / ramen_refactor_development_plan / umaai-main-refactor / umaai_air_redirector_integration / ramen_online_integration_plan / adapter_spec / 上游三层架构建议 / 性能与 GPU 方案 / handwritten_policy / 旧 issues 等）