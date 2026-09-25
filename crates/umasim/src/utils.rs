#[cfg(feature = "cli")]
use std::io::Write;
#[cfg(feature = "cli")]
use std::sync::Mutex;
use std::sync::OnceLock;

use anyhow::{Result, anyhow};
use colored::Colorize;
#[cfg(feature = "cli")]
use comfy_table::Table;
#[cfg(feature = "cli")]
use flexi_logger::{DeferredNow, Duplicate, FileSpec, style};
#[cfg(feature = "cli")]
use log::Record;
use log::{error, info};
use rand_distr::weighted::WeightedIndex;
#[cfg(feature = "cli")]
use serde::Serialize;

#[cfg(feature = "cli")]
use crate::gamedata::LOGGER;
use crate::{
    game::onsen::OnsenOrder,
    gamedata::{
        EventCollection,
        EventData,
        GAMECONSTANTS,
        GAMEDATA,
        GameConfig,
        OverrideConfig,
        OverrideGameConfig,
        OverrideMctsConfig
    }
};

pub type Array5 = [i32; 5];
pub type Array6 = [i32; 6];

/// 串行化首次 `flexi_logger::start()` 调用
///
/// 历史背景：早期用 `INIT_LOCK: Mutex<()>` + `LOGGER_INIT_DONE: AtomicBool`
/// 双重检查保护并行测试下的 flexi_logger 全局只能 init 一次的竞争。
/// 现已统一用 `std::sync::OnceLock::get_or_init` 替代——`OnceLock` 本身用 atomic 实现
/// "run-once + 同步"语义，比手写 Mutex + Atomic 更简洁。
///
/// `LOGGER_INIT_RESULT` 是占位 type（()），用于触发 OnceLock 的"只设一次"机制——
/// init 闭包真正返回的 `Result<()>` 失败信息通过 `expect()` 内部 panic：
/// log crate 启动失败属于 fatal，进程应立即退出，吞错反而难调。
///
/// 仅在 `cli` feature 下编译；core-only 构建（嵌入式/Android target）不需要此全局。
#[cfg(feature = "cli")]
static LOGGER_INIT: OnceLock<()> = OnceLock::new();

#[cfg(feature = "cli")]
pub fn log_format(w: &mut dyn Write, _now: &mut DeferredNow, record: &Record) -> Result<(), std::io::Error> {
    let level = record.level();
    write!(
        w,
        "{} {}",
        style(level).paint(level.to_string()[..1].to_string()),
        style(level).paint(record.args().to_string())
    )
}

/// 初始化日志系统（默认：写文件 + stderr）
#[cfg(feature = "cli")]
pub fn init_logger(app: &str, spec: &str) -> Result<()> {
    init_logger_with(app, spec, true)
}

/// 同 `init_logger`，但支持自定义是否输出到 stderr
///
/// - `duplicate_stderr=true`：写文件 + stderr（默认）
/// - `duplicate_stderr=false`：只写文件，不占用 stderr（TUI 兼容）
#[cfg(feature = "cli")]
pub fn init_logger_with(app: &str, spec: &str, duplicate_stderr: bool) -> Result<()> {
    LOGGER_INIT.get_or_init(|| {
        let logger = flexi_logger::Logger::try_with_str(spec)
            .expect("log spec 解析失败")
            .format_for_stderr(log_format)
            .log_to_file(FileSpec::default().directory("logs").basename(app));
        let logger = if duplicate_stderr {
            logger.duplicate_to_stderr(Duplicate::All).start()
        } else {
            // 只输出到文件，不干扰 stderr（TUI 玩家测试场景）
            logger.start()
        }
        .expect("flexi_logger start 失败（log crate 全局不可重复 init）");
        // LOGGER.set 可能失败（被其他线程抢先），但只要 start 成功，log crate 已被初始化
        let _ = LOGGER.set(Mutex::new(logger));
    });
    Ok(())
}

/// 初始化日志系统：只输出到 stdout，不写文件
///
/// 适用于 `ramen_manual` 等玩家测试场景：
/// - 日志与 `println!` 一起显示在 stdout，玩家可以直接看到训练/事件日志
/// - inquire 默认从 `/dev/tty` 读取，三者互不干扰
///
/// 注意：flexi_logger 的 `log_to_stdout` 与 `log_to_file` 互斥，
/// 所以 stdout 模式不写文件，调用方需自行处理日志持久化（如重定向 shell 输出）。
///
/// `app` 参数保留以维持公开签名稳定（与 `init_logger_with` 一致），但 stdout
/// 模式不写文件，实际不使用。
#[cfg(feature = "cli")]
pub fn init_logger_stdout(_app: &str, spec: &str) -> Result<()> {
    LOGGER_INIT.get_or_init(|| {
        let logger = flexi_logger::Logger::try_with_str(spec)
            .expect("log spec 解析失败")
            .format_for_stdout(log_format)
            .log_to_stdout()
            .start()
            .expect("flexi_logger start 失败");
        let _ = LOGGER.set(Mutex::new(logger));
    });
    Ok(())
}

/// 测试场景专用 logger：只输出到 stderr，不写文件。
///
/// 与 `init_logger` 的区别：
/// - 不写文件（避免大量测试日志堆积 `logs/test_<date>.log`）
/// - 输出到 stderr 由 cargo test 默认按测试名隔离捕获（天然不会交错）
///
/// 与生产 `init_logger` 共用：
/// - 共享全局 `LOGGER` 单例（`OnceLock<Mutex<LoggerHandle>>`）
/// - 并行测试首次 init 串行化由现有 `INIT_LOCK` + `LOGGER_INIT_DONE` 双重检查锁保证
///
/// 适用场景：仅在 `#[cfg(test)]` 模块中使用。业务 binary（umaai、ramen_manual 等）
/// 继续使用 `init_logger` / `init_logger_with` / `init_logger_stdout`。
#[cfg(feature = "cli")]
pub fn init_test_logger(spec: &str) -> Result<()> {
    LOGGER_INIT.get_or_init(|| {
        let logger = flexi_logger::Logger::try_with_str(spec)
            .expect("log spec 解析失败")
            .format_for_stderr(log_format)
            .log_to_stderr() // ⚠️ 只 stderr，不写文件
            .start()
            .expect("flexi_logger start 失败");
        // LOGGER.set 可能失败（被其他线程抢先），但只要 start 成功，log crate 已被初始化
        let _ = LOGGER.set(Mutex::new(logger));
    });
    Ok(())
}

/// Core-only 测试不携带 CLI 日志后端；保留同一公开签名，让测试无需按 feature
/// 重复分支。`log` facade 在未安装 logger 时会安全地丢弃记录。
#[cfg(all(test, not(feature = "cli")))]
pub fn init_test_logger(_spec: &str) -> Result<()> {
    Ok(())
}

/// 把当前工作目录修改为exe所在目录
pub fn check_working_dir() -> Result<()> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().expect("parent");
    println!("正在进入UmaAI所在目录: {exe_dir:?}");
    std::env::set_current_dir(exe_dir)?;
    Ok(())
}

/// 获取workspace根目录路径
///
/// 通过CARGO_MANIFEST_DIR环境变量定位workspace根目录，
/// 适用于测试和需要访问workspace级别资源（如gamedata目录）的场景。
///
/// # 返回值
/// 返回workspace根目录的PathBuf，如果无法定位则返回错误。
///
/// # 示例
/// ```rust
/// use umasim::utils::get_workspace_root;
///
/// let workspace_root = get_workspace_root().expect("无法获取workspace根目录");
/// println!("Workspace根目录: {:?}", workspace_root);
/// ```
pub fn get_workspace_root() -> Result<std::path::PathBuf> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow!("无法定位workspace根目录，请确保在正确的crate中运行"))?;
    Ok(workspace_root.to_path_buf())
}

/// 本次测试专属的临时目录（工作区 `target/test-tmp/` 下，每次调用唯一）
///
/// 不用系统临时目录的固定名字：`std::env::temp_dir()` 下写死名字时，
/// 同机并行跑的多个测试进程（`cargo test` 的多 target、或两个终端同时跑）会写到
/// 同一个路径上互相覆盖模型与旁车，出现「A 进程刚写完 B 进程就删掉」的假失败；
/// 而且清理时容易越界删到工作区以外。
///
/// 目录名为 `<tag>-<pid>-<纳秒>-<进程内序号>`，进程内序号保证同一进程里连续两次
/// 调用也不撞名。目录建在工作区 `target/` 下——`target/` 本来就是构建产物区，
/// 已被 `.gitignore` 覆盖，清理范围也天然限制在工作区内。
///
/// # 错误
///
/// 定位工作区根目录失败、取系统时间失败或建目录失败时报错。
#[cfg(test)]
pub fn unique_test_dir(tag: &str) -> Result<std::path::PathBuf> {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH}
    };

    /// 进程内自增序号：同一纳秒内的两次调用也不会撞名
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| anyhow!("取系统时间失败: {e}"))?
        .as_nanos();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = get_workspace_root()?
        .join("target")
        .join("test-tmp")
        .join(format!("{tag}-{}-{nanos}-{seq}", std::process::id()));
    fs_err::create_dir_all(&dir)?;
    Ok(dir)
}

/// 删除 [`unique_test_dir`] 建出来的目录，并**先核对它确实在预期范围内**
///
/// 只接受绝对路径且必须位于工作区 `target/test-tmp/` 之下；不满足时
/// **不删除**并报错——宁可留下垃圾目录，也不冒险递归删到别处。
///
/// # 错误
///
/// 路径不在 `target/test-tmp/` 之下，或删除失败时报错。
#[cfg(test)]
pub fn cleanup_test_dir(dir: &std::path::Path) -> Result<()> {
    let base = get_workspace_root()?.join("target").join("test-tmp");
    let abs = dir
        .canonicalize()
        .map_err(|e| anyhow!("无法规范化待删目录 {}: {e}", dir.display()))?;
    let base_abs = base
        .canonicalize()
        .map_err(|e| anyhow!("无法规范化 test-tmp 基准目录 {}: {e}", base.display()))?;
    if !abs.starts_with(&base_abs) || abs == base_abs {
        return Err(anyhow!(
            "拒绝删除 {}：不在 {} 之下（只清理本次建出来的子目录）",
            abs.display(),
            base_abs.display()
        ));
    }
    fs_err::remove_dir_all(&abs)?;
    Ok(())
}

/// 测试观测收集器：全程只打印，末尾汇总失败
///
/// `AGENTS.md` 规定测试用 `println` 而非 `assert` 宏——中途 panic 会丢掉后续诊断信息。
/// 但只打印不失败的话，回归时 `cargo test` 仍报 ok，等于没有防线。
/// 折中方案：每条观测都打印 `OK` / `NG`，**末尾**用 [`Checks::finish`] 汇总，
/// 有 NG 才返回 `Err`——既保留完整诊断输出，又保留失败能力。
///
/// # 示例
/// ```ignore
/// let mut c = Checks::new();
/// c.check(v.len() == INPUT_DIM, "维度等于 INPUT_DIM");
/// c.check(v.iter().all(|x| x.is_finite()), "不含 NaN / Inf");
/// c.finish()   // 有 NG 则 Err，列出全部失败项
/// ```
#[cfg(test)]
#[derive(Default)]
pub struct Checks {
    failed: Vec<String>
}

#[cfg(test)]
impl Checks {
    /// 新建一个空的观测收集器
    pub fn new() -> Self {
        Self { failed: Vec::new() }
    }

    /// 记录一条观测并打印 `OK` / `NG`
    pub fn check(&mut self, ok: bool, what: &str) {
        println!("  [{}] {what}", if ok { "OK" } else { "NG" });
        if !ok {
            self.failed.push(what.to_string());
        }
    }

    /// 汇总：有 NG 则返回 `Err`（列出全部失败项）
    pub fn finish(self) -> Result<()> {
        if self.failed.is_empty() {
            return Ok(());
        }
        Err(anyhow!("{} 项观测未通过: {}", self.failed.len(), self.failed.join(" / ")))
    }
}

/// 检测终端类型（Windows 平台彩色提示）
#[cfg(feature = "cli")]
pub fn check_windows_terminal() -> Result<()> {
    if !std::env::var("WT_SESSION").is_ok() {
        println!(
            "{}",
            "警告: 当前终端不是Windows Terminal或版本太老，可能出现乱码或显示不全".yellow()
        );
        println!(
            "{}",
            "UmaAI推荐使用最新版Windows Terminal终端，以获得更好的体验".bright_green()
        );
        pause()?;
    }
    Ok(())
}

/// 非 cli 模式下的 stub（保持公开签名稳定，调用方无需 cfg gate）
#[cfg(not(feature = "cli"))]
pub fn check_windows_terminal() -> Result<()> {
    Ok(())
}

/// 等待用户按 Enter 键继续（cli 容器，含 colored / comfy-table 等所有 CLI 子模块）
#[cfg(feature = "cli")]
pub fn pause() -> Result<()> {
    println!("按任意键继续...");
    std::io::stdin().read_line(&mut String::new())?;
    Ok(())
}

/// 把可序列化数组渲染为表格字符串（cli 容器，含 comfy-table）
#[cfg(feature = "cli")]
pub fn make_table<T: Serialize>(data: &[T]) -> Result<Table> {
    let mut table = Table::new();
    table.set_truncation_indicator("...");
    let mut has_headers = false;
    for row in data {
        if !has_headers {
            let header = serde_json::to_value(row)?;
            table.set_header(header.as_object().expect("map").keys());
            has_headers = true;
        }
        let row = serde_json::to_value(row)?;
        table.add_row(row.as_object().expect("row").values());
    }
    Ok(table)
}

/// 把"运气值"格式化为带颜色的字符串
///
/// colored 无条件加载；启用 `no-color` feature 时 `colored::Colorize` 编译期为
/// 纯字符串输出，业务代码不需要 cfg gate。
pub fn format_luck(prefix: &str, luck: f64) -> String {
    let luck_str = if luck < 0.0 {
        format!("{luck:.0}")
    } else {
        format!("+{luck:.0}")
    };
    if luck < -1600.0 {
        format!("{prefix} {}", luck_str.red())
    } else if luck < -400.0 {
        format!("{prefix} {}", luck_str.yellow())
    } else if luck < 400.0 {
        format!("{prefix} {luck_str}")
    } else if luck < 1600.0 {
        format!("{prefix} {}", luck_str.green())
    } else {
        format!("{prefix} {}", luck_str.bright_green())
    }
}

#[macro_export]
macro_rules! global {
    ($name:ident) => {
        $name.get().expect(concat!(stringify!($name), " not initialized"))
    };
}

pub fn global_events() -> &'static EventCollection {
    &global!(GAMEDATA).events
}
/// 复用全局只读常量表的随机事件分布，在首次随机事件采样时初始化。
pub fn global_event_distribution() -> &'static WeightedIndex<f64> {
    static DISTRIBUTION: OnceLock<WeightedIndex<f64>> = OnceLock::new();
    DISTRIBUTION.get_or_init(|| {
        WeightedIndex::new(global!(GAMECONSTANTS).get_event_distribution()).expect("event weights")
    })
}
/// 获得events.json里记载的指定system事件
pub fn system_event(key: &str) -> Result<&'static EventData> {
    global_events()
        .system_events
        .get(key)
        .ok_or_else(|| anyhow!("未知系统事件: {key}"))
}
/// 获得constants.json里记载的指定事件概率
pub fn system_event_prob(key: &str) -> Result<f64> {
    global!(GAMECONSTANTS)
        .event_probs
        .get(key)
        .map(|x| *x as f64)
        .ok_or_else(|| anyhow!("未知事件概率: {key}"))
}

pub trait AttributeArray {
    fn add_eq(&mut self, other: &Self) -> &mut Self;

    fn is_default(&self) -> bool;
}

impl<const N: usize> AttributeArray for [i32; N] {
    fn add_eq(&mut self, other: &Self) -> &mut Self {
        if self.len() != other.len() {
            error!("self: {self:?}, other: {other:?}");
            panic!("数组长度不匹配, 请检查调用代码");
        }
        for (i, x) in self.iter_mut().enumerate() {
            *x += other[i];
        }
        self
    }

    fn is_default(&self) -> bool {
        self.iter().all(|x| *x == 0)
    }
}

pub fn split_status(status_pt: &Array6) -> Result<(&Array5, i32)> {
    let left: &Array5 = status_pt[..5].try_into()?;
    let right = status_pt[5];
    Ok((left, right))
}

// ========== 路径常量（Phase 2 步骤 4：加载集中化） ==========
//
// 路径解析优先级（从高到低）：
//   1. 环境变量 `UMAI_DATA_DIR`：data 根目录（含 gamedata/default_config.toml）
//   2. 工作目录下 `gamedata/default_config.toml`（默认）
//
// 用户配置 `game_config.toml` 始终位于工作目录根（Phase 6 可考虑移至 `UMAI_DATA_DIR`）。

/// 默认配置（开发者默认值）相对于 data 根目录的相对路径
pub const DEFAULT_CONFIG_REL_PATH: &str = "default_config.toml";
/// 用户配置（覆盖层）相对于工作目录的相对路径
pub const USER_CONFIG_REL_PATH: &str = "game_config.toml";
/// data 根目录（gamedata/）相对于工作目录的相对路径
pub const DATA_DIR_REL_PATH: &str = "gamedata";
/// 环境变量名：覆盖 data 根目录的绝对路径
pub const ENV_DATA_DIR: &str = "UMAI_DATA_DIR";

/// 解析 data 根目录绝对路径：优先用环境变量 `UMAI_DATA_DIR`，否则用工作目录 + `gamedata/`
pub fn resolve_data_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var(ENV_DATA_DIR) {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from(DATA_DIR_REL_PATH)
    }
}

/// 解析默认配置绝对路径
pub fn resolve_default_config_path() -> std::path::PathBuf {
    resolve_data_dir().join(DEFAULT_CONFIG_REL_PATH)
}

/// 解析用户配置绝对路径
pub fn resolve_user_config_path() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_default().join(USER_CONFIG_REL_PATH)
}

/// 校验 GameConfig 关键字段（Phase 2 步骤 4：加载集中化）
///
/// 业务模块不应自行校验字段格式；统一在此处报错。当前覆盖：
/// - `scenario`：枚举合法性
/// - `trainer`：枚举合法性
/// - `cards`：长度 = 6
/// - `ramen_region_fixed`（fixed 策略时）：长度 = 1
pub fn validate_game_config(config: &GameConfig) -> Result<()> {
    match config.scenario.as_str() {
        "basic" | "onsen" | "ramen" => {}
        other => anyhow::bail!("未知 scenario={other:?}，应为 basic | onsen | ramen")
    }
    match config.trainer.as_str() {
        "manual" | "random" | "handwritten" | "collector" | "neuralnet" | "mcts" => {}
        other => anyhow::bail!("未知 trainer={other:?}")
    }
    if config.cards.len() != 6 {
        anyhow::bail!("cards 长度应为 6，实际 {}", config.cards.len());
    }
    if matches!(
        config.ramen_region_strategy,
        crate::gamedata::RamenRegionStrategy::Fixed
    ) {
        match &config.ramen_region_fixed {
            Some(fixed) if fixed.len() == 1 => {}
            Some(fixed) => anyhow::bail!(
                "ramen_region_strategy=fixed 但 ramen_region_fixed 长度 = {}（应为 1）",
                fixed.len()
            ),
            None => anyhow::bail!("ramen_region_strategy=fixed 但未设置 ramen_region_fixed")
        }
    }
    Ok(())
}

/// 用户配置文件不存在时的覆盖层：所有覆盖字段为 `None`，merge 后完整保留 `default_config.toml`。
///
/// 本函数**不走 serde**，因此 `#[serde(default)]` 碰不到这条路径；
/// `mcts` 必须是 [`OverrideMctsConfig::default`]（全 `None`），
/// 不能写成带代码缺省值的 `MctsConfig`。
pub(crate) fn fallback_override_game_config() -> OverrideGameConfig {
    OverrideGameConfig {
        onsen_order: OnsenOrder::default(),
        config_override: OverrideConfig {
            uma: None,
            cards: None,
            blue_count: None,
            extra_count: None,
            trainer: None,
            mcts_selected_onsen: None,
            log_level: None,
            num_threads: None,
            mcts_turn_bonus: None,
            pt_favor_rate: None,
            race_grades: None,
            luck_record: None,
            friend_complete_required: None
        },
        mcts: OverrideMctsConfig::default(),
        ramen_region_strategy: None,
        ramen_region_fixed: None,
        ramen_trainer_policy: None,
        ramen_nn_model_path: None
    }
}

/// 载入 gamedata/default_config.toml, 和 game_config.toml 合并
pub fn load_game_config() -> Result<GameConfig> {
    let def_path = resolve_default_config_path();
    let def_file = fs_err::read_to_string(&def_path)?;
    let default_config: GameConfig = toml::from_str(&def_file)?;

    let cfg_path = resolve_user_config_path();
    let override_config: OverrideGameConfig = if cfg_path.exists() {
        let cfg_file = fs_err::read_to_string(&cfg_path)?;
        toml::from_str(&cfg_file)?
    } else {
        info!(
            "用户配置不存在（{}），使用默认配置 + OverrideGameConfig 兜底",
            cfg_path.display()
        );
        fallback_override_game_config()
    };

    let merged = override_config.merge(&default_config);
    validate_game_config(&merged)?;
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 共享事件分布与原构造的抽样结果和随机流推进一致，重复取得时复用同一实例。
    #[test]
    fn test_global_event_distribution() -> Result<()> {
        use std::{env::set_current_dir, ptr};

        use rand::RngCore;
        use rand_distr::Distribution;

        use crate::{gamedata::init_global, rng::EventRng};

        set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let original = WeightedIndex::new(global!(GAMECONSTANTS).get_event_distribution())?;
        let mut original_rng = EventRng::new(61444);
        let mut shared_rng = original_rng.clone();
        let mut counts = [0; 4];
        let mut same = true;
        for _ in 0..1024 {
            let expected = original.sample(&mut original_rng);
            let actual = global_event_distribution().sample(&mut shared_rng);
            same &= actual == expected;
            counts[actual] += 1;
        }
        println!("共享事件分布抽样计数: {counts:?}");
        let mut c = Checks::new();
        c.check(same, "事件类别抽样序列一致");
        c.check(original_rng.next_u64() == shared_rng.next_u64(), "采样后随机流位置一致");
        c.check(ptr::eq(global_event_distribution(), global_event_distribution()), "事件分布实例复用");
        c.finish()
    }

    /// 缺文件兜底：手写构造路径 merge 后必须是生产值 12288 / 1.4，不是代码缺省 10240 / 2.0。
    ///
    /// 若把兜底改回 `MctsConfig::default()` 的等价物（`Some(10240)` / `Some(2.0)`），本测试必须红。
    #[test]
    fn test_missing_user_config_keeps_production_mcts() -> Result<()> {
        let root = get_workspace_root()?;
        let def_path = root.join("gamedata").join("default_config.toml");
        let default_config: GameConfig = toml::from_str(&fs_err::read_to_string(&def_path)?)?;
        let merged = fallback_override_game_config().merge(&default_config);
        println!(
            "缺文件兜底 merge: search_n={} radical_factor_max={} search_group_size={} expected_search_stdev={} rollout_batch_size={}",
            merged.mcts.search_n,
            merged.mcts.radical_factor_max,
            merged.mcts.search_group_size,
            merged.mcts.expected_search_stdev,
            merged.mcts.rollout_batch_size
        );
        let mut c = Checks::new();
        c.check(merged.mcts.search_n == 12288, "缺文件兜底 search_n == 12288（不是 10240）");
        c.check(
            merged.mcts.radical_factor_max == 1.4,
            "缺文件兜底 radical_factor_max == 1.4（不是 2.0）"
        );
        c.check(
            merged.mcts.search_group_size == default_config.mcts.search_group_size,
            "缺文件兜底不践踏 search_group_size"
        );
        c.finish()
    }

    #[test]
    fn test_validate_game_config_scenario_enum() {
        let mut cfg = GameConfig::default_for_init();
        cfg.scenario = "ramen".to_string();
        assert!(validate_game_config(&cfg).is_ok());

        cfg.scenario = "bogus".to_string();
        assert!(validate_game_config(&cfg).is_err());
    }

    /// 回归：default_config.toml 的 ramen_region_strategy/fixed 必须解析进 GameConfig
    ///
    /// 历史 bug：这两个顶层字段曾误落在 `[mcts]` 段内被吞掉，导致加载后策略恒为
    /// `All`（第3年枚举 120 组合）。顶层平铺字段必须在任意 `[xxx]` 段之前声明。
    #[test]
    fn test_default_config_ramen_region_fixed() {
        let root = get_workspace_root().expect("workspace root");
        std::env::set_current_dir(root).expect("set cwd");
        let cfg = load_game_config().expect("load_game_config 失败");
        println!(
            "ramen_region_strategy = {:?}, ramen_region_fixed = {:?}",
            cfg.ramen_region_strategy, cfg.ramen_region_fixed
        );
        println!(
            "应为 Fixed + Some([[11,14,15]])，实际 log_level = {}（也应来自 default_config 而非 serde 兜底）",
            cfg.log_level
        );
    }

    #[test]
    fn test_validate_game_config_trainer_enum() {
        let mut cfg = GameConfig::default_for_init();
        cfg.trainer = "manual".to_string();
        assert!(validate_game_config(&cfg).is_ok());

        cfg.trainer = "unknown".to_string();
        assert!(validate_game_config(&cfg).is_err());
    }

    /// `[config_override] trainer` 应能覆盖 `default_config.toml` 的 trainer。
    ///
    /// 回归：trainer 是 GameConfig 顶层字段，原本不在 OverrideConfig 里，
    /// game_config.toml 顶层写 trainer=... 会被 serde 默默忽略。
    /// 本测试是 OverrideConfig 收容 trainer 字段的合并守门。
    #[test]
    fn test_override_config_trainer_overrides_default() -> Result<()> {
        let root = get_workspace_root()?;
        let def_path = root.join("gamedata").join("default_config.toml");
        let default_config: GameConfig = toml::from_str(&fs_err::read_to_string(&def_path)?)?;

        let mut o = fallback_override_game_config();
        o.config_override.trainer = Some("mcts".to_string());
        let merged = o.merge(&default_config);
        println!(
            "覆盖前 default trainer = {}，覆盖后 merged trainer = {}",
            default_config.trainer, merged.trainer
        );
        let mut c = Checks::new();
        c.check(merged.trainer == "mcts", "OverrideConfig.trainer 覆盖 default");
        c.finish()
    }

    #[test]
    fn test_validate_game_config_ramen_region_fixed_length() {
        use crate::gamedata::RamenRegionStrategy;
        let mut cfg = GameConfig::default_for_init();
        cfg.ramen_region_strategy = RamenRegionStrategy::Fixed;
        cfg.ramen_region_fixed = Some(vec![[0, 1, 2]]);
        assert!(validate_game_config(&cfg).is_ok());

        cfg.ramen_region_fixed = Some(vec![[0, 1, 2], [3, 4, 5]]); // 长度=2，应拒绝
        assert!(validate_game_config(&cfg).is_err());

        cfg.ramen_region_fixed = None;
        assert!(validate_game_config(&cfg).is_err());
    }

    #[test]
    fn test_resolve_default_config_path() {
        let p = resolve_default_config_path();
        // 默认相对路径应以 "gamedata/default_config.toml" 结尾
        assert!(p.ends_with("default_config.toml"));
    }

    /// 用户配置路径应定位到工作目录根下的 game_config.toml（曾误写为
    /// `../game_config.toml` 导致用户配置从未被加载）。
    #[test]
    fn test_resolve_user_config_path_points_to_workspace_root() {
        let p = resolve_user_config_path();
        println!("用户配置路径: {}", p.display());
        assert!(p.ends_with("game_config.toml"));
        assert!(
            !p.to_string_lossy().contains(".."),
            "不应含上级目录跳转: {}",
            p.display()
        );
    }
}
