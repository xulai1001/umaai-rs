use std::{io::Write, sync::Mutex};

use anyhow::{Result, anyhow};
use colored::Colorize;
use comfy_table::Table;
use flexi_logger::{DeferredNow, Duplicate, FileSpec, style};
use log::Record;
use serde::Serialize;

use crate::gamedata::{EventCollection, EventData, GAMECONSTANTS, GAMEDATA, GameConfig, LOGGER, OverrideGameConfig};

pub type Array5 = [i32; 5];
pub type Array6 = [i32; 6];

pub fn log_format(w: &mut dyn Write, _now: &mut DeferredNow, record: &Record) -> Result<(), std::io::Error> {
    let level = record.level();
    write!(
        w,
        "{} {}",
        style(level).paint(level.to_string()[..1].to_string()),
        style(level).paint(record.args().to_string())
    )
}

pub fn init_logger(app: &str, spec: &str) -> Result<()> {
    let handle = flexi_logger::Logger::try_with_str(spec)?
        .format_for_stderr(log_format)
        .log_to_file(FileSpec::default().directory("logs").basename(app))
        .duplicate_to_stderr(Duplicate::All)
        .start()?;
    LOGGER
        .set(Mutex::new(handle))
        .map_err(|_| anyhow!("Logger init failed"))?;
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

/// 检测终端类型
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

pub fn pause() -> Result<()> {
    println!("按任意键继续...");
    std::io::stdin().read_line(&mut String::new())?;
    Ok(())
}

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
/// 获得events.json里记载的指定system事件
pub fn system_event(key: &str) -> Result<&'static EventData> {
    global_events()
        .system_events
        .get(key)
        .ok_or(anyhow!("未知系统事件: {key}"))
}
/// 获得constants.json里记载的指定事件概率
pub fn system_event_prob(key: &str) -> Result<f64> {
    global!(GAMECONSTANTS)
        .event_probs
        .get(key)
        .map(|x| *x as f64)
        .ok_or(anyhow!("未知事件概率: {key}"))
}

pub trait AttributeArray {
    fn add_eq(&mut self, other: &Self) -> &mut Self;

    fn is_default(&self) -> bool;
}

impl<const N: usize> AttributeArray for [i32; N] {
    fn add_eq(&mut self, other: &Self) -> &mut Self {
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

/// 载入 gamedata/default_config.toml, 和 game_config.toml 合并
pub fn load_game_config() -> Result<GameConfig> {
    let def_file = fs_err::read_to_string("gamedata/default_config.toml")?;
    let default_config: GameConfig = toml::from_str(&def_file)?;
    let cfg_file = fs_err::read_to_string("game_config.toml")?;
    let override_config: OverrideGameConfig = toml::from_str(&cfg_file)?;
    let ret = override_config.merge(&default_config);
    //println!("{ret:#?}");
    Ok(ret)
}