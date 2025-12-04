use std::io::Write;

use anyhow::{Result, anyhow};
use comfy_table::Table;
use env_logger::Builder;
use serde::Serialize;

use crate::gamedata::{EventCollection, EventData, GAMECONSTANTS, GAMEDATA};

pub type Array5 = [i32; 5];
pub type Array6 = [i32; 6];

/// 初始化日志系统
///
/// # 参数
/// - `level`: 日志级别，"debug" (完整显示) | "off" (全部关闭)
pub fn init_logger(level: &str) -> Result<()> {
    let filter = match level.to_lowercase().as_str() {
        "off" => log::LevelFilter::Off,
        _ => log::LevelFilter::Debug, // 默认显示全部日志
    };

    let mut builder = Builder::new();
    builder.format(|buf, record| {
        let level_sty = buf.default_level_style(record.level());
        let level_str = record.level().to_string();
        writeln!(
            buf,
            "{level_sty}{}{level_sty:#} {}",
            level_str.chars().next().expect("logger"),
            record.args()
        )
        /*
        let file_str = record.file().map(|f| f.replace("crates\\umasim\\src\\", "")).unwrap_or_default();
        let line_str = record.line().map(|l| format!(":{l}")).unwrap_or_default();
        writeln!(buf,
            "{} {level_sty}{:<5}{level_sty:#} {:<12}{} {}",
            format!("[{}", chrono::Local::now().format("%y-%m-%d %H:%M:%S.%3f")).bright_black(),
            record.level(),
            format!("{file_str}{line_str}").bright_black(),
            "]".bright_black(),
            record.args()
        )
        */
    });
    builder.filter_level(filter).try_init()?;
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
