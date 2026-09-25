use std::{
    sync::{Mutex, OnceLock},
    time::Duration
};

use anyhow::Result;
use crossterm::{
    self,
    event::{self, Event, KeyCode, KeyEventKind}
};
use log::warn;
use umasim::game::onsen::game::OnsenGame;
#[cfg(target_os = "windows")]
use windows::Win32::System::Threading::GetCurrentThreadStackLimits;

use crate::protocol::GameStatusOnsen;

pub static SAVED_GAME: OnceLock<Mutex<OnsenGame>> = OnceLock::new();
pub fn handle_f2() -> Result<()> {
    if let Some(mutex) = SAVED_GAME.get() {
        let game = mutex.lock().expect("saved game");
        let status = GameStatusOnsen::from(&*game);
        let filename = format!("logs/turn{}.json", game.turn);
        warn!("保存当前回合到 {filename}");
        fs_err::write(filename, serde_json::to_string_pretty(&status)?)?;
    } else {
        warn!("游戏未开始，无法保存游戏信息");
    }
    Ok(())
}

pub async fn hotkey_handler() {
    loop {
        if let Ok(true) = crossterm::event::poll(Duration::from_millis(100)) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.code == KeyCode::F(2) && k.kind == KeyEventKind::Release {
                    let _ = handle_f2().inspect_err(|e| log::error!("保存回合信息出错: {e:?}"));
                }
            }
        }
    }
}

/// 获取当前线程的栈大小，用于调试
#[cfg(target_os = "windows")]
pub fn get_stack_size() -> usize {
    unsafe {
        let mut low_limit = 0;
        let mut high_limit = 0;
        GetCurrentThreadStackLimits(&mut low_limit, &mut high_limit);
        high_limit - low_limit // 返回栈大小
    }
}

#[cfg(target_os = "linux")]
pub fn get_stack_size() -> usize {
    let mut limits = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    unsafe { libc::getrlimit(libc::RLIMIT_STACK, &mut limits) };
    limits.rlim_cur as usize
}

/// 本次测试专属的临时目录（工作区 `target/test-tmp/` 下，每次调用唯一）
///
/// 与 `umasim::utils::unique_test_dir` 同语义的本 crate 副本（那一份是
/// `#[cfg(test)]`，umaai 看不到）。
///
/// 不用系统临时目录的固定名字：写死名字时，同机并行跑的多个测试进程会写到
/// 同一路径上互相覆盖模型与旁车，出现「A 进程刚写完 B 进程就删掉」的假失败；
/// 清理时也容易越界删到工作区以外。目录名为
/// `<tag>-<pid>-<纳秒>-<进程内序号>`，建在工作区 `target/` 下（构建产物区，
/// 已被 `.gitignore` 覆盖，清理范围天然限制在工作区内）。
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
        .duration_since(UNIX_EPOCH)?
        .as_nanos();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = umasim::utils::get_workspace_root()?
        .join("target")
        .join("test-tmp")
        .join(format!("{tag}-{}-{nanos}-{seq}", std::process::id()));
    fs_err::create_dir_all(&dir)?;
    Ok(dir)
}

/// 删除 [`unique_test_dir`] 建出来的目录，并**先核对它确实在预期范围内**
///
/// 只接受位于工作区 `target/test-tmp/` 之下的绝对路径；不满足时**不删除**并报错
/// ——宁可留下垃圾目录，也不冒险递归删到别处。
///
/// # 错误
///
/// 路径不在 `target/test-tmp/` 之下，或删除失败时报错。
#[cfg(test)]
pub fn cleanup_test_dir(dir: &std::path::Path) -> Result<()> {
    let base = umasim::utils::get_workspace_root()?.join("target").join("test-tmp");
    let abs = dir.canonicalize()?;
    let base_abs = base.canonicalize()?;
    if !abs.starts_with(&base_abs) || abs == base_abs {
        anyhow::bail!(
            "拒绝删除 {}：不在 {} 之下（只清理本次建出来的子目录）",
            abs.display(),
            base_abs.display()
        );
    }
    fs_err::remove_dir_all(&abs)?;
    Ok(())
}

/// 测试用观测收集器（与 `umasim::utils::Checks` 同语义的本 crate 副本）
///
/// umasim 的那一份是 `#[cfg(test)]`，只在它自己的测试内可见，umaai 拿不到；
/// 这里放一份供本 crate 的测试共用，避免每个测试模块各写一遍。
///
/// 用法与项目测试规范一致：逐条 `println!` 观测结果，最后 [`Self::finish`]
/// 汇总；不使用 `assert!` 宏。
#[cfg(test)]
#[derive(Default)]
pub struct Checks {
    /// 未通过的观测描述
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
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误，错误信息列出全部失败项。
    pub fn finish(self) -> Result<()> {
        if self.failed.is_empty() {
            return Ok(());
        }
        anyhow::bail!("{} 项观测未通过: {}", self.failed.len(), self.failed.join(" / "))
    }
}
