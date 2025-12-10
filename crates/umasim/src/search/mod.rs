//! 搜索模块
//!
//! 提供扁平蒙特卡洛搜索，用于生成高质量训练数据。
//!
//! # 模块结构
//! - `config`: 搜索配置
//! - `result`: 搜索结果（分数分布统计）
//! - `flat_search`: 扁平蒙特卡洛搜索实现

mod config;
mod result;
mod flat_search;

pub use config::SearchConfig;
pub use result::{ActionResult, SearchOutput};
pub use flat_search::FlatSearch;

