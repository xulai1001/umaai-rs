//! 搜索配置
//!
//! 定义扁平蒙特卡洛搜索的参数。
use crate::gamedata::GameConfig;
/// 游戏总回合数
pub const TOTAL_TURN: usize = 78;

/// 搜索配置
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// 每个动作的搜索次数
    ///
    /// 搜索次数越多，结果越准确，但耗时也越长。
    /// 推荐值: 1024+
    pub search_n: usize,

    /// 最大搜索深度（回合数）
    ///
    /// - 0: 搜索到游戏结束（推荐）
    /// - >0: 搜索指定回合后用评估函数估值
    pub max_depth: usize,

    /// 激进度因子最大值
    ///
    /// 每次搜索按回合**确定性衰减**，不是随机生成：
    /// `radical_factor = (剩余回合 / 总回合)^0.5 * radical_factor_max`。
    /// 激进度越高，越倾向选择高分高风险的动作
    /// （`weighted_mean(radical_factor)` 的 rank 加权指数越大越偏向高分尾部）。
    /// C++ UmaAi 默认值: 50.0
    pub radical_factor_max: f64,

    /// Policy softmax 温度
    ///
    /// 用于将各动作的加权平均分转换为概率分布。
    /// 较小的值使分布更尖锐（更倾向最优动作）。
    /// C++ UmaAi 默认值: 100.0
    pub policy_delta: f64,

    // ========== UCB 搜索分配参数 ==========
    /// 是否启用 UCB 搜索分配
    ///
    /// - true: 使用 UCB 公式动态分配搜索资源（C++ 方式）
    /// - false: 均匀分配搜索次数给每个动作（当前方式）
    pub use_ucb: bool,

    /// UCB 每组搜索次数
    ///
    /// UCB 分配时，每次给选中的动作增加的搜索次数。
    /// C++ UmaAi 默认值: 256
    pub search_group_size: usize,

    /// UCB 探索常数 (cpuct)
    ///
    /// UCB 公式: search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    /// 越大越倾向探索搜索次数少的动作。
    /// C++ UmaAi 默认值: 1.0
    pub search_cpuct: f64,

    pub expected_search_stdev: f64,

    /// 是否在每个 `(回合, 阶段)` 边界重新播种 rollout 随机流（外挂 CRN 开关）
    ///
    /// **仅规则层未改造的剧本（onsen）生效**：拉面规则层已由无状态流接管
    /// （RNG Refactor Plan v2 §5.2），其 rollout 路径不再调用阶段重播种。
    ///
    /// 实测（onsen，回合 0 Train，7 候选 × 200 rollout）：关闭时平均配对相关 0.18、
    /// 等效 1.31 倍；开启后 0.69、等效 **3.65 倍**（区间 2.44–8.62）。
    pub crn_stage_reseed: bool,

    /// 是否按 rollout 序号保留有序原始分（仅教师数据采集用）
    ///
    /// 默认 `false`：生产路径不分配缓冲、不写入 [`crate::search::SearchOutput::ordered_rollouts`]，
    /// 对搜索结果零影响。开启后按候选、按序号对齐记录 `score` 轴原始分，
    /// 失败序号为 `None`（不能省略，否则后续元素会与其他候选的 CRN 配对错位）。
    pub record_ordered_rollouts: bool,

    /// 决策理由分差门限（保留字段，不再用作触发器）
    ///
    /// 当前每回合都输出决策理由；分差仅用于选择其他候选的显示颜色档位。
    /// 字段保留并写入 [`DecisionReasonData::threshold`]，供下游兼容与对照。
    pub reason_gap_threshold: f64,

    /// 决策理由最多显示选项数
    ///
    /// 全部候选按评分降序只取前 N 个进入显示与分析，其余**直接排除**：
    /// 不显示内容、不做原因分析。中选者一般也在前 N 内；若不在，渲染时
    /// 仍按"首选"在第 1 行单独显示。
    pub reason_max_display: usize,

    /// 拉面地区选择（`RegionSelect`）候选预过滤 top-K
    ///
    /// 仅对拉面 `RegionSelect` 生效：按手写地区先验（与 rollout 基策同源）
    /// 降序排序，只把前 K 个候选交给 MCTS。`0`（默认）= 不剪枝，
    /// 生产行为逐位不变。来源：`game_config.toml` 的
    /// `[mcts] ramen_region_prune_topk`。
    pub ramen_region_prune_topk: usize
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            search_n: 1024,
            max_depth: 0, // 搜到终局
            radical_factor_max: 50.0,
            policy_delta: 100.0,
            // UCB 参数（默认启用，使用UCB分配）
            use_ucb: true,
            search_group_size: 256,
            search_cpuct: 1.0,
            expected_search_stdev: 2200.0,
            crn_stage_reseed: true,
            record_ordered_rollouts: false,
            reason_gap_threshold: 150.0,
            reason_max_display: 5,
            ramen_region_prune_topk: 0
        }
    }
}

impl SearchConfig {
    /// 创建 UCB 搜索配置（C++ 风格）
    pub fn ucb() -> Self {
        Self {
            search_n: 1024,
            use_ucb: true,
            search_group_size: 256,
            search_cpuct: 1.0,
            expected_search_stdev: 2200.0,
            ..Default::default()
        }
    }

    /// 设置搜索次数
    pub fn with_search_n(mut self, n: usize) -> Self {
        self.search_n = n;
        self
    }

    /// 设置最大深度
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// 设置激进度最大值
    pub fn with_radical_factor_max(mut self, max: f64) -> Self {
        self.radical_factor_max = max;
        self
    }

    /// 设置 Policy softmax 温度
    pub fn with_policy_delta(mut self, delta: f64) -> Self {
        self.policy_delta = delta;
        self
    }

    /// 启用/禁用 UCB 搜索分配
    pub fn with_ucb(mut self, enabled: bool) -> Self {
        self.use_ucb = enabled;
        self
    }

    /// 设置 UCB 每组搜索次数
    pub fn with_search_group_size(mut self, size: usize) -> Self {
        self.search_group_size = size;
        self
    }

    /// 设置 UCB 探索常数
    pub fn with_search_cpuct(mut self, cpuct: f64) -> Self {
        self.search_cpuct = cpuct;
        self
    }

    /// 设置预期搜索标准差
    pub fn with_expected_search_stdev(mut self, stdev: f64) -> Self {
        self.expected_search_stdev = stdev;
        self
    }

    /// 启用/禁用按阶段重播种（外挂 CRN 开关，onsen 用，见 [`crn_stage_reseed`](Self::crn_stage_reseed)）
    pub fn with_crn_stage_reseed(mut self, enabled: bool) -> Self {
        self.crn_stage_reseed = enabled;
        self
    }

    /// 启用/禁用按 rollout 序号保留有序原始分（教师数据采集用，见 [`record_ordered_rollouts`](Self::record_ordered_rollouts)）
    pub fn with_record_ordered_rollouts(mut self, enabled: bool) -> Self {
        self.record_ordered_rollouts = enabled;
        self
    }

    /// 设置决策理由分差门限（`0` 等价禁用理由输出）
    pub fn with_reason_gap_threshold(mut self, threshold: f64) -> Self {
        self.reason_gap_threshold = threshold;
        self
    }

    /// 设置决策理由最多显示选项数（评分降序前 N 进入显示与分析）
    pub fn with_reason_max_display(mut self, max_display: usize) -> Self {
        self.reason_max_display = max_display;
        self
    }

    /// 设置拉面地区候选预过滤 top-K（`0` = 不剪枝）
    pub fn with_ramen_region_prune_topk(mut self, topk: usize) -> Self {
        self.ramen_region_prune_topk = topk;
        self
    }

    pub fn new_game_config(game_config: &GameConfig) -> Self {
        let search_config = SearchConfig::default()
            .with_search_n(game_config.mcts.search_n)
            .with_radical_factor_max(game_config.mcts.radical_factor_max)
            .with_max_depth(game_config.mcts.max_depth)
            .with_policy_delta(game_config.mcts.policy_delta)
            // UCB 参数
            .with_ucb(game_config.mcts.use_ucb)
            .with_search_group_size(game_config.mcts.search_group_size)
            .with_search_cpuct(game_config.mcts.search_cpuct)
            .with_expected_search_stdev(game_config.mcts.expected_search_stdev)
            .with_crn_stage_reseed(game_config.mcts.crn_stage_reseed)
            .with_reason_gap_threshold(game_config.mcts.reason_gap_threshold)
            .with_reason_max_display(game_config.mcts.reason_max_display)
            .with_ramen_region_prune_topk(game_config.mcts.ramen_region_prune_topk);
        search_config
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use fs_err::read_to_string;

    use super::*;
    use crate::utils::{Checks, get_workspace_root};

    /// 从 workspace 根目录加载真实的 `gamedata/default_config.toml`。
    fn load_real_default() -> Result<GameConfig> {
        let path = get_workspace_root()?.join("gamedata/default_config.toml");
        let text = read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    /// `new_game_config` 必须转发 `crn_stage_reseed`。
    ///
    /// 删掉 `.with_crn_stage_reseed(...)` 后 `SearchConfig::default()` 会留下 `true`，本测试必须红。
    #[test]
    fn test_new_game_config_follows_crn_stage_reseed() -> Result<()> {
        let mut game = load_real_default()?;
        game.mcts.crn_stage_reseed = false;
        let sc = SearchConfig::new_game_config(&game);
        println!(
            "new_game_config crn_stage_reseed = {} (GameConfig 侧 = {})",
            sc.crn_stage_reseed, game.mcts.crn_stage_reseed
        );
        let mut c = Checks::new();
        c.check(
            !sc.crn_stage_reseed,
            "crn_stage_reseed 跟随 GameConfig 的 false"
        );
        c.finish()
    }

    /// `record_ordered_rollouts` 必须默认关闭，且 toml 路径不会悄悄打开它。
    #[test]
    fn test_record_ordered_rollouts_defaults_off() -> Result<()> {
        let def = SearchConfig::default();
        let from_toml = SearchConfig::new_game_config(&load_real_default()?);
        println!(
            "default={} new_game_config={}",
            def.record_ordered_rollouts, from_toml.record_ordered_rollouts
        );
        let mut c = Checks::new();
        c.check(!def.record_ordered_rollouts, "SearchConfig::default 为 false");
        c.check(
            !from_toml.record_ordered_rollouts,
            "new_game_config 未接线时保持默认 false"
        );
        c.finish()
    }
}
