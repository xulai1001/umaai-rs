//! 搜索配置
//!
//! 定义扁平蒙特卡洛搜索的参数。

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
    /// 每次搜索会随机生成 [0, radical_factor_max] 范围的激进度。
    /// 激进度越高，越倾向选择高分高风险的动作。
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

    /// 预期搜索标准差
    ///
    /// UCB 公式中的常数，用于平衡探索和利用。
    /// C++ UmaAi 默认值: 2200.0
    pub expected_search_stdev: f64,

    /// 是否启用激进度随回合调整
    ///
    /// - true: 游戏快结束时降低激进度（C++ 方式）
    /// - false: 使用固定的随机激进度
    pub adjust_radical_by_turn: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            search_n: 1024,
            max_depth: 0,  // 搜到终局
            radical_factor_max: 50.0,
            policy_delta: 100.0,
            // UCB 参数（默认启用，使用UCB分配）
            use_ucb: true,
            search_group_size: 256,
            search_cpuct: 1.0,
            expected_search_stdev: 2200.0,
            adjust_radical_by_turn: true,  // 默认启用激进度调整
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
            adjust_radical_by_turn: true,
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

    /// 启用/禁用激进度随回合调整
    pub fn with_adjust_radical_by_turn(mut self, enabled: bool) -> Self {
        self.adjust_radical_by_turn = enabled;
        self
    }
}

