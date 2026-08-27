//! 拉面杯实验策略：在现有即时评分上增加长期训练结构与剧本 PT 阈值价值。
use std::sync::Mutex;

use anyhow::Result;
use rand::{SeedableRng, prelude::StdRng};

use crate::{
    game::{
        FriendOutState,
        Game,
        Person,
        PersonType,
        Trainer,
        ramen::{
            Operation,
            RamenAction,
            RamenGame,
            RamenStage,
            effects::calc_ramen_training_effect,
            policy::{RamenPolicy, RamenPolicyConfig, RamenPolicyOutput},
            rules::{
                calc_ramen_pt_gain, calc_region_bonus, consume_for_ramen, get_recipe, get_turn_special_feeling,
                list_special_targets_for
            }
        }
    },
    gamedata::{EventChoice, EventData, GAMECONSTANTS, ramen::RAMENDATA},
    global
};
use crate::trainer::ramen_handwritten_trainer::ramen_effective_stage;

#[derive(Debug, Clone)]
pub struct LocalRamenConfig {
    /// 低于 80 羁绊的普通支援卡每获得 1 点羁绊所折算的长期评分。
    ///
    /// 实际加分还会乘年度衰减系数，并受距离 80 羁绊的剩余空间限制；单位为策略评分/羁绊点。
    /// 设为 `0.0` 可关闭普通支援卡的早期羁绊估值。
    pub early_bond_value: f32,

    /// 点击带 Hint 支援卡时附加的即时 Hint 价值，单位为策略评分。
    ///
    /// 启用 [`Self::probabilistic_hint`] 时，会除以当前训练中带 Hint 的卡数，表达随机命中概率。
    pub hint_bonus: f32,

    /// 首次点击剧本友人卡、使其从未点击状态向外出解锁推进的长期价值，单位为策略评分。
    pub first_friend_click_value: f32,

    /// 剧本友人卡已点击但羁绊低于 60 时，每次点击的长期价值，单位为策略评分。
    ///
    /// 该值会乘年度衰减系数，避免后期继续高估尚未解锁完成的友人链。
    pub low_friend_bond_value: f32,

    /// 剧本友人卡进入活跃阶段后的每次点击价值，单位为策略评分。
    pub active_friend_value: f32,

    /// 旧版高失败率尾部惩罚的最大值，单位为策略评分。
    ///
    /// 仅当 [`Self::expected_fail`] 为 `false` 且基础失败率高于 15% 时使用；
    /// `0.0` 表示关闭该旧模型。当前推荐配置使用期望失败模型，保留此字段仅供消融实验。
    pub high_fail_penalty: f32,

    /// 诀窍总库存超过该数量后，开始给“立即吃面”增加溢出压力。
    ///
    /// 单位为诀窍个数；例如默认值 `8` 表示库存总数从第 9 个开始产生压力。
    pub feeling_overflow_threshold: i32,

    /// 每个超过 [`Self::feeling_overflow_threshold`] 的诀窍所产生的吃面奖励，单位为策略评分/诀窍。
    ///
    /// 用于避免库存接近上限时继续等待而丢弃最早获得的诀窍；`0.0` 表示关闭。
    pub overflow_value: f32,

    /// 长期结构评分相对基础即时评分最多允许牺牲的分数，单位为策略评分。
    ///
    /// 如果长期结构选出的动作比基础策略最佳动作低超过该值，则回退到基础策略动作，
    /// 防止羁绊、Hint 等启发式为了长期收益牺牲过多本回合收益。
    pub max_base_score_sacrifice: f32,

    /// 为未来固定事件和终盘奖励预留的最大属性空间，单位为原始属性点。
    ///
    /// 预留量会随剩余回合线性缩小；训练把属性推近上限时会产生软惩罚。
    /// `0.0` 表示关闭属性溢出预留模型。
    pub status_reserve_max: f32,

    /// 是否启用按五维完成度动态调整属性边际价值。
    ///
    /// 开启后会提高相对落后属性的精确评分边际，并在属性接近上限时降低继续堆叠的价值；
    /// 三张及以上同类型卡会放大对应属性的近上限衰减。默认关闭，仅供配对矩阵验证。
    pub dynamic_status_balance: bool,

    /// 短板追赶强度。1.0 表示完成度每落后最高维度 10%，该维精确属性评分边际增加 10%。
    pub status_gap_strength: f32,

    /// 近上限衰减强度。属性完成度超过 70% 后按平方曲线增长，并受同类型卡过量系数放大。
    pub status_overflow_strength: f32,

    /// 是否使用随回合变化的体力成本模型。
    ///
    /// `true` 时前期体力消耗更贵，终盘体力价值逐渐降低，并只补上相对基础策略
    /// `train_vital_value` 尚未计入的差额。
    pub dynamic_vital: bool,

    /// 是否把多个同时亮起的 Hint 视为随机命中，而不是每个 Hint 都按全额价值计算。
    pub probabilistic_hint: bool,

    /// 是否使用连续的失败期望损失模型。
    ///
    /// `true` 时按失败概率扣除小失败损失，并在失败率达到 20% 后加入大失败尾部风险；
    /// `false` 时可选择使用 [`Self::high_fail_penalty`] 的旧阈值模型。
    pub expected_fail: bool,

    /// 吃面跨越 `scenario_pt` 常驻效果档位后的持续价值倍率。
    ///
    /// 先计算训练加成、得意率、Hint 与地区词条的档位差，再乘当年剩余回合和该倍率。
    /// 这是无量纲缩放系数；`0.0` 表示关闭档位前瞻价值。
    pub checkpoint_scale: f32,

    /// 本次吃面首次跨过当年 RMJ 成功线时的一次性奖励，单位为策略评分。
    ///
    /// 只在吃面前低于成功线且吃面后达到或超过成功线时计入一次。
    pub rmj_cross_bonus: f32,

    /// 第三年本次吃面首次跨过 5000 剧本 PT 大成功线时的一次性奖励，单位为策略评分。
    pub great_cross_bonus: f32,

    /// 随机事后前向值的权重，无量纲。
    ///
    /// 前向会在状态副本中执行候选拉面、随机落地分身并比较吃面前后的最佳动作。
    /// 实验表明它会干扰当前真实训练窗口，当前推荐值为 `0.0`；保留字段用于回归消融。
    pub ramen_lookahead_weight: f32,

    /// 每个候选拉面执行随机事后前向时使用的独立样本数。
    ///
    /// 仅当 [`Self::ramen_lookahead_weight`] 大于 `0.0` 时生效；最小按 1 个样本处理。
    pub ramen_lookahead_samples: usize,

    /// 是否强制在存在可制作拉面时从拉面候选中选择，而不允许“不吃面”参与竞争。
    ///
    /// 该模式只用于实验；当前推荐策略为 `false`，由窗口价值正常决定吃面时机。
    pub eager_eat: bool,

    /// 当前真实训练窗口与候选拉面覆盖训练的耦合权重，无量纲。
    ///
    /// 窗口由当前训练原始收益、人数和彩圈数构成，再乘地区效果强度。
    /// 这是 v8 高收益的主要来源；配置 token `window10` 对应 `0.10`。
    pub ramen_window_weight: f32,

    /// 吃面-训练联动权重（训练侧显式项），无量纲。
    ///
    /// 当前吃面且训练位落在该面 `at_trains` 内时，`decide_train` 为该训练候选
    /// 加 `地区效果强度 × 本权重` 的显式加分。`calc_training_value` 已隐含地区效果
    /// （吃面覆盖位的训练数值更高），本项是**显式强化**：让策略在彩圈/羁绊/属性缺口
    /// 等其它收益占优时仍倾向兑现已付出的吃面成本。`0.0` 表示关闭（仅隐含）。
    /// 配置 token `couple50` 对应 `0.50`。
    pub ramen_train_coupling_weight: f32,

    /// 弱位训练偏好：吃面回合对 at_trains 覆盖的卡少位（card_type_count ≤ 1）
    /// 训练位加权，用于（1）`ramen_window_alignment` 评估"这碗面值得吃吗"时放大
    /// 弱位 raw 收益，让选面阶段倾向覆盖弱位的面；（2）`decide_train` 在耦合分支
    /// 之外对卡少位吃面训练候选加分，让训练阶段倾向练弱位。
    ///
    /// 区分吃面/不吃面——只在吃面回合（`current_ramen.is_some()`）生效，
    /// 避免不吃面时练卡少位（历史上证实会劣化）。配合 `card_type_count[t] ≤ 1`
    /// 限定"带卡少但非零"的弱位（卡 0 位的面通常 at_trains 不会覆盖）。
    ///
    /// `0.0` 关闭；推荐启动值由后续扫描定（用户：以后再调整）。
    /// 配置 token `weakboost150` 对应 `1.50`。
    pub ramen_weak_train_boost: f32,


    /// 隐藏风味饥饿加成权重，单位为策略评分/缺口点。
    ///
    /// 友人外出固定 +2 隐藏风味（上限 4，见 `RamenGame::do_friend_outing`）。
    /// 隐藏风味是吃面资源，被吃面持续消耗；`special_feeling` 缺口越大，
    /// 友人外出的补给价值越高，本权重把该价值计入 `dynamic_friend_outing_value`。
    /// 计算时扣除"未来 2 回合内固定发放量"（夏合宿 +2 / 年末 +1），避免在
    /// 即将自然补足时仍为饥饿付费（溢出浪费）。`0.0` 表示关闭。
    /// 配置 token `starve15` 对应 `15.0`。
    pub friend_hidden_starve_weight: f32,

    /// 隐藏风味"未来供给缺口"权重，无量纲。
    ///
    /// 饥饿加成只估当前库存缺口；本项前瞻未来吃面供需：按剩余回合 × 平均吃面频率 ×
    /// 每次消耗估算需求，减去未来固定发放（夏合宿/年末）与友人剩余次数供给，
    /// 得出未来缺口（风味数）。本次外出的 +2 风味按 `min(2, 缺口) × 每风味吃面收益
    /// (≈800，每次吃面评分约 1200 ÷ 平均消耗 1.5)` 计入友人价值——"友人 5 次 =
    /// 10 个隐藏风味 = 吃面资源"的长期估值。`0.0` 表示关闭。配置 token `fh1` 对应 `1.0`。
    pub friend_future_hidden_weight: f32,

    /// 友人"主动积极使用"价值，单位为策略评分。
    ///
    /// 友人外出是"带收益的休息"：事件给 30~50 体力 + 属性 + 心情 + 2 隐藏风味。
    /// 饥饿加成只在 `special_feeling` 缺口大时触发，导致体力尚可、不饥饿时策略
    /// 从不主动用友人——链拖到后期、体力线被训练打低后才被迫用。
    ///
    /// 本项在「未来 3 回合无固定发放（夏合宿 +2 / 年末 +1）」且「本次 +2 不溢出
    /// （`special_feeling ≤ 2`）」时给友人加固定价值，代表"主动用友人维持体力线 +
    /// 完链"的收益——体力正常/高时也愿意用，而不是等饥饿或被迫休息。
    /// `0.0` 关闭。配置 token `pro150` 对应 `150.0`。
    pub friend_proactive_weight: f32,


    /// 吃面必成价值权重，无量纲。
    ///
    /// 吃面回合训练失败率下降（`fail_rate_drop`：Y1 30% / Y2 50% / Y3 100%，
    /// 吃面后训练必成）。`decide_ramen` 按"本回合基础动作若是训练"计算其失败
    /// 期望损失（训练收益 × 失败率 + 失败惩罚 × 失败率），乘本权重计入吃面候选。
    /// 第三年低体力吃面训练已被 `y3_post_train_hard_floor` 等门禁兜底，
    /// 必成价值可以放心计入。`0.0` 表示关闭。配置 token `guarantee100` 对应 `1.0`。
    pub eat_guarantee_weight: f32,

    /// 策略评分是否采用吃面后的实际失败率下降。
    ///
    /// `true` 按当年拉面效果降低失败率；`false` 使用吃面前基础失败率作为保守风险预算。
    /// 游戏规则执行始终使用真实失败率，本开关只影响动作评分。
    pub effective_ramen_failure: bool,

    /// 第一年安全过渡门控允许救援的训练最低基础失败率，单位为百分比。
    ///
    /// 大于 `100.0` 表示完全关闭该实验功能；当前默认 `101.0` 即关闭。
    pub safety_bridge_min_fail: f32,

    /// 应用第一年 30% 相对失败率下降后，风险训练超过当前最佳动作所需的最低增益。
    ///
    /// 单位为策略评分，仅在安全过渡门控启用时生效。
    pub safety_bridge_min_gain: f32,

    /// 安全过渡选择拉面时，每损失一个事后可制作选项或消耗一个隐藏风味的成本。
    ///
    /// 单位为策略评分/资源单位，仅在安全过渡门控启用时生效。
    pub safety_bridge_stock_cost: f32,

    /// 田园杯 Cook2 凹函数材料估值适配到拉面诀窍库存后的总权重。
    ///
    /// 对 A/B/C 分别计算 `sqrt(吃前库存+2)-sqrt(吃后库存+2)`，隐藏风味另计灵活性成本，
    /// 再乘年度剩余比例与 RMJ 进度折扣。单位为策略评分缩放；当前最佳 `cook2-40` 为 `40.0`。
    pub cook2_stock_weight: f32,

    /// 是否把“吃面”和“本回合训练”视为不可拆分的事务。
    ///
    /// `true` 时先在不吃面的当前局面决定基础动作：若应休息、外出、治病或比赛，
    /// RamenSelect 直接选择不吃；一旦已经吃面，Train 阶段只在五种训练中比较，
    /// 不允许随后休息而浪费仅本回合生效的拉面加成。
    pub eat_requires_training: bool,

    /// 吃面后必须训练**该面 at_trains 覆盖位**（C 方案简化约束）
    ///
    /// 玩家 87% 吃面训练落在 at_trains 内 vs 自动 52%（seed=61444 决策日志解析），
    /// 差距根因之一是自动选面（`decide_ramen`）不前瞻"吃完练哪个位"，吃了面却练
    /// 不覆盖的位导致 youqing/xunlian/失败率下降加成浪费。
    ///
    /// 本开关在 `decide_ramen` 吃面候选打分时**预演**该面落地后的 `decide_train`
    /// （clone 当前状态 + 设 `current_ramen=Some(region_id)` 后跑完整训练打分），
    /// 若最优动作不是该面 at_trains 覆盖的训练位，则否决该吃面候选（`NEG_INFINITY`）。
    /// 体力低导致预演最优动作是休息时，`eat_requires_training` 已先否决吃面——
    /// 两个开关配合实现"吃面后必训练且训练位必须被面覆盖"。
    ///
    /// `false` 关闭（退化为仅 `eat_requires_training` 的事务门）；推荐 preset 开启。
    pub eat_requires_covered_train: bool,

    /// 每年吃面前希望具备的训练前体力，单位为体力点。
    ///
    /// 它回答“现在是否应该先恢复”。低于目标不会直接禁止吃面，而会按短缺量收费，
    /// 使极强窗口仍可突破保守线。`0` 表示关闭训练前体力预算。
    /// （字段名保留 `y3` 前缀为历史沿革；现每年吃面决策都评估。）
    pub y3_pre_train_vital_target: i32,

    /// 每年吃面并完成计划训练后希望保留的体力，单位为体力点。
    ///
    /// 它回答“本次训练会不会使下一回合崩盘”。智力训练同样参与计算，但因其体力变化
    /// 通常为正，训练后短缺自然较小；不再给予无条件豁免。`0` 表示关闭训练后预算。
    pub y3_post_train_vital_target: i32,

    /// 每年训练前/后体力每短缺 1 点对候选面的软惩罚，单位为策略评分/体力点。
    ///
    /// 总成本为 `max(pre_target-V0,0) + max(post_target-V1,0)` 再乘此权重。
    /// `0.0` 表示关闭联合体力预算。
    pub y3_vital_shortfall_weight: f32,

    /// 每年非智力训练后的极端安全底线，低于该值才硬禁止吃面。
    ///
    /// 与软目标分离：正常体力不足只扣分，只有接近打空时才保下限。智力训练也必须满足
    /// `V1 >= 0`，但不受此非智力硬底线。`0` 表示不额外硬拦。
    pub y3_post_train_hard_floor: i32,

    /// 是否按“距离下一次确定恢复前还有几个可训练回合”判断第三年体力崩盘。
    ///
    /// 当前规则中 turn=70 训练后，turn=71 为有马纪念，赛后固定恢复 40；随后
    /// turn=72 起超级拉面每回合开始恢复 20。因此 turn=70 可以把体力控到 0，
    /// 不应再为训练后低体力付费。更早回合若低体力会影响至少一个普通训练回合，
    /// 才计入崩盘成本。
    pub y3_recovery_horizon: bool,

    /// 当体力守门或正常打分原本选择休息时，是否优先用尚未完成的友人外出替代。
    ///
    /// 友人外出同样恢复体力，同时提供属性、干劲、Hint、隐藏风味和事件链进度；
    /// 仅替换本来就会消耗的休息回合，不为了赶链强行覆盖高价值训练。
    pub friend_outing_replaces_rest: bool,

    /// 友人第三次外出时，当前体力低于该值就选择恢复 50 体力的选项。
    ///
    /// 否则保留事件通用评分，可选无回复的属性/PT选项。`0` 表示关闭该低体力保护。
    pub friend_outing3_recovery_vital: i32,

    /// 各年结束前允许累计使用的友人外出次数上限。
    ///
    /// 五次外出是整局有限资源，每次还产生 2 个万能材料；不能因为第一年休息较多就一次用完。
    /// 例如 `[1, 3, 5]` 表示第一年最多用 1 次、第二年结束前最多累计 3 次、第三年可用完。
    /// `[5, 5, 5]` 等价于不做跨年配额；仅在 `friend_outing_replaces_rest=true` 时生效。
    pub friend_outing_cumulative_caps: [usize; 3],

    /// “休息→友人外出”替代时允许的最高当前万能材料数量。
    ///
    /// 外出固定获得 2 个万能材料且上限为 4；设为 2 可避免替代路径产生材料溢出。
    /// 原策略主动选择友人外出不受此门控，只受总次数配额约束。`4` 表示关闭。
    pub friend_rest_max_special: i32,

    /// RMJ/第三年5000目标在截止前的可达性紧迫度。
    pub deadline_urgency_scale: f32,

    /// SpecialSelect 是否按吃后库存、后续可制作集合和年末剩余价值动态选择。
    pub dynamic_special_targets: bool
}
impl Default for LocalRamenConfig {
    fn default() -> Self {
        Self {
            early_bond_value: 8.,
            hint_bonus: 6.,
            first_friend_click_value: 75.,
            low_friend_bond_value: 35.,
            active_friend_value: 8.,
            high_fail_penalty: 0.,
            feeling_overflow_threshold: 8,
            overflow_value: 8.,
            max_base_score_sacrifice: 140.,
            status_reserve_max: 0.,
            dynamic_status_balance: false,
            status_gap_strength: 0.0,
            status_overflow_strength: 0.0,
            dynamic_vital: false,
            probabilistic_hint: false,
            expected_fail: false,
            checkpoint_scale: 0.,
            rmj_cross_bonus: 0.,
            great_cross_bonus: 0.,
            ramen_lookahead_weight: 1.0,
            ramen_lookahead_samples: 12,
            eager_eat: false,
            ramen_window_weight: 0.0,
            ramen_train_coupling_weight: 0.0,
            ramen_weak_train_boost: 0.0,
            friend_hidden_starve_weight: 0.0,
            friend_future_hidden_weight: 0.0,
            friend_proactive_weight: 0.0,
            eat_guarantee_weight: 0.0,
            effective_ramen_failure: true,
            safety_bridge_min_fail: 101.0,
            safety_bridge_min_gain: 0.0,
            safety_bridge_stock_cost: 0.0,
            cook2_stock_weight: 0.0,
            eat_requires_training: false,
            eat_requires_covered_train: false,
            y3_pre_train_vital_target: 0,
            y3_post_train_vital_target: 0,
            y3_vital_shortfall_weight: 0.0,
            y3_post_train_hard_floor: 0,
            y3_recovery_horizon: false,
            friend_outing_replaces_rest: false,
            friend_outing3_recovery_vital: 0,
            friend_outing_cumulative_caps: [5, 5, 5],
            friend_rest_max_special: 4,
            deadline_urgency_scale: 0.0,
            dynamic_special_targets: false
        }
    }
}
pub struct LocalRamenTrainer {
    policy: RamenPolicy,
    config: LocalRamenConfig,
    /// 是否采集评分分解文本（供 `LoggingTrainer` 取用）
    ///
    /// 作为**搜索的 rollout 基策**时必须关掉：`stash` 每次决策都无条件
    /// `format!` 出全候选分解并锁同一把 `Mutex`，而所有 rayon 线程共享同一个
    /// rollout trainer。单次 rollout 约 170 次决策 × 24 线程 = 高频锁争用，
    /// 而 rollout 内部的分解文本没有任何消费者。不影响分数，只影响吞吐。
    /// （与 `RamenHandwrittenTrainer::collect_breakdown` 同构。）
    collect_breakdown: bool,
    last_breakdown: Mutex<Option<String>>
}
impl Default for LocalRamenTrainer {
    fn default() -> Self {
        Self::with_configs(RamenPolicyConfig::default(), LocalRamenConfig::default())
    }
}
impl LocalRamenTrainer {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_configs(policy: RamenPolicyConfig, config: LocalRamenConfig) -> Self {
        Self {
            policy: RamenPolicy::new(policy),
            config,
            collect_breakdown: true,
            last_breakdown: Mutex::new(None)
        }
    }
    /// 创建 rollout 专用实例（关闭分解采集，见 [`collect_breakdown`](Self::collect_breakdown)）
    pub fn for_rollout() -> Self {
        Self {
            collect_breakdown: false,
            ..Self::new()
        }
    }
    pub fn matrix_variant(name: &str) -> Result<Self> {
        let mut policy = RamenPolicyConfig::default();
        let mut local = LocalRamenConfig::default();
        let (mut p, mut s, mut m, mut f) = (false, false, false, false);
        for token in name.split('-') {
            if token == "rawfail" {
                policy.effective_ramen_failure = false;
                local.effective_ramen_failure = false
            } else if let Some(v) = token.strip_prefix("bridge") {
                local.safety_bridge_min_fail = v.parse()?
            } else if let Some(v) = token.strip_prefix("bgain") {
                local.safety_bridge_min_gain = v.parse()?
            } else if let Some(v) = token.strip_prefix("bcost") {
                local.safety_bridge_stock_cost = v.parse()?
            } else if let Some(v) = token.strip_prefix("cook2") {
                local.cook2_stock_weight = v.parse()?
            } else if let Some(v) = token.strip_prefix("vrest") {
                policy.vital_rest = v.parse()?
            } else if token == "eatguard" {
                local.eat_requires_training = true
            } else if let Some(v) = token.strip_prefix("y3pre") {
                local.y3_pre_train_vital_target = v.parse()?
            } else if let Some(v) = token.strip_prefix("y3post") {
                local.y3_post_train_vital_target = v.parse()?
            } else if let Some(v) = token.strip_prefix("y3vw") {
                local.y3_vital_shortfall_weight = v.parse()?
            } else if let Some(v) = token.strip_prefix("y3hard") {
                local.y3_post_train_hard_floor = v.parse()?
            } else if token == "y3horizon" {
                local.y3_recovery_horizon = true
            } else if token == "friendrest" {
                local.friend_outing_replaces_rest = true
            } else if let Some(v) = token.strip_prefix("friend3v") {
                local.friend_outing3_recovery_vital = v.parse()?
            } else if let Some(v) = token.strip_prefix("friendcap") {
                let digits = v.as_bytes();
                if digits.len() != 3 || !digits.iter().all(u8::is_ascii_digit) {
                    anyhow::bail!("friendcap 必须是三个数字，如 135: {v}");
                }
                local.friend_outing_cumulative_caps = [
                    (digits[0] - b'0') as usize,
                    (digits[1] - b'0') as usize,
                    (digits[2] - b'0') as usize
                ];
                let c = local.friend_outing_cumulative_caps;
                if c[0] > c[1] || c[1] > c[2] || c[2] > 5 {
                    anyhow::bail!("friendcap 必须单调且不超过5: {v}");
                }
            } else if let Some(v) = token.strip_prefix("friendspecial") {
                local.friend_rest_max_special = v.parse()?
            } else if let Some(v) = token.strip_prefix("deadline") {
                local.deadline_urgency_scale = v.parse::<f32>()? / 100.0
            } else if token == "specialdynamic" {
                local.dynamic_special_targets = true
            } else if token == "statusdyn" {
                local.dynamic_status_balance = true
            } else if let Some(v) = token.strip_prefix("gap") {
                local.status_gap_strength = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("over") {
                local.status_overflow_strength = v.parse::<f32>()? / 100.0
            } else if token == "failmodel" {
                local.expected_fail = true
            } else if token == "vital" {
                local.dynamic_vital = true
            } else if token == "hintprob" {
                local.probabilistic_hint = true
            } else if token == "structall" {
                local.status_reserve_max = 40.;
                local.dynamic_vital = true;
                local.probabilistic_hint = true;
                local.expected_fail = true
            } else if token == "eager" {
                local.eager_eat = true
            } else if token == "plain" {
                local.early_bond_value = 0.;
                local.hint_bonus = 0.;
                local.first_friend_click_value = 0.;
                local.low_friend_bond_value = 0.;
                local.active_friend_value = 0.;
                local.overflow_value = 0.;
                m = true
            } else if token == "long" || token == "base" {
                m = true
            } else if let Some(v) = token.strip_prefix("pt") {
                policy.pt_rate = v.parse()?;
                p = true
            } else if let Some(v) = token.strip_prefix("sac") {
                local.max_base_score_sacrifice = v.parse()?;
                s = true
            } else if let Some(v) = token.strip_prefix("reserve") {
                local.status_reserve_max = v.parse()?
            } else if let Some(v) = token.strip_prefix("fail") {
                local.high_fail_penalty = v.parse()?;
                f = true
            } else if let Some(v) = token.strip_prefix("ck") {
                local.checkpoint_scale = v.parse::<f32>()? / 100.
            } else if let Some(v) = token.strip_prefix("rmj") {
                local.rmj_cross_bonus = v.parse()?
            } else if let Some(v) = token.strip_prefix("great") {
                local.great_cross_bonus = v.parse()?
            } else if let Some(v) = token.strip_prefix("rpt") {
                policy.ramen_pt_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("align") {
                local.ramen_lookahead_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("window") {
                local.ramen_window_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("couple") {
                local.ramen_train_coupling_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("capd") {
                policy.cap_discount_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("starve") {
                local.friend_hidden_starve_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("fh") {
                local.friend_future_hidden_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("pro") {
                local.friend_proactive_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("guarantee") {
                local.eat_guarantee_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("weakboost") {
                // 弱位训练偏好（吃面前 + 吃面后训练阶段），值原样 /100，
                // `weakboost150` 对应 `1.50`。默认 0.0 关闭。
                local.ramen_weak_train_boost = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("look") {
                local.ramen_lookahead_weight = v.parse::<f32>()? / 100.0
            } else if let Some(v) = token.strip_prefix("samples") {
                local.ramen_lookahead_samples = v.parse()?
            } else {
                anyhow::bail!("未知矩阵变体字段: {token} ({name})")
            }
        }
        if !(p && s && m && f) {
            anyhow::bail!("矩阵变体字段不完整: {name}")
        }
        Ok(Self::with_configs(policy, local))
    }
    fn choose(o: &[RamenPolicyOutput]) -> usize {
        o.iter()
            .enumerate()
            .max_by(|(li, l), (ri, r)| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)))
            .map(|x| x.0)
            .unwrap_or(0)
    }
    fn stash(&self, o: &[RamenPolicyOutput]) {
        if !self.collect_breakdown {
            return;
        }
        let t = o
            .iter()
            .enumerate()
            .map(|(i, x)| format!("#{i} {:.0}[{}]", x.score, x.reason))
            .collect::<Vec<_>>()
            .join(" | ");
        if let Ok(mut b) = self.last_breakdown.lock() {
            *b = Some(t)
        }
    }
    fn phase(turn: i32) -> f32 {
        if turn < 24 {
            1.
        } else if turn < 48 {
            0.55
        } else {
            0.15
        }
    }
    fn reserve_penalty(&self, g: &RamenGame, gain: &[i32; 6]) -> f32 {
        if self.config.status_reserve_max <= 0. {
            return 0.;
        }
        let rem = (76 - g.turn()).max(0) as f32;
        let r = self.config.status_reserve_max * rem / 76.;
        let mut p = 0.;
        for i in 0..5 {
            let h = (g.uma.five_status_limit[i] - g.uma.five_status[i]).max(0) as f32;
            let b = (r - h).max(0.);
            let a = (r - (h - gain[i] as f32)).max(0.);
            p += (a * a - b * b) / (2. * r.max(1.));
        }
        p * 6.
    }
    fn dynamic_status_adjustment(&self, g: &RamenGame, gain: &[i32; 6]) -> f32 {
        if !self.config.dynamic_status_balance {
            return 0.0;
        }
        let completion: [f32; 5] = std::array::from_fn(|i| {
            let limit = g.uma.five_status_limit[i].max(1) as f32;
            (g.uma.five_status[i].max(0) as f32 / limit).clamp(0.0, 1.0)
        });
        let leading = completion.iter().copied().fold(0.0f32, f32::max);
        let cons = global!(GAMECONSTANTS);
        let mut adjustment = 0.0;
        for i in 0..5 {
            let limit = g.uma.five_status_limit[i].max(0) as usize;
            let cur = (g.uma.five_status[i].max(0) as usize).min(limit);
            let next = cur.saturating_add(gain[i].max(0) as usize).min(limit);
            let cur_score = cons.status_final_score(cur as i32) as f32;
            let next_score = cons.status_final_score(next as i32) as f32;
            let exact_margin = (next_score - cur_score) * self.policy.config.status_rate;
            let gap_bonus = self.config.status_gap_strength * (leading - completion[i]).max(0.0);
            let near_cap = ((completion[i] - 0.70) / 0.30).clamp(0.0, 1.0);
            let excess_cards = (g.card_type_count[i] - 2).max(0) as f32;
            let overflow = self.config.status_overflow_strength
                * near_cap
                * near_cap
                * (1.0 + 0.5 * excess_cards);
            let multiplier = (1.0 + gap_bonus - overflow).clamp(0.10, 2.00);
            adjustment += exact_margin * (multiplier - 1.0);
        }
        adjustment
    }

    /// 弱位偏好 effective boost（按 build 卡组结构自适应 + 实验 override 入口）
    ///
    /// 行为分支：
    /// - `config_override > 0.0`：所有 build 用该固定值（实验 override）。
    /// - `config_override <= 0.0`：按智卡数 `card_type_count[4]` 查表（推荐 preset 默认启用）：
    ///   - 智卡 ≤1（speed/stamina/spd2_gut0）：5.0 — 强化真弱位智训练
    ///   - 智卡 =2（speed_wisdom/sta0_wis2/power_wisdom 智卡=3 但智+力共三张）：
    ///     视 `card_type_count[4]` 而定，2→0.0（触发的位 count≤1 不在 at_trains 主选区，关）
    ///   - 智卡 ≥3（power_wisdom/wisdom 智=3 时）：2.0（智位已满，边际低，小值微调）
    ///
    /// 经验数据来源（stamina seed=61444 × 50 seed × 7 build）：
    /// | 智卡 | 代表 build | 最佳 boost | t | wins |
    /// |---|---|---|---|---|
    /// | 1 | speed/stamina/spd2_gut0 | 5–6 | 3.35 | 87–89/150 |
    /// | 2 | speed_wisdom/sta0_wis2 | 0.0（关闭） | – | – |
    /// | 3 | power_wisdom/wisdom | 2.0 | 2.90 | 46/100 |
    fn effective_weak_boost(g: &RamenGame, config_override: f32) -> f32 {
        if config_override > 0.0 {
            return config_override;
        }
        if config_override < 0.0 {
            // 显式关闭查表（测试/调试用，effective=0）
            return 0.0;
        }
        // 默认 (=0.0) 启用按 build 自适应查表（推荐 preset 默认行为）
        let w = g.card_type_count[4];
        if w <= 1 { 5.0 } else if w == 2 { 0.0 } else { 2.0 }
    }

    fn vital_factor(t: i32) -> f32 {
        if t >= 72 { 0.25 } else { 3.5 + (t as f32 / 72.) * 2. }
    }
    /// 本年是否仍有友人外出配额。配额按整局累计次数控制，而不是每年重置。
    fn friend_outing_within_pacing(&self, g: &RamenGame) -> bool {
        let year = (g.current_year() - 1).clamp(0, 2) as usize;
        let used = g.friend.out_used.iter().filter(|&&x| x).count();
        used < self.config.friend_outing_cumulative_caps[year]
    }

    /// 友人外出 +2 隐藏风味是否不溢出（上限 4，见 `do_friend_outing`）。
    ///
    /// 溢出时 +2 中超出部分浪费——隐藏风味补给价值归零，友人只剩体力/属性/完链
    /// 价值，与休息无本质区别；友人次数有限（[0,2,5]），应留给"不溢出 + 低体力"
    /// 的完整价值回合，溢出回合退回休息。
    fn friend_hidden_not_overflow(&self, g: &RamenGame) -> bool {
        g.ramen.special_feeling <= 2
    }


    /// 下一段友人外出的动态价值。
    ///
    /// 事件本体按当前体力/干劲裁掉溢出，第三段两个选项也在这里实时比较；万能材料固定按
    /// 2 个来源计价，即使当前计数已满也不把外出禁掉。跨年稀缺性只由累计配额控制。
    fn dynamic_friend_outing_value(&self, g: &RamenGame) -> Result<(f32, Vec<(String, f32)>, String)> {
        let used = g.friend.out_used.iter().filter(|&&x| x).count();
        if used >= 5 {
            return Ok((f32::NEG_INFINITY, vec![], "友人外出已完成".to_string()));
        }
        let data = RAMENDATA.get().ok_or_else(|| anyhow::anyhow!("RAMENDATA 未初始化"))?;
        let event = data
            .friend_events
            .get(&format!("outing{}", used + 1))
            .ok_or_else(|| anyhow::anyhow!("缺少友人外出事件 {}", used + 1))?;
        let (choice, event_value) = self.dynamic_friend_event_choice(g, &event.choices)?;

        // friend_outing_bonus 原本把“2万能材料+事件链”压成一个固定值。这里保留总尺度，
        // 但拆为固定材料来源价值和随段数/年份上升的完链价值。
        let material = self.policy.config.friend_outing_bonus * (2.0 / 3.0);
        let chain_urgency = 0.70 + used as f32 * 0.12 + (g.current_year() - 1) as f32 * 0.18;
        let chain = self.policy.config.friend_outing_bonus * (1.0 / 3.0) * chain_urgency;
        let base = self.policy.config.outing_base;
        // 隐藏风味饥饿加成：隐藏风味是吃面资源（上限 4，友人外出固定 +2），
        // 缺口越大补给价值越高；扣除未来 2 回合内固定发放（夏合宿 +2 / 年末 +1），
        // 避免在即将自然补足时仍为饥饿付费、导致溢出浪费。
        let starve = if self.config.friend_hidden_starve_weight > 0.0 {
            let gap = (4 - g.ramen.special_feeling).max(0) as f32;
            let future_gain = [1, 2]
                .iter()
                .map(|&d| get_turn_special_feeling(g.turn() + d).max(0) as f32)
                .sum::<f32>();
            (gap - future_gain).max(0.0) * self.config.friend_hidden_starve_weight
        } else {
            0.0
        };
        // 未来供给缺口：本次外出的 +2 风味按"未来吃面供需缺口"估值——
        // 需求 = 剩余回合 × 平均吃面频率 × 每次消耗；供给 = 未来固定发放 +
        // 友人剩余次数 × 2 + 当前库存。缺口越大，本次补给越接近"保住一次吃面"。
        // 每风味吃面收益 ≈ 800（每次吃面评分 ~1200 ÷ 平均消耗 1.5）。
        let supply = if self.config.friend_future_hidden_weight > 0.0 {
            let gap = self.hidden_future_gap(g);
            gap.min(2.0) * 800.0 * self.config.friend_future_hidden_weight
        } else {
            0.0
        };
        // 主动积极使用：未来 3 回合无固定发放（夏合宿 +2 / 年末 +1）且本次 +2 不溢出
        // （special ≤ 2）时，友人的"体力维持 + 完链"价值——体力正常/高时也愿意用，
        // 维持体力线、提前完链，而不是等饥饿或被迫休息。友人体力恢复实际按
        // vital_bonus 乘算（如骏川满破 +60% → 48~80 体力）。
        let proactive = if self.config.friend_proactive_weight > 0.0 {
            let upcoming = [1, 2, 3]
                .iter()
                .map(|&d| get_turn_special_feeling(g.turn() + d).max(0))
                .sum::<i32>();
            let not_overflow = g.ramen.special_feeling <= 2;
            if upcoming == 0 && not_overflow {
                self.config.friend_proactive_weight
            } else {
                0.0
            }
        } else {
            0.0
        };
        let total = base + event_value + material + chain + starve + supply + proactive;
        Ok((
            total,
            vec![
                ("outing_base".to_string(), base),
                ("friend_event_dynamic".to_string(), event_value),
                ("friend_material_required".to_string(), material),
                ("friend_chain_dynamic".to_string(), chain),
                ("friend_hidden_starve".to_string(), starve),
                ("friend_hidden_future".to_string(), supply),
                ("friend_proactive".to_string(), proactive),
            ],
            format!(
                "友人外出#{} 选项{} 动态事件{:.0} 材料+2(库存{}也不禁用) 饥饿+{:.0} 未来+{:.0} 主动+{:.0}",
                used + 1,
                choice + 1,
                event_value,
                g.ramen.special_feeling,
                starve,
                supply,
                proactive
            )
        ))
    }

    /// 未来隐藏风味供需缺口（风味数）：剩余普通回合的吃面需求 - 未来固定发放 -
    /// 友人剩余次数供给 - 当前库存。
    ///
    /// 平均吃面频率取 0.35 次/回合（实测 25~31 次/70 回合）、每次消耗 1.5 风味；
    /// 固定发放按 `get_turn_special_feeling` 对剩余回合逐回合累计（夏合宿 +2 / 年末 +1）。
    /// 负值截为 0（供给充足时本次外出的补给没有额外价值）。
    fn hidden_future_gap(&self, g: &RamenGame) -> f32 {
        let rem = (71 - g.turn()).max(0) as i32;
        if rem <= 0 {
            return 0.0;
        }
        let demand = rem as f32 * 0.35 * 1.5;
        let mut supply = g.ramen.special_feeling as f32;
        for d in 1..=rem {
            supply += get_turn_special_feeling(g.turn() + d).max(0) as f32;
        }
        let used = g.friend.out_used.iter().filter(|&&x| x).count();
        // 本次外出之后的剩余次数（本次 +2 不计入，因其价值正是本项在估）
        let remaining = (5 - used - 1).max(0) as f32;
        supply += remaining * 2.0;
        (demand - supply).max(0.0)
    }

    /// 按当前状态给友人事件选项评分。先复用通用事件评分，再扣除体力/干劲实际无法获得的
    /// 溢出；最大体力是永久收益，补回通用事件评分尚未覆盖的价值。
    fn dynamic_friend_event_choice(&self, g: &RamenGame, choices: &[Vec<EventChoice>]) -> Result<(usize, f32)> {
        // 友人卡词条乘数：「事件效果提高」作用于五维/PT、「恢复量提高」作用于正向体力
        // 与永久最大体力（与 apply_friend_bonus 规则一致），避免友人事件价值被低估。
        let event_mult = (100 + g.friend.event_bonus) as f32 / 100.0;
        let vital_mult = (100 + g.friend.vital_bonus) as f32 / 100.0;
        let mut values = Vec::with_capacity(choices.len());
        for group in choices {
            let mut val = 0.0;
            for c in group {
                let prob = if c.prob == 0 { 1.0 } else { c.prob as f32 / 100.0 };
                val += self.policy.score_friend_event_choice(g, c, event_mult, vital_mult)?;
                // 体力/干劲溢出修正：用乘算后的实际恢复量，避免高估溢出
                let max_after = g.uma.max_vital + c.value.max_vital;
                let requested_vital = (c.value.vital.max(0) as f32 * vital_mult) as i32;
                let realized_vital = requested_vital.min((max_after - g.uma.vital).max(0));
                val -= (requested_vital - realized_vital) as f32 * self.policy.config.event_vital_weight * prob;
                let requested_motivation = c.value.motivation.max(0);
                let realized_motivation = requested_motivation.min((5 - g.uma.motivation).max(0));
                val -= (requested_motivation - realized_motivation) as f32
                    * self.policy.config.event_motivation_weight
                    * prob;
            }
            values.push(val);
        }
        let choice = values
            .iter()
            .enumerate()
            .max_by(|(li, l), (ri, r)| l.total_cmp(r).then_with(|| ri.cmp(li)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok((choice, values.get(choice).copied().unwrap_or(0.0)))
    }

    fn decide_train(&self, g: &RamenGame, a: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let (mut guard, mut out) = self.policy.decide_train(g, a)?;
        let recovery_guard = self.config.friend_outing_replaces_rest
            && a.get(guard).is_some_and(|x| x.operation == Operation::Rest)
            && out.len() != a.len();
        if recovery_guard && a.iter().any(|x| x.operation == Operation::FriendOuting) {
            // 展开完整候选以便真正执行五段动态估值；最终仍只允许休息/友人恢复动作获胜。
            out = self.policy.score_train_actions(g, a)?;
            guard = a.iter().position(|x| x.operation == Operation::Rest).unwrap_or(guard);
        }
        if out.len() != a.len() {
            let ate_this_turn = self.config.eat_requires_training && g.ramen.current_ramen.is_some();
            let selected_is_train = a
                .get(guard)
                .is_some_and(|action| matches!(action.operation, Operation::Train(_)));
            if !ate_this_turn || selected_is_train {
                return Ok((guard, out));
            }
            // 已吃面但旧硬守门想休息/外出：重新计算全部候选，并只允许五种训练。
            // 生病/自选比赛通常不会经过吃面前门控；这里仍以“拉面只为训练使用”为最终不变量。
            out = self.policy.score_train_actions(g, a)?;
            let _ = out
                .iter()
                .enumerate()
                .filter(|(i, _)| a.get(*i).is_some_and(|x| matches!(x.operation, Operation::Train(_))))
                .max_by(|(li, l), (ri, r)| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)))
                .map(|(i, _)| i)
                .ok_or_else(|| anyhow::anyhow!("已吃面但 Train 阶段没有训练候选"))?;
        }
        if let Some(friend_idx) = a.iter().position(|x| x.operation == Operation::FriendOuting) {
            let (score, breakdown, reason) = self.dynamic_friend_outing_value(g)?;
            if let Some(friend) = out.get_mut(friend_idx) {
                friend.score = score;
                friend.breakdown = breakdown;
                friend.reason = reason;
            }
        }
        let base = out.iter().map(|x| x.score).collect::<Vec<_>>();
        let bb = Self::choose(&out);
        let ph = Self::phase(g.turn());
        for (act, o) in a.iter().zip(out.iter_mut()) {
            let Operation::Train(tt) = act.operation else { continue };
            let tr = tt as usize;
            let buffs = g.calc_training_buff(tr)?;
            let val = g.calc_training_value(&buffs, tr)?;
            let people = g
                .distribution()
                .get(tr)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&x| x >= 0 && (x as usize) < g.persons().len())
                .map(|x| x as usize)
                .collect::<Vec<_>>();
            let hn = people
                .iter()
                .filter(|&&i| g.persons()[i].hint() && matches!(g.persons()[i].person_type(), PersonType::Card))
                .count();
            let all_hint = g.is_hint_special_active_for_train(tr);
            let hp = if self.config.probabilistic_hint && hn > 0 && !all_hint {
                1. / hn as f32
            } else {
                1.
            };
            let mut lt = 0.;
            for i in people {
                let x = &g.persons()[i];
                match x.person_type() {
                    PersonType::ScenarioCard => {
                        lt += match g.friend.out_state {
                            FriendOutState::UnClicked => self.config.first_friend_click_value,
                            _ if x.friendship() < 60 => self.config.low_friend_bond_value * ph,
                            _ => self.config.active_friend_value
                        }
                    }
                    PersonType::Card if x.friendship() < 80 => {
                        let mut b = if g.uma.flags.aijiao { 9. } else { 7. };
                        if x.hint() {
                            b += 5. * hp
                        }
                        b = b.min((80 - x.friendship()) as f32);
                        lt += b * self.config.early_bond_value * ph;
                        if x.hint() {
                            let repeats = if all_hint && i < g.deck().len() {
                                1 + g.deck()[i].effect.hint_count_bonus
                            } else {
                                1
                            };
                            lt += self.config.hint_bonus * hp * repeats as f32
                        }
                    }
                    PersonType::Card if x.hint() => {
                        let repeats = if all_hint && i < g.deck().len() {
                            1 + g.deck()[i].effect.hint_count_bonus
                        } else {
                            1
                        };
                        lt += self.config.hint_bonus * hp * repeats as f32
                    }
                    _ => {}
                }
            }
            o.score += lt;
            o.add("local_long_term", lt);
            let rp = -self.reserve_penalty(g, &val.status_pt);
            o.score += rp;
            o.add("future_status_reserve", rp);
            let balance = self.dynamic_status_adjustment(g, &val.status_pt);
            o.score += balance;
            o.add("dynamic_status_balance", balance);
            if self.config.dynamic_vital {
                let c = (-val.vital).max(0) as f32;
                let z = -c * (Self::vital_factor(g.turn()) - self.policy.config.train_vital_value);
                o.score += z;
                o.add("dynamic_vital", z)
            }
            let base_fr = g.calc_training_failure_rate(&buffs, tr);
            let ramen_effect = calc_ramen_training_effect(g, tr, g.shining_count(tr) > 0);
            let fr = if self.config.effective_ramen_failure {
                (base_fr * (100.0 - ramen_effect.fail_rate_drop as f32) / 100.0).clamp(0.0, 100.0)
            } else {
                base_fr
            };
            if self.config.expected_fail && fr > 0. {
                let p = fr / 100.;
                let bp = if fr >= 20. { p } else { 0. };
                let z = -p * (150. + bp * 350. - self.policy.config.failure_penalty);
                o.score += z;
                o.add("expected_fail_layers", z)
            } else if fr > 15. && self.config.high_fail_penalty > 0. {
                let z = -((fr - 15.) / 85.).clamp(0., 1.) * self.config.high_fail_penalty;
                o.score += z;
                o.add("local_high_fail_tail", z)
            }
            // 吃面-训练联动（显式项）：当前吃面且 at_trains 覆盖该训练位 →
            // 加地区效果强度 × 权重。calc_training_value 已隐含数值加成，本项让
            // 策略在彩圈/羁绊/属性缺口占优时仍倾向兑现吃面成本。
            if self.config.ramen_train_coupling_weight > 0.0 {
                if let Some(rid) = g.ramen.current_ramen {
                    if let Some(region) = RAMENDATA
                        .get()
                        .and_then(|d| d.ramen_region_effect.get(rid))
                    {
                        if region.at_trains.contains(&(tt as i32)) {
                            let effect = (region.xunlian + region.youqing + region.pt_bonus) as f32
                                + region.hint_count as f32 * 10.0;
                            let bonus = effect * self.config.ramen_train_coupling_weight;
                            o.score += bonus;
                            o.add("ramen_train_coupling", bonus);
                        }
                    }
                }
            }
            // 弱位训练偏好（吃面后训练阶段）：仅在吃面回合（current_ramen.is_some()）
            // 且训练位被当前吃面 at_trains 覆盖且该位是**未满**的卡少位（card_type_count ≤ 1）时，
            // 按 youqing/xunlian × effective_boost × (2-card_count) 加分。effective_boost 来自
            // `Self::effective_weak_boost`：默认按智卡数查表，实验 override 用配置字段。
            //
            // 未满条件：已满位只剩 PT 收益（属性差分=0），弱位加成本意为"培养副属性"，
            // 对已满位无意义且会错误抬升其训练分（把该位从"无属性价值"变成"虚高最优"）。
            let weak_boost = Self::effective_weak_boost(g, self.config.ramen_weak_train_boost);
            if weak_boost > 0.0 {
                let tr = tt as usize;
                if g.card_type_count[tr] <= 1
                    && g.uma.five_status[tr] < g.uma.five_status_limit[tr]
                {
                    if let Some(rid) = g.ramen.current_ramen {
                        if let Some(region) = RAMENDATA
                            .get()
                            .and_then(|d| d.ramen_region_effect.get(rid))
                        {
                            if region.at_trains.contains(&(tt as i32)) {
                                let effect = (region.youqing + region.xunlian) as f32;
                                let weight = weak_boost
                                    * (2.0 - g.card_type_count[tr] as f32);
                                let bonus = effect * weight;
                                o.score += bonus;
                                o.add("ramen_weak_train_boost", bonus);
                            }
                        }
                    }
                }
            }
        }
        let lb = Self::choose(&out);
        let sacrifice = base[bb] - base[lb];
        let mut c = if sacrifice <= self.config.max_base_score_sacrifice {
            lb
        } else {
            bb
        };
        if recovery_guard {
            c = out
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    a.get(*i).is_some_and(|x| {
                        x.operation == Operation::Rest
                            || (x.operation == Operation::FriendOuting
                                && self.friend_outing_within_pacing(g)
                                // 体力低时应优先友人（恢复 48~80 体力 + 属性 + 完链，比休息值），
                                // 但隐藏风味不溢出时才行：溢出时友人 +2 补给浪费，只剩
                                // 体力/属性价值，与休息无本质区别；友人次数有限，应留给
                                // "不溢出 + 低体力"的完整价值回合。
                                && self.friend_hidden_not_overflow(g))
                    })
                })
                .max_by(|(li, l), (ri, r)| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)))
                .map(|(i, _)| i)
                .ok_or_else(|| anyhow::anyhow!("低体力守门没有合法恢复动作"))?;
        }
        if !self.friend_outing_within_pacing(g) && a.get(c).is_some_and(|x| x.operation == Operation::FriendOuting) {
            // 配额约束的是所有友人外出，而不只是“替代休息”路径。
            c = out
                .iter()
                .enumerate()
                .filter(|(i, _)| a.get(*i).is_some_and(|x| x.operation != Operation::FriendOuting))
                .max_by(|(li, l), (ri, r)| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)))
                .map(|(i, _)| i)
                .ok_or_else(|| anyhow::anyhow!("友人外出达到跨年总配额后没有其他合法动作"))?;
        }
        Ok((c, out))
    }
    fn pt_effect(pt: i32) -> Result<(i32, i32, i32)> {
        let d = RAMENDATA.get().ok_or_else(|| anyhow::anyhow!("RAMENDATA 未初始化"))?;
        let e = d
            .ramen_pt_effect
            .iter()
            .filter(|e| e.pt_min <= pt)
            .last()
            .or_else(|| d.ramen_pt_effect.first())
            .ok_or_else(|| anyhow::anyhow!("ramen_pt_effect 为空"))?;
        Ok((e.xunlian, e.deyilv, e.hint))
    }
    fn year_end(g: &RamenGame) -> i32 {
        if g.turn() < 24 {
            23
        } else if g.turn() < 48 {
            47
        } else {
            71
        }
    }
    fn scenario_threshold_value(&self, g: &RamenGame, post: i32) -> Result<(f32, f32, f32)> {
        let cur = g.ramen.scenario_pt;
        let rem = (Self::year_end(g) - g.turn()).max(0) as f32;
        let (a, b) = (Self::pt_effect(cur)?, Self::pt_effect(post)?);
        // 训练加成最直接，得意率与 Hint 使用较低近似权重；乘年度剩余回合表达提前跨档的持续价值。
        let delta = ((b.0 - a.0) as f32 * 4. + (b.1 - a.1) as f32 * 0.8 + (b.2 - a.2) as f32 * 0.4).max(0.);
        let region_delta = (calc_region_bonus(post) - calc_region_bonus(cur)).max(0) as f32 * 8.;
        let checkpoint = (delta + region_delta) * rem * self.config.checkpoint_scale;
        let year = (g.current_year() - 1) as usize;
        let d = global!(RAMENDATA);
        let threshold = d.ramen_success_pt[year];
        let rmj = if cur < threshold && post >= threshold {
            self.config.rmj_cross_bonus
        } else {
            0.
        };
        let great = if year == 2 && cur < 5000 && post >= 5000 {
            self.config.great_cross_bonus
        } else {
            0.
        };
        Ok((checkpoint, rmj, great))
    }
    /// 在不吃面的当前状态下返回真正会执行的基础动作。
    /// 用于在 RamenSelect 前决定本回合究竟是训练，还是应先休息/外出/治病/比赛。
    fn pre_eat_action(&self, g: &RamenGame) -> Result<Operation> {
        let mut preview = g.clone();
        preview.stage = RamenStage::Train;
        preview.ramen.current_ramen = None;
        preview.ramen.clear_pending();
        let actions = preview.list_actions()?;
        let (idx, _) = self.decide_train(&preview, &actions)?;
        actions
            .get(idx)
            .map(|a| a.operation)
            .ok_or_else(|| anyhow::anyhow!("吃面前训练决策索引越界: {idx}/{}", actions.len()))
    }

    /// "吃面后必训练 at_trains 覆盖位" 门控：该面落地后，最优训练位是否落在面的 at_trains 内
    ///
    /// 实现：clone 当前状态，设 `current_ramen = Some(region_id)`（等价吃面已落地，coupling /
    /// 弱位偏好等吃面后加分正常参与），跑完整 `decide_train`——若最优动作不是 `Train(_)`
    /// （体力崩 → 休息等，`eat_requires_training` 已挡在 RamenSelect 前，此处防御）或训练位
    /// 不在该面 `at_trains` 内，返回 `false`（调用方将该吃面候选降为 `NEG_INFINITY`）。
    ///
    /// 注意：与 `post_ramen_vital_transition` 的预演同一模式（`current_ramen` + `clear_pending`），
    /// 不落地随机分身（分身属策略流，预演不消费真实随机）。
    fn eat_covered_train_passes(&self, g: &RamenGame, region_id: usize) -> Result<bool> {
        let region = RAMENDATA
            .get()
            .and_then(|d| d.ramen_region_effect.get(region_id))
            .ok_or_else(|| anyhow::anyhow!("地区效果缺失: {region_id}"))?;
        let mut preview = g.clone();
        preview.stage = RamenStage::Train;
        preview.ramen.current_ramen = Some(region_id);
        preview.ramen.clear_pending();
        let actions = preview.list_actions()?;
        let (idx, _) = self.decide_train(&preview, &actions)?;
        match actions.get(idx).map(|a| a.operation) {
            Some(Operation::Train(tt)) => {
                let covered = region.at_trains.contains(&(tt as i32));
                if !covered {
                    crate::diag!(
                        "吃面/{} 落地后最优动作是训练位 {tt:?}，不在该面 at_trains {:?}——否决该面",
                        region.name,
                        region.at_trains
                    );
                }
                Ok(covered)
            }
            _ => Ok(false)
        }
    }

    /// 预演第三年某碗面落地后的最佳训练，返回 `(训练类型, 训练前体力, 训练后体力)`。
    ///
    /// 训练前体力回答“本回合是否应先恢复”，训练后体力回答“下一回合是否会崩盘”。
    /// 不落地随机分身，只使用当前可知面板与确定性拉面效果。
    /// 第三年本回合训练后，低体力是否还会伤害下一次普通训练。
    ///
    /// turn=70 后紧接 turn=71 有马纪念（赛后 +40），再进入 turn=72 超级拉面（回合开始 +20），
    /// 所以没有待保护的普通训练回合；此时体力归零也是合理终盘控制。
    fn y3_collapse_matters(&self, g: &RamenGame) -> bool {
        !self.config.y3_recovery_horizon || g.turn() < 70
    }

    fn post_ramen_vital_transition(&self, g: &RamenGame, region_id: usize) -> Result<Option<(usize, i32, i32)>> {
        // 每年吃面决策都评估吃面后的体力（turn>=72 超级拉面回合不吃面，防御性返回 None）
        if g.turn() >= 72 {
            return Ok(None);
        }
        let mut preview = g.clone();
        preview.stage = RamenStage::Train;
        preview.ramen.current_ramen = Some(region_id);
        preview.ramen.clear_pending();
        let actions = preview.list_actions()?;
        let (idx, _) = self.decide_train(&preview, &actions)?;
        let Some(action) = actions.get(idx) else {
            anyhow::bail!("吃面后预演索引越界: {idx}/{}", actions.len());
        };
        let Operation::Train(tt) = action.operation else {
            return Ok(None);
        };
        let train = tt as usize;
        let buffs = preview.calc_training_buff(train)?;
        let value = preview.calc_training_value(&buffs, train)?;
        let before = preview.uma.vital;
        Ok(Some((train, before, before + value.vital)))
    }

    fn best_action_score(&self, g: &RamenGame) -> Result<f32> {
        let actions = g.list_actions()?;
        let (idx, out) = self.decide_train(g, &actions)?;
        // 守门返回单项 MAX；吃面通常不改变治病/休息等守门结论，因此不把 MAX 计入前向增量。
        if out.len() != actions.len() {
            return Ok(0.0);
        }
        Ok(out.get(idx).map(|x| x.score).unwrap_or(0.0))
    }
    /// 精确复原 v8 的吃面前窗口信号，用于解释其收益来源。
    /// 它只查看候选地区 at_trains 当前已有的真实训练窗口，不预测分身。
    fn ramen_window_alignment(&self, g: &RamenGame, region_id: usize) -> Result<f32> {
        if self.config.ramen_window_weight <= 0.0 {
            return Ok(0.0);
        }
        let d = RAMENDATA.get().ok_or_else(|| anyhow::anyhow!("RAMENDATA 未初始化"))?;
        let region = d
            .ramen_region_effect
            .get(region_id)
            .ok_or_else(|| anyhow::anyhow!("地区效果缺失: {region_id}"))?;
        let mut best = 0.0f32;
        for &t in &region.at_trains {
            if !(0..5).contains(&t) {
                continue;
            }
            let tr = t as usize;
            let buffs = g.calc_training_buff(tr)?;
            let v = g.calc_training_value(&buffs, tr)?;
            let raw = v.status_pt[..5].iter().sum::<i32>() as f32 + v.status_pt[5] as f32 * 2.0;
            let people = g.distribution().get(tr).map(|x| x.len()).unwrap_or(0) as f32;
            let shining = g.shining_count(tr) as f32;
            // 弱位放大：at_trains 覆盖的**未满**卡少位（card_type_count ≤ 1）raw 按 boost 放大，
            // 让"吃面前"选面阶段倾向覆盖弱势属性的面（吃面前瞻，因果断在选面时成立）。
            // 智卡数分组（默认查找表）：1→5.0 / 2→0.0 / 3→2.0。`ramen_weak_train_boost > 0`
            // 时强制 override（实验用），≤0 则走查找表（推荐 preset）。
            // 未满条件与 `decide_train` 弱位 boost 一致：已满位无属性培养价值，放大只会虚高。
            let weak_boost = Self::effective_weak_boost(g, self.config.ramen_weak_train_boost);
            let weak_mult =
                if weak_boost > 0.0 && g.card_type_count[tr] <= 1 && g.uma.five_status[tr] < g.uma.five_status_limit[tr] {
                    weak_boost
                } else {
                    1.0
                };
            best = best.max(raw * weak_mult + people * 8.0 + shining * 35.0);
        }
        let effect = (region.xunlian + region.youqing + region.pt_bonus) as f32 + region.hint_count as f32 * 10.0;
        Ok(best * effect * self.config.ramen_window_weight / 100.0)
    }
    /// 在真正吃面前，用状态副本执行候选面并评估其事后最佳动作。
    /// 所有 region_id 走同一逻辑；不按人数、彩圈或拉面名称硬编码排序。
    fn ramen_lookahead(&self, g: &RamenGame, region_id: usize) -> Result<f32> {
        if self.config.ramen_lookahead_weight <= 0.0 {
            return Ok(0.0);
        }
        let mut no_eat = g.clone();
        no_eat.stage = RamenStage::Train;
        no_eat.ramen.current_ramen = None;
        no_eat.ramen.clear_pending();
        let baseline = self.best_action_score(&no_eat)?;
        let targets = list_special_targets_for(&g.ramen, region_id)?
            .into_iter()
            .min_by_key(|t| t.iter().sum::<i32>())
            .ok_or_else(|| anyhow::anyhow!("拉面 {region_id} 没有合法诀窍方案"))?;
        let n = self.config.ramen_lookahead_samples.max(1);
        let mut total = 0.0;
        for sample in 0..n {
            let mut preview = g.clone();
            preview.ramen.current_ramen = None;
            preview.ramen.pending_ramen = Some(region_id);
            preview.ramen.pending_special_targets = targets;
            // 种子只由吃面前已知状态、候选和样本编号构成；不会读取真实策略流的落点。
            let seed = (g.turn() as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ (g.ramen.scenario_pt as u64).rotate_left(17)
                ^ ((region_id as u64) << 32)
                ^ sample as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            preview.ground_ramen_effects(&mut rng)?;
            preview.stage = RamenStage::Train;
            // decide_train 会用 calc_training_buff/value/failure 对全部五个训练和其他合法动作统一评分。
            total += self.best_action_score(&preview)?;
        }
        Ok((total / n as f32 - baseline) * self.config.ramen_lookahead_weight)
    }
    /// Detect a narrow Y1 safety transition. The normal train policy stays conservative
    /// (raw failure); this only asks whether the shared 30% reduction would make a risky
    /// training overtake the current best action. If any craftable ramen already covers that
    /// training, normal window alignment owns the decision and this bridge stays off.
    fn safety_bridge(&self, g: &RamenGame, ramen_actions: &[RamenAction]) -> Result<Option<(usize, f32)>> {
        if g.current_year() != 1 || self.config.safety_bridge_min_fail > 100.0 {
            return Ok(None);
        }
        let mut preview = g.clone();
        preview.stage = RamenStage::Train;
        let actions = preview.list_actions()?;
        let (_, outs) = self.policy.decide_train(&preview, &actions)?;
        if outs.len() != actions.len() {
            return Ok(None);
        }
        let raw_best = outs.iter().map(|x| x.score).fold(f32::NEG_INFINITY, f32::max);
        let mut rescued: Option<(usize, f32)> = None;
        for (act, out) in actions.iter().zip(outs.iter()) {
            let Operation::Train(tt) = act.operation else { continue };
            let tr = tt as usize;
            let buffs = preview.calc_training_buff(tr)?;
            let fr = preview.calc_training_failure_rate(&buffs, tr);
            if fr < self.config.safety_bridge_min_fail {
                continue;
            }
            let fail_adj = out
                .breakdown
                .iter()
                .find(|(k, _)| k == "fail_adj")
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            let gross = out.score - fail_adj;
            let effective_fr = fr * 0.70;
            let effective_adj =
                -(gross * effective_fr / 100.0 + self.policy.config.failure_penalty * effective_fr / 100.0);
            let effective_score = gross + effective_adj;
            let gain = effective_score - raw_best;
            if gain >= self.config.safety_bridge_min_gain && rescued.map(|(_, old)| gain > old).unwrap_or(true) {
                rescued = Some((tr, gain));
            }
        }
        let Some((tr, gain)) = rescued else {
            return Ok(None);
        };
        let d = RAMENDATA.get().ok_or_else(|| anyhow::anyhow!("RAMENDATA 未初始化"))?;
        let covered = ramen_actions.iter().filter_map(|x| x.ramen).any(|rid| {
            d.ramen_region_effect
                .get(rid)
                .map(|r| r.at_trains.contains(&(tr as i32)))
                .unwrap_or(false)
        });
        Ok(if covered { None } else { Some((tr, gain)) })
    }

    /// Adaptation of Cook2::materialEvaluation. A unit from a scarce stock is worth more
    /// than one from a rich stock (concave sqrt utility). Unlike the farm scenario, ramen stock
    /// resets yearly, so its shadow price decays toward the RMJ boundary. Before reaching the
    /// annual success target we discount the price: spending to secure scenario progression is
    /// deliberately preferred, matching Cook2 Y1's aggressive cooking-until-target rule.
    fn cook2_ramen_stock_cost(&self, g: &RamenGame, region_id: usize) -> Result<f32> {
        if self.config.cook2_stock_weight <= 0.0 {
            return Ok(0.0);
        }
        let targets = list_special_targets_for(&g.ramen, region_id)?
            .into_iter()
            .min_by_key(|t| t.iter().sum::<i32>())
            .ok_or_else(|| anyhow::anyhow!("拉面 {region_id} 无合法 targets"))?;
        let recipe = get_recipe(region_id)?;
        let net = [recipe[0] - targets[0], recipe[1] - targets[1], recipe[2] - targets[2]];
        let year_end = Self::year_end(g);
        let remaining_fraction = ((year_end - g.turn()).max(0) as f32 / 21.0).clamp(0.0, 1.0);
        let year = (g.current_year() - 1) as usize;
        let d = RAMENDATA.get().ok_or_else(|| anyhow::anyhow!("RAMENDATA 未初始化"))?;
        let target = *d.ramen_success_pt.get(year).unwrap_or(&i32::MAX);
        let progression_discount = if g.ramen.scenario_pt < target { 0.35 } else { 1.0 };
        let mut marginal = 0.0;
        for i in 0..3 {
            let before = g.ramen.feeling_stock[i] as f32;
            let after = (g.ramen.feeling_stock[i] - net[i]).max(0) as f32;
            // Bias keeps the derivative finite, as in Cook2's sqrt(count + bias).
            marginal += (before + 2.0).sqrt() - (after + 2.0).sqrt();
        }
        // Hidden flavor is globally flexible, so charge it as two ordinary marginal units.
        let hidden = targets.iter().sum::<i32>() as f32;
        marginal += hidden * 0.50;
        Ok(marginal * self.config.cook2_stock_weight * remaining_fraction * progression_discount)
    }

    fn safety_bridge_candidate(&self, g: &RamenGame, region_id: usize, gain: f32) -> Result<f32> {
        let targets = list_special_targets_for(&g.ramen, region_id)?
            .into_iter()
            .min_by_key(|t| t.iter().sum::<i32>())
            .ok_or_else(|| anyhow::anyhow!("拉面 {region_id} 无合法 targets"))?;
        let used = targets.iter().sum::<i32>() as f32;
        let before = g
            .ramen
            .selected_regions
            .iter()
            .filter(|&&rid| {
                list_special_targets_for(&g.ramen, rid)
                    .map(|x| !x.is_empty())
                    .unwrap_or(false)
            })
            .count();
        let mut post = g.ramen.clone();
        consume_for_ramen(&mut post, region_id, &targets)?;
        let after = g
            .ramen
            .selected_regions
            .iter()
            .filter(|&&rid| {
                list_special_targets_for(&post, rid)
                    .map(|x| !x.is_empty())
                    .unwrap_or(false)
            })
            .count();
        let lost = before.saturating_sub(after) as f32;
        Ok(gain - (lost + used) * self.config.safety_bridge_stock_cost)
    }

    fn deadline_urgency(&self, g: &RamenGame, post: i32) -> Result<f32> {
        if self.config.deadline_urgency_scale <= 0.0 {
            return Ok(0.0);
        }
        let year = (g.current_year() - 1) as usize;
        let data = RAMENDATA.get().ok_or_else(|| anyhow::anyhow!("RAMENDATA 未初始化"))?;
        let normal = *data.ramen_success_pt.get(year).unwrap_or(&i32::MAX);
        let target = if year == 2 { 5000 } else { normal };
        if post >= target {
            return Ok(0.0);
        }
        let turns = (Self::year_end(g) - g.turn() + 1).max(1) as f32;
        let gain = calc_ramen_pt_gain(year, g.ramen.eat_count + 1)?.max(1) as f32;
        let bowls_needed = ((target - post) as f32 / gain).ceil();
        let pressure = (bowls_needed / turns).clamp(0.0, 1.5);
        Ok(pressure * (target - post) as f32 * self.config.deadline_urgency_scale)
    }

    fn decide_special_dynamic(&self, g: &RamenGame, a: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let (_, mut out) = self.policy.decide_special(g, a)?;
        for (act, score) in a.iter().zip(out.iter_mut()) {
            let Some(targets) = act.special_targets else { continue };
            let Some(region) = act.ramen else { continue };
            let mut post = g.ramen.clone();
            consume_for_ramen(&mut post, region, &targets)?;
            let craftable = g
                .ramen
                .selected_regions
                .iter()
                .filter(|&&rid| {
                    list_special_targets_for(&post, rid)
                        .map(|x| !x.is_empty())
                        .unwrap_or(false)
                })
                .count() as f32;
            let balance = post.feeling_stock.iter().map(|&x| (x as f32 + 2.0).sqrt()).sum::<f32>();
            let year_left = (Self::year_end(g) - g.turn()).max(0) as f32 / 21.0;
            let future = (craftable * 18.0 + balance * 4.0) * year_left;
            score.score += future;
            score.add("future_craftability", future);
            score.reason = format!("隐藏方案{:?} 后续可做{}种", targets, craftable as i32);
        }
        Ok((Self::choose(&out), out))
    }

    fn decide_ramen(&self, g: &RamenGame, a: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let (_, mut out) = self.policy.decide_ramen(g, a)?;
        let pre_action = self.pre_eat_action(g)?;
        let year = (g.current_year() - 1) as usize;
        let eat_post = g.ramen.scenario_pt + calc_ramen_pt_gain(year, g.ramen.eat_count)?;
        let deadline_exception = self.deadline_urgency(g, eat_post)? > 0.0
            && matches!(pre_action, Operation::Race | Operation::Rest | Operation::FriendOuting);
        if self.config.eat_requires_training && !matches!(pre_action, Operation::Train(_)) && !deadline_exception {
            let no_eat = a
                .iter()
                .position(|action| action.ramen.is_none())
                .ok_or_else(|| anyhow::anyhow!("需要休息/外出时 RamenSelect 却没有不吃面候选"))?;
            for (i, candidate) in out.iter_mut().enumerate() {
                if i == no_eat {
                    candidate.reason = "不吃面：本回合基础决策不是训练".to_string();
                } else {
                    candidate.score = f32::NEG_INFINITY;
                    candidate.reason = "禁止吃面：本回合应先休息/外出/治病/比赛".to_string();
                }
            }
            return Ok((no_eat, out));
        }
        let risk = (g.ramen.feeling_stock.iter().sum::<i32>() - self.config.feeling_overflow_threshold).max(0) as f32;
        let bridge = self.safety_bridge(g, a)?;
        // 吃面必成价值：本回合基础动作若是训练，吃面使训练必成（fail_rate_drop 生效），
        // 消除失败期望损失 = 失败率 ×（训练收益 × 0.5 + 失败惩罚）。所有吃面候选共享
        // 同一年度 fail_rate_drop，故在循环外统一计算一次。
        let guarantee = if self.config.eat_guarantee_weight > 0.0 {
            match pre_action {
                Operation::Train(tt) => {
                    let tr = tt as usize;
                    let buffs = g.calc_training_buff(tr)?;
                    let base_fr = g.calc_training_failure_rate(&buffs, tr);
                    if base_fr > 0.0 {
                        let val = g.calc_training_value(&buffs, tr)?;
                        let gain_val: f32 =
                            val.status_pt[..5].iter().sum::<i32>() as f32 + val.status_pt[5] as f32 * 2.0;
                        let loss = base_fr / 100.0 * (gain_val * 0.5 + self.policy.config.failure_penalty);
                        loss * self.config.eat_guarantee_weight
                    } else {
                        0.0
                    }
                }
                _ => 0.0
            }
        } else {
            0.0
        };
        for (act, o) in a.iter().zip(out.iter_mut()) {
            if let Some(region_id) = act.ramen {
                // 吃面后必训练 at_trains 覆盖位（C 方案简化约束）：预演该面落地后的最优训练位，
                // 若不在 at_trains 内则否决（吃面加成浪费——玩家 87% 覆盖 vs 自动 52%）。
                if self.config.eat_requires_covered_train
                    && !self.eat_covered_train_passes(g, region_id)?
                {
                    o.score = f32::NEG_INFINITY;
                    o.reason = "禁止吃面：吃完后最优训练位不在该面 at_trains 内".to_string();
                    o.add("eat_covered_train_gate", f32::NEG_INFINITY);
                    continue;
                }
                if let Some((train, pre_vital, post_vital)) = self.post_ramen_vital_transition(g, region_id)? {
                    if train != 4
                        && self.config.y3_post_train_hard_floor > 0
                        && post_vital < self.config.y3_post_train_hard_floor
                    {
                        o.score = f32::NEG_INFINITY;
                        o.reason = format!(
                            "禁止吃面：{}训练体力{}→{}低于硬底线{}",
                            ["速", "耐", "力", "根", "智"][train],
                            pre_vital,
                            post_vital,
                            self.config.y3_post_train_hard_floor
                        );
                        o.add("y3_vital_hard_guard", f32::NEG_INFINITY);
                        continue;
                    }
                    let pre_short = (self.config.y3_pre_train_vital_target - pre_vital).max(0) as f32;
                    let post_short = if self.y3_collapse_matters(g) {
                        (self.config.y3_post_train_vital_target - post_vital).max(0) as f32
                    } else {
                        0.0
                    };
                    let transition_cost = (pre_short + post_short) * self.config.y3_vital_shortfall_weight;
                    o.score -= transition_cost;
                    o.add(
                        "y3_pre_vital_shortfall",
                        -pre_short * self.config.y3_vital_shortfall_weight
                    );
                    o.add(
                        "y3_post_vital_shortfall",
                        -post_short * self.config.y3_vital_shortfall_weight
                    );
                }
                let pressure = risk * self.config.overflow_value;
                o.score += pressure;
                o.add("local_stock_pressure", pressure);
                let y = (g.current_year() - 1) as usize;
                let post = g.ramen.scenario_pt + calc_ramen_pt_gain(y, g.ramen.eat_count)?;
                let (ck, rmj, great) = self.scenario_threshold_value(g, post)?;
                let deadline = self.deadline_urgency(g, post)?;
                let window = self.ramen_window_alignment(g, region_id)?;
                let cook2_cost = self.cook2_ramen_stock_cost(g, region_id)?;
                let safety = if let Some((_, gain)) = bridge {
                    self.safety_bridge_candidate(g, region_id, gain)?
                } else {
                    0.0
                };
                let look = self.ramen_lookahead(g, region_id)?;
                o.score += ck + rmj + great + deadline + window + safety + look - cook2_cost + guarantee;
                o.add("scenario_checkpoint", ck);
                o.add("rmj_cross", rmj);
                o.add("great_cross", great);
                o.add("deadline_urgency", deadline);
                o.add("ramen_window", window);
                o.add("cook2_stock_cost", -cook2_cost);
                o.add("safety_bridge", safety);
                o.add("ramen_lookahead", look);
                o.add("eat_guarantee", guarantee)
            }
        }
        // 吃不吃与吃哪碗分层：eager 模式下，只要 RamenSelect 已列出可制作面，
        // 就在这些面之间 argmax；不扩展 selected_regions，也不枚举年度其他地区。
        // 吃完后的 Train 阶段仍根据真实落地状态重新比较全部合法动作。
        let chosen = if self.config.eager_eat {
            a.iter()
                .zip(out.iter())
                .enumerate()
                .filter(|(_, (act, _))| act.ramen.is_some())
                .max_by(|(li, (_, l)), (ri, (_, r))| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)))
                .map(|(i, _)| i)
                .unwrap_or_else(|| Self::choose(&out))
        } else {
            Self::choose(&out)
        };
        Ok((chosen, out))
    }
}

/// 当前经过配对基准验证的正式拉面杯手写策略。
///
/// 该类型把实验矩阵中表现最好的配置固化成一个可复用 preset，避免模拟器默认策略、
/// 蒙特卡洛 rollout 与 benchmark 各自复制参数后发生漂移。当前 preset 为：
///
/// - 分年技能 PT 权重：第一年 16，第二/三年 64；
/// - 长期结构最大即时分牺牲：140；
/// - 启用属性预留、动态体力、概率 Hint 与连续失败期望；
/// - 吃面 PT 权重：2.0；
/// - 当前真实训练窗口权重：0.10；
/// - 吃面-训练联动（训练侧显式项）权重：0.50；吃面必成价值权重：1.0；
/// - 动态属性平衡：五维完成度修正训练边际价值（短板追赶 0.5 + 近上限衰减 0.5）；
/// - 使用基础失败率作为保守决策风险预算（游戏规则仍应用真实减失败率）；
/// - Cook2 式诀窍边际库存权重：40；
/// - 关闭随机分身 lookahead；
/// - 回合级体力门限：吃面回合训练必成放掉门限（vital_rest_eating=0），不吃面回合
///   保持体力 30 硬休息（三年一致）；第三年吃面时按 y3 门禁（训练后硬底线 15 /
///   吃面前软目标 25 / 缺口软成本 0.5）防打空体力；
/// - 吃面前先决定是否训练；吃面后强制从训练候选中选择，禁止休息浪费加成；
/// - 第三年终盘允许有马前把体力控到 0，随后由赛后 +40 与超级拉面每回合 +20 接管；
/// - 本来要休息时按 0/2/5 跨年累计节奏使用友人外出；第一年不消耗次数，第二年累计 2 次，第三年完成 5 次；
///   隐藏风味缺口大时友人外出价值提高（饥饿加成 300，扣除未来 2 回合固定发放防溢出）；
/// - 五段事件按当前体力、干劲、属性/PT及完链进度动态估值，第三段不再使用硬体力阈值；
/// - 不使用 RMJ 截止期紧迫度加分：300 局同种子矩阵中 deadline20/35/50 完全同轨，
///   平均分 56960.7，显著低于 deadline0 的 58881.6；硬目标仍由规则和既有跨线价值保证。
///
/// 这个结构只负责按年份转发给三份不可变策略；所有字段含义仍由
/// [`LocalRamenConfig`] 与 [`RamenPolicyConfig`] 的 Rustdoc 定义。
pub struct RecommendedRamenTrainer {
    years: [LocalRamenTrainer; 3],
    /// 最近一次调用落在哪一年的策略，用于把对应 breakdown 暴露给 LoggingTrainer。
    last_year: Mutex<Option<usize>>,
    /// 是否记录 `last_year`。rollout 下关闭：24 线程共享同一实例，每次决策都抢同
    /// 一把 `Mutex`，而 `last_year` 的唯一读者是 [`Trainer::last_breakdown`]，
    /// 该场景下三份年策略的 `collect_breakdown` 也已关闭、必然返回 `None`。
    record_last_year: bool
}

impl RecommendedRamenTrainer {
    /// 从正式 preset 精确复制，只覆盖专项矩阵明确列出的评分参数。
    ///
    /// 吃面事务门、体力硬门、友人 0/2/5 节奏、动态事件、隐藏风味等结构逻辑
    /// 均逐字继承 `new()`，防止实验候选混入未声明的策略差异。
    pub fn with_experiment_overrides(
        pt_rates: [f32; 3],
        gap_strength: f32,
        overflow_strength: f32,
        max_base_score_sacrifice: f32,
        ramen_window_weight: f32,
        status_reserve_max: f32,
        early_bond_value: f32,
        hint_bonus: f32,
        weakboost: f32,
        region_weak_cover_weight: f32,
        eat_requires_covered_train: bool,
    ) -> Self {
        let mut trainer = Self::new();
        for (year, pt_rate) in trainer.years.iter_mut().zip(pt_rates) {
            year.policy.config.pt_rate = pt_rate;
            year.config.dynamic_status_balance = gap_strength != 0.0 || overflow_strength != 0.0;
            year.config.status_gap_strength = gap_strength;
            year.config.status_overflow_strength = overflow_strength;
            year.config.max_base_score_sacrifice = max_base_score_sacrifice;
            year.config.ramen_window_weight = ramen_window_weight;
            year.config.status_reserve_max = status_reserve_max;
            year.config.early_bond_value = early_bond_value;
            year.config.hint_bonus = hint_bonus;
            year.config.ramen_weak_train_boost = weakboost;
            year.policy.config.region_weak_cover_weight = region_weak_cover_weight;
            year.config.eat_requires_covered_train = eat_requires_covered_train;
        }
        trainer
    }

    /// 构造当前正式推荐 preset。
    pub fn new() -> Self {
        fn make(pt_rate: f32, vital_rest: i32, eating_rest: i32) -> LocalRamenTrainer {
            let mut policy = RamenPolicyConfig::default();
            policy.pt_rate = pt_rate;
            policy.ramen_pt_weight = 2.0;
            // 不吃面回合体力硬门限（防打空体力后下回合被迫休息/失败）。
            policy.vital_rest = vital_rest;
            // 吃面回合门限：fail_rate_drop 分年份——Y1 30% / Y2 50%（吃面训练并非必成，
            // 低体力仍可能失败），只有 Y3 100% 必成，故仅第三年吃面回合放掉门限（0），
            // 第一/二年吃面回合保留与不吃面相同的硬门限。
            policy.vital_rest_eating = eating_rest;
            // 保守风险预算：只影响策略打分，不改变规则层真实失败率。
            policy.effective_ramen_failure = false;
            // 残余收益折扣（方案 E）：主属性快满时打折副属性+PT，提前分流。初始 1.0 待矩阵验证。
            policy.cap_discount_weight = 1.0;

            let mut local = LocalRamenConfig::default();
            local.status_reserve_max = 40.0;
            local.dynamic_vital = true;
            local.probabilistic_hint = true;
            local.expected_fail = true;
            local.max_base_score_sacrifice = 140.0;
            local.ramen_window_weight = 0.10;
            // 吃面-训练联动（训练侧显式项）+ 吃面必成价值 + 隐藏风味饥饿加成：
            // 见 LocalRamenConfig 对应字段注释；数值经 base_seed=61444 配对矩阵调优
            // （starve 100 局峰值在 300，couple 保持 2.0，gap/over 0.5 最优）。
            local.ramen_train_coupling_weight = 2.0;
            local.eat_guarantee_weight = 3.0;
            local.friend_hidden_starve_weight = 300.0;
            // 友人主动积极使用：短期无固定发放 + 不溢出时给基础价值（体力维持 + 完链）。
            local.friend_proactive_weight = 150.0;
            // 未来供给缺口估值（方案2）经 100 局扫描为单调负收益（fh=0.2 -125 ~ fh=1.0 -767）：
            // 追求友人 5/5 的边际代价超过隐藏风味边际收益，4.6/5 是 starve=300 下的最优平衡。
            // 字段保留可配（matrix_variant `fh`），preset 关闭。
            local.friend_future_hidden_weight = 0.0;
            // 动态属性平衡：按五维完成度修正训练边际价值（短板追赶 + 近上限衰减）。
            local.dynamic_status_balance = true;
            local.status_gap_strength = 0.5;
            local.status_overflow_strength = 0.5;
            local.ramen_lookahead_weight = 0.0;
            local.ramen_lookahead_samples = 1;
            local.effective_ramen_failure = false;
            local.cook2_stock_weight = 40.0;
            local.eat_requires_training = true;
            // 吃面后必训练 at_trains 覆盖位（C 方案简化约束）：选面时预演"吃完练哪个位"，
            // 确保吃面加成不被浪费（玩家 87% 覆盖 vs 自动 52%，见 issues.md 对应条目）。
            local.eat_requires_covered_train = true;
            // 第三年回合级体力门禁（workbench_improve_1 §2）：吃面前软目标 25、
            // 训练后硬底线 15（非智）、缺口软成本 0.5/点——防吃面打空体力后
            // 下回合被迫休息/失败；turn≥70 由有马 +40 / 超级拉面 +20 接管。
            local.y3_pre_train_vital_target = 25;
            local.y3_post_train_vital_target = 0;
            local.y3_vital_shortfall_weight = 0.5;
            local.y3_post_train_hard_floor = 15;
            local.y3_recovery_horizon = true;
            local.friend_outing_replaces_rest = true;
            local.friend_outing3_recovery_vital = 0;
            local.friend_outing_cumulative_caps = [0, 2, 5];
            local.friend_rest_max_special = 4;
            local.deadline_urgency_scale = 0.0;
            local.dynamic_special_targets = true;
            LocalRamenTrainer::with_configs(policy, local)
        }

        Self {
            // 回合级体力门限：不吃面回合统一 40（base_seed=61444 配对 100 局扫描峰值，
            // 30→40 总加权 +318；45 回落——门限过高休息过多）；吃面回合仅第三年放掉
            // （Y3 fail_rate_drop=100% 必成），第一/二年保留 40（Y1/Y2 吃面训练仍可能失败）。
            years: [make(16.0, 40, 40), make(64.0, 40, 40), make(64.0, 40, 0)],
            last_year: Mutex::new(None),
            record_last_year: true
        }
    }

    /// 创建 rollout 专用实例（关闭三份年的 breakdown 采集）
    ///
    /// 搜索/批跑场景必须用本构造器：24 线程共享同一个 rollout trainer，
    /// `stash` 每次决策都无条件 `format!` 出全候选分解并锁同一把 `Mutex`，
    /// 而 rollout 内部的分解文本没有任何消费者——纯锁争用开销。
    /// 与 [`LocalRamenTrainer::for_rollout`] / [`RamenHandwrittenTrainer::for_rollout`] 同构。
    pub fn for_rollout() -> Self {
        let mut trainer = Self::new();
        for year in trainer.years.iter_mut() {
            year.collect_breakdown = false;
        }
        // 连 last_year 的写入一并关掉：只关 collect_breakdown 的话，三个 select_*
        // 每次决策仍会锁同一把 Mutex，「避免锁争用」只做了一半。
        trainer.record_last_year = false;
        trainer
    }

    fn year(game: &RamenGame) -> usize {
        if game.turn() < 24 {
            0
        } else if game.turn() < 48 {
            1
        } else {
            2
        }
    }
}

impl Default for RecommendedRamenTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer<RamenGame> for RecommendedRamenTrainer {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        let year = Self::year(game);
        if self.record_last_year {
            if let Ok(mut slot) = self.last_year.lock() {
                *slot = Some(year);
            }
        }
        self.years[year].select_action(game, actions, rng)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        let year = Self::year(game);
        if self.record_last_year {
            if let Ok(mut slot) = self.last_year.lock() {
                *slot = Some(year);
            }
        }
        self.years[year].select_choice(game, choices, rng)
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        let year = Self::year(game);
        if self.record_last_year {
            if let Ok(mut slot) = self.last_year.lock() {
                *slot = Some(year);
            }
        }
        self.years[year].select_event_choice(game, event, choices, rng)
    }

    fn last_breakdown(&self) -> Option<String> {
        let year = (*self.last_year.lock().ok()?)?;
        self.years[year].last_breakdown()
    }
}

impl Trainer<RamenGame> for LocalRamenTrainer {
    fn select_action(&self, g: &RamenGame, a: &[RamenAction], _r: &mut StdRng) -> Result<usize> {
        // 单个候选直接返回（无选择空间）；仍记录 breakdown 供决策日志展示
        if a.len() <= 1 {
            if self.collect_breakdown {
                if let Ok(mut slot) = self.last_breakdown.lock() {
                    *slot = Some(format!("仅1候选: {}", a[0]));
                }
            }
            return Ok(0);
        }
        // 阶段分派用 `ramen_effective_stage` 而非裸 `g.stage`：第 1 年地区选择（turn 2）
        // 由 `run_begin` 内联触发，此时 `g.stage` 仍是 Begin，裸分派会落入默认分支
        // 恒选候选 0（详见 ramen_handwritten_trainer.rs 的 ramen_effective_stage 注释）。
        let (c, o) = match ramen_effective_stage(g, a) {
            RamenStage::Train => self.decide_train(g, a)?,
            RamenStage::RamenSelect => self.decide_ramen(g, a)?,
            RamenStage::SpecialSelect => {
                if self.config.dynamic_special_targets {
                    self.decide_special_dynamic(g, a)?
                } else {
                    self.policy.decide_special(g, a)?
                }
            }
            RamenStage::RegionSelect => {
                let y = match g.turn() {
                    2 => 0,
                    23 => 1,
                    47 => 2,
                    _ => 0
                };
                self.policy.decide_region(g, y, a)?
            }
            // 缺此分支会落到 `_ => (0, vec![])`，选项二静默变成选项一
            RamenStage::SuperRamenSelect => self.policy.decide_super_ramen(g, a)?,
            _ => (0, Vec::new())
        };
        self.stash(&o);
        Ok(c)
    }
    fn select_choice(&self, g: &RamenGame, c: &[Vec<EventChoice>], _r: &mut StdRng) -> Result<usize> {
        let (i, o) = self.policy.decide_event(g, c)?;
        self.stash(&o);
        Ok(i)
    }
    fn select_event_choice(
        &self, g: &RamenGame, e: &EventData, c: &[Vec<EventChoice>], r: &mut StdRng
    ) -> Result<usize> {
        if (830305111..=830305115).contains(&e.id) && !c.is_empty() {
            let (choice, _) = self.dynamic_friend_event_choice(g, c)?;
            return Ok(choice);
        }
        self.select_choice(g, c, r)
    }
    fn last_breakdown(&self) -> Option<String> {
        self.last_breakdown.lock().ok().and_then(|b| b.clone())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::game::{Game, Trainer};

    use super::{LocalRamenConfig, LocalRamenTrainer, RamenPolicyConfig, RecommendedRamenTrainer};

    /// 第1年地区选择（turn 2 在 run_begin 内联触发、stage=Begin）必须走 decide_region 打分。
    ///
    /// 回归：LocalRamenTrainer::select_action 只按 `g.stage` 分派时，第1年地区选择
    /// 会落入默认分支恒选候选 0（详见 ramen_handwritten_trainer.rs 的 ramen_effective_stage 注释）。
    #[test]
    #[allow(clippy::panic)]
    fn recommended_region_select_year1_runs_policy() -> Result<()> {
        use rand::{SeedableRng, prelude::StdRng};

        use crate::{
            game::{
                InheritInfo,
                ramen::{Operation, RamenAction, RamenGame, rules::get_region_combinations}
            },
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let trainer = RecommendedRamenTrainer::new();
        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        game.base.turn = 2; // 第1年地区选择（run_begin 内联触发）
        let actions: Vec<RamenAction> = get_region_combinations(0)?
            .iter()
            .map(|&c| RamenAction::no_ramen(Operation::RegionSelect(c)))
            .collect();
        let mut rng = StdRng::seed_from_u64(42);
        let idx = trainer.select_action(&game, &actions, &mut rng)?;
        let bd = trainer.last_breakdown();
        println!(
            "第1年地区选择: stage={:?} 候选={} 选中={:?} breakdown={}",
            game.stage,
            actions.len(),
            actions[idx].operation,
            bd.clone().unwrap_or_default()
        );
        if bd.as_deref().unwrap_or_default().is_empty() {
            panic!(
                "第1年地区选择未走 decide_region（stage={:?} 落入默认分支），恒选候选 {idx}",
                game.stage
            );
        }
        Ok(())
    }

    /// 单候选决策点必须记录「仅1候选」breakdown（决策日志完整性）；
    /// `for_rollout` 实例关闭分解采集（搜索 rollout 高频锁争用，与
    /// `RamenHandwrittenTrainer::collect_breakdown` 同构）。
    #[test]
    #[allow(clippy::panic)]
    fn local_single_candidate_breakdown_and_for_rollout() -> Result<()> {
        use rand::{SeedableRng, prelude::StdRng};

        use crate::{
            game::{
                InheritInfo,
                ramen::{Operation, RamenAction, RamenGame}
            },
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        game.base.turn = 2;
        let actions = vec![RamenAction::no_ramen(Operation::RegionSelect([0, 1, 2]))];
        let mut rng = StdRng::seed_from_u64(42);

        let normal = LocalRamenTrainer::new();
        let idx = normal.select_action(&game, &actions, &mut rng)?;
        let bd = normal.last_breakdown().unwrap_or_default();
        println!("普通实例单候选: idx={idx} breakdown={bd}");
        if bd.is_empty() || !bd.contains("仅1候选") {
            panic!("单候选决策点应记录「仅1候选」breakdown，实际: {bd}");
        }

        let rollout = LocalRamenTrainer::for_rollout();
        let idx = rollout.select_action(&game, &actions, &mut rng)?;
        let bd = rollout.last_breakdown();
        println!("for_rollout 单候选: idx={idx} breakdown={bd:?}");
        if bd.is_some() {
            panic!("for_rollout 实例不应采集 breakdown，实际: {bd:?}");
        }
        Ok(())
    }

    /// `RecommendedRamenTrainer::for_rollout()` 只许省掉观测开销，不许改决策。
    ///
    /// 它关掉两样东西——三份年策略的 `collect_breakdown`、以及 `last_year` 的
    /// `Mutex` 写入。两者的唯一读者都是 [`Trainer::last_breakdown`]，决策链
    /// （`choose` / `select_*`）不消费任何一个，所以整局必须逐位相同。
    /// 这条守门存在的意义：将来若有人把某个字段挪进决策路径，这里会红。
    #[test]
    fn recommended_for_rollout_decisions_identical() -> Result<()> {
        use crate::{
            bench::seeded_rngs,
            game::{InheritInfo, ramen::RamenGame, traits::Game},
            gamedata::init_global,
            utils::{Checks, get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        let inherit =
            InheritInfo { blue_count: [15, 0, 0, 0, 3], extra_count: [10, 10, 20, 20, 20, 40] };

        // 同一 base_seed / run_idx ⇒ 决策 RNG 与规则 RNG 都逐位相同
        let mut run = |rollout: bool| -> Result<(i32, [i32; 5], i32, bool)> {
            let (mut rng, rule_master) = seeded_rngs(61444, 0);
            let mut game = RamenGame::newgame(102601, &DECK, inherit.clone())?;
            game.set_rule_master(rule_master);
            let trainer = if rollout {
                RecommendedRamenTrainer::for_rollout()
            } else {
                RecommendedRamenTrainer::new()
            };
            game.run_full_game(&trainer, &mut rng)?;
            Ok((
                game.uma.calc_score(),
                game.uma.five_status,
                game.uma.skill_pt,
                trainer.last_breakdown().is_some()
            ))
        };

        let (s_n, f_n, p_n, bd_n) = run(false)?;
        let (s_r, f_r, p_r, bd_r) = run(true)?;
        println!("new():        评分={s_n} 五维={f_n:?} PT={p_n} 有 breakdown={bd_n}");
        println!("for_rollout(): 评分={s_r} 五维={f_r:?} PT={p_r} 有 breakdown={bd_r}");

        let mut c = Checks::new();
        c.check(s_n == s_r, "整局评分逐位相同");
        c.check(f_n == f_r, "整局五维逐位相同");
        c.check(p_n == p_r, "整局技能点逐位相同");
        c.check(bd_n, "new() 仍暴露 breakdown（决策日志依赖）");
        c.check(!bd_r, "for_rollout() 不暴露 breakdown（rollout 无消费者）");
        c.finish()
    }

    /// 正式 preset 必须使用 v44 同种子回归胜出的友人跨年节奏。
    #[test]
    #[allow(clippy::panic)]
    fn recommended_ramen_uses_025_friend_pacing() {
        let trainer = RecommendedRamenTrainer::new();
        let actual = trainer
            .years
            .each_ref()
            .map(|year| year.config.friend_outing_cumulative_caps);
        let expected = [[0, 2, 5]; 3];
        println!("正式友人累计出门配额: {actual:?}");
        if actual != expected {
            panic!("正式 preset 应使用 {expected:?}，实际为 {actual:?}");
        }
    }

    /// 吃面-训练联动：当前吃面覆盖速位时，速训练候选获得显式 `ramen_train_coupling` 加分，
    /// 非覆盖位不加。`calc_training_value` 的隐含加成之外，策略应倾向兑现吃面成本。
    #[test]
    #[allow(clippy::panic)]
    fn train_coupling_bonus_on_eating() -> Result<()> {
        use crate::{
            game::{
                InheritInfo,
                ramen::{Operation, RamenGame, RamenStage}
            },
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut local = LocalRamenConfig::default();
        local.ramen_train_coupling_weight = 1.0;
        let trainer = LocalRamenTrainer::with_configs(RamenPolicyConfig::default(), local);
        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        game.ramen.current_ramen = Some(10); // 札幌-速 at_trains=[0]，youqing=50
        let mut preview = game.clone();
        preview.stage = RamenStage::Train;
        let actions = preview.list_actions()?;
        let (idx, outs) = trainer.decide_train(&preview, &actions)?;

        let mut speed_coupling = 0.0f32;
        let mut other_coupling_max = 0.0f32;
        let mut speed_found = false;
        for (act, o) in actions.iter().zip(outs.iter()) {
            if let Operation::Train(t) = act.operation {
                let c = o
                    .breakdown
                    .iter()
                    .find(|(k, _)| k == "ramen_train_coupling")
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                if t as usize == 0 {
                    speed_found = true;
                    speed_coupling = c;
                } else {
                    other_coupling_max = other_coupling_max.max(c);
                }
            }
        }
        println!(
            "吃面(速)状态: 速位 coupling={speed_coupling} 其它位 max={other_coupling_max} 选中={:?}",
            actions[idx].operation
        );
        if !speed_found || speed_coupling <= 0.0 {
            panic!("吃面覆盖速位时速训练应有 ramen_train_coupling>0，实际 {speed_coupling}");
        }
        if other_coupling_max != 0.0 {
            panic!("非覆盖位不应有 ramen_train_coupling，实际 {other_coupling_max}");
        }
        Ok(())
    }

    /// 友人隐藏风味饥饿加成：special_feeling 缺口越大友人外出价值越高；
    /// 夏合宿（turn 24 开始 +2）前缺口将被自然补足，饥饿加成应归零（防溢出）。
    #[test]
    #[allow(clippy::panic)]
    fn friend_hidden_starve_and_overflow_guard() -> Result<()> {
        use crate::{
            game::{InheritInfo, ramen::RamenGame},
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut local = LocalRamenConfig::default();
        local.friend_hidden_starve_weight = 15.0;
        let trainer = LocalRamenTrainer::with_configs(RamenPolicyConfig::default(), local);
        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;

        // 饥饿：special=0，无近期固定发放 → starve ≈ 4×15 = 60
        game.ramen.special_feeling = 0;
        game.base.turn = 30;
        let (total0, bd0, _) = trainer.dynamic_friend_outing_value(&game)?;
        let starve0 = bd0
            .iter()
            .find(|(k, _)| k == "friend_hidden_starve")
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        println!("special=0 turn=30: starve={starve0} total={total0:.0}");

        // 防溢出：turn=23（turn 24 夏合宿 +2），special=2 → 缺口 2 被未来发放 2 扣除 → starve=0
        game.base.turn = 23;
        game.ramen.special_feeling = 2;
        let (total1, bd1, _) = trainer.dynamic_friend_outing_value(&game)?;
        let starve1 = bd1
            .iter()
            .find(|(k, _)| k == "friend_hidden_starve")
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        println!("special=2 turn=23: starve={starve1} total={total1:.0}");

        if starve0 < 45.0 {
            panic!("隐藏风味耗尽时友人饥饿加成应显著（>=45），实际 {starve0}");
        }
        if starve1 > 0.5 {
            panic!("夏合宿前缺口将被自然补足，饥饿加成应归零，实际 {starve1}");
        }
        Ok(())
    }

    /// 吃面后必训练 at_trains 覆盖位（C 方案简化约束）：
    /// 1. 该面落地后最优训练位在 at_trains 内 → `eat_covered_train_passes` 通过 → 吃面候选保留
    /// 2. 最优训练位不在该面 at_trains 内 → 门控拒绝（吃面加成将浪费）
    /// 3. 门控关闭（preset 默认开）且构造同一局面时，吃面候选不会被否决
    #[test]
    #[allow(clippy::panic)]
    fn eat_covered_train_gate_blocks_mismatched_ramen() -> Result<()> {
        use crate::{
            game::{
                InheritInfo,
                ramen::{Operation, RamenAction, RamenGame, RamenStage, action::list_ramen_select_actions}
            },
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        // stamina-风格卡组? 用推荐默认卡组（speed 向）——构造"最优训练位=速/耐"但吃"智面"（id 9 at_trains=[4]）
        // 让 at_trains 覆盖位明显不优 → 门控应拒绝智面
        let mut local = LocalRamenConfig::default();
        local.eat_requires_covered_train = true;
        local.ramen_window_weight = 0.10;
        let policy = RamenPolicyConfig::default();
        let on = LocalRamenTrainer::with_configs(policy.clone(), local);
        let mut local_off = LocalRamenConfig::default();
        local_off.eat_requires_covered_train = false;
        local_off.ramen_window_weight = 0.10;
        let off = LocalRamenTrainer::with_configs(policy, local_off);

        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        // year2 中期，体力充足：pre_action 应倾向训练
        // turn 12：无自选比赛（race_grades[12]=0），训练为唯一最佳动作，避免比赛干扰门控断言
        game.base.turn = 12;
        game.uma.vital = 100;
        game.ramen.special_feeling = 2;
        game.ramen.feeling_stock = [2, 2, 2]; // 库存充足，确保候选面可做
        game.ramen.selected_regions = [0, 1, 4]; // 第1年地区：0 速面 / 1 耐面 / 4 智面

        // 局面 A：速低有空间 + 智满 → 最优训练必非智 → 智面 (id 4) 应被门控拒绝
        // 「满」一律从实际上限取，不写字面量：上限 = 剧本基值 + 继承，会随剧本数据与
        // 蓝因子变化，写死数字会让夹具在上限变动后静默失去「满」的语义。
        game.uma.five_status = [600, 1000, 1000, 1000, game.uma.five_status_limit[4]];
        let pass_rid4 = on.eat_covered_train_passes(&game, 4)?;
        println!("局面A(智满): 智面通过={pass_rid4}");
        if pass_rid4 {
            panic!("智已满且最优训练非智时，智面 (id 4) 应被 eat_covered_train_passes 拒绝");
        }

        // 局面 B：其他位全满 + 智低 → 最优训练必是智 → 智面 (id 4) 应通过、速面 (id 0) 拒绝
        // 打印落地面后的候选分布确认最优位
        game.uma.five_status = game.uma.five_status_limit;
        game.uma.five_status[4] = 600;
        {
            let mut preview = game.clone();
            preview.stage = RamenStage::Train;
            preview.ramen.current_ramen = Some(4);
            preview.ramen.clear_pending();
            let acts = preview.list_actions()?;
            let (idx, outs) = on.decide_train(&preview, &acts)?;
            println!("局面B 智面落地最优: {:?} score={:.1}", acts[idx].operation, outs[idx].score);
        }
        let pass_rid4_b = on.eat_covered_train_passes(&game, 4)?;
        let pass_rid0_b = on.eat_covered_train_passes(&game, 0)?;
        println!("局面B(智低): 智面通过={pass_rid4_b} 速面通过={pass_rid0_b}");
        if !pass_rid4_b {
            panic!("其他位全满、只剩智位有空间时，覆盖智位的面 (id 4) 应通过门控");
        }
        if pass_rid0_b {
            panic!("其他位全满、最优训练为智时，不覆盖智位的面 (id 0 速) 应被门控拒绝");
        }
        Ok(())
    }

    /// 吃面必成价值：本回合基础动作是训练且失败率>0 时，吃面候选应计入
    /// `eat_guarantee`（消除失败期望损失）；体力充足失败率为 0 时不计。
    #[test]
    #[allow(clippy::panic)]
    fn eat_guarantee_value_on_risky_train() -> Result<()> {
        use crate::{
            game::{
                InheritInfo,
                ramen::{RamenGame, action::list_ramen_select_actions, policy::RamenPolicyConfig}
            },
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut policy = RamenPolicyConfig::default();
        policy.vital_rest = 0; // 取消体力守门，让低体力训练进入打分
        let mut local = LocalRamenConfig::default();
        local.eat_guarantee_weight = 1.0;
        let trainer = LocalRamenTrainer::with_configs(policy, local);

        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        // 用无自选比赛候选的回合（can_self_race 需 turn>12）：
        // 本测试要验证的是「低体力训练失败率>0 → 吃面必成价值>0」这一机制，
        // 而非比赛/训练的取舍——若放在 G1 回合，自由比赛真实收益会让策略
        // 正确改选比赛（pre_action=Race），吃面必成价值按设计降为 0，前提不成立。
        game.base.turn = 12;
        game.uma.vital = 45; // 速位失败率 (100-45)*(52-45)/40 ≈ 9.6% > 0
        // 智已满上限（训练边际≈0），其余中段属性训练收益高且失败率>0，
        // 策略本回合打算训练（而非休息）→ 吃面必成价值应>0
        game.uma.five_status = [1000, 1000, 1000, 1000, 2400];
        game.ramen.selected_regions = [6, 7, 8];
        game.ramen.special_feeling = 2;
        game.ramen.feeling_stock = [2, 2, 2]; // 库存充足，确保候选面可做

        let actions = list_ramen_select_actions(&game.ramen, &game.ramen.selected_regions);
        let pre = trainer.pre_eat_action(&game)?;
        let (_, outs) = trainer.decide_ramen(&game, &actions)?;
        let mut guarantee = 0.0f32;
        let mut has_eat = false;
        for (act, o) in actions.iter().zip(outs.iter()) {
            if act.ramen.is_some() {
                has_eat = true;
                let g = o
                    .breakdown
                    .iter()
                    .find(|(k, _)| k == "eat_guarantee")
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                guarantee = guarantee.max(g);
            }
        }
        println!(
            "turn=12 vital=45: 吃面候选={has_eat} eat_guarantee={guarantee} 候选数={} pre_action={:?}",
            actions.len(),
            pre
        );
        if !has_eat {
            panic!("测试构造失败：无吃面候选（special={} selected_regions={:?}）", game.ramen.special_feeling, game.ramen.selected_regions);
        }
        if guarantee <= 0.0 {
            panic!("低体力训练失败率>0 时吃面必成价值应>0，实际 {guarantee}");
        }
        Ok(())
    }

    /// 正式 preset 应启用四项新机制：吃面-训练联动、必成价值、隐藏风味饥饿、动态属性平衡。
    #[test]
    #[allow(clippy::panic)]
    fn recommended_ramen_new_mechanisms_enabled() {
        let trainer = RecommendedRamenTrainer::new();
        for (i, year) in trainer.years.each_ref().iter().enumerate() {
            let c = &year.config;
            println!(
                "year{i}: couple={} starve={} guarantee={} statusdyn={} gap={} over={}",
                c.ramen_train_coupling_weight,
                c.friend_hidden_starve_weight,
                c.eat_guarantee_weight,
                c.dynamic_status_balance,
                c.status_gap_strength,
                c.status_overflow_strength
            );
            if c.ramen_train_coupling_weight <= 0.0
                || c.friend_hidden_starve_weight <= 0.0
                || c.eat_guarantee_weight <= 0.0
                || !c.dynamic_status_balance
                || c.status_gap_strength <= 0.0
                || c.status_overflow_strength <= 0.0
            {
                panic!("year{i} 未启用全部新机制: {c:?}");
            }
        }
    }

    /// 未来供给缺口：早期剩余回合多、需求大且友人未用完时 gap>0，本次外出的
    /// +2 风味计入"保住吃面"价值；后期固定发放 + 剩余次数供给充足时 gap=0。
    #[test]
    #[allow(clippy::panic)]
    fn friend_future_hidden_supply() -> Result<()> {
        use crate::{
            game::{InheritInfo, ramen::RamenGame},
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut local = LocalRamenConfig::default();
        local.friend_future_hidden_weight = 1.0;
        let trainer = LocalRamenTrainer::with_configs(RamenPolicyConfig::default(), local);
        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;

        // 早期：turn=30（第二年），特殊=0，未用过友人 → 剩余回合多、需求大 → gap>0
        game.base.turn = 30;
        game.ramen.special_feeling = 0;
        let (_, bd1, _) = trainer.dynamic_friend_outing_value(&game)?;
        let supply1 = bd1
            .iter()
            .find(|(k, _)| k == "friend_hidden_future")
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        println!("turn=30 special=0 used=0: friend_hidden_future={supply1}");

        // 后期：turn=55（第三年），特殊=2，已用 3 次友人 → 固定发放+剩余供给充足 → gap=0
        game.base.turn = 55;
        game.ramen.special_feeling = 2;
        game.friend.out_used = vec![true, true, true, false, false];
        let (_, bd2, _) = trainer.dynamic_friend_outing_value(&game)?;
        let supply2 = bd2
            .iter()
            .find(|(k, _)| k == "friend_hidden_future")
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        println!("turn=55 special=2 used=3: friend_hidden_future={supply2}");

        if supply1 <= 0.0 {
            panic!("早期友人未用时未来缺口应>0，实际 {supply1}");
        }
        if supply2 > 0.5 {
            panic!("后期供给充足时未来缺口应为0，实际 {supply2}");
        }
        Ok(())
    }

    /// 残余收益折扣（方案 E，policy 层）：主属性快满时，训练该位的副属性收益
    /// 打折（cap_discount_weight=1 的 attr < 0 的 attr）；远离上限时两者相同。
    #[test]
    #[allow(clippy::panic)]
    fn cap_discount_ratio_behavior() -> Result<()> {
        use crate::{
            game::{
                InheritInfo,
                ramen::{Operation, RamenGame, RamenStage, TrainingType}
            },
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut policy_off = RamenPolicyConfig::default();
        policy_off.cap_discount_weight = 0.0;
        let mut policy_on = RamenPolicyConfig::default();
        policy_on.cap_discount_weight = 1.0;
        let off = LocalRamenTrainer::with_configs(policy_off, LocalRamenConfig::default());
        let on = LocalRamenTrainer::with_configs(policy_on, LocalRamenConfig::default());

        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        game.base.turn = 50;
        game.uma.vital = 100;

        fn speed_attr(trainer: &LocalRamenTrainer, game: &RamenGame) -> Result<f32> {
            let mut preview = game.clone();
            preview.stage = RamenStage::Train;
            let actions = preview.list_actions()?;
            let (_, outs) = trainer.decide_train(&preview, &actions)?;
            for (act, o) in actions.iter().zip(outs.iter()) {
                if let Operation::Train(TrainingType::Speed) = act.operation {
                    return Ok(o
                        .breakdown
                        .iter()
                        .find(|(k, _)| k == "attr")
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0));
                }
            }
            Ok(0.0)
        }

        // 速位接近上限（剩余 10 < 2×训练值）：打折生效 → on 的 attr < off
        game.uma.five_status[0] = game.uma.five_status_limit[0] - 10;
        let attr_off_near = speed_attr(&off, &game)?;
        let attr_on_near = speed_attr(&on, &game)?;
        println!("速位剩余10: attr_off={attr_off_near} attr_on={attr_on_near}");

        // 速位远离上限（剩余 1500）：不打折 → 两者相同
        game.uma.five_status[0] = game.uma.five_status_limit[0] - 1500;
        let attr_off_far = speed_attr(&off, &game)?;
        let attr_on_far = speed_attr(&on, &game)?;
        println!("速位剩余1500: attr_off={attr_off_far} attr_on={attr_on_far}");

        if attr_on_near >= attr_off_near {
            panic!("速位快满时打折应降低 attr，实际 off={attr_off_near} on={attr_on_near}");
        }
        if (attr_off_far - attr_on_far).abs() > 1e-3 {
            panic!("速位远离上限时不应打折，实际 off={attr_off_far} on={attr_on_far}");
        }
        Ok(())
    }

    /// 弱位训练偏好（双层级，吃面前 + 吃面后）：boost=0 时无副作用；boost>0 且训练位
    /// 是卡少位且被当前吃面 at_trains 覆盖时，ramen_window_alignment 放大该位 raw、
    /// decide_train 给该位训练候选加 (youqing+xunlian)*boost*(2-card_count) 分。
    #[test]
    #[allow(clippy::panic)]
    fn ramen_weak_train_boost_effect() -> Result<()> {
        use crate::{
            game::{
                InheritInfo,
                ramen::{Operation, RamenGame, RamenStage, TrainingType}
            },
            gamedata::{init_global, ramen::RAMENDATA},
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        // stamina build [2,2,0,0,1]：智位 card_type_count[4]=1（卡少位）。
        // 找一个 at_trains 含智位（index 4）的拉面区域作为弱位覆盖面 → id 4/9/14/17/19。
        let rid = 9; // 小仓-智，at_trains=[4], youqing=50
        assert!(RAMENDATA.get().unwrap().ramen_region_effect[rid].at_trains.contains(&4));

        // 构造关 off / on 两个 trainer（policy 一样，仅 local.ramen_weak_train_boost 不同）
        // ramen_window_alignment 在 ramen_window_weight=0 时直接 return 0，故测试时打开 window。
        let mut cfg_off = LocalRamenConfig::default();
        cfg_off.ramen_window_weight = 0.10; // 标准推荐值，让 window 进入评估循环
        cfg_off.ramen_weak_train_boost = -1.0; // 显式关闭查表（让 off=0 effective，测 override 字段生效性）
        let mut cfg_on = LocalRamenConfig::default();
        cfg_on.ramen_window_weight = 0.10;
        cfg_on.ramen_weak_train_boost = 1.5;
        let off = LocalRamenTrainer::with_configs(RamenPolicyConfig::default(), cfg_off);
        let on = LocalRamenTrainer::with_configs(RamenPolicyConfig::default(), cfg_on);

        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 0, 0, 0, 3], extra_count: [10, 10, 20, 20, 20, 40] }
        )?;
        game.base.turn = 30; // year2 中期，吃面落地前
        game.uma.vital = 100;
        game.ramen.current_ramen = Some(rid); // 模拟已吃面（用于 evaluate decide_train）

        // 层级 A：ramen_window_alignment（吃面前瞻）
        // —— boost>0 时，覆盖卡少位（智）的面（这里 region.at_trains=[4] 就是智），
        //    best 应当被 weak_mult 放大 raw。
        let win_off = off.ramen_window_alignment(&game, rid)?;
        let win_on = on.ramen_window_alignment(&game, rid)?;
        println!("ramen_window_alignment[rid={rid}]: off={win_off} on={win_on}");
        if win_on <= win_off {
            panic!("boost>0 时 ramen_window_alignment 应对卡少位覆盖面加分，实际 off={win_off} on={win_on}");
        }

        // 层级 B：decide_train 中弱位训练加分（吃面后）
        // —— boost>0 且训练位是卡少位（智位 card_count=1）且被吃面 at_trains 覆盖时，
        //    智训练候选的 breakdown 出现 ramen_weak_train_boost 项。
        let mut preview = game.clone();
        preview.stage = RamenStage::Train;
        preview.ramen.current_ramen = Some(rid);
        preview.ramen.clear_pending();
        let actions = preview.list_actions()?;
        let (_, outs_off) = off.decide_train(&preview, &actions)?;
        let (_, outs_on) = on.decide_train(&preview, &actions)?;
        for (act, (o_off, o_on)) in actions.iter().zip(outs_off.iter().zip(outs_on.iter())) {
            if let Operation::Train(TrainingType::Wisdom) = act.operation {
                let bonus_off = o_off.breakdown.iter().find(|(k, _)| k == "ramen_weak_train_boost").map(|(_, v)| *v).unwrap_or(0.0);
                let bonus_on = o_on.breakdown.iter().find(|(k, _)| k == "ramen_weak_train_boost").map(|(_, v)| *v).unwrap_or(0.0);
                let score_diff = o_on.score - o_off.score;
                println!("智训练: off_score={:.1} on_score={:.1} diff={:.1} weakboost_off={:.1} weakboost_on={:.1}",
                    o_off.score, o_on.score, score_diff, bonus_off, bonus_on);
                if bonus_off != 0.0 {
                    panic!("boost=0 时不应有 ramen_weak_train_boost 项: {bonus_off}");
                }
                if bonus_on <= 0.0 {
                    panic!("boost>0 且卡少位被吃面覆盖时应有 ramen_weak_train_boost 加分: {bonus_on}");
                }
            }
        }

        // 反例：boost>0 但当前不吃面（current_ramen=None）→ 弱位不加分（区分吃面/不吃面）
        let mut no_eat = game.clone();
        no_eat.ramen.current_ramen = None;
        let (_, outs_no_eat) = on.decide_train(&no_eat, &actions)?;
        for (act, o) in actions.iter().zip(outs_no_eat.iter()) {
            if let Operation::Train(TrainingType::Wisdom) = act.operation {
                let bonus = o.breakdown.iter().find(|(k, _)| k == "ramen_weak_train_boost").map(|(_, v)| *v).unwrap_or(0.0);
                println!("不吃面时智训练 weakboost: {bonus}");
                if bonus != 0.0 {
                    panic!("不吃面时不应有 ramen_weak_train_boost: {bonus}");
                }
            }
        }

        Ok(())
    }

    /// Top 函数精确 microbench
    ///
    /// pprof 采样给出占比估算，但单次真实耗时需 wall-clock 直测。
    /// 在固定局面（speed build turn=30 seed=61444）下，对 sim_profiler
    /// 测出的 top 函数逐个直接调用 N=100000 次，记总/最小/平均时间。
    ///
    /// 输出单位：纳秒；3 轮取 min/mean 减小调度噪声。
    ///
    /// 跑法：`cargo test --release microbench_top_fns -- --ignored --nocapture`
    ///
    /// `#[ignore]`：本测试 `set_current_dir` 改的是**进程级全局 CWD**，与并行跑的其他
    /// 测试互相污染；且 N=100000×3 轮在 debug 下极慢。它本就是手动剖析工具，不是守门。
    #[ignore]
    #[test]
    fn microbench_top_fns() {
        use std::{hint::black_box, time::Instant};

        use crate::{
            bench, game::{Game, InheritInfo, ramen::RamenGame}, gamedata::init_global_with_config, trainer::{
                LoggingTrainer, RecommendedRamenTrainer
            }, utils::{get_workspace_root, load_game_config}
        };

        const N: usize = 100_000;
        const UMA: u32 = 102_601;
        const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        const INHERIT: InheritInfo = InheritInfo {
            blue_count: [15, 0, 0, 0, 3],
            extra_count: [0, 10, 30, 10, 30, 40]
        };

        let workspace_root = get_workspace_root().unwrap();
        std::env::set_current_dir(workspace_root).unwrap();
        init_global_with_config(&load_game_config().unwrap()).unwrap();

        let (mut rng, rule_master) = bench::seeded_rngs(61444, 30);
        let mut game = RamenGame::newgame(UMA, &DECK, INHERIT).unwrap();
        game.set_rule_master(rule_master);
        // 推进到 turn 30（避开 turn 0-1 边界、地区选择、第 1 年体力波动）
        let mut trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), 30);
        trainer.set_logging(false);
        while game.turn() < 30 {
            if !game.next() {
                break;
            }
            game.run_stage(&trainer, &mut rng).unwrap();
        }
        let local = LocalRamenTrainer::new();
        let gain_sample: [i32; 6] = [10, 5, 0, 0, 5, 0];

        // 每函数：warmup + 3 轮 × N 次
        fn run<F: FnMut()>(name: &str, mut f: F, n: usize) -> (u128, f64) {
            // Warmup
            for _ in 0..1000 {
                black_box(f());
            }
            let mut min_total = u128::MAX;
            let mut mean_sum = 0.0f64;
            for round in 0..3 {
                let start = Instant::now();
                for _ in 0..n {
                    black_box(f());
                }
                let total = start.elapsed().as_nanos();
                min_total = min_total.min(total);
                mean_sum += total as f64 / n as f64;
                println!("  {} 轮 {}: total={} ns, mean={:.1} ns/call", name, round + 1, total, total as f64 / n as f64);
            }
            (min_total, mean_sum / 3.0)
        }

        println!("\n=== Top 函数 microbench (speed build turn=30 seed=61444) ===");
        println!("采样函数单位：ns/op；3 轮取 min/mean\n");

        // 1. reserve_penalty
        let (min1, mean1) = run("LocalRamenTrainer::reserve_penalty", || {
            let _ = black_box(local.reserve_penalty(&game, &gain_sample));
        }, N);
        println!(">>> reserve_penalty           min/单轮={} ns   mean/3轮={:.1} ns/call\n", min1, mean1);

        // 2. default_calc_training_buff
        let (min2, mean2) = run("RamenGame::default_calc_training_buff(0)", || {
            let _ = black_box(game.default_calc_training_buff(0).unwrap());
        }, N);
        println!(">>> default_calc_training_buff   min/单轮={} ns   mean/3轮={:.1} ns/call\n", min2, mean2);

        // 3. calc_training_value（先用 buff 准备）
        let buffs = game.default_calc_training_buff(0).unwrap();
        let (min3, mean3) = run("RamenGame::calc_training_value", || {
            let _ = black_box(game.calc_training_value(&buffs, 0).unwrap());
        }, N);
        println!(">>> calc_training_value         min/单轮={} ns   mean/3轮={:.1} ns/call\n", min3, mean3);

        // 4. SupportCard::calc_training_effect
        let sample_card = &game.deck()[0];
        let (min4, mean4) = run("SupportCard::calc_training_effect", || {
            let _ = black_box(sample_card.calc_training_effect(&game, 0).unwrap());
        }, N);
        println!(">>> SupportCard::calc_training_effect  min/单轮={} ns   mean/3轮={:.1} ns/call\n", min4, mean4);

        // 5. CardTrainingEffect::clone
        let (min5, mean5) = run("CardTrainingEffect::clone", || {
            let _ = black_box(buffs.clone());
        }, N);
        println!(">>> CardTrainingEffect::clone    min/单轮={} ns   mean/3轮={:.1} ns/call\n", min5, mean5);

        // 6. Trainer::select_action（LocalRamenTrainer）—— 整段打分耗时
        let train_actions: Vec<crate::game::ramen::RamenAction> = (0..5)
            .map(|tr| {
                use crate::game::ramen::{Operation, TrainingType};
                crate::game::ramen::RamenAction::no_ramen(Operation::Train(match tr {
                    0 => TrainingType::Speed,
                    1 => TrainingType::Stamina,
                    2 => TrainingType::Power,
                    3 => TrainingType::Guts,
                    _ => TrainingType::Wisdom,
                }))
            })
            .collect();
        use rand::SeedableRng;
        let mut action_rng = rand::rngs::StdRng::seed_from_u64(42);
        let (min6, mean6) = run("LocalRamenTrainer::select_action(train)", || {
            let _ = black_box(local.select_action(&game, &train_actions, &mut action_rng).unwrap());
        }, N);
        println!(">>> LocalRamenTrainer::select_action  min/单轮={} ns   mean/3轮={:.1} ns/call\n", min6, mean6);

        println!("\n=== 对比 pprof ticks 数据（1000 局，no diag feature）===");
        println!("reserve_penalty:               148 ticks (~17.2%) [private, 不可直测]");
        println!("default_calc_training_buff:     64 ticks (~7.4%) [Game trait]");
        println!("calc_training_value:            40 ticks (~4.6%) [Game trait]");
        println!("SupportCard::calc_training_effect: 20 ticks (~2.3%) [public]");
        println!("LocalRamenTrainer::select_action  n/a [含整段打分链路]");
        println!("\n注意：reserve_penalty 是 LocalRamenTrainer private 方法，从外部不可直测。");
        println!("select_action 总耗时 - reserve_penalty 预估 ≈ 其他打分项。");
    }
}

