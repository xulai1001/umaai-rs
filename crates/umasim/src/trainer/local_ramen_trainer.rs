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
            policy::{RamenPolicy, RamenPolicyConfig, RamenPolicyOutput, TrainEvalCache},
            rules::{
                calc_ramen_pt_gain, calc_region_bonus, consume_for_ramen, get_recipe, get_turn_special_feeling,
                min_special_targets
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

    /// 预留惩罚的增益口径（`reserve_penalty` 用哪个增量计算"训练后透支"）。
    ///
    /// - `0`：原始未截断增量（旧行为）。已满位属性收益实为 0，但此处仍按原始
    ///   gain（如 +355 速）计入透支，惩罚被虚增——终盘（r→0）时雪崩放大。
    /// - `1`（A 修复）：按实际生效增量 `min(gain, h)` 截断，已满位增量 0、不产生
    ///   透支，接近满位按真实可增量温和处罚。
    /// - `2`（B 修复）：已满位（`h <= 0`）直接跳过惩罚，其余位保持旧逻辑。
    ///
    /// 调参按 `matrix_variant` / `with_tokens` 的 `reserve`（调 max）与 `rgn`（调口径）
    /// 两个 token 实验；`0` 为基准默认，改动经基准判别后决定是否入 preset。
    pub reserve_gain_mode: u8,

    /// 是否启用按五维完成度动态调整属性边际价值。
    ///
    /// 开启后会提高相对落后属性的精确评分边际，并在属性接近上限时降低继续堆叠的价值；
    /// 三张及以上同类型卡会放大对应属性的近上限衰减。默认关闭，仅供配对矩阵验证。
    pub dynamic_status_balance: bool,

    /// 短板追赶强度。1.0 表示完成度每落后最高维度 10%，该维精确属性评分边际增加 10%。
    pub status_gap_strength: f32,

    /// 近上限衰减强度。属性完成度超过 70% 后按平方曲线增长，并受同类型卡过量系数放大。
    pub status_overflow_strength: f32,

    /// EXP-006e：power（i==2）专属近上限衰减强度覆盖。
    /// `f32::NAN` = 不覆盖（用统一 [`Self::status_overflow_strength`]）。
    /// 诊断依据：EXP-006d 盘面 P90−P10 power=−204 唯一负值 ⇒ 力量过度喂养。
    pub power_overflow_strength: f32,

    /// EXP-006e：power（i==2）专属短板追赶强度覆盖（可为负=允许更落后）。
    /// `f32::NAN` = 不覆盖（用统一 [`Self::status_gap_strength`]）。
    pub power_gap_strength: f32,

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

    /// 实验：**第三年**替代休息路径允许的最高隐藏风味（现行口径 = 2）。
    ///
    /// 第三年夏合宿（turn 60 +2 / 61-63 各 +1）会把风味顶到 4，之后要等吃面把风味
    /// 吃到 ≤2 才打开替代路径，晚回合窗口被闸门吃掉，第 5 次出行常走不完。
    /// 调到 `3` 可提早约 1 次吃面的窗口；代价是该次出行 +2 中有 1 点溢出、且饥饿
    /// 加成（库存 ≥3 时已为 0）不参与，出行只剩体力/干劲/属性/完链价值。
    pub friend_y3_overflow_cap: i32,

    /// 实验：第三年剩余配额走不完时，是否强制补足出行次数。
    ///
    /// `false` = 现行口径（只受 [`Self::friend_outing_cumulative_caps`] 硬上限约束，
    /// 用不用得完完全交给动态估值）。`true` 时若"剩余出行次数 > 剩余可出行回合数"
    /// 且本次出行不溢出（受 [`Self::friend_y3_overflow_cap`] 限制），则该回合直接
    /// 判定出行胜利，保证 5 次走完。
    pub friend_y3_urgency_force: bool,

    /// 实验：第三年**必须走完的次数**（`0` = 按配额自动取 `caps[2] - caps[1]`）。
    ///
    /// 与 [`Self::friend_y3_urgency_force`] 配合：当"第三年还该走的次数 ≥ 剩余可出行
    /// 回合数"时强制让出行赢，保证当年配额走完（如 `[0,2,5]` 第三年必须走 3 次）。
    pub friend_y3_force_remaining: i32,

    /// 友人出行是否**必须走完 5 次**（完成硬门限）。
    ///
    /// 由 `game_config.toml` / `default_config.toml` 的 `friend_complete_required`
    /// 落到此开关（默认开）。开时隐藏风味闸门不再阻断出行（完成优先于风味利用），
    /// 且"剩余出行次数 ≥ 剩余可出行回合数"时强制出行，保证 5 次走完；
    /// 关时回到纯动态估值口径（可能主动跳过第 5 次）。
    pub friend_complete_required: bool,

    /// RMJ/第三年5000目标在截止前的可达性紧迫度。
    pub deadline_urgency_scale: f32,

    /// 实验（第八轮）：逐卡 Hint 精确估值倍率（百分比）。
    ///
    /// 0.0 = 关闭，沿用固定 hint_bonus；正数 = 用 P/100 x hint_event_expected_value
    /// 替换固定值，让卡面 hint_level 0~5（1~6 级 Hint）与属性分支按终局评分真实折算。
    pub hint_card_aware: f32,

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
            reserve_gain_mode: 0,
            dynamic_status_balance: false,
            status_gap_strength: 0.0,
            status_overflow_strength: 0.0,
            power_overflow_strength: f32::NAN,
            power_gap_strength: f32::NAN,
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
            friend_y3_overflow_cap: 2,
            friend_y3_urgency_force: false,
            friend_y3_force_remaining: 0,
            friend_complete_required: false,
            deadline_urgency_scale: 0.0,
            dynamic_special_targets: false,
            hint_card_aware: 0.0
        }
    }
}
/// 同一基础局面的训练与策略评估；只有当前拉面可以在预演间改变。
#[derive(Default)]
struct LocalTrainCache {
    training: TrainEvalCache,
    long_term: [[Option<f32>; 2]; 5],
    friend: Option<RamenPolicyOutput>
}

pub struct LocalRamenTrainer {
    policy: RamenPolicy,
    config: LocalRamenConfig,
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
            last_breakdown: Mutex::new(None)
        }
    }
    /// 创建 rollout 专用实例，关闭评分分解、原因字符串和日志文本采集。
    pub fn for_rollout() -> Self {
        let mut trainer = Self::new();
        trainer.policy.collect_details = false;
        trainer
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
            } else if let Some(v) = token.strip_prefix("fov3") {
                // 第三年替代休息路径的隐藏风味上限（fov33 → 3；fov34 → 4 基本关闸）
                local.friend_y3_overflow_cap = v.parse()?
            } else if token == "furg3" {
                // 第三年配额紧迫度强制补足（配合 fcap 使用，如 fcap035-furg3）
                local.friend_y3_urgency_force = true
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
            } else if let Some(v) = token.strip_prefix("rgn") {
                // reserve 增益口径：0=原始（基准） / 1=A截断 / 2=B满位豁免
                let mode: u8 = v.parse()?;
                if mode > 2 {
                    anyhow::bail!("rgn 仅支持 0/1/2: {v}");
                }
                local.reserve_gain_mode = mode
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
        if !self.policy.collect_details {
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
    /// 预留门限（与 `dynamic_status_adjustment` 一对；同 microbench 用）
    ///
    /// 注：原为私有方法，提升为 `pub` 是给 `tools/data_collection/calc_training_value_microbench.rs`
    /// 性能调优工具用的。
    pub fn reserve_penalty(&self, g: &RamenGame, gain: &[i32; 6]) -> f32 {
        if self.config.status_reserve_max <= 0. {
            return 0.;
        }
        let rem = (76 - g.turn()).max(0) as f32;
        let r = self.config.status_reserve_max * rem / 76.;
        let mut p = 0.;
        for i in 0..5 {
            let h = (g.uma.five_status_limit[i] - g.uma.five_status[i]).max(0) as f32;
            // 已满位豁免（B 修复）：该维已无空间，训练不可能透支未来预留，跳过。
            if self.config.reserve_gain_mode == 2 && h <= 0. {
                continue;
            }
            let b = (r - h).max(0.);
            // 参与透支计算的增量（A 修复）：按实际生效增量截断——已满位增量 0，
            // 接近满位只按真实可增部分计，避免把被 cap 截断掉的溢出当透支再罚一次。
            let eff_gain = if self.config.reserve_gain_mode == 1 {
                (gain[i] as f32).min(h)
            } else {
                gain[i] as f32
            };
            let a = (r - (h - eff_gain)).max(0.);
            p += (a * a - b * b) / (2. * r.max(1.));
        }
        p * 6.
    }
    /// 注：原为私有方法，提升为 `pub` 是给 `tools/data_collection/calc_training_value_microbench.rs`
    /// 性能调优工具用的。
    pub fn dynamic_status_adjustment(&self, g: &RamenGame, gain: &[i32; 6]) -> f32 {
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
            // EXP-006e：power 位用专属覆盖（NAN=回退统一值，base 逐位不变）
            let gap_strength = if i == 2 && !self.config.power_gap_strength.is_nan() {
                self.config.power_gap_strength
            } else {
                self.config.status_gap_strength
            };
            let overflow_strength = if i == 2 && !self.config.power_overflow_strength.is_nan() {
                self.config.power_overflow_strength
            } else {
                self.config.status_overflow_strength
            };
            let gap_bonus = gap_strength * (leading - completion[i]).max(0.0);
            let near_cap = ((completion[i] - 0.70) / 0.30).clamp(0.0, 1.0);
            let excess_cards = (g.card_type_count[i] - 2).max(0) as f32;
            let overflow = overflow_strength
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
        // 完成硬门限开启时，出行完成优先于隐藏风味利用：风味溢出不再是阻断条件
        // （否则晚回合窗口被闸门吃掉，5 次走不完）。
        if self.config.friend_complete_required {
            return true;
        }
        let cap = if g.current_year() >= 3 { self.config.friend_y3_overflow_cap } else { 2 };
        g.ramen.special_feeling <= cap
    }

    /// 第三年还剩几个**可出行回合**（不含必赛、夏合宿、超级拉面回合）。
    ///
    /// 只用于第三年的配额紧迫度判断（[`Self::friend_y3_urgency_force`]）；
    /// 统一按拉面剧本的固定回合结构（夏合宿 60-63、超级拉面 72-77）与本人赛程计算。
    fn friend_outing_turns_left(&self, g: &RamenGame) -> i32 {
        (g.turn()..=71)
            .filter(|&t| !g.uma.is_race_turn(t) && !(60..64).contains(&t))
            .count() as i32
    }


    /// 下一段友人外出的动态价值。
    ///
    /// 事件本体按当前体力/干劲裁掉溢出，第三段两个选项也在这里实时比较；万能材料固定按
    /// 2 个来源计价，即使当前计数已满也不把外出禁掉。跨年稀缺性只由累计配额控制。
    fn dynamic_friend_outing_value(&self, g: &RamenGame) -> Result<(f32, Vec<(&'static str, f32)>, String)> {
        let used = g.friend.out_used.iter().filter(|&&x| x).count();
        if used >= 5 {
            return Ok((f32::NEG_INFINITY, vec![], if self.policy.collect_details {
                "友人外出已完成".to_string()
            } else {
                String::new()
            }));
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
            if self.policy.collect_details {
                vec![
                    ("outing_base", base),
                    ("friend_event_dynamic", event_value),
                    ("friend_material_required", material),
                    ("friend_chain_dynamic", chain),
                    ("friend_hidden_starve", starve),
                    ("friend_hidden_future", supply),
                    ("friend_proactive", proactive),
                ]
            } else {
                Vec::new()
            },
            if self.policy.collect_details {
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
            } else {
                String::new()
            }
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
        let mut best: Option<(usize, f32)> = None;
        for (i, group) in choices.iter().enumerate() {
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
            if best.is_none_or(|(_, best_val)| val.total_cmp(&best_val).is_gt()) {
                best = Some((i, val));
            }
        }
        Ok(best.unwrap_or((0, 0.0)))
    }

    /// 单个带 Hint 人头的价值：默认固定 hint_bonus，启用第八轮实验后按精确模型折算。
    fn hint_person_value(&self, g: &RamenGame, person_index: usize, tr: usize) -> f32 {
        if self.config.hint_card_aware > 0.0 {
            self.config.hint_card_aware * self.hint_event_expected_value(g, person_index, tr)
        } else {
            self.config.hint_bonus
        }
    }

    /// 单次 Hint 事件的期望终局评分（第八轮实验，见 LocalRamenConfig::hint_card_aware）。
    ///
    /// 规则层（RamenGame::handle_hint_event -> push_hint_event）在训练成功且人头带 Hint
    /// 时必然推 1 个事件（hint_count_bonus 另按次数计）：event_probs.hint_attr = 0.25 走
    /// 属性事件 hint_event_value[train]，其余走技能事件，给 min(5, 1 + 卡面 hint_level)
    /// 级 Hint（再受 max_hint_per_card - total_hints 截断；等级 <= 0 时只推属性事件）。
    ///
    /// 两者都按终局评分折算：属性增量走 RamenPolicy::status_gain（与训练同一凹凸曲线），
    /// Hint 等级走 hint_pt_rate x pt_score_rate（6.5 x 2.0 = 13 分/级，见 Uma::total_pt 与
    /// Uma::score_parts）。固定 hint_bonus 无法表达卡面 hint_level 差异，也表达不了
    /// “这一位现在还有几级 Hint 可拿”；本函数是逐人头精确模型，无卡人头按 1 级处理。
    fn hint_event_expected_value(&self, g: &RamenGame, person_index: usize, tr: usize) -> f32 {
        let cons = global!(GAMECONSTANTS);
        let attr_prob = crate::utils::system_event_prob("hint_attr").unwrap_or(0.25) as f32;
        let mut attr_value = 0.0;
        if let Some(row) = cons.hint_event_value.get(tr) {
            for (i, &inc) in row.iter().take(5).enumerate() {
                if inc > 0 {
                    attr_value += self.policy.status_gain(g, i, inc);
                }
            }
        }
        let levels = match Game::deck_index_of(g, person_index) {
            Some(di) => (1 + g.deck()[di].card_value().hint_level)
                .min(5)
                .min(cons.max_hint_per_card - g.deck()[di].total_hints),
            None => 1
        };
        if levels <= 0 {
            // 卡面 Hint 已满：规则层只推属性事件。
            return attr_value;
        }
        let per_level = cons.hint_pt_rate * cons.pt_score_rate;
        attr_value * attr_prob + (1.0 - attr_prob) * levels as f32 * per_level
    }

    /// 按原人头顺序计算羁绊与 Hint 的长远价值，隐藏 Hint 模式由当前训练位决定。
    fn train_long_term(&self, g: &RamenGame, tr: usize, all_hint: bool) -> f32 {
        let ph = Self::phase(g.turn());
        let people = g
            .distribution()
            .get(tr)
            .into_iter()
            .flatten()
            .copied()
            .filter(|&x| x >= 0 && (x as usize) < g.persons().len())
            .map(|x| x as usize);
        let hn = people
            .clone()
            .filter(|&i| g.persons()[i].hint() && matches!(g.persons()[i].person_type(), PersonType::Card))
            .count();
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
                        lt += self.hint_person_value(g, i, tr) * hp * repeats as f32
                    }
                }
                PersonType::Card if x.hint() => {
                    let repeats = if all_hint && i < g.deck().len() {
                        1 + g.deck()[i].effect.hint_count_bonus
                    } else {
                        1
                    };
                    lt += self.hint_person_value(g, i, tr) * hp * repeats as f32
                }
                _ => {}
            }
        }
        lt
    }

    /// 整回合 Train 阶段打分（5 train 候选 + 修复路径）
    ///
    /// 注：原为私有方法，提升为 `pub` 是给 `tools/data_collection/calc_training_value_microbench.rs`
    /// 性能调优工具用的；产品路径仍走 `select_action`。
    pub fn decide_train(&self, g: &RamenGame, a: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let mut eval_cache = LocalTrainCache::default();
        let mut out = Vec::new();
        let chosen = self.decide_train_cached(g, a, g.ramen.current_ramen, &mut eval_cache, &mut out)?;
        Ok((chosen, out))
    }

    /// 复用同一局面的训练评估；选面预演只改变当前面时，由 policy 刷新各面的训练上层值。
    fn decide_train_cached(
        &self, g: &RamenGame, a: &[RamenAction], ramen: Option<usize>, eval_cache: &mut LocalTrainCache,
        out: &mut Vec<RamenPolicyOutput>
    ) -> Result<usize> {
        let mut guard = self.policy.decide_train_cached(g, a, ramen, &mut eval_cache.training, out)?;
        let recovery_guard = self.config.friend_outing_replaces_rest
            && a.get(guard).is_some_and(|x| x.operation == Operation::Rest)
            && out.len() != a.len();
        if recovery_guard && a.iter().any(|x| x.operation == Operation::FriendOuting) {
            // 展开完整候选以便真正执行五段动态估值；最终仍只允许休息/友人恢复动作获胜。
            self.policy.score_train_actions_cached(g, a, ramen, &mut eval_cache.training, out)?;
            guard = a.iter().position(|x| x.operation == Operation::Rest).unwrap_or(guard);
        }
        if out.len() != a.len() {
            let ate_this_turn = self.config.eat_requires_training && ramen.is_some();
            let selected_is_train = a
                .get(guard)
                .is_some_and(|action| matches!(action.operation, Operation::Train(_)));
            if !ate_this_turn || selected_is_train {
                return Ok(guard);
            }
            // 已吃面但旧硬守门想休息/外出：重新计算全部候选，并只允许五种训练。
            // 生病/自选比赛通常不会经过吃面前门控；这里仍以“拉面只为训练使用”为最终不变量。
            self.policy.score_train_actions_cached(g, a, ramen, &mut eval_cache.training, out)?;
            let _ = out
                .iter()
                .enumerate()
                .filter(|(i, _)| a.get(*i).is_some_and(|x| matches!(x.operation, Operation::Train(_))))
                .max_by(|(li, l), (ri, r)| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)))
                .map(|(i, _)| i)
                .ok_or_else(|| anyhow::anyhow!("已吃面但 Train 阶段没有训练候选"))?;
        }
        if let Some(friend_idx) = a.iter().position(|x| x.operation == Operation::FriendOuting) {
            let cached = match &mut eval_cache.friend {
                Some(cached) => cached,
                slot => {
                    let (score, breakdown, reason) = self.dynamic_friend_outing_value(g)?;
                    slot.insert(RamenPolicyOutput { score, breakdown, reason, ..Default::default() })
                }
            };
            if let Some(friend) = out.get_mut(friend_idx) {
                *friend = cached.clone();
            }
        }
        let bb = Self::choose(out);
        let best_base = out[bb].score;
        // 同训练位复用相同 eval，重复候选的原分相同；非训练分数不参与下方调整。
        let mut train_base = [0.0; 5];
        for (act, o) in a.iter().zip(out.iter_mut()) {
            let Operation::Train(tt) = act.operation else { continue };
            let tr = tt as usize;
            train_base[tr] = o.score;
            // B2：复用 policy 层已算好的 eval（`decide_train_cached` 首次求值）
            let eval = eval_cache.training[tr].as_ref().expect("Train eval 应由 policy 层预填");
            let val = &eval.value;
            let all_hint = g.is_hint_special_active_for_train_with_ramen(tr, ramen);
            let lt = *eval_cache.long_term[tr][usize::from(all_hint)]
                .get_or_insert_with(|| self.train_long_term(g, tr, all_hint));
            o.score += lt;
            let rp = -self.reserve_penalty(g, &val.status_pt);
            o.score += rp;
            let balance = self.dynamic_status_adjustment(g, &val.status_pt);
            o.score += balance;
            if self.policy.collect_details {
                o.add("local_long_term", lt);
                o.add("future_status_reserve", rp);
                o.add("dynamic_status_balance", balance);
            }
            if self.config.dynamic_vital {
                let c = (-val.vital).max(0) as f32;
                let z = -c * (Self::vital_factor(g.turn()) - self.policy.config.train_vital_value);
                o.score += z;
                if self.policy.collect_details {
                    o.add("dynamic_vital", z);
                }
            }
            let base_fr = eval.fail_rate;
            let ramen_effect = &eval.ramen_effect;
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
                if self.policy.collect_details {
                    o.add("expected_fail_layers", z);
                }
            } else if fr > 15. && self.config.high_fail_penalty > 0. {
                let z = -((fr - 15.) / 85.).clamp(0., 1.) * self.config.high_fail_penalty;
                o.score += z;
                if self.policy.collect_details {
                    o.add("local_high_fail_tail", z);
                }
            }
            // 吃面-训练联动（显式项）：当前吃面且 at_trains 覆盖该训练位 →
            // 加地区效果强度 × 权重。calc_training_value 已隐含数值加成，本项让
            // 策略在彩圈/羁绊/属性缺口占优时仍倾向兑现吃面成本。
            if self.config.ramen_train_coupling_weight > 0.0 {
                if let Some(rid) = ramen {
                    if let Some(region) = RAMENDATA
                        .get()
                        .and_then(|d| d.ramen_region_effect.get(rid))
                    {
                        if region.at_trains.contains(&(tt as i32)) {
                            let effect = (region.xunlian + region.youqing + region.pt_bonus) as f32
                                + region.hint_count as f32 * 10.0;
                            let bonus = effect * self.config.ramen_train_coupling_weight;
                            o.score += bonus;
                            if self.policy.collect_details {
                                o.add("ramen_train_coupling", bonus);
                            }
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
                    if let Some(rid) = ramen {
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
                                if self.policy.collect_details {
                                    o.add("ramen_weak_train_boost", bonus);
                                }
                            }
                        }
                    }
                }
            }
        }
        let lb = Self::choose(out);
        let local_base = match a[lb].operation {
            Operation::Train(t) => train_base[t as usize],
            _ => out[lb].score
        };
        let sacrifice = best_base - local_base;
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
        // 实验：第三年配额紧迫度强制补足（`friend_y3_urgency_force`）。
        //
        // 第三年可出行回合有限（必赛 + 夏合宿占掉大半），而动态估值只按"这一次值不值"
        // 定价，没有任何"不补就作废"的紧迫项——实测第 5 次出行常因风味闸门/估值不敌
        // 训练而走不完。这里在"剩余次数 > 剩余可出行回合数"时直接让出行赢，
        // 且仍要求本次不溢出（受 `friend_y3_overflow_cap` 约束）。
        if self.config.friend_y3_urgency_force
            && g.current_year() >= 3
            && a.get(c).is_some_and(|x| x.operation != Operation::FriendOuting)
        {
            let out_done = g.friend.out_used.iter().filter(|&&x| x).count() as i32;
            let left = 5 - out_done;
            // 第三年必须走完的次数：显式目标（`friend_y3_force_remaining`）优先，
            // 否则用"第三年配额允许的剩余次数"= caps[2] - caps[1]。
            let caps = self.config.friend_outing_cumulative_caps;
            let y3_target = if self.config.friend_y3_force_remaining > 0 {
                self.config.friend_y3_force_remaining
            } else {
                (caps[2].saturating_sub(caps[1])) as i32
            };
            let must_do = y3_target.min(left).max(0);
            if must_do > 0 && must_do >= self.friend_outing_turns_left(g) && self.friend_hidden_not_overflow(g) {
                if let Some(fi) = a.iter().position(|x| x.operation == Operation::FriendOuting) {
                    if self.friend_outing_within_pacing(g) {
                        c = fi;
                    }
                }
            }
        }
        // 完成硬门限（`friend_complete_required`，默认开）：保证 5 次走完。
        //
        // 触发条件：剩余出行次数 ≥ 剩余可出行回合数 ⇒ 本回合不走就来不及了，直接
        // 让出行赢（含第一年的下限检查：若剩余次数已超过后续全部可出行回合数，
        // 则第一年也必须补）。不接受"来不及"的极端局面——该分支每回合都会重新
        // 检查，故只要"剩余次数 ≤ 剩余可出行回合数"成立，就一定能在某个回合补足。
        if self.config.friend_complete_required
            && a.get(c).is_some_and(|x| x.operation != Operation::FriendOuting)
            && self.friend_outing_within_pacing(g)
        {
            let out_done = g.friend.out_used.iter().filter(|&&x| x).count() as i32;
            let left = 5 - out_done;
            let turns_left = self.friend_outing_turns_left(g);
            if left > 0 && left >= turns_left {
                if let Some(fi) = a.iter().position(|x| x.operation == Operation::FriendOuting) {
                    c = fi;
                }
            }
        }
        // 诊断：给最终中选的友人出行候选标注决策路径与次优对照。
        //
        // 用于区分两种性质完全不同的友人出行：
        // - `恢复`（低体力守门内与休息二选一）：替换的是**休息**，体力缺口本来就要补，
        //   机会成本最低；三年皆可承担。
        // - `常规`（自由打分胜出）：替换的是**训练/比赛**，真实机会成本，
        //   第三年自由回合少时最伤。
        // `次优=` 给出不出行时本会选的动作及其分数，`Δ` 为友人分减次优分（负数表示
        // 该回合友人是被守门路径选中、并非打分胜出）。纯日志，不参与打分。
        if self.policy.collect_details && a.get(c).is_some_and(|x| x.operation == Operation::FriendOuting) {
            let path = if recovery_guard { "恢复" } else { "常规" };
            let best_alt = out
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != c)
                .max_by(|(li, l), (ri, r)| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)));
            if let Some((ai, alt)) = best_alt {
                let tag = match a[ai].operation {
                    Operation::Train(t) => format!("{:?}训练", t),
                    Operation::Race => "比赛".to_string(),
                    Operation::Rest => "休息".to_string(),
                    Operation::NormalOuting => "普通出行".to_string(),
                    Operation::Clinic => "治病".to_string(),
                    _ => "其他".to_string()
                };
                out[c].reason = format!(
                    "{} | 路径={} 次优={}({:.0}) Δ={:+.0}",
                    out[c].reason,
                    path,
                    tag,
                    alt.score,
                    out[c].score - alt.score
                );
            }
        }
        Ok(c)
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
    /// 借用原局面与训练候选预演指定面，复用评分空间，不落地分身或消费随机流。
    fn preview_train(
        &self, g: &RamenGame, actions: &[RamenAction], ramen: Option<usize>, eval_cache: &mut LocalTrainCache,
        scores: &mut Vec<RamenPolicyOutput>
    ) -> Result<Operation> {
        let idx = self.decide_train_cached(g, actions, ramen, eval_cache, scores)?;
        actions
            .get(idx)
            .map(|a| a.operation)
            .ok_or_else(|| anyhow::anyhow!("预演训练决策索引越界: {idx}/{}", actions.len()))
    }

    /// "吃面后必训练 at_trains 覆盖位" 门控：该面落地后，最优训练位是否落在面的 at_trains 内
    ///
    /// 与体力评估复用同一次预演；非训练动作或未覆盖的训练位返回 `false`。
    fn eat_covered_train_passes(&self, operation: Operation, region_id: usize) -> Result<bool> {
        let region = RAMENDATA
            .get()
            .and_then(|d| d.ramen_region_effect.get(region_id))
            .ok_or_else(|| anyhow::anyhow!("地区效果缺失: {region_id}"))?;
        match operation {
            Operation::Train(tt) => {
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

    /// 第三年本回合训练后，低体力是否还会伤害下一次普通训练。
    ///
    /// turn=70 后紧接 turn=71 有马纪念（赛后 +40），再进入 turn=72 超级拉面（回合开始 +20），
    /// 所以没有待保护的普通训练回合；此时体力归零也是合理终盘控制。
    fn y3_collapse_matters(&self, g: &RamenGame) -> bool {
        !self.config.y3_recovery_horizon || g.turn() < 70
    }

    /// 从已有预演计算训练前后体力，返回 `(训练类型, 训练前体力, 训练后体力)`。
    fn post_ramen_vital_transition(
        &self, g: &RamenGame, operation: Operation, eval_cache: &LocalTrainCache
    ) -> Result<Option<(usize, i32, i32)>> {
        // 每年吃面决策都评估吃面后的体力（turn>=72 超级拉面回合不吃面，防御性返回 None）
        if g.turn() >= 72 {
            return Ok(None);
        }
        let Operation::Train(tt) = operation else {
            return Ok(None);
        };
        let train = tt as usize;
        let value = &eval_cache.training[train]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("预演训练缺少评估结果: {train}"))?
            .value;
        let before = g.uma.vital;
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
    /// 它只查看候选地区 at_trains 当前已有的真实训练窗口，不预测分身；同次选面共用各训练位的窗口分量。
    fn ramen_window_alignment(
        &self, g: &RamenGame, region_id: usize, window_cache: &mut [Option<f32>; 5]
    ) -> Result<f32> {
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
            if let Some(value) = window_cache[tr] {
                best = best.max(value);
                continue;
            }
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
            let value = raw * weak_mult + people * 8.0 + shining * 35.0;
            window_cache[tr] = Some(value);
            best = best.max(value);
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
        let recipe = get_recipe(region_id)?;
        let targets = min_special_targets(&g.ramen, recipe)
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
            let gross = out.score - out.train_fail_adj;
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
        let recipe = get_recipe(region_id)?;
        let targets = min_special_targets(&g.ramen, recipe)
            .ok_or_else(|| anyhow::anyhow!("拉面 {region_id} 无合法 targets"))?;
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
        let recipe = get_recipe(region_id)?;
        let targets = min_special_targets(&g.ramen, recipe)
            .ok_or_else(|| anyhow::anyhow!("拉面 {region_id} 无合法 targets"))?;
        let used = targets.iter().sum::<i32>() as f32;
        let before = g
            .ramen
            .selected_regions
            .iter()
            .filter(|&&rid| {
                get_recipe(rid)
                    .map(|recipe| min_special_targets(&g.ramen, recipe).is_some())
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
                get_recipe(rid)
                    .map(|recipe| min_special_targets(&post, recipe).is_some())
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
                    get_recipe(rid)
                        .map(|recipe| min_special_targets(&post, recipe).is_some())
                        .unwrap_or(false)
                })
                .count() as f32;
            let balance = post.feeling_stock.iter().map(|&x| (x as f32 + 2.0).sqrt()).sum::<f32>();
            let year_left = (Self::year_end(g) - g.turn()).max(0) as f32 / 21.0;
            let future = (craftable * 18.0 + balance * 4.0) * year_left;
            score.score += future;
            if self.policy.collect_details {
                score.add("future_craftability", future);
                score.reason = format!("隐藏方案{:?} 后续可做{}种", targets, craftable as i32);
            }
        }
        Ok((Self::choose(&out), out))
    }

    fn decide_ramen(&self, g: &RamenGame, a: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let (_, mut out) = self.policy.decide_ramen(g, a)?;
        // 仅限此次只切换当前面的预演；真实吃面落地或局面改变后必须使用新缓存。
        let mut eval_cache = LocalTrainCache::default();
        let train_actions = g.list_train_actions();
        let mut train_scores = Vec::new();
        let pre_action = self.preview_train(g, &train_actions, None, &mut eval_cache, &mut train_scores)?;
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
                    if self.policy.collect_details {
                        candidate.reason = "不吃面：本回合基础决策不是训练".to_string();
                    }
                } else {
                    candidate.score = f32::NEG_INFINITY;
                    if self.policy.collect_details {
                        candidate.reason = "禁止吃面：本回合应先休息/外出/治病/比赛".to_string();
                    }
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
        let mut window_cache = [None; 5];
        for (act, o) in a.iter().zip(out.iter_mut()) {
            if let Some(region_id) = act.ramen {
                let preview = self.preview_train(g, &train_actions, Some(region_id), &mut eval_cache, &mut train_scores)?;
                // 吃面后必训练 at_trains 覆盖位（C 方案简化约束）：预演该面落地后的最优训练位，
                // 若不在 at_trains 内则否决（吃面加成浪费——玩家 87% 覆盖 vs 自动 52%）。
                if self.config.eat_requires_covered_train
                    && !self.eat_covered_train_passes(preview, region_id)?
                {
                    o.score = f32::NEG_INFINITY;
                    if self.policy.collect_details {
                        o.reason = "禁止吃面：吃完后最优训练位不在该面 at_trains 内".to_string();
                        o.add("eat_covered_train_gate", f32::NEG_INFINITY);
                    }
                    continue;
                }
                if let Some((train, pre_vital, post_vital)) = self.post_ramen_vital_transition(g, preview, &eval_cache)? {
                    if train != 4
                        && self.config.y3_post_train_hard_floor > 0
                        && post_vital < self.config.y3_post_train_hard_floor
                    {
                        o.score = f32::NEG_INFINITY;
                        if self.policy.collect_details {
                            o.reason = format!(
                                "禁止吃面：{}训练体力{}→{}低于硬底线{}",
                                ["速", "耐", "力", "根", "智"][train],
                                pre_vital,
                                post_vital,
                                self.config.y3_post_train_hard_floor
                            );
                            o.add("y3_vital_hard_guard", f32::NEG_INFINITY);
                        }
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
                    if self.policy.collect_details {
                        o.add(
                            "y3_pre_vital_shortfall",
                            -pre_short * self.config.y3_vital_shortfall_weight
                        );
                        o.add(
                            "y3_post_vital_shortfall",
                            -post_short * self.config.y3_vital_shortfall_weight
                        );
                    }
                }
                let pressure = risk * self.config.overflow_value;
                o.score += pressure;
                if self.policy.collect_details {
                    o.add("local_stock_pressure", pressure);
                }
                let y = (g.current_year() - 1) as usize;
                let post = g.ramen.scenario_pt + calc_ramen_pt_gain(y, g.ramen.eat_count)?;
                let (ck, rmj, great) = self.scenario_threshold_value(g, post)?;
                let deadline = self.deadline_urgency(g, post)?;
                let window = self.ramen_window_alignment(g, region_id, &mut window_cache)?;
                let cook2_cost = self.cook2_ramen_stock_cost(g, region_id)?;
                let safety = if let Some((_, gain)) = bridge {
                    self.safety_bridge_candidate(g, region_id, gain)?
                } else {
                    0.0
                };
                let look = self.ramen_lookahead(g, region_id)?;
                o.score += ck + rmj + great + deadline + window + safety + look - cook2_cost + guarantee;
                if self.policy.collect_details {
                    o.add("scenario_checkpoint", ck);
                    o.add("rmj_cross", rmj);
                    o.add("great_cross", great);
                    o.add("deadline_urgency", deadline);
                    o.add("ramen_window", window);
                    o.add("cook2_stock_cost", -cook2_cost);
                    o.add("safety_bridge", safety);
                    o.add("ramen_lookahead", look);
                    o.add("eat_guarantee", guarantee);
                }
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
    /// 该场景下三份年策略的 `collect_details` 也已关闭、必然返回 `None`。
    record_last_year: bool
}

impl RecommendedRamenTrainer {
    /// 从正式 preset 精确复制，只覆盖专项矩阵明确列出的评分参数。
    ///
    /// 吃面事务门、体力硬门、友人 0/2/5 节奏、动态事件、隐藏风味等结构逻辑
    /// 均逐字继承 `new()`，防止实验候选混入未声明的策略差异。
    ///
    /// `region_weak_cover_weight` 走三态语义（见
    /// [`RamenPolicy::effective_region_weak_cover`]）：`0.0`=按智卡数查表（方案Ⅰ，
    /// 与 preset 默认一致）、`<0.0`=显式关闭（老行为）、`>0.0`=固定值（实验）。
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

    /// 地区打分权重覆盖（三年统一；`None` = 保持 preset 值）。
    ///
    /// 只覆盖地区选择相关权重，其余策略参数逐字继承 `new()`。
    /// `youqing_weight` 对应 [`RamenPolicyConfig::region_youqing_weight`]（卡组构成×友情词条），
    /// `waste_penalty` 对应 [`RamenPolicyConfig::region_waste_penalty`]（覆盖无卡位惩罚），
    /// `weak_cover_weight` 三态语义同 `with_experiment_overrides` 的第 10 参数
    /// （`None` = preset 默认按智卡数查表），
    /// `main_bias_bonus` 对应 [`RamenPolicyConfig::region_main_bias_bonus`]（C2 主训位翻倍）。
    pub fn with_region_weights(
        mut self,
        youqing_weight: Option<f32>,
        waste_penalty: Option<f32>,
        weak_cover_weight: Option<f32>,
        main_bias_bonus: Option<f32>,
    ) -> Self {
        for year in self.years.iter_mut() {
            if let Some(v) = youqing_weight {
                year.policy.config.region_youqing_weight = v;
            }
            if let Some(v) = waste_penalty {
                year.policy.config.region_waste_penalty = v;
            }
            if let Some(v) = weak_cover_weight {
                year.policy.config.region_weak_cover_weight = v;
            }
            if let Some(v) = main_bias_bonus {
                year.policy.config.region_main_bias_bonus = v;
            }
        }
        self
    }

    /// 第 3 年地区选择"单点偏好"强度（实验扫参，见
    /// [`RamenPolicyConfig::region_y3_single_focus`]）。
    ///
    /// 三年统一写入该字段，但字段只在 `year_idx == 2`（第 3 年）被消费，
    /// 第 1/2 年行为不受影响——配对实验中其余年份两臂逐位等价，Δ 纯归因第 3 年。
    /// `0` = 现状（默认），`1..=3` = 组合内至少含 1..=3 个单点地区。
    pub fn with_region_y3_single_focus(mut self, focus: u8) -> Self {
        for year in self.years.iter_mut() {
            year.policy.config.region_y3_single_focus = focus;
        }
        self
    }

    /// EXP-006c：从 token 串构造 preset 变体（逐 token 覆盖三年同配置）。
    ///
    /// - `wisfN`：智力训练体力豁免下限 = N（见 [`RamenPolicyConfig::wisdom_vital_floor`]）
    /// - `capdN`：副属性残余收益折扣 = N/100（[`RamenPolicyConfig::cap_discount_weight`]）
    /// - `ckN`：剧本 PT 档位前瞻倍率 = N/100（[`LocalRamenConfig::checkpoint_scale`]；preset 现值 0=关闭）
    /// - `cookN`：诀窍边际库存权重 = N（[`LocalRamenConfig::cook2_stock_weight`]；调高=材料更保守）
    /// - `g1N/g2N/g3N`：第 1/2/3 年短板追赶强度 = N/100（分年动态属性平衡）
    /// - `o1N/o2N/o3N`：第 1/2/3 年近上限衰减强度 = N/100
    /// - `poN`：power 近上限衰减统一覆盖 = N/100（EXP-006e）；`p1o/p2o/p3oN` 分年
    /// - `pg[m]N`：power 短板追赶覆盖 = ±N/100（m 前缀=负号，'-' 是 token 分隔符不能用；EXP-006e）
    /// - `base`：无覆盖（对照）
    /// - `supermodeN`：0默认二，1固定一，2固定三，3按终盘缺口和卡数选范围。
    /// - `ptblendN`：近上限 PT 连续定价窗口 N/100 次训练，0 关闭（实验）。
    /// - `hintlvN`：逐卡 Hint 精确估值倍率 = N/100，0 关闭（实验；见 [`LocalRamenConfig::hint_card_aware`]）。
    ///
    /// 未识别 token 直接报错，防止实验名拼错静默跑成 base。
    /// 设置"友人出行必须走完 5 次"的完成硬门限（对应 `game_config.toml` 的
    /// `friend_complete_required`）。返回 `self` 便于链式构造。
    pub fn with_friend_complete_required(mut self, required: bool) -> Self {
        for year in self.years.iter_mut() {
            year.config.friend_complete_required = required;
        }
        self
    }

    pub fn with_tokens(tokens: &str) -> Result<Self> {
        let mut trainer = Self::new();
        for token in tokens.split('-') {
            if token == "base" {
                continue;
            } else if let Some(v) = token.strip_prefix("supermode") {
                let mode: u8 = v.parse()?;
                anyhow::ensure!(mode <= 3, "supermode 需要0..3");
                for year in trainer.years.iter_mut() { year.policy.config.super_choice_mode = mode; }
            } else if let Some(v) = token.strip_prefix("ptblend") {
                let turns: f32 = v.parse::<f32>()? / 100.0;
                anyhow::ensure!(turns.is_finite() && (0.0..=10.0).contains(&turns), "ptblend 必须在 0..=1000");
                for year in trainer.years.iter_mut() {
                    year.policy.config.pt_cap_blend_turns = turns;
                }
            } else if let Some(v) = token.strip_prefix("wisf") {
                let floor: i32 = v.parse()?;
                for year in trainer.years.iter_mut() {
                    year.policy.config.wisdom_vital_floor = floor;
                }
            } else if let Some(v) = token.strip_prefix("capd") {
                let weight: f32 = v.parse::<f32>()? / 100.0;
                for year in trainer.years.iter_mut() {
                    year.policy.config.cap_discount_weight = weight;
                }
            } else if let Some(v) = token.strip_prefix("trdsh") {
                // 已满位训练有彩圈 PT 定价（N/100，见 RamenPolicyConfig::pt_tradeoff_shining）
                let f: f32 = v.parse::<f32>()? / 100.0;
                for year in trainer.years.iter_mut() {
                    year.policy.config.pt_tradeoff_shining = f;
                }
            } else if let Some(v) = token.strip_prefix("trds") {
                // 超级拉面回合（72-77）已满位 PT 定价（N/100，见 pt_tradeoff_super）
                let f: f32 = v.parse::<f32>()? / 100.0;
                for year in trainer.years.iter_mut() {
                    year.policy.config.pt_tradeoff_super = f;
                }
            } else if let Some(v) = token.strip_prefix("trd") {
                // 已满位训练普通档 PT 定价（N/100，见 RamenPolicyConfig::pt_tradeoff）
                let f: f32 = v.parse::<f32>()? / 100.0;
                for year in trainer.years.iter_mut() {
                    year.policy.config.pt_tradeoff = f;
                }
            } else if let Some(v) = token.strip_prefix("ck") {
                let scale: f32 = v.parse::<f32>()? / 100.0;
                for year in trainer.years.iter_mut() {
                    year.config.checkpoint_scale = scale;
                }
            } else if let Some(v) = token.strip_prefix("cook") {
                let w: f32 = v.parse()?;
                for year in trainer.years.iter_mut() {
                    year.config.cook2_stock_weight = w;
                }
            } else if let Some(v) = token.strip_prefix("g1") {
                let s: f32 = v.parse::<f32>()? / 100.0;
                trainer.years[0].config.status_gap_strength = s;
            } else if let Some(v) = token.strip_prefix("g2") {
                let s: f32 = v.parse::<f32>()? / 100.0;
                trainer.years[1].config.status_gap_strength = s;
            } else if let Some(v) = token.strip_prefix("g3") {
                let s: f32 = v.parse::<f32>()? / 100.0;
                trainer.years[2].config.status_gap_strength = s;
            } else if let Some(v) = token.strip_prefix("o1") {
                let s: f32 = v.parse::<f32>()? / 100.0;
                trainer.years[0].config.status_overflow_strength = s;
            } else if let Some(v) = token.strip_prefix("o2") {
                let s: f32 = v.parse::<f32>()? / 100.0;
                trainer.years[1].config.status_overflow_strength = s;
            } else if let Some(v) = token.strip_prefix("o3") {
                let s: f32 = v.parse::<f32>()? / 100.0;
                trainer.years[2].config.status_overflow_strength = s;
            } else if let Some(v) = token.strip_prefix("po") {
                // EXP-006e：power 近上限衰减，全部年统一覆盖（po150 → 1.50）
                let s: f32 = v.parse::<f32>()? / 100.0;
                for year in trainer.years.iter_mut() {
                    year.config.power_overflow_strength = s;
                }
            } else if let Some(v) = token.strip_prefix("p1o") {
                trainer.years[0].config.power_overflow_strength = v.parse::<f32>()? / 100.0;
            } else if let Some(v) = token.strip_prefix("p2o") {
                trainer.years[1].config.power_overflow_strength = v.parse::<f32>()? / 100.0;
            } else if let Some(v) = token.strip_prefix("p3o") {
                trainer.years[2].config.power_overflow_strength = v.parse::<f32>()? / 100.0;
            } else if let Some(v) = token.strip_prefix("pg") {
                // EXP-006e：power 短板追赶覆盖。负值用 'm' 前缀编码（'-' 是 token
                // 分隔符）：pgm30 → −0.30，pg30 → +0.30。
                let (sign, digits) = match v.strip_prefix('m') {
                    Some(d) => (-1.0f32, d),
                    None => (1.0f32, v)
                };
                let s: f32 = sign * digits.parse::<f32>()? / 100.0;
                for year in trainer.years.iter_mut() {
                    year.config.power_gap_strength = s;
                }
            } else if let Some(v) = token.strip_prefix("reserve") {
                // 预留上限空间（调低=减轻终盘已满位惩罚；C 修复入口）
                let max: f32 = v.parse()?;
                for year in trainer.years.iter_mut() {
                    year.config.status_reserve_max = max;
                }
            } else if let Some(v) = token.strip_prefix("rgn") {
                // reserve 增益口径：0=原始（基准） / 1=A截断 / 2=B满位豁免
                let mode: u8 = v.parse()?;
                if mode > 2 {
                    anyhow::bail!("rgn 仅支持 0/1/2: {v}");
                }
                for year in trainer.years.iter_mut() {
                    year.config.reserve_gain_mode = mode;
                }
            } else if let Some(v) = token.strip_prefix("fcap") {
                // 友人出行累计配额（三位数按年编码，如 fcap135 → [1, 3, 5]）
                anyhow::ensure!(v.len() == 3, "fcap 必须是三个数字，如 135: {v}");
                let mut caps = [0usize; 3];
                for (i, ch) in v.chars().enumerate() {
                    caps[i] = ch
                        .to_digit(10)
                        .ok_or_else(|| anyhow::anyhow!("fcap 含非数字字符: {v}"))?
                        as usize;
                }
                anyhow::ensure!(
                    caps.windows(2).all(|w| w[0] <= w[1]) && caps[2] <= 5,
                    "fcap 必须单调且不超过5: {v}"
                );
                for year in trainer.years.iter_mut() {
                    year.config.friend_outing_cumulative_caps = caps;
                }
            } else if let Some(v) = token.strip_prefix("fov3") {
                // 第三年替代休息路径的隐藏风味上限（fov33 → 3；fov34 → 4 基本关闸）
                let cap: i32 = v.parse()?;
                for year in trainer.years.iter_mut() {
                    year.config.friend_y3_overflow_cap = cap;
                }
            } else if token == "freq" {
                // 友人完成硬门限（默认由 config 决定；此 token 显式打开）
                for year in trainer.years.iter_mut() {
                    year.config.friend_complete_required = true;
                }
            } else if token == "freqoff" {
                // 显式关闭完成硬门限（对照实验）
                for year in trainer.years.iter_mut() {
                    year.config.friend_complete_required = false;
                }
            } else if let Some(v) = token.strip_prefix("frem3") {
                // 第三年必须走完的次数（frem33 → 必须走 3 次；配合 furg3 使用）
                let n: i32 = v.parse()?;
                for year in trainer.years.iter_mut() {
                    year.config.friend_y3_force_remaining = n;
                }
            } else if token == "furg3" {
                // 第三年配额紧迫度强制补足（配合 fcap/frem3 使用，如 fcap025-furg3）
                for year in trainer.years.iter_mut() {
                    year.config.friend_y3_urgency_force = true;
                }
            } else if let Some(v) = token.strip_prefix("hintlv") {
                let weight = v.parse::<f32>()? / 100.0;
                anyhow::ensure!(weight.is_finite() && (0.0..=10.0).contains(&weight), "hintlv 必须在 0..=1000");
                for year in trainer.years.iter_mut() {
                    year.config.hint_card_aware = weight;
                }
            } else {
                anyhow::bail!("未知 token: {token}（完整: {tokens}）");
            }
        }
        Ok(trainer)
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
            // 副属性残余收益折扣（方案 E）：0 = 主属性接近上限时不再给副属性/PT 打折。
            // 0.0 为第十二轮配对验收的组合值（token capd0）。
            policy.cap_discount_weight = 0.0;

            let mut local = LocalRamenConfig::default();
            // 预留上限空间：40 → 157（第十二轮配对验收组合值，token reserve157）。
            local.status_reserve_max = 157.0;
            // 预留惩罚按实际可增长量计算，已满位不再虚增透支（token rgn1）。
            local.reserve_gain_mode = 1;
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
            // 短板追赶 / 近上限衰减强度：0.5 / 0.5 → 4.98 / 3.04（第十二轮组合值，
            // 对应 token g1498-g2498-g3498-o1304-o2304-o3304）。
            local.status_gap_strength = 4.98;
            local.status_overflow_strength = 3.04;
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
            // 友人出行跨年配额定档 [0,3,5]（2026-09-21 用户拍板）：第 1 年不启用
            // （第一年出行实测在葛城王牌上硬亏 -628，t=-8.6）、第 2 年放宽到 3
            // 以消化"提前的休息替代"、第 3 年补满 5。实测：第 2 年配额 2→3 两马娘
            // 均在噪声内（+44 / -51），用满率 77%→80%；第 1 年开配额为纯亏。
            local.friend_outing_cumulative_caps = [0, 3, 5];
            local.friend_rest_max_special = 4;
            local.deadline_urgency_scale = 0.0;
            local.dynamic_special_targets = true;
            // ===== GA 方向定稿（2026-09-17 ga_lab 合并，9 旋钮组合档）=====
            // 来源：ga_lab fork (479fb38) 跨轮一致 GA 方向；本地 CRN 配对验证
            // （4 马 × 2 种子块 = 420 局配对，组合档 Δ=+1394 t=12.55，8/8 单元显著）。
            // 注意：单项均≤0/惰性，收益来自组合交互；weakboost(ramen_weak_train_boost)
            // 单独 -1017 且拖累组合 → 明确不采纳。
            // ===== 第十二轮配对验收（2026-09-18）=====
            // 两个独立随机卡组池（各 160 副全新卡组 × 300 局，同卡组同种子配对）上，本组合相对
            // 上一版推荐组合（ptblend200-capd0-rgn1-supermode3-hintlv600）随机组 +130.4 [+93.6,+167.2]
            // 与 +147.6 [+108.2,+187.0]；相对本文件改动前的默认 preset 累计 +517 / +552（同批配对均值可加）。
            // 消融、验收池说明与限制见 experiments/validated_policy/README.md。
            policy.pt_tradeoff = 44.25;             // 满位普通档 16→37（GA）→44.25（第十二轮）
            policy.pt_tradeoff_super = 34.5;        // 超拉面回合 0→35（GA）→34.5（第十二轮）
            policy.pt_tradeoff_shining = 25.0;      // 已满位有彩圈 36→25（第十二轮）
            policy.pt_cap_blend_turns = 8.0;        // 近上限连续 PT 定价窗口 = 8.00 次训练（第十二轮）
            policy.super_choice_mode = 3;           // 超级拉面按终盘属性缺口与卡型数选范围（第十二轮）
            policy.outing_base = 0.0;               // 外出基准分 15→0（第十一轮消融 +15.2 [+12.1,+18.3]）
            local.hint_card_aware = 6.0;            // 逐卡 Hint 倍率 0→6.00（沿用已发布档 hintlv600，第十二轮复核 675→600 零损失）
            policy.region_weak_cover_weight = 35.0; // 弱位覆盖 查表→35（GA 97%；>0 直值）
            policy.region_youqing_weight = 0.4;     // 友情词条 1.5→0.4（GA top 100% 降）
            local.hint_bonus = 8.0;                 // 掌握度 6→8
            local.max_base_score_sacrifice = 200.0; // 140→200
            local.ramen_window_weight = 0.15;       // 0.10→0.15
            local.checkpoint_scale = 0.15;          // 剧本PT前瞻 0→0.15
            LocalRamenTrainer::with_configs(policy, local)
        }

        Self {
            // 回合级体力门限：不吃面回合统一 40（base_seed=61444 配对 100 局扫描峰值，
            // 30→40 总加权 +318；45 回落——门限过高休息过多）；吃面回合仅第三年放掉
            // （Y3 fail_rate_drop=100% 必成），第一/二年保留 40（Y1/Y2 吃面训练仍可能失败）。
            years: [make(56.0, 40, 40), make(64.0, 40, 40), make(64.0, 40, 0)],
            last_year: Mutex::new(None),
            record_last_year: true
        }
    }

    /// 创建 rollout 专用实例，关闭评分分解、原因字符串、日志文本和 `last_year` 记录。
    ///
    /// rollout 不消费这些观测数据；决策评分和训练失败损失保留。
    /// `recommended_for_rollout_decisions_identical` 核对整局动作和事件记录一致。
    pub fn for_rollout() -> Self {
        let mut trainer = Self::new();
        for year in trainer.years.iter_mut() {
            year.policy.collect_details = false;
        }
        // 年份转发记录同样只供日志文本读取。
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

    /// 复用同年手写地区策略的候选打分（供 MCTS 地区候选预过滤使用同一先验）。
    ///
    /// 返回 `(手写 argmax 下标, 与 `actions` 严格同长同序的打分)`；`.score` 即
    /// `RamenPolicy::decide_region` 的地区分（第 3 年含 `region_y3_single_focus`
    /// 否决：不合格候选记 0）。调用点与手写路径一致：turn 2/23/47 → 年 idx 0/1/2。
    pub fn region_prior(
        &self, game: &RamenGame, actions: &[RamenAction]
    ) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let year_idx = match game.turn() {
            2 => 0,
            23 => 1,
            47 => 2,
            _ => Self::year(game)
        };
        self.years[year_idx].policy.decide_region(game, year_idx, actions)
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
            if self.policy.collect_details {
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

    use super::{LocalRamenConfig, LocalRamenTrainer, LocalTrainCache, RamenPolicyConfig, RecommendedRamenTrainer};

    /// 第1年地区选择（turn 2 在 run_begin 内联触发、stage=Begin）必须走 decide_region 打分。
    ///
    /// 回归：LocalRamenTrainer::select_action 只按 `g.stage` 分派时，第1年地区选择
    /// 会落入默认分支恒选候选 0（详见 ramen_handwritten_trainer.rs 的 ramen_effective_stage 注释）。
    #[test]
    #[allow(clippy::panic)]
    fn recommended_region_select_year1_runs_policy() -> Result<()> {
        use crate::utils::Checks;
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
        let rollout = RecommendedRamenTrainer::for_rollout();
        let rollout_idx = rollout.select_action(&game, &actions, &mut rng)?;
        let mut c = Checks::new();
        c.check(idx == rollout_idx, "普通实例与 rollout 的第1年地区选择一致");
        c.check(matches!(trainer.last_year.lock().as_deref(), Ok(Some(0))), "普通实例记录本次决策年份");
        c.check(matches!(rollout.last_year.lock().as_deref(), Ok(None)), "rollout 跳过决策年份记录");
        c.check(rollout.last_breakdown().is_none(), "rollout 不采集原因日志");
        c.finish()
    }

    /// `with_region_y3_single_focus` 把强度写入三年（字段只在第 3 年被消费），
    /// 且经完整 `select_action` 路径后第 3 年选中组合确实全为单点地区、
    /// 与默认档（focus=0）选区不同。
    #[test]
    #[allow(clippy::panic)]
    fn recommended_region_y3_single_focus_end_to_end() -> Result<()> {
        use crate::{
            gamedata::{init_global, ramen::RAMENDATA},
            utils::{Checks, get_workspace_root, init_test_logger}
        };
        use crate::game::{
            InheritInfo,
            ramen::{Operation, RamenAction, RamenGame, rules::get_region_combinations}
        };
        use rand::{SeedableRng, prelude::StdRng};

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let data = RAMENDATA.get().expect("init_global 后 RAMENDATA 已装载");
        let is_single = |rid: usize| data.ramen_region_effect[rid].at_trains.len() == 1;

        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let inherit = InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] };
        let actions: Vec<RamenAction> = get_region_combinations(2)?
            .iter()
            .map(|&c| RamenAction::no_ramen(Operation::RegionSelect(c)))
            .collect();

        let mut game = RamenGame::newgame(102601, &deck, inherit)?;
        game.base.turn = 47; // 第 3 年地区选择
        let mut rng = StdRng::seed_from_u64(42);

        let trainer = RecommendedRamenTrainer::new();
        let idx0 = trainer.select_action(&game, &actions, &mut rng)?;
        let f3 = RecommendedRamenTrainer::new().with_region_y3_single_focus(3);
        let idx3 = f3.select_action(&game, &actions, &mut rng)?;
        println!("Y3 选区: focus=0 → {:?} / focus=3 → {:?}", actions[idx0].operation, actions[idx3].operation);

        let combo3 = match actions[idx3].operation {
            Operation::RegionSelect(c) => c,
            _ => return Err(anyhow::anyhow!("focus=3 选中不是 RegionSelect"))
        };
        let combo0 = match actions[idx0].operation {
            Operation::RegionSelect(c) => c,
            _ => return Err(anyhow::anyhow!("focus=0 选中不是 RegionSelect"))
        };
        let mut c = Checks::new();
        c.check(
            combo3.iter().all(|&rid| is_single(rid)),
            "focus=3 经完整 select_action 选中组合全为单点地区"
        );
        c.check(combo0 != combo3, "focus=3 与 focus=0 选区不同");
        c.check(
            trainer.years.iter().all(|y| y.policy.config.region_y3_single_focus == 0)
                && f3.years.iter().all(|y| y.policy.config.region_y3_single_focus == 3),
            "with_region_y3_single_focus 三年统一写入"
        );
        c.finish()
    }

    /// 单候选决策点必须记录「仅1候选」breakdown；rollout 关闭原因文本采集。
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

    /// 普通实例与 rollout 的整局动作、事件和终局数值一致；仅普通实例提供原因日志。
    #[test]
    fn recommended_for_rollout_decisions_identical() -> Result<()> {
        use crate::{
            bench::seeded_rngs,
            game::{InheritInfo, ramen::RamenGame, traits::Game},
            gamedata::init_global,
            output::decision_log::DecisionLog,
            trainer::LoggingTrainer,
            utils::{Checks, get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        let inherit =
            InheritInfo { blue_count: [15, 0, 0, 0, 3], extra_count: [10, 10, 20, 20, 20, 40] };

        /// 一局跑完后的终局数值与完整决策日志。
        struct Observed {
            score: i32,
            five: [i32; 5],
            skill_pt: i32,
            log: DecisionLog
        }

        // 同一 base_seed / run_idx ⇒ 决策 RNG 与规则 RNG 都逐位相同
        let run = |rollout: bool| -> Result<Observed> {
            let (mut rng, rule_master) = seeded_rngs(61444, 0);
            let mut game = RamenGame::newgame(102601, &DECK, inherit.clone())?;
            game.set_rule_master(rule_master);
            let trainer = if rollout {
                RecommendedRamenTrainer::for_rollout()
            } else {
                RecommendedRamenTrainer::new()
            };
            let logged = LoggingTrainer::new(trainer, 61444);
            game.run_full_game(&logged, &mut rng)?;
            Ok(Observed {
                score: game.uma.calc_score(),
                five: game.uma.five_status,
                skill_pt: game.uma.skill_pt,
                log: logged.take_records()
            })
        };

        let mut n = run(false)?;
        let mut r = run(true)?;
        println!(
            "new():         评分={} 五维={:?} PT={} 决策记录={}",
            n.score, n.five, n.skill_pt, n.log.rows.len()
        );
        println!(
            "for_rollout(): 评分={} 五维={:?} PT={} 决策记录={}",
            r.score, r.five, r.skill_pt, r.log.rows.len()
        );

        let mut c = Checks::new();
        c.check(n.score == r.score, "整局评分逐位相同");
        c.check(n.five == r.five, "整局五维逐位相同");
        c.check(n.skill_pt == r.skill_pt, "整局技能点逐位相同");
        c.check(!n.log.rows.is_empty() && !r.log.rows.is_empty(), "两种实例均记录完整决策轨迹");
        c.check(
            n.log.rows.iter().filter(|row| row.stage != "Event").all(|row| {
                row.score_breakdown.as_deref().is_some_and(|text| !text.is_empty())
            }),
            "普通实例的每次动作决策均向 LoggingTrainer 提供原因文本"
        );
        c.check(r.log.rows.iter().all(|row| row.score_breakdown.is_none()), "rollout 的全部决策日志均不含原因文本");
        for row in n.log.rows.iter_mut().chain(&mut r.log.rows) {
            row.elapsed_us = 0;
            row.score_breakdown = None;
        }
        c.check(n.log == r.log, "排除耗时和原因文本后，整局动作及事件记录完全一致");
        c.finish()
    }

    /// 正式 preset 必须使用定档的友人跨年节奏 [0,3,5]（2026-09-21 拍板替换旧 [0,2,5]）。
    #[test]
    #[allow(clippy::panic)]
    fn recommended_ramen_uses_035_friend_pacing() {
        let trainer = RecommendedRamenTrainer::new();
        let actual = trainer
            .years
            .each_ref()
            .map(|year| year.config.friend_outing_cumulative_caps);
        let expected = [[0, 3, 5]; 3];
        println!("正式友人累计出门配额: {actual:?}");
        if actual != expected {
            panic!("正式 preset 应使用 {expected:?}，实际为 {actual:?}");
        }
    }

    /// 吃面-训练联动：当前吃面覆盖速位时，速训练候选获得显式 `ramen_train_coupling` 加分，
    /// 非覆盖位不加。`calc_training_value` 的隐含加成之外，策略应倾向兑现吃面成本。
    /// 候选重排、重复训练和纯非训练列表均遵守基础分牺牲上限。
    #[test]
    #[allow(clippy::panic)]
    fn train_coupling_bonus_on_eating() -> Result<()> {
        use crate::{
            game::{
                InheritInfo,
                ramen::{Operation, RamenAction, RamenGame, RamenStage, TrainingType}
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
        let mut trainer = LocalRamenTrainer::with_configs(RamenPolicyConfig::default(), local);
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
                    .find(|(k, _)| *k == "ramen_train_coupling")
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
        preview.uma.five_status[0] = preview.uma.five_status_limit[0];
        trainer.config.ramen_train_coupling_weight = 10.0;
        let mut reordered = actions;
        reordered.reverse();
        reordered.push(RamenAction::new(Operation::Train(TrainingType::Speed)));
        let non_train = [Operation::Rest, Operation::NormalOuting, Operation::Clinic].map(RamenAction::new);
        let mut checks = crate::utils::Checks::new();
        for candidates in [reordered.as_slice(), non_train.as_slice()] {
            let mut base = trainer.policy.score_train_actions(&preview, candidates)?;
            if let Some(i) = candidates.iter().position(|action| action.operation == Operation::FriendOuting) {
                base[i].score = trainer.dynamic_friend_outing_value(&preview)?.0;
            }
            let base_best = LocalRamenTrainer::choose(&base);
            for limit in [0.0, 140.0, 100_000.0] {
                trainer.config.max_base_score_sacrifice = limit;
                let (chosen, adjusted) = trainer.decide_train(&preview, candidates)?;
                let local_best = LocalRamenTrainer::choose(&adjusted);
                let sacrifice = base[base_best].score - base[local_best].score;
                let expected = if sacrifice <= limit { local_best } else { base_best };
                println!(
                    "候选{} 牺牲上限{limit} 基础最优{base_best} 调整后最优{local_best} 牺牲{sacrifice} 选择{chosen}",
                    candidates.len()
                );
                checks.check(chosen == expected, "候选顺序与重复训练不改变基础分牺牲约束");
            }
        }
        checks.finish()
    }

    /// 友人事件按价值取最高项，平局取首项，并保留全负收益和空候选的结果。
    #[test]
    fn friend_event_choice_values_and_ties() -> Result<()> {
        use std::env::set_current_dir;

        use crate::{
            game::{InheritInfo, ramen::RamenGame},
            gamedata::{ActionValue, EventChoice, init_global},
            utils::{Checks, get_workspace_root}
        };

        set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let trainer = LocalRamenTrainer::new();
        let game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        let mut c = Checks::new();
        let choices = [0, 10, 10].map(|friendship| vec![EventChoice {
            value: ActionValue { friendship, ..Default::default() },
            ..Default::default()
        }]);
        let best = trainer.dynamic_friend_event_choice(&game, &choices)?;
        println!("友人事件正收益及平局: {best:?}");
        c.check(best == (1, 50.0), "选择最高收益，平局保留首项");

        let losses = [-10, -5].map(|friendship| vec![EventChoice {
            value: ActionValue { friendship, ..Default::default() },
            ..Default::default()
        }]);
        let best = trainer.dynamic_friend_event_choice(&game, &losses)?;
        println!("友人事件全负收益: {best:?}");
        c.check(best == (1, -25.0), "全负收益时选择损失较小项并保留负分");
        c.check(trainer.dynamic_friend_event_choice(&game, &[])? == (0, 0.0), "空候选返回 (0, 0)");
        c.finish()
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
            .find(|(k, _)| *k == "friend_hidden_starve")
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        println!("special=0 turn=30: starve={starve0} total={total0:.0}");

        // 防溢出：turn=23（turn 24 夏合宿 +2），special=2 → 缺口 2 被未来发放 2 扣除 → starve=0
        game.base.turn = 23;
        game.ramen.special_feeling = 2;
        let (total1, bd1, _) = trainer.dynamic_friend_outing_value(&game)?;
        let starve1 = bd1
            .iter()
            .find(|(k, _)| *k == "friend_hidden_starve")
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
    /// 4. 复用预演与独立副本结果一致，覆盖与体力评估均不改变原局面
    #[test]
    #[allow(clippy::panic)]
    fn eat_covered_train_gate_blocks_mismatched_ramen() -> Result<()> {
        use crate::{
            game::{
                FriendOutState,
                InheritInfo,
                PersonType,
                ramen::{Operation, RamenAction, RamenGame, RamenStage, action::list_ramen_select_actions}
            },
            gamedata::init_global,
            utils::{Checks, get_workspace_root, init_test_logger}
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
        game.stage = RamenStage::RamenSelect;
        game.ramen.pending_ramen = Some(0);
        let train_actions = game.list_train_actions();
        let mut train_scores = Vec::new();

        // 局面 A：速低有空间 + 智满 → 最优训练必非智 → 智面 (id 4) 应被门控拒绝
        // 「满」一律从实际上限取，不写字面量：上限 = 剧本基值 + 继承，会随剧本数据与
        // 蓝因子变化，写死数字会让夹具在上限变动后静默失去「满」的语义。
        game.uma.five_status = [600, 1000, 1000, 1000, game.uma.five_status_limit[4]];
        let mut eval_cache = LocalTrainCache::default();
        let preview = on.preview_train(&game, &train_actions, Some(4), &mut eval_cache, &mut train_scores)?;
        let pass_rid4 = on.eat_covered_train_passes(preview, 4)?;
        println!("局面A(智满): 智面通过={pass_rid4}");
        if pass_rid4 {
            panic!("智已满且最优训练非智时，智面 (id 4) 应被 eat_covered_train_passes 拒绝");
        }

        // 局面 B：其他位全满 + 智低 → 最优训练必是智 → 智面 (id 4) 应通过、速面 (id 0) 拒绝
        // 打印落地面后的候选分布确认最优位
        game.uma.five_status = game.uma.five_status_limit;
        game.uma.five_status[4] = 600;
        let original = game.clone();
        {
            let mut preview = game.clone();
            preview.stage = RamenStage::Train;
            preview.ramen.current_ramen = Some(4);
            preview.ramen.clear_pending();
            let acts = preview.list_actions()?;
            let (idx, outs) = on.decide_train(&preview, &acts)?;
            println!("局面B 智面落地最优: {:?} score={:.1}", acts[idx].operation, outs[idx].score);
        }
        let mut eval_cache = LocalTrainCache::default();
        let preview = on.preview_train(&game, &train_actions, Some(4), &mut eval_cache, &mut train_scores)?;
        let pass_rid4_b = on.eat_covered_train_passes(preview, 4)?;
        let vital = on.post_ramen_vital_transition(&game, preview, &eval_cache)?;
        println!("局面B 智面预演体力: {vital:?}");
        let mut checks = Checks::new();
        checks.check(
            matches!(vital, Some((4, before, after)) if before == original.uma.vital && after > before),
            "智面覆盖与体力评估使用同一智力训练，训练前体力来自原局面且训练后恢复"
        );
        let preview = on.preview_train(&game, &train_actions, Some(0), &mut eval_cache, &mut train_scores)?;
        let pass_rid0_b = on.eat_covered_train_passes(preview, 0)?;
        println!("局面B(智低): 智面通过={pass_rid4_b} 速面通过={pass_rid0_b}");
        if !pass_rid4_b {
            panic!("其他位全满、只剩智位有空间时，覆盖智位的面 (id 4) 应通过门控");
        }
        if pass_rid0_b {
            panic!("其他位全满、最优训练为智时，不覆盖智位的面 (id 0 速) 应被门控拒绝");
        }
        let mut eval_cache = LocalTrainCache::default();
        for ramen in [None, Some(4), Some(0), None, Some(4)] {
            let mut fresh = game.clone();
            fresh.stage = RamenStage::Train;
            fresh.ramen.current_ramen = ramen;
            fresh.ramen.clear_pending();
            let actions = fresh.list_actions()?;
            let (fresh_chosen, fresh_scores) = on.decide_train(&fresh, &actions)?;
            let fresh_operation = actions[fresh_chosen].operation;
            let reused = on.preview_train(&game, &train_actions, ramen, &mut eval_cache, &mut train_scores)?;
            println!("预演 {ramen:?}: 借用={reused:?} 独立={fresh_operation:?}");
            checks.check(train_actions == actions, "训练候选不受原阶段或 pending 面影响");
            checks.check(reused == fresh_operation, "借用原局面的训练决策与独立副本一致");
            checks.check(train_scores == fresh_scores, "逐碗复用基础值后完整评分与独立计算一致");
            checks.check(
                train_scores.iter().zip(&fresh_scores).all(|(a, b)| a.score.to_bits() == b.score.to_bits()),
                "逐碗最终评分的浮点位模式一致"
            );
            let fresh_vital = if let Operation::Train(tt) = fresh_operation {
                let train = tt as usize;
                let buffs = fresh.calc_training_buff(train)?;
                let value = fresh.calc_training_value(&buffs, train)?;
                Some((train, fresh.uma.vital, fresh.uma.vital + value.vital))
            } else {
                None
            };
            checks.check(
                on.post_ramen_vital_transition(&game, reused, &eval_cache)? == fresh_vital,
                "复用已选训练值后的体力变化一致"
            );
        }
        let actions = list_ramen_select_actions(&game.ramen, &game.ramen.selected_regions);
        let (_, outputs) = on.decide_ramen(&game, &actions)?;
        checks.check(
            actions.iter().zip(&outputs).any(|(action, output)| action.ramen == Some(0) && output.score == f32::NEG_INFINITY),
            "速面被覆盖门控拒绝"
        );
        checks.check(
            actions.iter().zip(&outputs).any(|(action, output)| action.ramen == Some(4) && output.score.is_finite()),
            "前面的候选被拒绝后，后续智面仍完成评分"
        );
        checks.check(game == original, "覆盖与体力预演不改变原局面");

        // 第三年非合宿、多 Hint 与友人已解锁的局面，当前面覆盖位会切换长期价值的 Hint 模式。
        game.base.turn = 64;
        game.stage = RamenStage::Train;
        game.uma.motivation = 5;
        game.friend.out_state = FriendOutState::AfterUnlock;
        game.base.distribution = vec![vec![0, 1], vec![2], vec![3], vec![4], Vec::new()];
        for person in &mut game.persons {
            if person.person_type == PersonType::Card {
                person.friendship = 40;
                person.is_hint = true;
            }
        }
        for card in &mut game.base.deck {
            card.friendship = 40;
        }
        let mut local = on.config.clone();
        local.probabilistic_hint = true;
        for collect_details in [true, false] {
            let mut trainer = LocalRamenTrainer::with_configs(on.policy.config.clone(), local.clone());
            trainer.policy.collect_details = collect_details;
            let mut cache = LocalTrainCache::default();
            let actions = game.list_train_actions();
            let mut scores = Vec::new();
            for ramen in [None, Some(15), Some(17), None, Some(15)] {
                let operation = trainer.preview_train(&game, &actions, ramen, &mut cache, &mut scores)?;
                let mut fresh = game.clone();
                fresh.stage = RamenStage::Train;
                fresh.ramen.current_ramen = ramen;
                fresh.ramen.clear_pending();
                let fresh_actions = fresh.list_actions()?;
                let (fresh_chosen, fresh_scores) = trainer.decide_train(&fresh, &fresh_actions)?;
                checks.check(
                    actions == fresh_actions && operation == fresh_actions[fresh_chosen].operation && scores == fresh_scores,
                    "Local 借用预演的动作、数值分解与原因完整一致"
                );
                checks.check(
                    scores.iter().zip(&fresh_scores).all(|(a, b)| {
                        a.score.to_bits() == b.score.to_bits() && a.train_fail_adj.to_bits() == b.train_fail_adj.to_bits()
                    }),
                    "Local 复用的最终分数与失败损失逐位一致"
                );
            }
            println!("Local 复用 details={collect_details} speed_hint={:?} friend={}", cache.long_term[0], cache.friend.is_some());
            checks.check(
                matches!(cache.long_term[0], [Some(normal), Some(special)] if normal.to_bits() != special.to_bits()),
                "多 Hint 训练实际覆盖两种不同的长期价值"
            );
            checks.check(cache.friend.is_some(), "已解锁友人的动态估值实际参与复用");

            let mut guarded = game.clone();
            guarded.ramen.current_ramen = None;
            guarded.uma.vital = 0;
            let mut cache = LocalTrainCache::default();
            let guarded_actions = guarded.list_train_actions();
            let chosen = trainer.decide_train_cached(&guarded, &guarded_actions, None, &mut cache, &mut scores)?;
            checks.check(
                cache.training.iter().all(Option::is_none)
                    && cache.long_term.iter().flatten().all(Option::is_none)
                    && cache.friend.is_none()
                    && scores.len() == 1
                    && guarded_actions[chosen].operation == Operation::Rest,
                "休息守门直接返回时不提前计算训练或友人估值"
            );
        }
        game.base.turn = 11;
        game.stage = RamenStage::RamenSelect;
        let actions = game.list_train_actions();
        let mut cache = LocalTrainCache::default();
        for ramen in [None, Some(4)] {
            let operation = on.preview_train(&game, &actions, ramen, &mut cache, &mut train_scores)?;
            let mut fresh = game.clone();
            fresh.stage = RamenStage::Train;
            fresh.ramen.current_ramen = ramen;
            fresh.ramen.clear_pending();
            let fresh_actions = fresh.list_actions()?;
            let (fresh_chosen, fresh_scores) = on.decide_train(&fresh, &fresh_actions)?;
            checks.check(
                actions == vec![RamenAction::no_ramen(Operation::Race)]
                    && actions == fresh_actions
                    && operation == fresh_actions[fresh_chosen].operation
                    && train_scores == fresh_scores,
                "必赛回合借用原选面阶段预演时仍只有比赛动作"
            );
        }
        checks.finish()
    }

    /// 吃面必成价值：本回合基础动作是训练且失败率>0 时，吃面候选应计入
    /// `eat_guarantee`（消除失败期望损失）；安全桥与评分在 rollout 中保持相同。
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
        let mut trainer = LocalRamenTrainer::with_configs(policy, local);

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
        let mut eval_cache = LocalTrainCache::default();
        let mut train_scores = Vec::new();
        let pre = trainer.preview_train(&game, &game.list_train_actions(), None, &mut eval_cache, &mut train_scores)?;
        let (_, outs) = trainer.decide_ramen(&game, &actions)?;
        let mut guarantee = 0.0f32;
        let mut has_eat = false;
        for (act, o) in actions.iter().zip(outs.iter()) {
            if act.ramen.is_some() {
                has_eat = true;
                let g = o
                    .breakdown
                    .iter()
                    .find(|(k, _)| *k == "eat_guarantee")
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
        trainer.config.safety_bridge_min_fail = 1.0;
        let mut rollout = LocalRamenTrainer::with_configs(trainer.policy.config.clone(), trainer.config.clone());
        rollout.policy.collect_details = false;
        let bridge = trainer.safety_bridge(&game, &[])?;
        let quiet_bridge = rollout.safety_bridge(&game, &[])?;
        let (chosen, scores) = trainer.decide_ramen(&game, &actions)?;
        let (quiet_chosen, quiet_scores) = rollout.decide_ramen(&game, &actions)?;
        println!("安全桥 normal={bridge:?} rollout={quiet_bridge:?}");
        let mut c = crate::utils::Checks::new();
        c.check(bridge.is_some(), "低体力场景实际启用安全桥训练收益评估");
        c.check(
            bridge.map(|(train, gain)| (train, gain.to_bits()))
                == quiet_bridge.map(|(train, gain)| (train, gain.to_bits())),
            "rollout 安全桥训练与收益逐位一致"
        );
        c.check(chosen == quiet_chosen, "rollout 吃面选择一致");
        c.check(
            scores.len() == quiet_scores.len()
                && scores.iter().zip(&quiet_scores).all(|(normal, quiet)| {
                    normal.score.to_bits() == quiet.score.to_bits()
                        && normal.train_fail_adj.to_bits() == quiet.train_fail_adj.to_bits()
                }),
            "rollout 吃面候选评分逐位一致"
        );
        c.check(
            quiet_scores.iter().all(|score| score.breakdown.is_empty() && score.reason.is_empty()),
            "rollout 吃面候选不构造分解和原因"
        );
        c.finish()
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
            .find(|(k, _)| *k == "friend_hidden_future")
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
            .find(|(k, _)| *k == "friend_hidden_future")
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
                        .find(|(k, _)| *k == "attr")
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
            utils::{Checks, get_workspace_root, init_test_logger}
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
        let win_off = off.ramen_window_alignment(&game, rid, &mut [None; 5])?;
        let win_on = on.ramen_window_alignment(&game, rid, &mut [None; 5])?;
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
                let bonus_off = o_off.breakdown.iter().find(|(k, _)| *k == "ramen_weak_train_boost").map(|(_, v)| *v).unwrap_or(0.0);
                let bonus_on = o_on.breakdown.iter().find(|(k, _)| *k == "ramen_weak_train_boost").map(|(_, v)| *v).unwrap_or(0.0);
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
                let bonus = o.breakdown.iter().find(|(k, _)| *k == "ramen_weak_train_boost").map(|(_, v)| *v).unwrap_or(0.0);
                println!("不吃面时智训练 weakboost: {bonus}");
                if bonus != 0.0 {
                    panic!("不吃面时不应有 ramen_weak_train_boost: {bonus}");
                }
            }
        }

        let mut checks = Checks::new();
        let mut window_cache = [None; 5];
        for region in [5, 6, 7, 6] {
            let shared = on.ramen_window_alignment(&no_eat, region, &mut window_cache)?;
            let fresh = on.ramen_window_alignment(&no_eat, region, &mut [None; 5])?;
            println!("地区{region}窗口：复用={shared}，独立={fresh}");
            checks.check(shared.to_bits() == fresh.to_bits(), "重叠地区复用原局面窗口分量后逐位一致");
        }
        checks.finish()
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
    // AGENTS.md：测试内允许 unwrap，但需显式标注
    #[allow(clippy::unwrap_used)]
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

        // 4. SupportCard::calc_training_effect（已简化签名，去 Result 包裹）
        let sample_card = &game.deck()[0];
        let (min4, mean4) = run("SupportCard::calc_training_effect", || {
            let _ = black_box(sample_card.calc_training_effect(&game, 0));
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

    #[test]
    fn test_with_tokens_tradeoff_parsing() -> anyhow::Result<()> {
        // trd / trdsh / trds = N/100 刻度（与 9/14 扫参 trd1600/trdsh3600 命名一致）
        let t = RecommendedRamenTrainer::with_tokens("trd800")?;
        println!("trd800 → pt_tradeoff={}", t.years[0].policy.config.pt_tradeoff);
        assert_eq!(t.years[0].policy.config.pt_tradeoff, 8.0);
        let t = RecommendedRamenTrainer::with_tokens("trdsh3000")?;
        println!("trdsh3000 → pt_tradeoff_shining={}", t.years[0].policy.config.pt_tradeoff_shining);
        assert_eq!(t.years[0].policy.config.pt_tradeoff_shining, 30.0);
        let t = RecommendedRamenTrainer::with_tokens("trds2400")?;
        println!("trds2400 → pt_tradeoff_super={}", t.years[0].policy.config.pt_tradeoff_super);
        assert_eq!(t.years[0].policy.config.pt_tradeoff_super, 24.0);
        let bad = RecommendedRamenTrainer::with_tokens("trdxx");
        println!("未知 token trdxx 是否报错: {}", bad.is_err());
        assert!(bad.is_err(), "无法解析的 trd 值必须报错");
        Ok(())
    }

    /// 二轮组合：只按真实可增长属性计算预留惩罚，保留远离上限时的原结果。
    #[test]
    fn test_round2_clipped_reserve_boundaries() -> Result<()> {
        use crate::{game::{InheritInfo, ramen::RamenGame}, gamedata::init_global,
            utils::{Checks, get_workspace_root}};
        std::env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut game = RamenGame::newgame(102601,
            &[302424,302894,303044,302924,303024,303054],
            InheritInfo { blue_count:[15,3,0,0,0], extra_count:[0,30,0,0,30,30] })?;
        game.base.turn=38;
        // 2026-09-18：两条臂显式钉住 capd/reserve，机制验证不再随 new() 的 preset 漂移。
        let base=RecommendedRamenTrainer::with_tokens("ptblend200-capd100-reserve40-rgn0")?;
        let candidate=RecommendedRamenTrainer::with_tokens("ptblend200-capd0-reserve40-rgn1")?;
        let mut c=Checks::new();
        for y in 0..3 {
            let mut expected=base.years[y].policy.config.clone();
            expected.cap_discount_weight=0.0;
            c.check(candidate.years[y].policy.config==expected,"policy 只取消副属性折扣");
            c.check(candidate.years[y].config.reserve_gain_mode==1,"三年均按实际增量算预留");
            c.check(candidate.years[y].config.status_reserve_max==base.years[y].config.status_reserve_max,
                "保留预留阈值，不等于关闭预留机制");
        }
        let gain=[100,0,0,0,0,0];
        // turn38 时 r=40*(76-38)/76=20；剩余10时，仅新增10有效。
        for (space,want) in [(0,0.0),(10,45.0),(200,0.0)] {
            game.uma.five_status[0]=game.uma.five_status_limit[0]-space;
            let old=base.years[1].reserve_penalty(&game,&gain);
            let new=candidate.years[1].reserve_penalty(&game,&gain);
            println!("space={space}: old={old} clipped={new}");
            c.check((new-want).abs()<0.001,"符合独立手算的实际增量惩罚");
            if space==200 { c.check(old==new,"窗口外惩罚不变"); }
            else { c.check(old>new,"溢出增量不再被重复惩罚"); }
        }
        c.finish()
    }

    /// 第八轮实验：逐卡 Hint 精确估值（token hintlvW）的 token 隔离、非法值与数值方向检查。
    #[test]
    fn test_round8_hint_card_aware() -> anyhow::Result<()> {
        use crate::utils::Checks;
        use crate::{
            game::{InheritInfo, ramen::RamenGame},
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let base = RecommendedRamenTrainer::new();
        let variant = RecommendedRamenTrainer::with_tokens("hintlv100")?;
        let mut checks = Checks::new();
        for year in 0..3 {
            let got = &variant.years[year].config;
            checks.check(base.years[year].config.hint_card_aware == 6.0, "preset 启用第十二轮验收的逐卡 Hint 估值");
            checks.check(got.hint_card_aware == 1.0, "hintlv 写入三年");
            checks.check(got.hint_bonus == base.years[year].config.hint_bonus
                && got.max_base_score_sacrifice == base.years[year].config.max_base_score_sacrifice
                && got.eat_requires_covered_train == base.years[year].config.eat_requires_covered_train,
                "hintlv 不改动其他本地字段");
        }
        for bad in ["hintlvNaN", "hintlv-1", "hintlv1001"] {
            checks.check(RecommendedRamenTrainer::with_tokens(bad).is_err(), "非法实验值报错");
        }

        // 满属性下属性分支为 0：每个带 Hint 人头的价值 = 0.75 x 等级 x (6.5 x 2.0)。
        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            InheritInfo { blue_count: [15, 3, 0, 0, 0], extra_count: [0, 30, 0, 0, 30, 30] }
        )?;
        for i in 0..5 {
            game.uma.five_status[i] = game.uma.five_status_limit[i];
        }
        let on = &variant.years[2];
        let off_trainer = RecommendedRamenTrainer::with_tokens("hintlv0")?;
        let off = &off_trainer.years[2];
        let mut multi = false;
        for person_index in 0..game.base.deck.len() {
            let levels = (1 + game.base.deck[person_index].card_value().hint_level).min(5);
            let got = on.hint_event_expected_value(&game, person_index, 0);
            let want = 0.75 * levels as f32 * 6.5 * 2.0;
            println!("card {person_index} hintLv={levels} value={got}");
            checks.check((got - want).abs() < 0.01, "满属性下只剩技能分支");
            checks.check(off.hint_person_value(&game, person_index, 0) == off.config.hint_bonus, "关闭时沿用固定值");
            multi |= levels > 1;
        }
        checks.check(multi, "固定卡组存在多级 Hint 卡面");
        checks.finish()
    }

    /// 实验参数仅改变三年的连续窗口，默认关闭且拒绝越界/非有限值。
    #[test]
    fn test_pt_cap_blend_token_isolation() -> anyhow::Result<()> {
        use crate::utils::Checks;
        let base = RecommendedRamenTrainer::new();
        let variant = RecommendedRamenTrainer::with_tokens("ptblend200")?;
        let mut checks = Checks::new();
        for year in 0..3 {
            let mut expected = base.years[year].policy.config.clone();
            checks.check(expected.pt_cap_blend_turns==8.0,"preset 采用第十二轮验收窗口");
            expected.pt_cap_blend_turns=2.0;
            checks.check(variant.years[year].policy.config==expected,"只覆盖窗口字段");
        }
        for bad in ["ptblendNaN","ptblendinf","ptblend1001","ptblendbad"] {
            checks.check(RecommendedRamenTrainer::with_tokens(bad).is_err(),"非法实验值报错");
        }
        println!("ptblend200 三年隔离与非法参数检查完成");
        checks.finish()
    }

    /// 超级拉面开关只覆盖对应字段，固定对照和自适应模式都应用于三年策略。
    #[test]
    fn test_super_choice_token_isolation() -> Result<()> {
        use crate::utils::Checks;
        let base = RecommendedRamenTrainer::new();
        let mut checks = Checks::new();
        for mode in 0..=3 {
            let variant = RecommendedRamenTrainer::with_tokens(&format!("supermode{mode}"))?;
            for year in 0..3 {
                let mut expected = base.years[year].policy.config.clone();
                checks.check(expected.super_choice_mode == 3, "preset 采用第十二轮验收的自适应范围");
                expected.super_choice_mode = mode;
                checks.check(variant.years[year].policy.config == expected, "只覆盖超级拉面模式");
            }
        }
        for bad in ["supermode4", "supermodeNaN", "supermode256", "supermode-1", "evreal3"] {
            checks.check(RecommendedRamenTrainer::with_tokens(bad).is_err(), "非法或未保留开关报错");
        }
        println!("超级拉面模式的三年隔离与参数校验完成");
        checks.finish()
    }

    /// 正式 preset 必须与第十二轮验收用的组合串逐位一致。
    ///
    /// 该串就是两个独立池上验收的那只（out/rmb/r3f 三项等于 preset 现值，无需 token）；
    /// 窗口、PT 定价、缺口/溢出强度、预留口径与超级拉面模式一旦漂移，本测试直接失败，
    /// 避免验收的是一个组合、线上跑的是另一个组合。
    #[test]
    fn preset_matches_validated_round12_combo() -> anyhow::Result<()> {
        use crate::utils::Checks;
        const COMBO: &str = "ptblend800-capd0-hintlv600-trd4425-trds3450-trdsh2500-g1498-g2498-g3498-o1304-o2304-o3304-reserve157-rgn1-supermode3";
        let preset = RecommendedRamenTrainer::new();
        let combo = RecommendedRamenTrainer::with_tokens(COMBO)?;
        let mut checks = Checks::new();
        for year in 0..3 {
            let (p, c) = (&preset.years[year], &combo.years[year]);
            checks.check(p.policy.config == c.policy.config, "三年策略配置与验收串逐位一致");
            checks.check(p.config.hint_card_aware == c.config.hint_card_aware, "逐卡 Hint 倍率一致");
            checks.check(p.config.status_gap_strength == c.config.status_gap_strength, "短板追赶强度一致");
            checks.check(p.config.status_overflow_strength == c.config.status_overflow_strength, "近上限衰减强度一致");
            checks.check(p.config.status_reserve_max == c.config.status_reserve_max, "预留上限一致");
            checks.check(p.config.reserve_gain_mode == c.config.reserve_gain_mode, "预留口径一致");
        }
        println!("preset 与第十二轮验收组合串逐位一致");
        checks.finish()
    }
}

