# 拉面杯手写策略配对矩阵挖矿计划

## 背景

v44 交接文档（`.trae/documents/ramen_policy_handoff_v44.md`）第 6 节列出了一批
"fork 实验通道一个都没碰过"的旋钮。本计划用**同 seed 配对矩阵**（A/B 测试）
逐个验证这些旋钮是否能提升评分，目标是将三种主流配卡
（2速1耐2智 / 3速1耐1智 / 2速1力1根1智）的评分提升到 6.5w-7w。

## 方法

对每个旋钮的每个候选值，跑 N 局同 seed 配对：
- baseline：`RecommendedRamenTrainer::new()`（正式 preset）
- candidate：`RecommendedRamenTrainer::with_mine_overrides(...)`（只覆盖被测旋钮）

记录配对胜/平/负、均分差、属性分差、PT 差。只有**胜 > 负 且 Δ分 > 0** 的候选
才认为有矿。

## 已验证旋钮清单

### 第一批（单点位）

| 旋钮 | 当前值 | 结论 |
|---|---|---|
| `cap_discount_weight` | 1.0 | 无矿（≥0.5 后饱和） |
| `cook2_stock_weight` | 40 | ✅ 有矿，最优因配卡而异（60/80/20） |
| `y3_post_train_hard_floor` | 15 | ✅ 有矿，最优因配卡而异（20/10/10） |

### 第二批（联合扫）

| 旋钮组合 | 当前值 | 结论 |
|---|---|---|
| `friend_proactive × friend_hidden_starve` | 150/300 | ✅ 有矿，最优因配卡而异 |
| `y3_pre_vital × y3_shortfall` | 25/0.5 | 无矿（全部 0/N/0 平局） |

### 第三批（事件三选一）

| 旋钮 | 当前值 | 结论 |
|---|---|---|
| `event_vital_weight` | 2.2 | ✅ 3速1耐1智 有大矿（6.0，+649 分） |
| `event_motivation_weight` | 40 | 基本无矿 |
| `event_bad_flag_penalty` | 300 | 无矿（未触发） |

## 配卡自适应规律

挖矿发现的最优值因配卡而异，可用 `card_type_count` 自适应：

| 旋钮 | 速卡≥3 | 速卡=2且智卡=2 | 速卡=2且智卡≤1 |
|---|---|---|---|
| `cook2_stock_weight` | 80 | 60 | 20 |
| `y3_post_train_hard_floor` | - | 20 | 10 |
| `friend_proactive_weight` | 300 | 0 | 0 |
| `friend_hidden_starve_weight` | 300 | 300 | 150 |
| `event_vital_weight` | 6.0 | 2.2 | 2.2 |

## 后续工作

1. 将配卡自适应规律写入 `RecommendedRamenTrainer::new()` 的 `effective_*` 方法
2. 用更多 seed 验证配卡自适应的稳定性
3. 探索 `policy.decide_event` 事件三选一的长期价值评估
