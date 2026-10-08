# 拉面整局 golden 基线记录

模拟器行为（策略、评分表、剧本数值口径）每次变动，同种子整局快照都会整体位移。
为保持代码里只留「基线值 + 一句口径说明」，把历次重抓的原因与数值变动集中记录在本文件。

## 基线位置

| 测试 | 文件 | 固定条件 | 快照内容 |
|---|---|---|---|
| `test_yearly_observability_full_game_and_csv` | `crates/umasim/src/bench.rs` | 纯 `RecommendedRamenTrainer`，seed=42 / run_idx=0 / `TEST_DECK` | `score` + `five_status` |
| `test_ramen_three_stage_action_unchanged` | `crates/umasim/src/search/flat_search.rs` | 拉面根局面，三阶段分别搜，seed=42 | 7 个候选的 `(n, mean)` |
| `test_combined_gate_off_full_game` | `crates/umasim/src/trainer/ramen_mcts_trainer.rs` | `ramen_and_special_stages()` + `combined=false`，seed=42 | `score` / `five` / `skill_pt` / `searched_count` |
| `test_combined_on_skips_special_search` | `crates/umasim/src/trainer/ramen_mcts_trainer.rs` | 合并搜索开启，seed=42 | `SpecialSelect` 调用数 / 重搜数 |

> 注：`test_combined_gate_off_full_game` 的 gate-off **不是**纯推荐策略跑局——
> `ramen_and_special_stages()` 下 ramen/special 仍在搜；纯推荐对照见
> `test_stages_none_matches_recommended` 与 bench.rs 的快照。

## 重抓历史

### bench.rs `BASELINE_SCORE` / `BASELINE_FIVE`

- 2026-08-25：不在判定与得意率解耦 + 地区分身缺席优先，模拟数值变化，基线作废重抓。
- 2026-08-27（两次叠加）：
  1. 五维上限剧本化——基值前置 + 删 `min(2800)` + 保住开局继承，速度上限 2958→3337；
  2. bench 与基线测试切到 `RecommendedRamenTrainer`（手写策略正式推荐版），吃面-训练联动 /
     体力门限 / 友人节奏 / 动态属性平衡等全机制接管。
     上游在 (1) 之前抓的 63532 / `[3258,...]` 作废（speed 3258 即缺陷 B 未修值）。
     本分支叠加后重抓：五维只有速度位 3258→3337（正是 (1) 保住的开局继承增量），
     其余四维逐位不变，分数 +804。
- 2026-09：吃面 PT 增量 / `eat_count` 延后到 NextTurn（后经 2026-10-07 判定为错误方向），
  训练阶段用吃面前 PT 算 `ramen_pt_effect` / `region_bonus` 档位，纯推荐策略整局偏低，基准重抓。
- 2026-09-17：合宿训练诀窍全 MAX 修复（d9374e8）+ 近期策略调整，改动前基线
  （63870 / `[3337,2293,2200,1086,829]`）在干净 master 上已过期，重抓。
- 2026-09-17 二次：GA 方向 9 旋钮组合档定稿进 preset（pt_rate Y1 56 / pt_tradeoff 37 / 超拉面 35 /
  弱位覆盖 35 / 友情 0.4 / hint 8 / max_sac 200 / ramen_window 0.15 / ck 0.15），同种子 68118→70138。
- 2026-09-18 三次：第十二轮配对验收组合档进 preset（ptblend 8 / capd 0 / hintlv 600 / trd 44.25 /
  trds 34.5 / trdsh 25 / gap 4.98 / overflow 3.04 / reserve 157 / rgn 1 / supermode 3 / out 0），
  同种子 70138→69219。
- 2026-09-21 四次：友人出行跨年配额定档 `[0,3,5]`（替换 `[0,2,5]`），同种子 69219→69232。
- 2026-10-04 五次：超级拉面效果修正——只保留 RMJ + finals（去掉误叠加的 pt_effect / basic
  试食会效果），并接入选中选项的 +100 训练上限；同种子 69232→68964（五维仅智 1184→1129）。
- 2026-10-06 六次：PT 上段上限口径修正——普通回合 `ramen_basic_effect.status_limit` 同时抬属性
  与 PT 上限（「獲得上限アップ」Y2 +20 / Y3 +40），吃面回合 PT 上限下降；同种子 68964→68564（五维不变）。
- 2026-10-06 七次：五维评分表换用 URA `StatusToPoint`（raw 表 3802 项），仅显示值 ≥2001 区间下调；
  同种子 68564→68258（五维不变）。
- 2026-10-07 八次（本次）：拉面 PT 口径定稿——吃面 PT 增量 / `eat_count` 回到**吃面当刻**入账，
  训练阶段按**吃面后** PT 取 `ramen_pt_effect` / `region_bonus` 档位；PT 上层友情**剔除** RMJ 结算部分
  （属性仍用完整友情）。同种子 65538→67720，五维 `[3337,2250,2200,1097,1080]`→`[3337,2435,2200,1163,1127]`。

  > 期间曾短暂取「吃面 PT 延后 / 按吃面前 PT / PT 友情不剔 RMJ」的 M2 口径（同种子 65288→65538），
  > 经实机对拍证伪后回退，此处不再保留中间态。

### flat_search.rs `test_ramen_three_stage_action_unchanged`

- 2026-08-25：不在判定与得意率解耦 + 地区分身缺席优先，rollout 数值变化，基准重抓。
- 2026-08-27（两次叠加）：(1) 五维上限剧本化，速度上限 2958→3337，rollout 终局分整体抬升；
  (2) `searchable.rs` RolloutTrainer 切到 `RecommendedRamenTrainer`，均值再上移 ~10k。
- 2026-09：吃面 PT 增量延后到 NextTurn，rollout 数值整体下移，基准重抓。
- 2026-09-17：合宿训练诀窍全 MAX 修复（d9374e8）后 rollout 数值上移。
- 2026-09-17 二次：GA 方向 9 旋钮组合档进 preset 后 rollout 数值再移。
- 2026-09-18：第十二轮组合档进 preset，rollout 数值再移。
- 2026-09-21：友人出行跨年配额定档 `[0,3,5]`，rollout 数值再移。
- 2026-10-01：切者/小切按 1.1 / 1.04 缩放终局评分 PT 项（`UmaFlags::pt_score_rate_factor`），
  rollout 内抽到切者的事件让终局分上移，均值 +118 左右。
- 2026-10-04：超级拉面效果修正（只保留 RMJ + finals、接入选中选项 +100 上限），URA 训练数值变化，
  均值整体下移约 750。
- 2026-10-06：PT 上段上限口径修正（普通回合 `basic.status_limit` 同时抬 PT 上限），吃面回合 PT 上限下降，
  rollout 终局 PT 变少，均值再下移约 120~190。
- 2026-10-06：五维评分表换 URA `StatusToPoint`，终局五维分（显示值 ≥2001 区间）下调，
  均值再下移约 180~300。
- 2026-10-07（本次）：拉面 PT 口径定稿（吃面后 PT + PT 友情剔 RMJ），rollout 数值整体位移，基准重抓。

### ramen_mcts_trainer.rs `test_combined_gate_off_full_game`

- 2026-08-25：不在判定与得意率解耦 + 地区分身缺席优先，模拟数值变化。
- 2026-08-27（两次叠加）：(1) 五维上限剧本化，速度上限 2958→3337；
  (2) fallback 与 rollout 均切到 `RecommendedRamenTrainer`。上游 (2) 抓的 66705 / `[3258,...]`
  是在 (1) 之前测的，叠加后在本分支重抓。
- 2026-09：吃面 PT 增量 / `eat_count` 延后到 NextTurn，训练阶段用吃面前 PT 取档位，
  拉面效果变弱导致整局偏低，基准重抓。
- 2026-09-18：上一版数值早于 preset 定稿（本次改动实测逐位不变，仅为同步）。
- 2026-09-21：友人出行跨年配额定档 `[0,3,5]`，整局路径变化。
- 2026-10-04：超级拉面效果修正（只保留 RMJ + finals、接入选中选项 +100 上限），URA 训练数值变化，
  整局路径与终局数值变化。
- 2026-10-06：PT 上段上限口径修正（普通回合 `basic.status_limit` 同时抬 PT 上限），
  整局路径与终局数值变化。
- 2026-10-06：五维评分表换 URA `StatusToPoint`，手写策略 marginal gain 下调，决策路径与终局数值整体变化。
- 2026-10-07（本次）：拉面 PT 口径定稿（吃面后 PT + PT 友情剔 RMJ），整局路径与终局数值变化。

### ramen_mcts_trainer.rs `test_combined_on_skips_special_search`

- 2026-09：吃面 PT 增量延后到 NextTurn，本回合 PT 档位提升延后生效，整局搜索路径微小变化，
  SpecialSelect 调用 / 重搜数基线重抓。
- 2026-09-18：上一版快照早于 preset 定稿（本次改动实测逐位不变，仅为同步）。
- 2026-09-21：友人出行配额定档 `[0,3,5]`，SpecialSelect 调用 28→29（重搜仍为 0）。
- 2026-10-04：超级拉面效果修正（RMJ + finals、选中选项 +100 上限），整局搜索路径变化，
  SpecialSelect 调用 29、重搜 0。
- 2026-10-06：PT 上段上限口径修正，吃面回合 PT 上限下降 → 决策倾向变化，
  SpecialSelect 调用 29→26、重搜 0→2。
- 2026-10-06：五维评分表换 URA `StatusToPoint`，决策倾向再变，SpecialSelect 调用 26→27、重搜 2→1。
- 2026-10-07（本次）：拉面 PT 口径定稿（吃面后 PT + PT 友情剔 RMJ），决策倾向再变，
  SpecialSelect 调用 27→28、重搜 1→0。

## 拉面 PT 口径校准证据（2026-10-07）

实机帧对拍（`logs/game6261/`）：

| 帧 | turn | source | `last_ramen` | `scenario_pt` | `active_effect_array` 数 |
|---|---|---|---|---|---|
| `game6261_turn47_2.json` | 47 | command | 8 | 5400 | 0 |
| `game6261_turn47_3.json` | 47 | command | 9 | **6000** | 6 |

本回合吃面后 `scenario_pt` 从 5400 涨到 6000（+600），且档位立刻按新值生效——证明协议帧的
`scenario_pt` 是**吃面后**当年累计值，实机训练用的就是吃面后 PT。

其余校准点（当年 `scenario_pt` → `region_bonus` 档位，系数表 `0/3/5/7/9/10`）：

| 帧 | `scenario_pt` | tier | bonus |
|---|---|---|---|
| `turn48_3` | 500 | 0 | 0 |
| `turn54_2` | 1650 | 1 | 3 |
| `turn55_3` | 2300 | 2 | 5 |
| `turn57_2` | 3000 | 3 | 7 |

要点：
- `region_bonus` 档位 = `(scenario_pt / 1000).min(5)`（旧实现按 `/300`，在同一批数据上会算出
  3/10/10/10，明显偏大）。
- `region_bonus` **只加到友情**（进属性与 PT 的友情乘子），**不加到 PT 加成**：
  `active_effect_array` id52 虽显示为 `pt_bonus + region_bonus`，PT 上层公式实际只用 region 基础
  `pt_bonus`。
- PT 上层友情**剔除 RMJ 结算部分**（`ramen_success_effect` / `ramen_fail_effect` 的 youqing），
  属性上层仍用完整友情。
- 训练阶段按**吃面后** PT 取档位（`ramen_pt_effect` / `region_bonus`）——M2 曾改为按吃面前 PT，
  经实机对拍证伪后回退。