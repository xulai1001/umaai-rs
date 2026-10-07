# 拉面杯追加笔记
注意：本文档存在大量术语，必须参考术语表`glossary.md`中的定义

----

## 基础游戏机制强调

### 友情训练（彩圈）
- 指一张支援卡位于其卡类型的位置，例如速卡位于速训练。此时该卡会显示为`彩圈`。此时进行的训练称为`友情训练`
- 只在友情训练时，乘算`youqing`（友情）加成。
- 同一训练上有多张卡触发友情训练，每个卡的`youqing`（友情）加成会`累乘`
- `得意率`（deyilv）会改变支援卡的随机分布，增加其出现彩圈（位于友情训练位置）的概率

### 友人解锁
- 友人从第2回合开始出现在训练中
- 首次选择友人在的训练时（即“点击友人”），触发”友人登场“事件(830305101)
- “友人登场“触发后，再次点击友人，概率触发“点击友人”事件（830305102），概率为 `event_probs.friend_click`
- “友人登场”触发后，回合开始时判定，随机触发“友人解锁”事件（830305103）
- 友人羁绊<60时，概率为 `event_probs.friend_unlock_low`；羁绊>=60时为 `event_probs.friend_unlock_high`
- 触发“友人解锁”事件后，可以选择“友人出行”动作，该动作可进行5次，固定依次触发“友人出行”的五个事件(830305111 - 830305115)，都触发完后，无法选择“友人出行”动作。
- 友人出行每次完成后，额外获得2个隐藏风味（新友人情况）
- 友人羁绊效果（ramen_basic_effect.jiban）对卡组中所有6张支援卡生效，不只是训练中的人物

### 剧本机制初始化（第2回合开始时）
根据携带友人卡的情况，初始化诀窍值和隐藏诀窍的数量：
- 有新友人卡（30305）：诀窍值每种=2，隐藏诀窍=2
- 有旧友人卡：诀窍值每种=1，隐藏诀窍=1
- 没有满足条件的卡：诀窍值每种=0，隐藏诀窍=0
- 初始化时同时选择第1年地区（regions 0-4）
- 初始化后立即重新打印回合信息

### 训练等级
- 起始为1级，最高5级
- 每点击一次某种训练，计数+1
- 训练等级 = floor(计数/4) + 1 + 剧本加成
- 剧本加成初始为0；`剧本点数结算`时如果result=2，则剧本加成+1
- RMJ结算成功时，训练等级剧本加成+1（累积）

----

## 核心剧本机制

### 诀窍（feeling, 诀窍点）
`诀窍`有3种，记为A/B/C. 
总上限为10；需要记录“获得诀窍的顺序”队列，当总`诀窍点`>10时，移除最早获得的诀窍点，直到留下10个。

### 诀窍槽(feeling gauge)
有A/B/C三个诀窍槽，上限为7
剧本机制开启后，每回合会填充诀窍槽，`诀窍槽`满7时，清零，并增加1个对应种类的`诀窍`
`诀窍槽`无论溢出多少，都只能增加1个`诀窍`并清零
诀窍槽获取算法，参见“特殊机制”节的“诀窍槽获取算法”

### 隐藏风味(special_feeling)
剧本的固定回合，或者友人出行后，会得到`隐藏风味`；隐藏风味上限为4

### 获取隐藏风味
得到的`隐藏风味`数量随友人卡稀有度不同。当携带`新友人`(id: 30305)时，固定回合和出行后都可以得到2个`隐藏风味`。
当携带`旧友人`(chara_id: 9001 or 9008, id != 30305)时，得到1个`隐藏风味`
没有符合条件的卡时，仅在固定回合得到1个`隐藏风味`，出行无法获取。

### 拉面（做面/吃面）
剧本机制开启后，在任意回合的训练/比赛前，都可以做面
固定消耗5个`诀窍点`=1拉面，但配方不同，记录在`region_feeling`中。
例如，`region_feeling = [2, 2, 1]`, 表示消耗2个A，2个B，1个C=1拉面。
`隐藏风味`可以**替代任意普通诀窍点**；每次做面时可以使用0-2个`隐藏风味`。
做面后，仅在当回合内享受拉面提供的加成效果。加成效果为：基础效果`ramen_basic_effect`和地区效果`ramen_region_effect`的和.
**PT 增量/eat_count 延后到 NextTurn 阶段**：`ground_ramen_effects`（SpecialSelect→Train 过渡时触发）只设 `current_ramen` / 消耗诀窍 / 生成分身 / 羁绊效果，不立即累加 `scenario_pt` 也不 `eat_count += 1`——这两步统一在 `Game::next()` 的 `NextTurn` 阶段（清空 `current_ramen` 之前）做。这样训练阶段的 `calc_ramen_training_effect` 读到的是「吃面前」的 `scenario_pt`，`ramen_pt_effect` / `region_bonus` 档位不会因本次吃面立即跨档抬升；PT 增量的档位收益从下一回合才参与计算。RMJ 归档与 `check_rmj` 看到的是吃面后 PT，行为不变。

> **显示层 vs 结算层（2026-10-07 补充）**：游戏内是「先按前一档算效果数值，再增加并显示地区面板」——面板 / `active_effect_array` 显示的是**吃面后**档位（含本次吃面新增 PT），但公式（结算层）用的是**吃面前** PT。因此 `umaai` 从协议快照（协议 `scenario_pt` 为吃面后值）重建局面时需回退到吃面前 PT（`into_game` 内 `restore_pre_eat_pt`），与引擎语义对齐；面板值看着像吃面后，属显示层与公式不同步。

### 剧本点数（ramen_pt）
剧本点数影响剧本的`全局加成`，初始为0，每次做面都会增加剧本点数。
增加量随`年份`和`当年内做面次数`增加，记录在`gain_pt_base`和`gain_pt_delta`中。叠加5次后，这个增量不再增加，下一年重置。
例如，第1年，第4次吃面，增加量为 300 + 30 * (4 - 1) = 390。第6次及以上，增加量固定为450。
全局加成记载在`ramen_pt_effect`

### 剧本点数结算（RMJ结算）
每年年终（回合23, 47, 71的结束阶段），需要检查剧本点数是否满足要求。
| 年份 | 所需Pt |
|---|---|
| 1 | 1500 |
| 2 | 3000 |
| 3 | 3500, >=5000为大成功 |

如果通过检测，则当前的`RMJ`事件触发`result=2`，下一年常驻享受前一年的`ramen_success_effect`，否则`result=1`，应用`ramen_fail_effect`。

**RMJ 事件**（在回合 23/47/71 末立即触发）：
- 第1年：401404
- 第2年：401405
- 第3年：401406

事件 choice 结构：选项组内含 2 个分支（`result=2` 成功 / `result=1` 失败），根据 `rmj_results[year_idx]` 选择对应分支并应用其 value。RMJ 事件没有 `player_select=true`，由代码自动按 RMJ 结算结果选择。

**RMJ 结算后 scenario_pt 归零**：下一年（年2 / 年3 / URA 阶段）的剧本 PT 从 0 重新累计。注意：
- `ramen_success_effect / ramen_fail_effect` 已经基于 `rmj_results` 可读取（下一年常驻生效）
- 新一年的 `ramen_pt_effect` 档位和 `region_bonus` 基于 PT=0 重新计算（pt_min=0 档位：`deyilv=50`，`region_bonus=0`）

**RMJ结算效果生效时间**：
- 第1年结算（rmj_results[0]）→ 第2年生效
- 第2年结算（rmj_results[1]）→ 第3年生效
- 第3年结算（rmj_results[2]）→ URA期间（回合72-77）生效

### 地区效果(ramen_region_effect)
记载选择地区拉面的额外效果。基础值记录在`ramen_region_effect`
地区效果的选择和年份有关，第1-2年，对应id: 0-9；第3年，对应id: 10-19。
每一年地区的选择范围固定，玩家需要从中选择3个地区，对应不同的拉面效果。接下来直到当年底都只能使用这三种拉面
地区效果中，`友情`和`PT加成`会随着`当年内获得的剧本点数`增加而增加。增加量记录在`region_bonus`中：`scenario_pt` 每满 **1000** 点提升一档（`region_bonus[pt/1000]`，上限 5 档，取值 0/3/5/7/9/10）。
例如，第3年，札幌的地区效果为 `id=10, youqing=50, pt_bonus=50`，在当年内获得1000点剧本PT时，region_bonus=3，面板总效果为 `youqing=53, pt_bonus=53`。

> 口径校准（2026-10-07，game6261 第3年 `active_effect_array` 反推）：`pt=500→0 / 1650→3 / 2300→5 / 3000→7`，即按 **1000 点一档**（旧述「每300点」有误）。
> 另注：面板/`active_effect_array` 会把 `region_bonus` 同时加到 `youqing` 与 `pt_bonus`，但**属性与 PT 的上层公式使用的 pt_bonus 只取 region 基础值（不含 region_bonus）**；`youqing` 则含 region_bonus。
> 且 **PT 口径（2026-10-07 定稿，M2）**：友情加成正常情况下**同时作用于属性与 PT**，PT 上层公式**不剔除** RMJ 结算的友情（推翻早前 M1「RMJ 友情只作用于属性」的方案）。另：吃面回合的 `ramen_pt_effect` / `region_bonus` 档位按**吃面前** `scenario_pt` 取（见「拉面（做面/吃面）」节的显示层说明）。

### 超级拉面(super_ramen)
在URA回合（回合72-77），每回合自动享受超级拉面效果，记录在`finals_effect`中
其中，`extra`效果为满足`支援卡种类>=4`时额外生效；`training_limit_options`为按玩家选择
* 超级拉面期间不可以吃其他面，不享受地区效果
* 超级拉面期间**只叠加「第3年RMJ结算效果 + finals_effect」**；试食会（吃面）专属效果一律不生效：
  - `ramen_pt_effect`（盛り上がりPt奖励，xunlian/deyilv/hint）**不生效**
  - `ramen_basic_effect`、`ramen_region_effect`（吃面才有的基础/地区效果）**不生效**
  - 第3年RMJ结算效果（ramen_success_effect 或 ramen_fail_effect）生效
  - `finals_effect.base` 效果生效（友情+150 等）
  - `finals_effect.extra` 效果仅在支援卡种类 >= 4 时生效（PT+100 / PT上限+100 / 分身+1）
* 净训练效果（第3年RMJ成功、支援卡种类>=4）：`xunlian=0`、`youqing=175`（RMJ 25 + finals 150）、`pt_bonus=100`、`pt_limit=100`
* 上层属性数值上限（`status_limit`）：选中选项（`training_limit_options[super_ramen]`）覆盖的 4 个训练位 **+100**（**仅抬属性上限**；未选选项/选项越界则不生效）
  * 原始资料写作「○○以外**獲得上限**+100」（属性），与 extra 的「**SP獲得上限**+100」（PT）是两个独立口径：超级拉面的属性上限走选项、PT 上限走 extra，各 +100，互不叠加

**finals_effect.base 自动应用**（每个 URA 回合 Begin 阶段）：
- `vital`（体力恢复，+20）：每个 URA 回合（turn=72-77）都生效，每回合 +20
- `motivation`（干劲提升，+1）：每个 URA 回合都生效，每回合 +1
- `saihou`（赛后加成，+100）：**仅一次性 +100**，之后回合（turn=73-77）保留已生效值，不重复累加
  - 实现：`RamenGame::apply_super_ramen_saihou()`——**幂等**（`turn>=72` 且已选超级拉面时 `uma.race_bonus += finals.base.saihou`），由非序列化标记 `super_ramen_saihou_applied` 保证合计只生效一次
  - 调用点：模拟路径在 turn 72 进入超级拉面时经 `run_begin`；**协议重放路径**在 `GameStatusRamen::into_game` 补调（协议帧不含 `raceBonus`，不补会让 URA 段比赛收益按 ×1.55 而非 ×2.55 被系统性低估）
  - 例如初始 race_bonus=60（来自支援卡），turn=72 后变为 60+100=160，turn=73-77 比赛都用 160 乘算

### NPC
剧本机制启动后，在普通训练中会出现`NPC`类型的`Person`，为了方便，NPC的支援卡ID固定为五个链接角色的chara_id:
- 1022(美妙)
- 1058(怒涛)
- 1060(内恰)
- 1077(成田路)
- 1120(金镇)

注意，虽然卡组也会包括这些卡（如SSR怒涛的chara_id=1058），但卡组的支援卡ID不会是这几个数字，Person类型也可以区分。
NPC仅影响训练的`人数`，没有其他效果，卡名统一为"NPC".
- NPC显示为`[NPC]`，不显示角色名字
- NPC不计算羁绊（羁绊固定为0）

### 地区选择机制
- 第1年地区选择在回合2的Begin阶段进行
- 第2年地区选择在回合23的NextTurn阶段（RMJ结算后）进行
- 第3年地区选择在回合47的NextTurn阶段（RMJ结算后）进行
- 通过Trainer统一接口（select_action）决策，可选组合为C(n,3)
- 年3地区(ID 10-19)的诀窍配方复用年1地区(0-9)的配方（取模映射）

### 休息心得（refresh_mind）
- 友人解锁事件后，`refresh_mind` 设置为1
- 每回合开始时，如果 `refresh_mind > 0`：
  - 体力+5
  - `refresh_mind` 计数+1
  - 根据 `group_buff_end_prob` 概率判定是否结束
- 结束时 `refresh_mind` 重置为0

### 设施等级显示
- 设施等级显示为加上剧本加成后的实际等级
- 实际等级 = floor(计数/4) + 1 + 剧本加成（上限5）

----

## 特殊规则

### 夏合宿
夏合宿期间（回合36-39和60-63），有以下特殊规则：
- 所有训练等级均为5
- 不会发生支援卡事件或掉心情事件
- 回合37-39和61-63开始时触发夏合宿奖励事件
- **诀窍槽**：带新友人(30305)时，所有动作（训练/比赛/休息/外出）都让三种类型的槽直接填满到上限（每种清零 +1 诀窍，即"全 MAX"）
- **动作菜单**：不允许普通外出、友人出行、治病（治病的场景由休息替代）
- **休息**：自动清除 `ill` 和 `bad_trainer` flag（等同 Clinic 效果）

### 训练分布的得意率计算
支援卡出现在训练位置的概率受总得意率加成：
- 总得意率 = **支援卡本身得意率 + 剧本得意率加成**
- 剧本得意率加成来源：`ramen_pt_effect`（常驻，当前档）+ RMJ 结算后的 `ramen_success_effect` / `ramen_fail_effect`（按生效年份）
- 基础权重为 [100, 100, 100, 100, 100]（速/耐/力/根/智），支援卡对应训练位置的权重加上总得意率后随机分配

**注**：本剧本 `ramen_basic_effect` 和 `ramen_region_effect` 字段都不含 deyilv（参考下方数据表），仅 pt 效果和 RMJ 结算效果对得意率有贡献。

### 分身
分身（clone）是指在训练中，一个支援卡的`Person`被复制，形成一个新的`Person`。

### 分身通用规则
- 分身和本体共享相同的 `person_index`（使用本体ID）
- 分身只能在本体的得意训练位置闪耀（train_type == train && friendship >= 80）
- 分身在非本体训练位置时不闪耀，友情加成不生效
- 分身增加人头数，影响训练效果（每多一人 +5%）
- 分身贡献支援卡效果（buff）
- 分身可以触发 hint 事件和友人点击事件
- 分身增加本体的羁绊（视为同一个人物）
- 如果当前训练已经存在相同卡的`Person`，则不能创建分身
- 如果当前训练已经满5个人，则分身会优先"挤"掉NPC；如果已经包含5个非NPC的人物，则不能创建分身

### 分身产生方式 - 独立规则

**方式(1) 地区拉面分身**
- **触发条件**：`地区效果` id >= 5，且`支援卡种类>=4`时，`at_trains`字段记录需要创建分身的位置，每个位置创建一个。例如，`at_trains = [1, 3]`，表示在耐、根的位置各创建1个分身。
- **分身来源**：卡组的支援卡，且**不包含友人卡**。
- **分配算法**：对于 `at_trains` 中的每个训练位置，随机选择一个不重复的支援卡分配分身
- **特殊规则**：
  - 即使某张卡在一开始的训练中**没有出现（判定为`未出现`），也可以创建分身**。
  - 同一训练不能存在相同卡的`Person`和分身（例如，耐位置原本已经有"怒涛(id:1058)"分身，则不能添加"怒涛(id:1058)"分身）
  - 不同位置可能出现同一张卡的分身（例如，耐、根位置原本均没有"怒涛(id:1058)"，则有可能各创建1个"怒涛(id:1058)"分身）

**方式(2) 超级拉面分身**
- **触发条件**：`超级拉面`且`支援卡种类>=4`时，每个支援卡固定额外出现一次
- **分身来源**：**包含友人卡**
- **分配算法**：出现的训练范围由`training_limit_options`指定，随机选择训练位置，分配失败则重新随机
- **特殊规则**：同一训练不能存在相同卡的`Person`和分身

### 诀窍槽获取算法
诀窍槽获取量是`基础值`和`训练加成`的和
但有一个例外：训练失败时不会获得任何诀窍槽.

1. 基础值
无论做什么动作，诀窍槽都会增加基础值.

- 基础值总和 base_sum

| 友人类型 | 基础值总和 |
|---|---|
| 新友人(id: 30305) | 10 |
| 旧友人 | 5 |
| 无 | 3 |

- 年初分配到三种类型(A/B/C)
重要！基础值会根据**每年选择的三个`地区拉面配方`的总`诀窍`消耗比例**，分配到三种类型上
计三个配方的总消耗为 `[A, B, C]`，基础值分配量为 `[a, b, c]`，且 A >= B >= C
a = round(A * base_sum / 15)
b = round(B * base_sum / 15)
c = round(C * base_sum / 15)
这只是初步计算。如果`a+b+c != base_sum`，需要根据实际情况调整。具体在实现时再细化. 通常只考虑新友人(base_sum = 10)的情况

ramen_memo里记录的典型的分配结果为：（左-总消耗，右-分配结果）
| 1位 | 2位 | 3位 |  → | 1位 | 2位 | 3位 | 地区  |
|---:|---:|---:|---:|---:|---:|---:|:-------|
|  8 |  6 |  1 |    |  5 |  4 |  1 | 新潟福島中京 |
|  7 |  7 |  1 |    |  5 |  4 |  1 | 札幌福島中京 |
|  8 |  5 |  2 |    |  5 |  3 |  2 | 札幌福島小倉 |
|  7 |  6 |  2 |    |  5 |  4 |  1 | 札幌中京小倉 |
|  9 |  3 |  3 |    |  6 |  2 |  2 | 福島京都小倉 |
|  8 |  4 |  3 |    |  5 |  3 |  2 | 札幌福島京都 |
|  7 |  5 |  3 |    |  5 |  3 |  2 | 札幌中京京都 |
|  6 |  6 |  3 |    |  4 |  4 |  2 | 札幌函館中京 |
|  6 |  5 |  4 |    |  4 |  3 |  3 | 札幌東京中京 |
|  7 |  4 |  4 |    |  5 |  3 |  2 | 札幌中山中京 |
|  5 |  5 |  5 |    |  4 |  3 |  3 | 中山中京京都 |

2. 选择"训练"时额外加成
- 每种训练会增加一个随机`类型`角标(A/B/C)，表示进行训练时，额外得到的诀窍类型
- 诀窍角标每种(A/B/C)至少出现1次（前3个训练位置打乱保证覆盖，后2个随机）
- 额外得到该类型诀窍槽的量为 `1 + 支援卡数量 + floor(NPC数量/2)`
    注意：支援卡数量不包括NPC，记者和理事长
- 如果为友情训练（彩圈）则三种全部再+2

3. 算例
- 基础值为 [3, 4, 3], 训练角标为C, 支援卡数量=2，NPC数量=3，非友情训练
    额外得到C类型诀窍槽的量为 `1 + 2 + 3/2 = 4`
    => 总诀窍槽增量为 [3, 4, 7]

- 基础值为 [2, 3, 5], 训练角标为B，支援卡数量=2，NPC数量=2，友情训练
    额外得到B类型诀窍槽的量为 `1 + 2 + 2/2 = 4`
    友情训练，三种类型都+2，但上限不超过7
    => 总诀窍槽增量为 [4, 7, 7]

----

## 训练计算公式

训练数值分为下层数值`lower_value`和上层数值`upper_value`

### 下层数值
- 不计算剧本加成，按基础公式 `default_calc_training_value` 计算出的训练数值
- 下层数值上限固定为100

### 上层数值
- 首先计算有剧本加成的总训练数值`training_value_ramen`
- upper_value = training_value_ramen - lower_value
- 之后进行上层数值上限约束，上限基础为100，随剧本buff增加
- 约束后，实际的最终训练数值 training_value = lower_value + upper_value

### 剧本加成
- 生效范围：ramen_pt_effect 常驻生效（**超级拉面期间不生效**）；ramen_basic_effect, ramen_region_effect 仅在**吃面后**，在 at_trains 标注的训练位置生效，不吃面时不生效
- 训练加成 xunlian: ramen_pt_effect, ramen_basic_effect, ramen_region_effect, 求和（超级拉面期间恒为 0）
- 友情加成 youqing: 来自 ramen_success_effect / ramen_fail_effect, ramen_basic_effect, ramen_region_effect 求和。仅在友情训练时生效，非友情训练时 youqing=0（超级拉面期间只来自 ramen_success/fail_effect + finals_effect.base）
- PT加成 pt_bonus：来自 ramen_region_effect（超级拉面期间来自 finals_effect.extra）
- 上层数值上限加成（**属性与 PT 各自结算；普通回合两者同值，超级拉面两者来源不同**）：
  - 属性上段上限 = `100 + status_limit`：普通回合来自 `ramen_basic_effect.status_limit`（Y2 +20 / Y3 +40），超级拉面来自选项（training_limit_options）的「獲得上限+100」
  - PT 上段上限 = `100 + pt_limit`：普通回合来自 `ramen_basic_effect.status_limit`（「獲得上限アップ」对属性/PT 同值生效），超级拉面来自 `finals_effect.extra` 的「SP獲得上限+100」
  - 超级拉面期间：属性上限走选项、PT 上限走 extra，两者各 +100，互不叠加
- 属性训练上层数值 training_value_ramen = lower_value * (100 + xunlian)/100.0 * (100+youqing)/100.0
- PT训练上层数值 training_value_ramen = lower_value * (100+xunlian)/100.0 * (100+youqing)/100.0 * (100+pt_bonus)/100.0
- Hint出现率：在分配人物时计算，不参与训练数值计算。基础值为 7.5%,随剧本Buff增加，例如+30就是 7.5* (1+30%)=9.75%
- Hint率 = base_hint_rate * (100 + card_hint_bonus) / 100 * (1 + scenario_hint_bonus / 100)
- scenario_hint_bonus = ramen_pt_effect.hint + ramen_success/fail_effect.hint
- hint_special表示“支援卡类型>=4时，除了友人、团队卡以外的所有支援卡都出现Hint，且训练后发动所有的Hint事件”，在事件逻辑里处理

### 拉面效果显示
- 普通回合：显示当前拉面的效果（包含基础效果和地域效果）
- 超级拉面回合：只显示 RMJ + finals 带来的加成，包括：
  - 友情加成（友情+）
  - 得意率（得意+）
  - PT加成（PT+）
  - PT上限（PT上限+）
  - hint率（hint+）
  - 分身数（分身+）
  （训练加成/失败率/羁绊/属性上限/hint全卡 属试食会基础效果，超级拉面期间不显示）

----

## 剧本数据 (scenario_ramen.json)

### 基本信息

- 剧本ID: 14
- 链接角色: 1022(美妙), 1058(怒涛), 1060(内恰), 1077(成田路), 1120(金镇), 9001(绿帽), 9008(B95)

### 剧本点数检测 (ramen_success_pt)

每年年终的剧本点数检测要求：

| 年份 | 1 | 2 | 3 |
|---|---|---|---|
| 所需Pt | 1500 | 3000 | 3500 |

第三年点数>=5000为大成功

### 吃面Pt获取

- `gain_pt_base`: 每次吃面基础Pt增量，每年 [300, 400, 500]
- `gain_pt_delta`: 随吃面次数叠加的修正值，根据年份每次 [30, 40, 50]，最多叠加5次

实际吃一次面增加量为 `gain_pt_base + gain_pt_delta * (吃面次数 - 1)`

### 诀窍槽上升量 (feeling_gauge_gain_base)

[3, 5, 10]

分别为：不带友人，带旧友人，带新友人（id: 30305）。为了简化，只考虑带新友人的情况

### 支援卡隐藏风味 (card_special_feeling)

| 卡ID | 获得量 |
|---|---|
| 10021 | 1 |
| 30021 | 1 |
| 10083 | 1 |
| 30052 | 1 |
| 30305 | 2 |

### 回合隐藏风味 (turn_special_feeling)

指定回合开始时获得的隐藏风味数量（回合从0开始）：

| 回合 | 数量 |
|---|---|
| 2 | 2 |
| 23 | 2 |
| 35 | 2 |
| 36 | 1 |
| 37 | 1 |
| 38 | 1 |
| 47 | 2 |
| 59 | 2 |
| 60 | 1 |
| 61 | 1 |
| 62 | 1 |

### 拉面基础效果 (ramen_basic_effect)

每年的基础效果词条：

| 年份 | xunlian | youqing | deyilv | fail_rate_drop | jiban | status_limit | hint_special |
|---|---|---|---|---|---|---|---|
| 1 | 15 | 0 | 0 | 30 | 10 | 0 | false |
| 2 | 15 | 30 | 0 | 50 | 0 | 20 | false |
| 3 | 15 | 45 | 0 | 100 | 0 | 40 | true |

说明：
- xunlian: 训练加成
- youqing: 友情训练加成
- deyilv: 得意率（本剧本三年均为 0；得意率加成来自 `ramen_pt_effect` 和 RMJ 结算效果，参见"训练分布的得意率计算"小节）
- fail_rate_drop: 失败率下降
- jiban: 羁绊增加
- status_limit: 上段数值上限增加（「獲得上限アップ」，属性与 PT 同值生效：Y2 +20 / Y3 +40；超级拉面选项则是属性专属 +100）
- hint_special: 仅第三年生效的特殊hint效果

### RMJ成功效果 (ramen_success_effect)

| 年份 | youqing | deyilv | hint |
|---|---|---|---|
| 1 | 5 | 80 | 30 |
| 2 | 10 | 120 | 75 |
| 3 | 25 | 250 | 125 |

### RMJ失败效果 (ramen_fail_effect)

| 年份 | youqing | deyilv | hint |
|---|---|---|---|
| 1 | 3 | 30 | 15 |
| 2 | 5 | 60 | 30 |
| 3 | 15 | 150 | 75 |

### 剧本Pt常驻加成 (ramen_pt_effect)

| pt_min | xunlian | deyilv | hint |
|---|---|---|---|
| 0 | 0 | 50 | 0 |
| 250 | 3 | 55 | 30 |
| 500 | 5 | 60 | 40 |
| 1000 | 8 | 63 | 50 |
| 1500 | 10 | 65 | 60 |
| 2000 | 12 | 68 | 70 |
| 2500 | 14 | 70 | 80 |
| 3000 | 16 | 73 | 90 |
| 3500 | 18 | 75 | 100 |
| 4000 | 20 | 78 | 110 |
| 5000 | 20 | 80 | 120 |

### 地区诀窍配方 (region_feeling)

每个地区的固定诀窍配方 [feeling0, feeling1, feeling2]：

| 地区ID | 地名 | feeling |
|---|---|---|
| 1 | 札幌 | [2, 2, 1] |
| 2 | 函馆 | [1, 2, 2] |
| 3 | 新潟 | [3, 1, 1] |
| 4 | 福岛 | [2, 3, 0] |
| 5 | 东京 | [1, 1, 3] |
| 6 | 中山 | [2, 0, 3] |
| 7 | 中京 | [3, 2, 0] |
| 8 | 京都 | [0, 3, 2] |
| 9 | 阪神 | [2, 1, 2] |
| 10 | 小仓 | [1, 3, 1] |

### 地区词条加成 (region_bonus)

随本年度剧本Pt增加的地区词条加成量：

[0, 3, 5, 7, 9, 10]

**档位计算**：每获得 **1000** 点剧本PT，提升一档（`region_bonus[pt/1000]`，上限 5 档）。档位对应关系：
- 0-999 PT：0档（加成0）
- 1000-1999 PT：1档（加成3）
- 2000-2999 PT：2档（加成5）
- 3000-3999 PT：3档（加成7）
- 4000-4999 PT：4档（加成9）
- 5000+ PT：5档（加成10）

> 口径校准见上节 `ramen_region_effect` 下方注释（旧述「每300点一档」有误，实测按 1000 点一档）。

### 训练基础值 (training_basic_value)

每个训练类型Lv1-5的基础值 [速, 耐, 力, 根, 智, SP, 体力]：

| 训练 | Lv1 | Lv2 | Lv3 | Lv4 | Lv5 |
|---|---|---|---|---|---|
| 速度 | [11,0,2,0,0,7,-20] | [12,0,2,0,0,7,-21] | [13,0,2,0,0,7,-22] | [14,0,3,0,0,7,-24] | [15,0,4,0,0,7,-25] |
| 耐力 | [0,10,0,3,0,7,-20] | [0,11,0,4,0,7,-21] | [0,12,0,4,0,7,-22] | [0,13,0,5,0,7,-24] | [0,14,0,6,0,7,-26] |
| 力量 | [0,6,9,0,0,7,-21] | [0,6,10,0,0,7,-22] | [0,6,11,0,0,7,-23] | [0,7,12,0,0,7,-25] | [0,8,13,0,0,7,-27] |
| 根性 | [2,0,2,11,0,7,-21] | [2,0,2,12,0,7,-22] | [2,0,2,13,0,7,-23] | [3,0,3,14,0,7,-25] | [3,0,3,15,0,7,-27] |
| 智力 | [3,0,0,0,7,7,5] | [3,0,0,0,8,7,5] | [3,0,0,0,9,7,5] | [4,0,0,0,10,7,5] | [5,0,0,0,11,7,5] |

合宿使用Lv5数值。

### 地域拉面效果 (ramen_region_effect)

每个地域对应的拉面效果，id从0开始（对应region_feeling数组索引）：

| id | 效果类型 | 数值 | 发动Hint数 | 训练位置(at_trains) |
|---|---|---|---|---|
| 0-4 | xunlian | 20 | 1 | 分别为 [0], [1], [2], [3], [4] |
| 5 | youqing | 10 | 2 | [0, 1, 2, 3, 4] |
| 6 | youqing | 50 | 2 | [2, 3] |
| 7 | youqing | 50 | 2 | [1, 3] |
| 8 | youqing | 50 | 2 | [1, 2] |
| 9 | youqing | 50 | 2 | [4] |
| 10 | youqing+pt_bonus | 50+50 | - | [0] |
| 11-14 | youqing+pt_bonus | 60+50 | - | 分别为 [1], [2], [3], [4] |
| 15 | youqing+pt_bonus | 40+50 | - | [0, 2, 4] |
| 16 | youqing+pt_bonus | 40+50 | - | [0, 2, 3] |
| 17 | youqing+pt_bonus | 40+50 | - | [0, 1, 4] |
| 18 | youqing+pt_bonus | 40+50 | - | [0, 1, 2] |
| 19 | youqing+pt_bonus | 40+50 | - | [0, 3, 4] |

**字段说明：**
- xunlian: 训练加成
- youqing: 友情训练加成
- pt_bonus: PT奖励加成
- hint_count: 发动Hint数量
- at_trains: 生效的训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）

### 超级拉面效果 (finals_effect)

超RMJ極的效果，分为基础效果、额外效果和单独效果：

**基础效果：**
- 体力(vital): +20
- 干劲(motivation): +1
- 赛后(race_bonus): +100
- 友情(youqing): +150

**额外效果**（ramen_success_pt达到5000时）：
- PT奖励(pt_bonus): +100
- PT训练上限(pt_limit): +100
- 全支援卡分身数(clone_count): +1

**注意**：超级拉面的分身 和 地区效果的分身 不同，地区效果是“指定训练增加一个随机分身，不包含友人卡，不计算得意率”，超级拉面是“每个支援卡增加一个分身到随机训练，包含友人卡，计算得意率”

**单独效果选项**（增加训练上限100）：
- 选项1: 速/耐/根/智 [0,1,3,4]
- 选项2: 速/耐/力/智 [0,1,2,4]
- 选项3: 速/力/根/智 [0,2,3,4]

----

## 实现说明（2026-08-16）

### 分离决策模型
动作采用分离决策模型，将吃面和训练分为两个阶段：
- **阶段1**：选择吃面决策（不吃面/吃面X/Y/Z）
- **阶段2**：选择基础操作（所有操作都可以）

这样可以大幅减少搜索空间（从40降到14），因为吃面后的分身是随机的，无法提前感知。

### 隐藏风味替换
做面时可以手动指定隐藏风味替换哪几种诀窍，通过 `special_targets: [i32; 3]` 参数控制。

### RMJ结算结果
使用 `RmjResult` 枚举表示结算结果：
- `Fail`: 失败
- `Success`: 成功
- `GreatSuccess`: 大成功（第三年 pt >= 5000）

### 隐藏风味回合表（已修正）
- 2, 24, 36, 48, 60 → 获得2个
- 37, 38, 39, 61, 62, 63 → 获得1个

### 固定触发事件清单（2026-08-19 补）

**回合开始时触发**（Begin 阶段，按 turn 选择）：
- `turn=0` 开始：400000400 马娘登场（来自 `global_events().story_events`，`player_select=false`）
- `turn=24` 开始：4009 经典年-新年（来自 `global_events().story_events`，`player_select=true`，3 选 1）
- `turn=48` 开始：4010 古马年-新年（来自 `global_events().story_events`，`player_select=true`，3 选 1）

**回合结束时触发**（NextTurn / AfterTrain 阶段，push 到 `unresolved_events`）：
- `turn=23` 末：401404 第一年RMJ（RMJ 结算后立即 apply）
- `turn=47` 末：401405 第二年RMJ（RMJ 结算后立即 apply）
- `turn=71` 末：401406 第三年RMJ（RMJ 结算后立即 apply）
- `turn=48` 末：4011 新年抽签（`system_events["ticket"]`，4 个 result 分支按 prob 加权选）
- `turn=77` 末：401407 育成结束 + 5011 ending（system_events） + 友人结束事件（ramen_data.friend_events["end"]）

**实现细节**：
- 回合开始事件：通过 `RamenGame::generate_events` 触发，遍历 `global_events().story_events` + `ramen_data.scenario_events` 的 Fixed 触发回合
- 回合结束事件：在 `add_mandatory_events`（turn=48 ticket, turn=77 ending+401407+友人结束）和 `next()` 的 RMJ 结算（turn=23/47/71）分支中 push 到 `base.unresolved_events`，由 AfterTrain 阶段统一消费
- **race_turn 短路修复**：`run_ramen_select` 的 race_turn 短路路径（turn=72/73/74/75/76/77）在 apply Race 后立即调用 `run_after_train` 处理 unresolved_events，再 `stage=NextTurn`。否则 AfterTrain 会被 next() 跳过，turn=77 的 ending/401407/友人结束事件漏触发

### 事件 player_select（2026-08-19 补）

`EventData.player_select` 字段控制是否由 Trainer 决策事件选项：
- `player_select=true`：调用 Trainer 的 `select_event_choice` 在 `choices` 间选择（如 4009、4010 的 3 选 1）
- `player_select=false`：直接选第 0 组选项，由 `apply_event` 按 prob/result 内部决定具体分支（如 RMJ 事件 401404-401406、抽签 4011）

`Game::run_event` 默认实现的判断条件：`event.player_select && event.choices.len() > 1`（不仅用 `choices.len() > 1`，避免把"单选项 + prob 加权"的事件错送给 Trainer）。

### RMJ 结算流程（2026-08-19 补）

`Game::next()` 的 NextTurn 阶段处理 RMJ 结算（turn=23/47/71），流程：
1. `check_rmj` 写入 `rmj_results`（success/fail），计算 train_level_bonus
2. 重置 `eat_count = 0`
3. `find_rmj_event(year_idx)` 找到 401404/401405/401406 事件，**立即 apply**（不是 push 到 unresolved_events 等下一回合）
   - RMJ 事件没有 `player_select=true`，直接 `apply_event(event, 0, rng)`
   - apply_event 中根据 `rmj_results[year_idx]`（true=result=2, false=result=1）选择对应分支
4. `scenario_pt = 0`（下一年重新累计 PT）

早期实现错误：把 RMJ 事件 push 到 unresolved_events，导致在 `turn=N+1` 的 AfterTrain 阶段才执行（晚 1 个回合）。修复为立即 apply。

### Trainer 事件选项决策（select_event_choice）

Trainer trait 的 `select_event_choice` 是 `select_choice` 的扩展接口，传入完整 `EventData` 便于上下文决策：
```rust
fn select_event_choice<G: Game>(
    &self, game: &G, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
) -> Result<usize>;
```

默认实现回退到 `select_choice`。MctsTrainer / HandwrittenTrainer 可按事件 ID / 选项 value 决定策略。

### 超级拉面自动恢复（2026-08-19 补）

`run_begin` 阶段对 `is_super_ramen_turn()` 的处理：
- 每个 URA 回合（turn=72-77）Begin 阶段：`self.uma.add_value(&ActionValue { vital: finals.base.vital, motivation: finals.base.motivation })`，即体力+20、干劲+1
- 仅一次性：调用 `self.apply_super_ramen_saihou()`（幂等，`turn>=72` 且已选超级拉面时 `uma.race_bonus += finals.base.saihou`，合计只加一次，避免持续增长）；协议重放路径另在 `into_game` 补调同一方法
- 实现位置：`crates/umasim/src/game/ramen/game.rs` 的 `apply_super_ramen_saihou` / `run_begin`；重放补调见 `crates/umaai/src/protocol/ramen.rs` 的 `into_game`