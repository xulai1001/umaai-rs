# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。记录应尽量精简，每条修改一行，不包含代码细节。

## 2026-08-27（本轮）
- **五维属性上限剧本化**：上限基值改为随构造参数传入（`Uma::new` / `BaseGame::new` 新增 `limit_base`），顺序固定为"先写剧本基值、再加继承"，三个剧本各自从自己的 `scenario_*.json` 取值，`constants.json` 同名字段降级为 basic 与缺字段兜底。原先"先写全局值、再由各剧本事后修正"的打补丁式设计全部删除——拉面的整体赋值发生在累加开局继承之后，会把继承增量擦掉；温泉的 `min(2800)` 是速度基值 2600 时代的防御值，基值提高后变成硬截断，且在继承事件后还会再截一次。补丁写法本身就是这两个缺陷的来源，新剧本照抄必然复现。温泉基值补入 `scenario_onsen.json`（此前无该字段，一直吃全局值再被截断）。**改变拉面与温泉模拟数值，基线作废**
- **终局评分查表口径统一**：新增 `GameConstants::status_final_score`，越界一律饱和到表末。此前三处消费点行为各异——裸下标越界 panic、`unwrap_or(0)` 越界静默返回 0。后者最坏：属性增益按查表差分计算，返回 0 会让该维收益变成巨大负值，手写策略永久回避该维且不报错。评分表长度有限而上限＝剧本基值＋继承三次，蓝因子拉满即可越界。顺带修 `status_gain` 中负增量 `as usize` 回绕溢出（当前取值恒正打不到）
- **上限相关守门与契约测试**：新增跨三剧本的开局上限守门测试（期望值从各剧本 JSON 推导，故改代码会红、改数据不误报）、剧本基值字面量契约测试（守数据漂移，并锁两剧本基值必须不同——拉面与全局常量当前数值相同，误接全局的回归只有它能抓）、查表越界饱和测试。`expected_score_parts` 保持不调用生产查表函数，维持独立对照。修正 `eat_covered_train_gate_blocks_mismatched_ramen` 夹具写死旧上限当"满"的问题，改为从实际上限取值；三处硬守门快照基线随上限变化重抓
- **MCTS rollout 与 fallback 切到正式推荐策略**：搜索评分原用机制残缺的策略核心，改为 `RecommendedRamenTrainer`，门控全关时与推荐策略逐位等价；`for_rollout()` 关 breakdown 采集避免线程锁争用。附两个切换验证工具
- **搜索掉分归因**：搜索低于纯推荐策略的原因是缺省 `radical_factor_max=50` 的加权均值有效样本量恒 3.9%（与 search_n 无关），选择偏差压过搜索收益；取 rf=0 后方向反转。缺省值不动（温泉在用），由调用方指定
- **硬守门测试基线重抓与收紧**：随 trainer 切换，4 处分数/五维/PT/searched_count 基线重抓；`test_combined_on_skips_special_search` 的重搜断言改回逐位快照 + 占比上界（先前放宽成「调用数 > 搜索数」，29 次调用搜 28 次也绿，等于没有守门）；`for_rollout` 补「与 `new()` 决策逐位相同」守门，并把 `last_year` 的锁写入一并关掉
- **rollout 加速 −29% CPU**：复用现有 diag feature，编译期消掉 rollout 路径上 5 处 explain 类屏幕输出调用；分数逐位一致。**仅在关 diag 时生效**——`default` 含 diag，`umaai` 已 `default-features = false`，umasim 自己的 bin 需显式 `--no-default-features --features cli`
- **perf 诊断工具与 Windows 可构建性**：新增 `sim_profiler` 定位打分链热点；pprof-rs 用 Unix API、Windows 编译不过，故收进可选 `profiler` feature 不进 default。`microbench_top_fns` 加 `#[ignore]`——它改进程级 CWD，与并行测试互相污染

## 2026-08-26
- **吃面后必训练 at_trains 覆盖位（C 方案）**：新增 `LocalRamenConfig.eat_requires_covered_train`（推荐 preset 开启）——`decide_ramen` 对每个吃面候选预演"落地后最优训练位"，不在该面 `at_trains` 内则否决，实现"吃面后必训练覆盖位、不训练就不吃面"。吃面训练覆盖实测 80%→99%，总分与技能点双升。**改变拉面模拟数值，基线作废**
- **弱位 boost 补"未满"条件**：`ramen_weak_train_boost` 与 `ramen_window_alignment` 的弱位放大仅在 `five_status < limit` 时生效——已满位只剩 PT 收益，放大只会虚高训练分。**改变拉面模拟数值**
- **地区选择弱位覆盖参数 + 配置覆盖修复**：`score_region` 新增 `region_weak_cover_weight`（默认 0.0，实验入口）；game_config.toml 顶层 `ramen_region_strategy/fixed` 覆盖修复（字段须写在所有 `[...]` 段之前，原注释位置被 `[mcts]` 段吸收导致不生效）

## 2026-08-26
- **搜索终局多维记录（P2）**：rollout 返回值扩为 `RolloutOutcome<T>`，新增 `search_with_terminal` 与 `MomentResult` 按候选累加终局观测量；`CandidateAccum` 收拢三条统计使其只在成功分支推进；UCB 失败计数统一末尾告警。**纯观测出口，模拟数值逐位不变**
- **拉面终局 25 维与诊断出口**：在 rollout 内部归约阈值类维度（PT 达成率等），避免均值丢信息；RMJ 直接读规则层；维度键名与顺序冻结（FROZEN_DIM_KEYS + 守门测试），合作伙伴用于手写策略前后对比
- **超级拉面纳入搜索**：补 `SuperRamenSelect` 阶段分支，新增 `Operation::SuperRamenSelect`；手写与 Local 同步补分支避免默认分支静默换选项。**门控默认关闭**
- **第 1 年地区纳入搜索**：拆出 `BeginAfterRegionSelect` 阶段边界，回合 2 走 `Begin → RegionSelect → BeginAfterRegionSelect → Distribute`；修 `encode_regions` 未选出时被编三份「地区 0」。**门控默认关闭；`all()` 语义变真，历史基线作废**
- **超级拉面搜索平局回退**：`deck_can_split == false` 时改为仅在确实平局时向选项二回退，判定跟随 `selection`
- **地区候选生成抽为纯函数**：`region_select_combos` 显式传参，守门测试直接调它，避免 `test_year1_2_always_all_regardless_of_strategy` 空转仍绿
- **补回 `test_combined_gate_off_full_game` 的 `#[test]`**：上次提交插入观察壳占用属性行导致该测试静默不运行，加静态扫描核对
- **拉面 MCTS 诊断出口接线**：主二进制单局开启 verbose，补 `#[ignore]` 整局观察壳；观察壳须自行设日志 info 级

## 2026-08-25
- **自由比赛收益真实衡量**：`race_grade_weight`（等级×常数）退役，改走训练同管线折算（真实收益 + 赛程压力叠加）；折扣经实测削弱至 0.3。**改变拉面模拟数值，基线作废**
- **bench handwritten 档切到正式推荐策略**：自动局表现失真，改为 `RecommendedRamenTrainer`；核心保留作 rollout 组件对照
- **方案 E 确认 PT 不打折**：残余折扣只作用于副属性，PT 独立计分；单点启发式无法观测的跨回合项留给 MCTS
- **拉面五维上限硬截断移除**：speed 恢复 3100，玩家高分档不再受 2800 截断拖累；bench 强制地区策略 All 不受手动模式影响
- **弱位训练偏好 + 按 build 自适应查表**：双层级（吃面前 / 吃面后）放大 at_trains 卡少位 raw；按智卡数查表（推荐 preset 默认启用），build 异质性极强
- **体力门限上调（30→40）**：300 局配对总加权 +397（7/7 build 正），失败率 1.5%→0.3%；y3 门禁改为每年评估，仅第三年吃面放掉硬门限
- **支援卡连续事件增强（用户手动）**：8001/8002 事件数值上调（体力 5→10、五维/PT/hint 增强）
- **地区权重重新评估**：当前策略下 300 局配对，`region_youqing_weight` 1.0→1.5（speed Y3 +387）
- **友人词条加成 + 主动使用**：词条 bonus（体力×1.6 / 属性×1.3），不溢出时主动用友人；失败率 2.4%→1.6%，友人 4.9/5
- **残余收益折扣（方案 E）**：主属性快满时副属性打折（PT 保留），300 局 +84
- **手写策略四项提分机制**：吃面联动 / 必成价值 / 友人饥饿 300 / 动态属性平衡，100 局 +749
- **地区选择修正公式 + 验证**：`bias×youqing - waste×10`；全 101 种验证：真实 build +99.9 / 残缺 -7.3
- **region_matrix 诊断工具 + test_region_selection_per_build**：按 build 打印三年选区 + 占比；7 build × 3 年人工审查
- **LocalRamenTrainer 补齐第 1 年地区选择打分**：不再恒选候选 0；基线作废
- **拉面动作空间不变量 + 终局分分解（MCTS P0 安全网）**
- **搜索层拉面合并动作落地（P1.1+P1.2）**：一次搜完 ramen×targets；拉面基线作废
- **拉面搜索阶段缺省补 `ramen`**：42 局配对 +2306
- **测试有效性审查修补**：缺省守门测试、结构恒等式、删无效测量壳
- **不在判定与得意率解耦**：distribute_person 两步算法，缺席名单入 RamenState
- **地区拉面分身缺席优先**：缺席卡优先补分身位；拉面基线作废

## 2026-08-24
- **训练人数加成按人头类型计数**：`1 + 0.05 × 人数` 乘区改按 `PersonType` 判定（替代硬编码下标），抽出 `count_training_persons`，负数与越界下标一并不计。**改变拉面模拟数值，基线作废；温泉与 base 逐位不变**
- **超级拉面分身补上友人卡**：候选收集改全扫全体人头（不再写死卡组下标范围），同时加「每训练一个友人」约束。**改变拉面模拟数值**
- **RecommendedTrainer 改进方案文档**：新增 `workbench_improve_1.md`，规划地区打分三指标、第三年体力门禁回合差异化、`matrix_variant` DSL 重构三件事。**文档规划，未实施代码**
- **配置层三处接线修复**：`[mcts]` 改全 Option + `deny_unknown_fields`；主二进制 onsen 改调既有 `SearchConfig::new_game_config`；`expected_search_stdev` 补注为 UCB 缩放标尺非实测统计量
- **搜索层 CRN 与 UCB 三处修正**：CRN 对照轴改按「候选间是否共享 `rule_master`」分臂（双种子 rollout 入口拆开决策流与规则主种子）；失败样本改按原始序号交集配对；UCB 首组步长收进 `search_n`。**生产语义与分数逐位不变**
- **拉面规则层四处数值修复**：分身分配改合法集直选（消除概率重试假失败）+ 按回合派生局部流使策略流消耗归零；训练人数加成改按人头类型计数；超级拉面分身补上友人卡与「每训练一个友人」约束。**改变拉面模拟数值，基线作废**
- **拉面杯逐年观测出口**：`scenario_pt` / `eat_count` / 地区选择改归零前按年归档，CSV 换逐年三列。**纯观测出口，模拟数值逐位不变**
- **第三方库引用规范化（续）**：bench 模块中 anyhow 宏的全名引用改 use 导入

## 2026-08-23
- **拉面杯 MCTS 训练员**：按阶段门控的搜索训练员，命中的决策点走扁平搜索、其余转发手写策略，门控全关时与纯手写逐位一致
- **拉面局面特征编码器**：新增 features 模块，把局面编码为定长向量（global / cards / persons 三段），较温泉版补齐成长率与属性上限并开启人头分支
- **人头下标与卡组槽位解耦**：拉面下人头顺序与卡组顺序不一致，原先按 person_index 直接当卡组下标的调用点全部改为按 card_id 反查。**改变拉面模拟数值，基线与落盘教师数据作废**
- **手写策略地区打分覆盖第 1 年 + build 自适应**：新增有效阶段判定使回合开始阶段内联触发的第 1 年地区选择也进入打分；`score_region` 纳入 youqing 项并按卡组 bias 统一缩放。**改变手写策略基线数值**
- **测试观测收集器**：新增 `utils::Checks`，测试全程 println 记 OK/NG、末尾汇总有失败才报错；既有裸断言与重复本地实现一并归拢

## 2026-08-22
- **基准新增自选比赛达标维度**：新增任意时点重比各区间完成场数的判定（原判定只在区间结束回合的下一回合执行，且不达标即终止育成），bench 结果与 CSV 加达标率并在每局 / 分组 / 总览打印；配套补两个守门测试（不改策略逻辑），逐回合扫描触发点以免随常量表调整失效
- **搜索层可复现 + 真 CRN + 泛型化（NN 管线 Phase 1，已完成）**：rollout 种子改为按序号确定性派生（候选索引不参与，否则协方差归零），移除全部随机播种，失败由静默丢弃改为计数告警；新增按阶段边界重播种的真 CRN（默认开启，可从 toml 关），实测朴素共享起始种子几乎无收益、按阶段重播种才显著；搜索结构泛型化并保留默认类型参数使活跃入口零改动，采用「公共内核 + rollout 闭包」规避泛型方法解析导致温泉特判静默失效；顺带修 NN leaf 微批路径漏重播种、UCB 终止判据用成功数会死循环两处缺陷，并把 rollout 基策的调试缓存改 Mutex 以满足跨线程共享
- **局面采样器（NN 管线 Phase 2 上半）**：为教师数据制造根局面——分层的采样空间、按工作项序号确定性导出采样任务（分片 / 续跑 / 改并行度均不变）、轨迹随机扰动、走真实决策路径截断捕获；根局面限定在阶段入口，回合开始阶段内联执行的决策点会破坏搜索的阶段推进契约
- **第三方库引用规范化**：搜索层与采样器中 anyhow 宏的全名引用改为 use 导入后直接调用
- **支援卡类型注释订正**：card_type 原注释与卡片数据实测相反（5 是友人、6 是团队）

- **RNG 受控重构（v3 三流，已实施）**：新增顶层 `rng.rs`（splitmix64 唯一实现 / 加法派生无状态流 SplitmixRng / 类型隔离三流 TurnFixedRng+EventRng+StrategyRng）；规则层随机改从 self 流取（run_distribute 独占局面流=角标/人头分布/hint 触发位，回合开始事件链走事件流，训练/分身/比赛走策略流），Trainer 决策流保持 StdRng；bench 局号进种子 `seeded_rngs(base,idx)→(StdRng,rule_master)`；拉面 CRN 由规则层接管（fork_for_rollout 注入 rule_master，simulate_common 退役阶段重播种），onsen 保留外挂 CRN；未注入 rule_master 时回退旧行为。验收：层 2/3 集成测试 `rng_consistency.rs`——跨策略 20 回合角标/分布/固定流消费量逐位一致（0 不一致），事件增量逐位一致；方案文档 `rng_refactor_plan.md` 更新为 v2/v3 并归档 v1，`rng_reply.md`（上游 CRN 评审意见）归档
- **umasim 主二进制接入拉面杯剧本**：main.rs 此前仅支持 onsen/basic（`scenario="ramen"` 时实际落 basic），新增 `run_ramen_once` 与 ramen 分发分支（random/handwritten/mcts 回退/默认 manual 均支持），handwritten 分支使用 RamenHandwrittenTrainer；`GameConfig::scenario` 注释补 ramen。实测主二进制跑通 77 回合拉面杯（UB2 49442 / PT 7941）
- **issues 更新**：第三年地区选择无 build 自适应（score_region 对第三年地区无区分度，实测各 build 同选一组合；方案已定待实施，含临时验证测试）
- **ramen_manual 屏幕输出整理（Agent 对话文本流风格）**：新增 turn_flow 渲染层与固定种子基线测试；候选内联预览（训练数值 / 吃面完整效果 / 诀窍配方）并分层着色；事件三段式、回合状态去重；ramen_manual 接入实时候选栏与选择确认；训练诊断输出暂屏蔽
- **第3年地区选择修复**：ramen_region 配置字段落错 TOML 段导致预设失效（恒枚举 120 组合），移回顶层后 fixed 预设生效
- **comfy-table custom_styling**：修复彩色表格 ANSI 宽度错乱
- **自选比赛守门 + 决策日志 breakdown**：等级过滤 / 摆烂判定 / 达标后停止，候选评分分解入决策日志
- **诀窍槽 NPC 按实际人数计算**、game_config.toml 加载修复、cargo-husky 撤销与 fmt 手动化、bench 玩家 build 外置与分组跑批
- **显示微调（用户）**：比赛加成信息亮品红；清理未使用 import
- **文档归档**：config_refactor_plan / log_refactor_plan 移入 archive

## 2026-08-21

- **bench 设施与全卡型基准**：新增 `umasim::bench` 公共设施（双 RNG 分裂 / 单局运行 / 统计 / CSV / 代表性选卡）+ `bench_compositions`（101 种卡组构成跑批），bench_base / bench_compositions 复用瘦身
- **手写策略规划文档**：新增 handwritten_policy 目录：定位（MCTS rollout 基策）、策略形态（参数化利于调参）、输出分层（决策日志 / DecisionInfo / GameView）、玩家经验标签
- **手写策略三步交付**：① 地基：bench_base + 决策日志 + 规则层可复现性修复（Random 基线 mean=30432）② 核心：RamenPolicy 各阶段打分 + RamenHandwrittenTrainer（较 Random +39%）③ 自选比赛守门 + 打分自洽性修正（实测 +18.5%）
- **rustfmt 规则固化 + AGENTS.md 微调（用户）**：明确 Nightly 格式、stable 禁跑 cargo fmt；需求澄清与安全注意事项表述精简

## 2026-08-20

- **注释精简**：umasim/Cargo.toml 注释 38→14 行；Rust 长注释压缩 6 处（文件头、重复的 1121 维清单去重），保留 13 处高价值文档（公式 / 索引映射 / 机制契约）
- **colored 无条件加载**：colored 从 cli feature 移出改为无条件依赖（非 Windows 纯 std 实现，Android / 嵌入式交叉编译无风险），消除 9 个文件约 20 处彩色双版本 cfg gate 重复代码；no-color 编译期无色语义不变
- **Phase 4 步骤1：依赖边界整理 + feature 拆分**：删除 analyzer crate；umasim feature 三层设计（default = cli + diag，新增 no-color / onnx）；15+ 文件 cfg gate 治理；nn 模块整体 cfg gate 到 onnx；umaai 依赖瘦身（去掉 tract-onnx）；四种编译组合通过；暂不抽 umasim-core
- **日志模块重构（Phase 3）**：新增 output 模块（diag! 宏 / GameView）；142 处规则层日志迁至 diag!；GameView 扩至 8 字段并删除 disable_log / enable_log；LOGGER 锁合并为 OnceLock，release 编译零 warning
- **测试日志简化**：新增 init_test_logger（只输出 stderr 不写文件），100+ 处测试迁移
- **友人事件词条生效修复**：apply_event 应用"事件效果提高 / 恢复量提高"词条，三剧本统一生效
- **排名数据补全**：rank_scores / rank_names 补齐至 LS24，速度档位上调
- **第3年地区选择默认 Fixed**：走固定组合 [[11,14,15]]，跳过 120 组合枚举
- **拉面杯回合规则收紧**：回合 0-12 无自选比赛；回合 0-1 与超级拉面回合跳过吃面阶段
- **其他**：友人高羁绊概率 0.3→0.25；ramen_manual 改密码学随机种子；新增 tests_overview.md

## 2026-08-19

- **吃面效果立即落地**：选完面与隐藏诀窍用法后立即消耗诀窍、效果生效并生成分身，玩家选训练前可见完整 buff
- **hint_special 全员触发**：第三年吃面且支援卡种类达标时，相关训练位置全部支援卡强制出 Hint
- **ManualTrainer 玩家测试**：支持真实终端交互与 mock 两种模式；新增完整 77 回合与 hint_special 路径的端到端测试
- **修复并发测试日志初始化竞争**
- **配置系统 Phase 2**：用户可调项迁至 default_config.toml（步骤1）；GameConfig 五子配置分组（步骤2+3）；配置加载集中化 + 统一校验（步骤4）；拉面杯第3年地区选择策略接入 PolicyConfig + TOML 精简（步骤5）；文档收尾（步骤7）
- **文档整理**：project_context 按实况更新，旧 issues 归档

## 2026-08-18

- **剧本 PT 每年归零**：RMJ 结算后归零重新累计，URA 阶段不再累计
- **RMJ 事件时机修正**：结算当回合立即触发；超级拉面基础效果 URA 回合自动生效（赛后加成仅首次）
- **事件补全**：RMJ 结算成功 / 失败事件 + 固定触发事件（登场 / 新年 / 抽签 / 结局），修复比赛回合事件漏触发
- **训练分布剧本得意率加成修复**（含 RMJ 效果）
- **夏合宿规则实现**：诀窍槽全 MAX、禁用普通 / 友人外出与治病、休息自动清除不良状态
- **决策重构**：新增"选面 + 吃法"一次性合并决策接口；动作阶段扩展为"选面 → 选诀窍用法 → 训练"三阶段

## 2026-08-17

- **umaai 跨平台构建支持**：可在 Ubuntu / Linux 下编译运行（Windows 专用依赖按平台限定）
- **拉面杯模块机制修正、显示改进与架构重构**：友人事件 / 分身系统 / 地区选择 / RMJ 结算 / 超级拉面 / 诀窍角标等
- **训练数值端到端观测测试**：固定回合打印吃面 / 不吃面场景的训练分布与数值

## 2026-08-16

- **拉面杯模块 1d 最小闭环**：回合 0-77 完整阶段流转、组合动作生成、事件处理、动态人头管理、回合边界处理
- **1b 核心游戏机制 + 1c 动作预览和手写策略**：诀窍 / 做面吃面 / RMJ 结算 / 地区选择 / 分身 / 隐藏风味 / 友人事件；"吃面选择 × 基础操作"分离决策模型
- **1a 核心类型定义 + 1b-1 诀窍系统**：拉面杯模块结构与核心类型；诀窍槽基础值分配、库存溢出、训练 / 友情加成
- **拉面重构计划调整**：Phase 合并为 1a-1d，归档旧规划文档、统一领域术语（食材→诀窍等）

## 2026-08-15

### 拉面剧本机制完善

- 补充友人解锁机制、诀窍槽算法、分身规则等核心机制文档
- 补充剧本机制初始化规则（第2回合开始时）
- 补充夏合宿规则（训练等级、事件触发）
- 补充超级拉面期间限制（不可吃其他面）
- 更新gamedata数据：调整事件概率、添加地域名称、完善超级拉面效果
- 更新AGENTS.md项目规则：完善提交规范和工作流程
- 添加ramen_story_flow.md拉面剧本流程文档
- 更新术语表：添加诀窍槽、友人解锁、复合宿等新术语
- 整理文档目录：将规划类文档移至opt子目录

## 2026-08-14

### 拉面剧本事件数据补充

- 在scenario_ramen.json中添加scenario_events和friend_events数据
- 更新RamenScenarioData结构体，添加对应的事件字段
- 添加单元测试验证事件数据加载

### EventData触发类型重构

- 新增TriggerType枚举：Random/Code/Fixed三种触发类型
- 移除EventData中的start_turn/end_turn/max_trigger_time字段
- 更新JSON数据文件和触发逻辑代码

## 2026-08-13

### 文档整理

- 创建了AGENTS.md项目规则总结文档
- 在.trae/documents/目录下整理相关文档

### 测试规范完善

- 在umasim::utils中新增get_workspace_root()函数，用于获取workspace根目录
- 修改了多个测试文件，在测试中使用get_workspace_root()切换到workspace根目录

### 拉面剧本数据完善

- 更新ramen_basic_effect：添加jiban/status_limit/hint_special字段，填充3年效果数据
- 添加finals_effect：定义超级拉面(含RMJ成功)的基础/额外/单独效果
- 添加ramen_region_effect：记录20条地域拉面效果数据
- 更新Rust结构体：添加RamenBasicEffect结构体
- 更新ramen_memo_cn.md文档：补充效果说明和字段定义
