# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。记录应尽量精简，每条修改一行，不包含代码细节。

## 2026-09-24
- **新增拉面杯神经网络可选项**：`ramen_trainer_policy` 可选 `mcts`（默认，行为不变）/ `mcts_nn_hint`（执行不变，每步另显示网络的推荐）/ `nn`（整局动作决策交给网络、不搜索，以分数换速度）；umaai 新增 `onnx` feature，未开 feature 或缺模型时启动报错
- **网络模型加载加固**：加载时校验 ONNX 图输出契约，并按固定 batch 编译（更快，输出逐位不变）
- **无搜索评分决策的来源标注**：网络模式下的决策照实显示为「神经网络」或「自选比赛守门」而非「手写逻辑」；luck 挂载改为合并决策自带的附加信息而不是覆盖
- **测试**：新增生成式 ONNX 测试模型，网络装配与客户端输出流程测试不再依赖未入库的权重；依赖真实权重的测试改为默认忽略、显式运行缺模型即报错

## 2026-09-23
- **persona 约束微调（用户）**：执行偏离一律只作校验说明、不数落部下（不再限于 AIRedirector 自动局）
- **执行推断训练阈值下修 13→12**：新局数据（game1444）验证发现开局低等级训练的属性增量可低于文档实测下限（turn 0 智训练 +12 被误判未知），下修后该局一致率回升且原校准局不受影响；另将「继承回合前一回合的执行窗口受继承落地污染」记入口径速查的已知限制
- **新增单局复盘分析引擎 `umaai_review`（crate + bin）**：对局包解包与角色识别（两种布局通吃）、快照只反序列化取字段、决策 CSV 按表头解析与链推断、gamedata 多级解析与降级，产出 digest.json（紧凑强类型 schema）与 report.html（minijinja 外置模板 + 自绘 SVG 三图三表，零 JS）
- **实际执行动作推断**：AI 建议与实际执行逐回合对照，分类口径经实测校准（必赛回合兜底判比赛、主增量阈值判训练以覆盖智训练不耗体力、体力回升判休息），game6234 一致率与文档校准吻合，偏离全部为真实偏离
- **检查项引擎**：伪波动标记（年界 / 继承 / RMJ 自动检测 / 开局第 1 年地区选择；turn 72 双属性既标记也计入超拉统计）、超级拉面期盈亏判定、坏手法已验证三项与训练失败候选清单；运气分读法口径写入 digest（已知 bug 与固定波动区）
- **继承质量分析**：窗口取前一回合末至继承回合首并剥离前一回合行动贡献，game6234 两次继承实测与文档逐位一致
- **分身彩圈观测口径修正**：彩圈判定改为分身新增落位（剔除本体占位假象）、A/B 两类分开统计、B 类只统计训练卡，有效彩圈按随机（好运气）/ 规则（好策略）二分来源
- **新增 Trae skill `umaai_review`**：六问通俗化叙事框架、马娘玩家术语优先、归因口径（输赛掉干劲与超拉连亏归运气）、语气基调（LG 档鼓励、坏局减负、继承根因点到为止）、篇幅纪律；附口径速查与可选秋川理事长人设（删除即回退默认口吻）
- **文档同步**：replay_review.md 更名与口径对齐实现现状（CLI、彩圈、伪波动、报告结构、进度）；project_context 补 crate 与 skill 结构
- **决策理由显示新增「候选高于首选」颜色档（用户）**：候选均值高于中选策略时红底亮黄高亮提示估值不一致，首选行样式同步调整，附 colored 输出临时验证脚本；text_data_dict 数据更新（用户）
- **执行推断口径升级（用户）**：训练判定改分年分段阈值（第 1 年取训练基础值表最低主维，第 2 年起固定 12）并要求非大回复，休息判定体力阈值提至 +25，继承窗口锚点标注「继承混合」不参与一致率；replay_review.md 口径段同步
- **A 类彩圈吃到判定**：分身明细新增 used 字段——彩圈新增回合的当回合训练命中彩圈位即吃到收益；训练匹配改前缀匹配以兼容「继承混合」后缀（此前漏判）
- **五维显示值换算**：timeline 全序列新增显示值字段（真实值超 1200 的部分减半折算），报告概览卡改用显示值口径
- **报告模板瘦身 + 叙述回填机制**：移除决策明细表与检查项表、口径说明简化为速览；模板留 4 个叙述占位标记，由 skill 回填总体 / 运气走势 / 检查项与偏离 / 总结四块（继承与分身叙述写在数据表后），卡片改为可折叠默认展开
- **skill 叙事纪律迭代（多轮用户反馈）**：全胜战绩不提、安抚性元话语不出口、玩家已知口径不解释、自选比赛两类分开、干劲恢复写具体动作、超拉彩圈只报汇总；结构化数据表格化、口径备注分条、运气数字带符号、digest 读取踩坑入档；persona 精简并同步元话语禁令
- **报告背景图装饰（用户素材）**：yayoi.png 入 skill reference；skill 回填报告时铺 20% 透明度背景、前景卡片调 70% 透明度
- **测试修复**：pack 测试临时目录并行互删竞争改为按测试命名隔离

## 2026-09-21
- **修复协议层 `failureRateBias` 语义反问题**：解析侧 `failureRateBias < 0 → good_trainer=true`（曾与上游相反，buff 被读成"不擅长训练"，MCTS 估值时 `calc_training_failure_rate` 加 +2 失败率人为偏高）、导出侧 `good_trainer → -2` 同步反向——与 `traits.rs:calc_training_failure_rate` 内部 `good_trainer → bias=-2` 同源；game421 turn 25+ 反复推「不吃面+休息」（vital 88~108 全休息）的根因，实测 turn 61 `rest_pair_probe` 不吃面领先 +351 → 落后 −449，turn 62 +645 → 落后 −223。补回归 `test_failure_rate_bias_parse`（三组 frb/-2/+2/0 钉解析方向）
- **新增在线单局决策日志的高体力休息审计脚本**：`scripts/analyze_rest_picks_online.py`（与 `scripts/analyze_rest_picks.py` 互补：前者跑批决策日志、本工具吃 `logs/game{id}/decisions.csv` + 回合 thisTurn.json 还原 vital），决策时体力从快照反查；与 `rest_pair_probe` 共同补齐「高体力休息」类排查链路
- **新增 `rest_pair_probe` 单回合诊断 bin**：对单回合 thisTurn.json 跑两遍 FlatSearch——A) 完整合并候选（含「不吃面」），B) 仅吃面候选（屏蔽不吃面）——CRN 共享种子 4096 rollout，输出两组的每候选 mean / n / 选中，用于隔离"不吃面 vs 吃面"在终局估值上的真实差距（game421 turn 61 修复前 +351、修复后 −449）
- **合宿/满体力「一选休息」回归复测（未复现）**：spd2_sta0 / speed_wisdom ×10 局（search_n=4096 生产口径、同种子）：合宿期 MCTS 无整局全休息、高体力（>休息目标线）休息仅约 0.35% 决策点、手写对照零次——9-17 合宿诀窍修复后未复现；新增 `scripts/analyze_rest_picks.py` 审计入口（合宿回合/体力分桶休息率，与规则层判定同口径）
- **决策日志/跑批观测增强**：决策日志追加「决策时体力」列（开发格式列尾演进，手动录制路径占位）；bench_base 新增 `--builds` 过滤（与 `--deck` 互斥）；bench mcts 档补接友人完成硬门限（与在线生产同口径，原漏接）
- **友人出行跨年配额定档 `[0,3,5]` + 新增「5 次必须走完」配置项**：preset 由 `[0,2,5]` 改为 `[0,3,5]`（第 1 年不启用、第 2 年放宽到 3 以消化提前的休息替代、第 3 年补满）；新增 `friend_complete_required`（`game_config.toml` / `gamedata/default_config.toml`，默认开）＝完成硬门限，开启时隐藏风味闸门不再阻断出行、剩余次数达到剩余可出行回合数即强制出行，保证 5 次走完；该开关经 `main.rs` 同时作用于 MCTS 的 fallback 手写策略与搜索 rollout 基策，`bench_base` 同口径读取。实测（700 局/单元同种子配对）：走完率 77%/92%→99.6%/100%，相对旧 `[0,2,5]` 配对差 −114/−86（不显著），硬门限项自身净代价 −159/−35
- **新增友人出行实验 token 与观测列**：token `fcap`（跨年配额）/`fov3`（第三年风味闸门）/`furg3`+`frem3`（第三年强制补足）/`freq`·`freqoff`（完成硬门限）；决策日志给友人出行加「决策路径（恢复/常规）+ 次优动作与分数差」标注，用于区分"替换休息"与"替换训练"；结果 CSV 新增逐年友人出行次数、用满标记、隐藏风味溢出浪费等观测列
- **重抓 preset 变更影响的整局快照**：`test_yearly_observability`（score/五维）、`test_ramen_three_stage_action_unchanged`（rollout 均值表）、MCTS gate-off 整局快照与 SpecialSelect 调用数、两处超级拉面选项断言；跨年配额单测改为钉 `[0,3,5]`

## 2026-09-19
- **MCTS vs 手写整局配对基准（固化评估入口）**：新增 `ramen_mcts_pair_bench` bin——同 (build, 种子, 局号) 下 MCTS 训练员与正式推荐手写策略各跑整局（共享规则主种子强行配对，两局 `rule_seed` 不一致即报错），配对差 Δ = 评分_mcts − 评分_handwritten 逐局落 CSV + 按 build/全局的均值、SE、95% CI、胜负汇总；MCTS 参数默认**取生产实际值**（game_config `[mcts]` + `ramen_search_stages`，与在线构造同款），`--search-n` 等覆盖仅限对照实验；bin 内冒烟测试（配对守卫 / 同参两次逐位可复现 / CSV 结构），project_context / tests_overview 同步
- **experiments/validated_policy 迁至 scripts/validated_policy**：`scripts/` 是官方脚本区，挪过来与 `plot_*.py`、`bench_commit_compare.py` 同级；内容不变
- **生产训练员门控含 region**：`gamedata/default_config.toml` 的 `ramen_search_stages` 由 `train,ramen` 改为 `train,ramen,region`——第 1 年地区（turn 2）和第 2/3 年地区（turn 23/47）纳入搜索，与 online 路径同款
- **`bench_commit_compare.py` 加 `--force-bench / --bench-runs`**：两版都有 perf_probe 时也可强制走 bench 模式（手写整局遍历全部 player_builds、同 seed 配对、同时输出耗时与评分配对），bench 默认局数 100；补充 example 命令
- **AGENTS.md 工作纪律微调**：精简对话规则段、删除独立的「需求澄清」条目（融入「项目特定上下文」）；明确 commit 前必读 git log + project_context + changelog 的项目启动纪律；文档清单与更新纪律同步精简
- **region 决策点 top-K 候选 dump 工具**：新增 `ramen_region_topk` bin——手写策略推进到 region 决策点（turn 2/23/47）后跑 FlatSearch，输出 top-K 候选的 (mean, stdev, count, weighted_mean, was_chosen) CSV + top1-top2 mean gap / stdev 中位数 / mean vs radical 排序差异汇总；MCTS 参数默认取生产实际值，bin 内冒烟测试（推进到 RegionSelect + dump 结构）——用途：研究 region 门控纳入搜索后 top 选项的均值差与方差分布；首轮扫测 3 turn × 7 build × 3 seed = 63 region 点：top1-top2 Δmean 中位数 ≤ 0.6%，top1 vs top2 stdev 差 ±5% 且正负不定（**top-K 候选方差无系统差异**），radical 加权排序 vs mean 排序在 21 个 (turn, build) 组合中有 5 个出现内部 swap（多在第 3 年 power_wisdom/wisdom/speed_wisdom 等 build）

## 2026-09-17
- **拉面杯逐卡 Hint 精确估值（可选）**：把固定 Hint 价值换成按卡面 Hint 等级与剩余可得的逐人头精确折算，默认关闭；一批全新 160 副随机卡组独立验收随机组 +174.0 [+153.0,+195.0]、预设 +77.7、固定卡组 −121.0 未证实，定位与 supermode3 同为“随机卡组可选”，关闭路径逐位不变。
- **拉面杯可选策略与验证**：新增近属性上限的连续技能PT估值和按终盘缺口选择超级拉面范围的可选开关，保留默认行为；提供固定/随机卡组配对工具、冻结复现配置、结果分析及边界回归，注明固定卡组和线上/MCTS适用限制。
- **ga_lab 最优策略合并（默认卡组）**：game_config 卡组切换 GA 通解骨架（速2耐1智2：待兼诗歌剧/千明代表/名将怒涛/优秀素质/青春永驻 + 骏川手纲），8 马 340 局同种子配对显著（Δ+516，t=3.36）
- **ga_lab 最优策略合并（9 旋钮组合档进 preset）**：已满位普通/超拉面 PT 定价上调（16→37 / 0→35）、弱位覆盖查表→35 直值、友情词条 1.5→0.4、掌握度 6→8、牺牲上限 140→200、拉面窗 0.10→0.15、剧本PT前瞻 0→0.15、Y1 pt_rate 16→56——4 马×2 种子块 420 局配对 Δ+1394 t=12.55（8/8 单元显著）；ramen_weak_train_boost 单测大负（−1017）明确不采纳
- **bench_base 新增 `--deck` 覆盖卡组入口**：显式 idrank 串指定卡组跑批（跳过 preset builds，标签 custom_deck），配卡对照实验用
- **重抓两个存量漂移基线**：`test_yearly_observability`（68118→70138，五维同步）与 `test_ramen_three_stage_action_unchanged`（7 组 rollout 均值）——GA 方向定稿进 preset 后当前行为
- **合宿训练诀窍填充修复**：`fill_feeling_gauge` 门控放宽为「合宿无条件三种槽全 MAX」——协议合宿回合角标全 0（None）时训练也能每种 +1 诀窍（此前在线对局合宿训练比休息/出行少 3 诀窍/回合，MCTS 合宿训练被系统性低估，是「高体力一选休息」的根因）；同步 URA 回合（72-77）角标照常抽签但不落库，训练不再产生诀窍槽（与在线一致）
- **新增合宿/URA 诀窍填充守门单测 ×2**
- **智力体力豁免语义修正为白名单**：`wisdom_vital_floor` 豁免带内只放行 智训练/休息/普通外出/治病（原实现=整个 40 门放开，速度/耐力等高位失败率训练在低体力下大亏；默认 MAX 仍不豁免，preset 行为逐位不变）
- **补齐已满位 PT 定价实验 token**：`trd/trdsh/trds`（N/100，文档早有命名从未实现）接入 `with_tokens` + 解析单测
- **重抓两个存量漂移基线**：`test_ramen_three_stage_action_unchanged`（7 组期望均值）与 `test_yearly_observability`（BASELINE_SCORE/FIVE）——最近策略调整与合宿诀窍修复后当前行为（干净 master 亦红，score/five 与本次改动逐位无关，纯同步）

## 2026-09-18
- **拉面杯第十二轮组合档进 preset（默认值）**：把放宽搜索界后经两个独立随机卡组池验收的组合设为 `RecommendedRamenTrainer::new()` 默认值（近上限连续定价窗口、已满位三类 PT 定价、状态缺口/溢出强度、预留上限与口径、超级拉面自适应范围、外出基准分、逐卡 Hint 倍率），随机组相对上一版 preset +130.4/+147.6、相对更早的 base 累计约 +517/+552，固定卡组不显著；同步重抓受影响的 6 处冻结快照（其中 2 处进入 preset 前就已过期）、把预留口径单测改为显式钉 token 以免随 preset 漂移，并按新默认值刷新 final-check 期望表
- **跨 commit CPU 耗时监测工具链**：新增 `perf_probe` bin（固定 Train 根整根搜索耗时 + 手写整局耗时，输出含根局面真实评分 `root_score` 记录两版评分区别）与 `scripts/bench_commit_compare.py`（worktree 检出两版构建、逐轮交替配对、probe/bench 双模式自动选择、gamedata 与搜索工作量一致性守卫、逐步 run.log）；`project_context.md` 新增「性能监测」节记录用法
- **局末自动打包本局游戏记录**：umaai 末回合第 2 份快照（拉面 `turn77_2`）处理完写 meta + SVG 后，把 `logs/game{id}/` 打成 `logs/game{id}.zip`（包内条目相对原目录，去掉 `game{id}/` 外壳）并清理原目录——zip crate 依赖进 workspace；切局/退出兜底（`switch` / `process_exit`）不打包，中途停止局保留原目录方便人工排查；终端 stderr 绿色绝对路径（dunce 去 `\\?\`）+ `info!` 日志一份（json 模式友好）
- **文档整理（重构完成后）**：完成重构的方案/草案文档移入 `.trae/documents/archive/`（拉面重构开发计划 / 上游三层架构建议 / main.rs 拆分 / RNG 重构 v2 / 在线对接计划 / AIRedirector 集成 / adapter_spec / handwritten_policy）；`project_context.md` 按 umasim/umaai 现状全面重写（模块结构、三流 RNG、搜索层、trainer 家族、config 五子结构、通道层细节）；AGENTS.md 相关文档链接同步为 archive 路径

## 2026-09-16
- **第3年地区单/多点整局配对扫描**：新增 `region_y3_single_focus` 扫参入口；seed42×100 全 101 构成 + seed61444×50 复测——混合档显著负、纯单点档玩家真实 build 大亏（≈−1100，t≈−5）→ 不采纳，维持现有按 build 自适应公式
- **运气分重放分析工具 + 三项修复**：新增 `luck_replay`（快照重放 → 明细/波动 CSV）、`luck_probe` 探针、`scripts/plot_luck_trend.py` 趋势图；修复 `selected_regions` 空数组整份丢弃、只吃面回合计不到运气分、年度 RMJ 派生状态缺失（第 2/3 年期望虚降 ~2300）
- **在线决策记录（每局一目录）**：umaai 实时运行按 `single_mode_chara_id` 落 `logs/game{id}/`（`thisTurn.json` 原文 + `decisions.csv` + `meta.json`，`luck_record` 默认开可关）；明细 schema 抽到 lib 与 `luck_replay` 共用，CSV 改走 `csv` crate 结构化写入（离线输出逐字节不变）
- **局末自动出 SVG**：收到末回合第 2 份快照（拉面 `turn77_2`）处理完即写 meta（`game_end`）并生成 `logs/game{id}/luck_trend.svg`（3 子图：边框/回合数轴/纵刻度与竖排标题/图例「蒙特卡洛估分…」/署名；切局退出兜底；终端绿色绝对路径可跳转）
- **比赛回合手写 fallback 上屏修复**：`HumanReadableSink` 对空分 fallback 决策（地区选择 / 比赛回合单候选 / RamenSelect 单候选）统一打印「选择…（手写逻辑）」，比赛回合不再静默（JSON 模式不受影响）

## 2026-09-15
- **地区弱位覆盖按智卡数查表（方案Ⅰ 固化）**：`region_weak_cover_weight` 改三态（0=按智卡数查表 智≤1→12/智≥2→0、<0=关闭、>0=固定值实验），与弱位训练偏好查表配套；全 101 种构成配对验证智≥2 零变化、智≤1 加权 +220
- **地区无卡位惩罚参数化**：`score_region` 硬编码 -10 提为 `region_waste_penalty`（默认 10 行为不变），扫描 5/15/20/30 均负收益，确认 10 为平衡点
- **地区 youqing 权重复核**：1.0~3.0 间为公共线性缩放无区分度（逐位不变），0.5 仅扰动 7 构成且方向不可靠，维持 1.5
- **主训位翻倍加分实验（未采纳）**：`region_main_bias_bonus` 让地区覆盖 build 卡最多位时 bias_sum 再 + 该位卡数，全 101 构成验证全档大负（3speed 类受伤最重），维持线性 bias_sum，字段保留可配
- **地区打分移除恒量项**：`score_region` 删 pt_bonus/hint_count 项（同年候选间恒定，不改变 argmax，验证逐位不变），删除孤儿字段 `region_pt_weight`；`region_hint_weight` 保留供吃面选择路径
- **地区权重扫描入口**：`bench_compositions` 新增 recommended 档（正式推荐 preset）与 `--region-weak-cover / --region-youqing-weight / --region-waste-penalty / --region-main-bias` 参数；`RecommendedRamenTrainer::with_region_weights` 覆盖入口
- **MCTS pt_favor_rate 扫参定档 2.0**：1.0~12.0 七档 N=1024 扫参（2.0~3.0 免费换 PT、4.0 起转亏、PT 增益 8.0 后饱和）+ 1.0/2.0/2.5 三档 N=4096 深测（5 速系卡组 10 局配对：2.0 评分损失不显著 t=−0.04、PT 增益显著 t=+6.3，2.5 边际仅多 +41pt）；正式默认 1.0→2.0，新增扫参汇总/绘图脚本
- **拉面 MCTS 决策候选分改真实评分**：`stash_last_summary` 缓存 `candidate_scores` 从 score_pt 轴（含 pt_favor_rate 缩放）改为 calc_score 轴——运气分 baseline、action_luck 与 AIRed 候选显示不再随 pt_favor_rate 虚增；选择仍走 score_pt 不变
- **urafile 读取改生产者-消费者模型（用户）**：thisTurn.json 监听从「收到事件后合并排空 + contents 去重」改为后台 producer 线程每次写事件读完整 JSON 进异步队列、主循环逐个消费——取消主观丢弃中间回合快照的路径（内容相同的冗余事件除外），notify 缓冲溢出错误事件改重读兜底
- **清理多余配置（用户）**：删除遗留 constants 快照文件；ramen_turn_inspect 展示同步 RamenMctsTrainer 字段变更（selection → use_combined_ramen_select）

## 2026-09-14
- **拉面手写策略评分换PT参数**：新增可调已满位训练 PT 折算价（`pt_tradeoff` 普通档 / `pt_tradeoff_shining` 有彩圈分级 / `pt_tradeoff_super` 超拉面档）——训练位主属性已满时属性收益为 0、只剩 PT，策略按独立价重估该训练候选，避免按 `pt_rate` 高估后终盘贪练已满位
- **评分换PT玩家配置**：新增顶层可调字段 `ramen_pt_sacrifice_score`（为多拿总 PT 最多愿意牺牲的总评分，默认 0 = 评分优先），按实测标定分段映射到已满位有彩圈定价（0→36/≤60→44/≤160→52/其余→64），`default_config.toml` 注明范围与对照表、`game_config.toml` 顶层可覆盖
- **实测调优结论**：彩圈分级实验确认 PT 产出由彩圈数主导（0圈≈40 / ≥1圈267-340）且与属性是否满无关（已满232 vs 未满225）；最优档=有彩圈定价36，100局 7 build 平均 +533 分、skill_pt -90；定价>64 会导致评分与 PT 双降
- **策略变体跑批入口**：`bench_base --tokens` 支持 `RecommendedRamenTrainer::with_tokens` 变体（trd/trds/trdsh/ptrate 等），实验不混入 preset
- **MCTS 评分换PT公式重设计**：`RamenGame::search_score()` 覆盖 trait 默认，`score_pt` 基于 `score_parts()` 使用干净公式 `skill_score + skill_pt × 2.0 × pt_favor_rate + five_status`（无 ×0.37 缩放），`pt_favor_rate=1.0` 等价 `calc_score()`；`pt_favor_rate` 代码默认 8.0→1.0
- **移除 RamenSelection 枚举**：ramen MCTS 统一走 `best_action_pt_idx()`（score_pt 口径），删除 `bench_base --search-selection` CLI、`RamenMctsTrainer::with_selection()`、`RamenSelection` re-export
- **友人卡 rank=0（未突破）合法化**：`newgame` 校验补 rank≤4 范围检查，修正测试断言（rank=0 应合法、rank=5-9 非法）

## 2026-09-13
- **拉面链式决策 JSON 输出顺序修正**：链式决策#1（不吃面 / 训练 turn1）在 `compute_next_step` 之前先 emit，下游先收决策结果再收"计算中"通知
- **拉面 RamenSelect 决策输出补齐**：吃面 / 不吃面在合并搜索路径下均合成 `ramen_select` 决策信息下发（不吃面同时作为链式决策#1 单独输出）
- **每轮计算完成信号**：拉面 / 温泉决策输出后 emit `compute_done`，无决策的 Begin 早退回合（事件 / 结算 / 数据不全）也补发，保证 `compute_start → compute_done` 成对
- **watcher 快速连续写入健壮性**：notify 错误事件不再退出进程，空 / 半截 / 文件锁读取带重试与两次一致校验，唤醒后排空事件队列合并为一次处理最新内容
- **Windows 构建脚本**：重新启用图标资源编译和栈链接参数输出（与 09-11 停用相反）

## 2026-09-11
- **模拟状态与规则计算**：内联继承因子和卡组计数，支援卡面板借用全局只读卡表，共享事件采样分布，简化做面可行性计算，系统事件和概率查询仅在失败时构造错误，减少复制与分配。
- **选面与训练评分**：预演借用原局面并显式指定候选面，复用训练候选、评分空间、训练基础值、Hint 与羁绊估值、友人动态估值、体力结果和地区窗口分量，各面的实际加成与最终评分分别计算；rollout 省略评分明细，普通决策日志和协议输出保持完整。
- **性能验证**：补充状态隔离、逐碗计算、候选顺序、事件随机流、Hint 切换、必赛分支和普通策略日志的一致性检查，记录生产参数正反序配对测量及源码、配置和产物校验信息。
- **Windows 构建脚本**：停用构建脚本中的图标资源编译和栈链接参数输出。

## 2026-09-10
- **本机 Release 性能优化**：启用速度优先优化、ThinLTO 和本机 CPU 指令集，减少模拟、评分与搜索中的重复分配、复制和计算，复用事件队列与地区评分，候选内部模拟并行执行并按原序归并；rollout 跳过原因文本生成，保留数值评分分解、正常决策日志与协议输出。
- **性能基准与验证**：基准继承游戏线程和搜索配置，补充分配复用、评分、随机流、动作与事件轨迹、协议输出及跨线程失败槽一致性检查，记录生产参数整局与高预算搜索根的配对测量口径和复测步骤。
- **main.rs 按职责拆分重构**：主程序收敛为薄调度（CLI / 初始化 / watch 循环分发）；新增 `decision/` 目录（决策后处理：luck 计算与决策输出，`luck_score` 一并移入）与 `scenario/` 目录（温泉 / 拉面各一幕块：含 newgame 检测、切局、决策计算与 emit）；行为等价
- **连续决策中间状态输出时机修正**：中间决策的「计算后续动作」提示与 `compute_next_step` 通知从主循环移到决策循环内部、在真正执行下一步决策（可能耗时）之前发出
- **比赛回合策略输出修复**：拉面比赛回合仅一个固定动作、MCTS 不搜索导致无输出——为固定动作合成决策信息使其在屏幕 / JSON 上可见，不挂 luck
- **`--json` 开始接受数据时发 `connected`**：仅 json 模式、watcher 就绪进入监听时 `emit_info("connected")` 通知 AIRed 连接成功

## 2026-09-09
- **决策间 `decision_kind` 顶层字段 + `scenario_extra.ramen_action`**：partial decision 类型分发——main.rs 在 select_action 前按 `RamenStage` 填 `decision_kind`，onsen 填 `train`/`event`；`ramen_action` 由 `RamenAction.to_string()` 给出含吃面 + 隐藏诀窍 + 操作三阶段动作串（AIRed 端只显示不解析），合并路径一条 JSON 表达、三阶段路径按 chain 顺序分条
- **新增 `candidate_descriptions` 字段**：与 `candidate_scores` / `candidate_n` 严格同长同序同截断，供 AIRed 映射拉面组合动作名（onsen 取自 `SearchOutput.actions[i]`、拉面 MCTS / 手写策略分别缓存到 `LastSearchSummary` / `LastDecisionSummary`）
- **`DecisionInfo` 简化 + `scenario_extra.reason` 挂载**：删 5 个 stub 字段（reason / search_depth / visit_count / score_breakdown / elapsed_ms），保留 candidate_scores / candidate_n；拉面从 `LastReasonSink` 取 `DecisionReasonData` 挂到 `scenario_extra.reason` 完整透传 human reason 信息
- **`--json` 输出类型扩展**：stdout 顶层 `type` 区分三类消息（`decision` / `info` / `error`），去掉 `schema_version`；`info` 仅 event 取值、`error` 仅 message——在 watcher init、watch loop 入口、拉面链式决策中间、切局、失败五处按需发射
- **AI 不再推进游戏状态**：拉面 calc_ramen_training 与温泉 calc_onsen_training / calc_onsen_event 改为只调一次 select_action 出推荐、不再 apply / next；主循环每次 watch 收到新 JSON 后从零重建 game 重新计算，两次 JSON 间不互相依赖
- **温泉 / 拉面回合头部打印**：human mode 在每次计算后打印马娘状态 / 剧本信息 / 训练分布；json mode 跳过这些屏幕输出
- **JSON 模式 stdout 净化**：计算完成提示「计算完成，等待新数据...」、启动横幅走 stderr；`[按 F2 保存当前回合状态]` 等人类调试提示在 json mode 跳过
- **ctrl-s 热键功能临时停用**：tokio::spawn(hotkey_handler) 注释掉（crossterm 无限 poll 占用 worker 配额、AI 通道下无意义），后续重构时按 feature gate 恢复
- **版本号 / 横幅升级**：`umasim` / `umaai` Cargo.toml version 升 0.2.x → 0.14.0；启动横幅 "UMAAI 0.26" 改为 "UMAAI-Ramen"
- **拉面 / 温泉连续决策**：特定场景在前一决策基础上继续生成下一决策——应用上一决策并推进到下一阶段后再出推荐（第 1 回合训练后接地区选择、选"不吃面"后接训练选择），一回合内可连续下发多段决策
- **连续决策触发条件修正**：仅在第 1 回合训练时继续，避免其余回合误触发
- **拉面内部状态字段清理**：移除仅用于状态导入内部判断、无需持久化的字段及对应类型，协议不再写出
- **拉面当前生效面修正在下**：仅在效果列表非空时才透传，同步更新样本导入测试断言
- **启动日志精简**：去掉"载入用户配置 / 载入默认配置"的日志输出
- **human 输出改为运气行**：删去 AI 选择 / 理由两行，输出期望评分与运气分（本局 / 本回合），四舍五入为整数，本局运气按区间着色
- **期望评分叠加每回合加成**：期望评分与运气分统一计入"每回合比手写逻辑多的分数"，按剩余回合数加权，初始基线按第 0 回合计算
- **拉面决策理由输出解耦**：理由的原始数据始终下发供宿主使用，可读文字仅在诊断模式上屏，避免双打印
- **feeling_guage → feeling_gauge 改名**：上游已修复误拼，协议与 umaai 端 `RamenStatus` 字段随迁 `feeling_gauge` / `feeling_gauge_gains` / `feeling_gauge_gain_base`，文档同步更新
- **拉面 into_game 严格按协议重建 base**：弃 `RamenGame::newgame` 打补丁，改由 `parse_basegame` 重建（Uma / Friend 走 `parse_uma` / `parse_friend`、`five_status_limit` 取协议值、丢弃 `friend_event_ids`，友人事件 / 五维上限视为外部输入）；新增 `RamenGame::from_base_game`；`parse_basegame` 卡组循环补 persons 越界守卫
- **地区选择（手写 fallback）决策可输出**：`region` 门控关闭时 `last_decision()` 为 None 导致无结果——`decide` **仅对 `RegionSelect`** 合成决策信息入链 emit（其余 None 阶段保持旧行为不合成）、不挂 luck；human 紫色显示「选择地区[...]（手写逻辑）」

## 2026-09-08
- **新增 adapter_spec 文档**：整理 SendGameStatusPlugin 与 umaai 协议对接的易混淆点（feeling_guage 拼错 / persons/personDistribution 适配 / playing_state 含义 / 数据获取不全判定 / 超级拉面回合处理 / 阶段来源三态等）
- **Step 7 拉面剧本协议与主流程接入**：阶段派发按 source / active_effect / playing_state 三方联合；turn ≤ 1 直接进 Train；playing_state=45 进地区选择；超级拉面回合按 active_effect 区分丢包/决策；数据获取不全 warn + 不派发
- **拉面 persons layout 与协议对齐**：理事长 / 记者 / NPC 按 adapter_spec 排布；记者出现回合修正为 turn > 12；151 样本驱动测试同步更新
- **GameStatusBase 协议字段扩展**：新增 source（snake_case）、single_mode_chara_id（snake_case，单调递增切局键），兼容旧 JSON
- **拉面 AI 主循环接入**：main loop 拉面分支从占位升级为完整流程；切局检测改用 single_mode_chara_id，缺失时退化到 uma_id
- **拉面单回合诊断 binary**：新增 CLI 二进制，指定单个 ramen JSON 即可跑完整 into_game + MCTS，human-readable 输出
- **集成文档 §3.4 同步**：Step 7 实装方式（协议层 into_game + 单回合诊断工具）替代规划期描述

## 2026-09-07
- **Step 8 AIRedirector C# 端极小改动**（独立仓库 URA_Plugins/AIRedirector，已合 a8edca8）：`UmaAiProcessStartInfo.Create` 加 `jsonMode` 参数（`--json` 开关）；`AIRedirectorConfig` 加 `Ramen` / `Ramen_Path` 字段；`Class1.StartProcess` 加 `jsonMode` 形参 + 拉面分支；`HandleOutput` 试 `TryParseUmaAiDecision` 解析（`schema_version` 识别）后路由 `ApplyDecision`；`UmaAiDecision` record struct；配置文件 UI 加拉面分支（`ConfigAction.EditRamen`）；smoke test 加 4 个测试（拉面 / 温泉 / 非 JSON / 缺 schema_version）—— 实际 Windows 编译与运行测试延后到切回 Windows
- **Step 7 RamenGame::from_external_state 完整覆写**：`RamenState` 新增 `feeling_guage_gains` / `next_scenario_pt` / `feeling_guage_gain_base` / `active_effect_array: Vec<ActiveEffectEntry>`；`ActiveEffectEntry` 协议类型（category 语义搁置）；`protocol/ramen.rs::into_game` 完整实现（base 字段 + 5 人卡组 + 友人/理事长/记者 + events + 12 个 ramen 段字段 + stage dispatch 按 playing_state 1/5/45/46/48）；驱动 151 份 `logs/GameStatusSend_Ramen` 样本 round-trip 校验 scenario_pt / current_ramen / selected_regions / super_ramen 全透传，max_scenario_pt=7500 / stage 分布 Train 145 + Settlement 5 + SuperRamenSelect 1 与预期一致
- **Step 6 parse_game scenarioId 分发**：protocol/ramen.rs 加 GameStatusRamen 骨架（scenario_id=14） + mod.rs `ParsedGame` 枚举 + `parse_game_by_scenario`，main 按 12/14 分发（拉面侧 Step 7 接入 AI 主流程）；7 个 protocol 测试
- **Step 5 LuckScoreTracker + emit_with_luck 接线**：luck_score.rs 新增 tracker + 切局检测 + 按局数加权 baseline，main emit 走 `emit_with_luck` 挂 scenario_extra；移除 ratatui（utils 只用 crossterm）；5 个 luck_score 测试
- **健壮性 fix 三件套**：watcher 路径/env 缺失降级为 warn + 空字符串 + `release-pause` feature gate（发布版启用 `--features release-pause`）；注释 check_windows_terminal；延迟 spawn hotkey_handler（避免失败路径 runtime drop hang）
- **Step 4 CLI --json 分流 + sink 接线**：umaai 用 lexopt 替 clap 解析 `--json`/`-h`/`--help`；JSON 模式关 ANSI + 启动横幅 eprintln；按模式选 StdoutJsonSink / HumanReadableSink
- **Step 3 DecisionSink 三实现**：sink.rs 新增 trait + EmptySink / HumanReadableSink / StdoutJsonSink；reason::NoopSink 改名 DecisionReasonNoopSink 与 sink::EmptySink 区分
- **Step 2 last_decision override 三 trainer**：DecisionInfo 加 candidate_n（与 scores 同长同截断供 luck 按局数加权）；MctsTrainer / RamenMctsTrainer / RamenHandwrittenTrainer override last_decision()；集成文档 §3.3.1 改"按局数加权"

## 2026-09-04
- **吃面 PT 增量延后到 NextTurn**：`ground_ramen_effects` 不再立即 `scenario_pt += pt_gain` / `eat_count += 1`，训练阶段 `calc_ramen_training_effect` 用吃面前 PT 算 `ramen_pt_effect` / `region_bonus` 档位；PT 增量与 eat_count 在 `next()` 的 `NextTurn` 阶段（清空 `current_ramen` 之前）统一处理，RMJ 归档与 `check_rmj` 行为不变
- **三处基线重抓 + 一条新守门**：`bench.rs` BASELINE_SCORE 64336→63870 / BASELINE_FIVE `[3337,2328,2200,1101,829]`→`[3337,2293,2200,1086,829]`、`flat_search.rs` 三阶段根搜索 7 候选 mean 重抓、`ramen_mcts_trainer` 两测试基线同步；新增 `test_eat_ramen_pt_gain_defers_to_next_turn` 钉「吃面 ground 后 scenario_pt 不变 / calc_ramen_training_effect 用吃面前 PT 算增量 / NextTurn 后累加」三条边界

## 2026-09-03
- **EXP-006h 复现合并**：handwritten token 入口接通 bench、闭环 Δ+67 t+4 显著，本地分支留档
- **`bench_compositions` 改加权口径**：满破面板「友情×2 + 干劲×0.5 + 训练」加权和，默认 pool_size=10 / min_panel=80（最新 10 张候选池留 5 张缓冲），池内加权降序、并列按 card_id 倒序取前 3
- **`ramen_space_bench` 固定地区策略为 `all`**：与 `bench_base`、`ramen_teacher_collect` 一致，不再跟随 `game_config.toml`，避免基线静默换分布
- **`ramen_handwritten_choice` 补转发 `select_event_choice`**：避免重放轨迹与采集时不一致
- **若干健壮性与文档修正**：补 `.npy` 长度检查、删 `ChoiceRow` 阶段字段 dead_code、NN 测试在 `saved_models/` 缺模型时跳过、修 `ramen_special_root` 死链等
- **训练评估单源化（B2）消除 policy↔local 双重计算**：`score_train_action` 拆 eval/other、`decide_train` 用 `TrainEvalCache` 同回合同 train 的 calc 链收口为 1 遍，decide_train 整回合 **-14%**、整局 **-10%**，平均分逐位一致
- **训练评估确定性守门**：`test_train_eval_deterministic_and_cached_consistent` 三条——`eval_train` 重复一致 / cached 与 uncached 决策逐项一致 / trait 双路径逐位等价
- **mcts_profiler bin 注册补 `required-features=["profiler"]`**：修复默认 `cargo check` `main not found` 硬错误
- **UCB search_group_size 2048→512**：`default_config.toml` 改值、断言同步

## 2026-09-02
- **`SupportCard::calc_training_effect` 简化签名 + 起点改基础面板**：去 `Result` 包裹与 `is_locked` 短路，起点改 `CardTrainingEffect::from` 保 fresh cumulative，`is_locked` 字段与 `effect = eff.clone()` 回写保留（NN feature 兼容）
- **deyilv 路径去掉 `eff.clone()` + microbench burn-in 清理**：三处 override `let deyilv = eff.deyilv; effect = eff;` move 而非 clone；microbench 删 `burn_in_lock_cards` 与 `CT_MICROBENCH_LOCKED`，数据基本不变
- **`Game::deyilv` trait 简化签名**：`Result<f32>` → `f32`（无 fail 路径），三处 impl override + 5 处 test caller + `distribute_person` 一处内部 caller 全部去 `?`；与 `calc_training_value`/`calc_training_buff` 保留 `Result` 的分工清晰
- **`calc_training_value` 微基准 bin**：新增 `calc_training_value_microbench.rs`（speed build + friendship 全 100 + 4 段：distribute_all / calc_training_buff / calc_training_value / 端到端），pprof 之外的"逐段基线"快速回归工具
- **`calc_training_value` microbench 扩到 7 段 + private 改 pub**：补 `score_train_action`/`decide_train`/`calc_ramen_training_effect` 覆盖 d10872a + perf_profiling pprof Top 20 缺口；`RamenPolicy::score_train_action`/`status_gain` 与 `LocalRamenTrainer::decide_train`/`dynamic_status_adjustment`/`reserve_penalty` 提 pub 供调优工具用
- **`perf_profiling.md` 第 7 章固化按段拆解基线**：新增 §7"按段拆解的最坏路径基线（calc_training_value_microbench）" + 与 pprof Top20 交叉校验 + 新优化优先级（distribute_all 缓存化、status_gain 两条新增）；附录 E 列 `CT_MICROBENCH_RUNS`/`CT_MICROBENCH_WARMUP`
- **`perf_profiling.md` 旧描述清理 / 重写**：§1 改"两类"→"三类"性能工具（加 microbench）；§3.1 改 `d10872a` commit hash + 实测数字；§3.3 旧优先级保留并加"原列表与 §7.6 互为补充"；§5 拆 §5.1 三 bin + §5.2 对比表；§6.4 加 microbench 复现 command
- **`perf_profiling.md` 多轮均值替换旧 pprof 单局**：§3 改"性能分析结论（当前方法论）"（microbench × N std ≤ 2.5%）；§3.2 旧"次要发现"标已弃用；附录 B/C 旧 pprof 单局 / cargo flamegraph 数据整体删除；§7.2/§7.3 7 段 × 3 轮（800.97/260.20/270.39/1398.90/449.82/4175.40/9.63 ns/iter）+ 附录 B 6 函数 × 3 轮；附录 D/E 上移
- **`perf_profiling.md` 重复内容整合**：原 §7 6 子节合并为 3（§7.1/§7.2/§7.3）；原 §7.5 复现命令删并引 §6.4；§7.6 优化优先级并入 §3.3 为 8 条（6 pprof + 2 microbench）
- **`distribute_person` 采样改零分配**：`WeightedIndex::new` 换 `Uniform::new + sample` + 5 元素线性分桶（新 `sample_bucket`），消 person clone 与负权重检查；7 组合 × 5 万次守门 + 全量数值逐位不变，每次采样 -31~-42%（A/D ≈ 53%）

## 2026-09-01
- **预声明 LR 日程**：train.py 新增 `--lr-schedule {plateau,cosine}`（默认 plateau）+ `--lr-warmup-steps`/`--lr-final-factor`，cosine 按 optimizer step 线性 warmup + 余弦退火；解决 ReduceLROnPlateau 按轮计数导致 LR 衰减次数随数据量漂移、对照差异无法归因
- **`--max-steps` 精确截断**：以 batch 边界精确截到指定步数取代"轮数向上取整"，`run_epoch` 增实跑步数、checkpoint 增 `global_step`，保证续跑对齐
- **采样空间指纹**：`SamplingSpace::content_hash()`（顺序敏感）+ `GEN1_SPACE_HASH_V1` 钉死，manifest 增 `sampling_space_hash`，导出器校验空间一致后才导出；防止扩空间后 plan_count 静默改写导致留出集切分整体错位；刻意不并入 `recipe_hash`（同空间新旧数据合并需要）
- **分布外采样空间**：`SamplingSpace::custom` + `--shape`/`--extra-card` 出口（不与 `--shape` 同用则不与教师数据同分布），输出显式标注分布外，用于检验网络未训练卡组流派泛化

## 2026-08-31
- **评估列窗口**：eval.py 的 `evaluate_model` 与命令行新增 `--eval-columns LO HI`，候选价值只用该段 rollout 列重算（data.py 的 NpyShard 相应惰性 mmap `cand_scores`/`cand_valid`，窗口内无有效列的样本整条跳过并计数）。原先只有全列 `cand_mean` 一个口径，而 `best.pt` 正是按它挑的，被结算的列因此参与了模型选择
- **eval.py 沿用 checkpoint 的切分粒度**：独立评估此前恒按默认的 combo 重切，按 sample 训练的模型会被静默换成另一套留出集
- **训练侧诊断设施**：train.py 新增 `--eval-columns`（每轮额外记一份限定列的留出指标到 `evaluation_a`，不参与早停与 LR 调度）、`--checkpoint-steps`（在指定 optimizer step 处存 `step_XXXXXX.pt`）、`--ema-halflife-steps`（按步维护带偏差校正的权重 EMA 并一并存盘）、`--no-early-stop`（训满轮数上限）。metrics 增记 `global_step`，run.json 增记 `diagnostics` 段。四项都不改变梯度，不给新参数时逐位复现旧结果
- **训练随机轴拆分**：train.py 新增 `--split-seed`（样本身份：划分与抽稀）与 `--init-seed`（优化路径：初始化、dropout、minibatch 顺序），未给出时都回落到 `--seed`，旧命令行为逐位不变。此前两者绑在一起，每换一个种子就换掉约 10% 训练样本；实测固定 split 后闭环 sd 从 2521 降到 156
- **早停按 optimizer step 计**：新增 `--patience-steps` 与 `--max-steps`，按每轮步数换算成轮。按轮计数时数据量翻倍会让同样的轮数变成两倍步数，学习曲线各点的训练时长口径不一致
- **闭环 bench 局号偏移**：ramen_space_bench 新增 `--run-offset`。相邻基种子会撞随机世界（`derive_seed` 是 XOR 后 splitmix，`base ^ r == base + r`），此前三个相邻种子的 12600 局实际只有 5248 个唯一世界，标准误被低估约 1.5 倍。改由固定基种子、按局号区间切分
- **选择集 / 验收集分离**：新增 scripts/ramen_nn/compare_bench.py，按世界去重做配对比较，并断言两集零重叠。同一批对局既挑 checkpoint 又报成绩会带 winner's curse
- **因子化吃面输出头**：model.py 新增 `factorized_eat_head`，把 `[1,201)` 联合格拆成「地区 + 用法 + 零初始化交互」。输出布局与 ONNX 算子集不变。三训练种子下无可测效果，默认关闭
- **SpecialSelect 联合决策根还原**：新增 trainer/ramen_special_root——把 `SpecialSelect` 局面写回 `pending_ramen` / `pending_special_targets` / `stage` 还原成它所来自的 `RamenSelect` 联合决策根，供网络在正确状态上读联合格位。教师在 `RamenSelect` 根上搜的是联合动作（地区 × 隐藏风味用法），policy 格 `[1,201)` 也是联合格，而训练集里 `SpecialSelect` 阶段样本数为 0；真实对局却把决策拆成两拍，第二拍的阶段 one-hot 在全部训练样本里恒为 0，网络输出属外推。模块不带 onnx 门控，否则守门测试在默认 feature 下不会运行。测试断言还原后特征逐位相同、且**不还原时必须不同**——后者保证规则层新增字段时测试会红而不是静默失效
- **NN 训练员 SpecialSelect 三档口径**：新增 `SpecialSelectMode::{Raw, Canonical, Handwritten}` 与 `with_special_mode`，默认 `Canonical`；`Raw` 保留作对照，`Handwritten` 把该阶段整个交给手写策略以给可恢复上限定界。`ramen_space_bench` 与 `ramen_advantage_probe` 都加 `--special-mode`。ramen_advantage_probe 的 required-features 补 onnx——无 onnx 时它没有任何可用功能，让 cargo 直接跳过该目标好过编一个只会报错的空壳；ramen_space_bench 仍须在默认 feature 下可构建（handwritten/random 是主用途），故其 special-mode 解析走 onnx 门控
- **拉面 NN 策略训练员**：新增 ramen_nn_trainer——把 ONNX 模型接到 `Trainer<RamenGame>`，编码定长特征后按冻结格位表给当前候选打分 argmax；choice 头未训练，事件选项委托推荐手写策略。仅在 onnx feature 下编译
- **自选比赛硬守门供网络复用**：`RamenPolicy::free_race_gate` 的判定本体抽成自由函数 `free_race_gate_index`，NN 训练员在 Train 阶段先过同一层守门再读网络输出。自选比赛不达标直接判育成失败，是硬性义务而非价值权衡，任何策略都要过；判定语义逐字保持不变，手写策略行为不变
- **采样空间基准接入网络**：ramen_space_bench 新增 `--trainer nn` 与 `--model`，模型在进程启动时加载一次由各局共享而非每局重载；`--no-race-shield` 可关掉守门，仅供研究守门能否移除，不作为验收口径。策略分派由字符串匹配改为预构造枚举，未知策略名在开跑前报错而不是每局重判
- **on-policy 配对 advantage 探针**：新增 ramen_advantage_probe bin——用指定策略跑完整局，在网络与手写选择不同的决策点上做配对 rollout（两动作共享同一张 CRN 种子表，rollout 基策为手写），按性能差分恒等式估计 `J(π) − J(H)`。两策略选同一动作的点贡献恒为 0，直接跳过不搜索；分歧点按蓄水池等概率抽样，单局估计按分歧点总数加权还原。`--rollin` 可把 roll-in 换成手写，用于量测占用分布错配。此前训练侧的 `expected_regret` 算在手写 roll-in 加扰动的分布上，与闭环结果反向，不能用来排序训练方案
- **训练集稳定抽稀**：data.py 新增 `subsample_train_refs`，train.py 新增 `--max-train-samples`——按 `splitmix64(sample_id, seed)` 排序取前 N 条，同一种子下各数据量点严格嵌套且验证集不变，曲线上的差异只来自数据量。checkpoint 的 split 段增记 `full_train_size` 与 `max_train_samples`
- **Train 阶段动作重加权（可选）**：train.py 新增 `--train-action-reweight`，按 policy 软标签主动作施加截断逆平方根样本权重（上限 4，归一到均值 1），只作用于 Train 阶段；权重与计数写入 run.json 与 checkpoint。默认关闭
- **onnx feature 编译修复**：neural_net_evaluator 补 `use rand::Rng`

## 2026-08-30
- **采样地区配额**：采样器新增地区配额与「只捕获指定阶段」开关，按工作项序号确定性分配、走独立随机频道，改配额不影响其余样本的截断回合。此前第 2/3 年的地区选择几乎采不到——它们在回合末，同回合的吃面/训练决策先命中采样白名单，实测 1200 次采样 turn 23 命中 0 条、turn 47 只有 9 条
- **拉面教师样本容器**：新增 training_sample 模块——定长特征 + 元信息 + 变长候选表，每候选按 rollout 序号存定长分数槽位并配有效性位图，失败的 rollout 留空而不是跳过，否则候选之间的 CRN 配对会整体错位；统计量由原始 f64 累加，均值与标准差和 ActionResult 同口径；附 pilot 用的 bincode 批次落盘。PolicySlots 补 serde 派生
- **搜索层保留有序 rollout**：SearchConfig 新增 `record_ordered_rollouts` 开关，默认关闭时不分配缓冲、不改变搜索结果；开启后按 rollout 序号定长记录 score 轴原始分，随 SearchOutput 一并输出根种子。失败的 rollout 留空而不是跳过，UCB 路径同样按序号写入，否则候选之间的 CRN 配对会整体错位
- **拉面版 export_sample**：搜索输出可直接导成教师样本——定长特征 + 元信息 + 按 rollout 序号对齐的候选分，不计算 policy/value 标签（标签是离线可再生的 sidecar）；未开启有序 rollout 记录时直接报错，不退化成用直方图回填
- **教师数据采集驱动**：新增 ramen_teacher_collect bin——采样局面、搜索、导出样本、分片落盘并写 manifest。四条运行时前提（记录有序 rollout / 关闭 UCB / 显式 radical_factor_max / 地区策略 all）由 bin 强制设置，manifest 记的是它们的实际取值，另存游戏数据签名与 git 提交以便复现；支持按 manifest 断点续跑，`--count` 是从 `--start` 起算的累计目标，区间为空时报错并保持 manifest 不变
- **教师数据 NumPy 导出**：新增 ramen_export_npy bin——把多个采集目录的 bincode 分片摊平成一组 .npy 数组供 Python 训练侧读取，候选维用 CSR 偏移表示变长。合并前校验各目录的采集配方哈希与 git 提交一致、样本 id 不重复，维度常数与本次编译不符时报错。只导原始量不导标签，软标签配方与 value 归一化留在训练侧。`--raw` 额外导出每次 rollout 的原始分数与槽位有效性。npy 头部定长占位、收尾回填行数，全程流式不驻留内存
- **采样空间基准**：新增 ramen_space_bench bin——遍历采样空间全部 (马娘, 卡组) 计划各跑若干整局，按构成与马娘分组给出均分、标准差与标准误。此前唯一的基准 bench_base 用的马娘不在采样空间内且无自选比赛要求，测出的手写基线不能当作网络验收门槛；本 bin 与教师数据同分布，两边数字才可比
- **手写策略选择记录**：新增 ramen_handwritten_choice bin——重放教师样本的每个局面、记录手写策略的选择并落到 policy 格位，供训练侧算出同一批局面同一 Q 口径下手写自己的后悔值。训练侧的后悔值是相对搜索教师的，不回答「网络比手写强还是弱」。RamenSelect 按真实对局分两阶段问再合成，因为手写的吃面决策不读万能风味用法，直接喂合并候选表等于让它随机挑
- **导出器记录采样计划数**：manifest 增记 plan_count，训练侧据此按 (马娘, 卡组) 组合切留出集，不再在 Python 侧硬编码组合数
- **Python 训练侧**：新增 scripts/ramen_nn——标签生成、模型、训练、评估、ONNX 导出与 mmap 多目录加载器。policy 标签取配对 Bayesian bootstrap 的最优概率而非温度化 softmax，value 走逐 rollout leave-one-out 的选择—估值以消除选择乐观偏差，地区选择按组合概率边缘化到三格，稀疏阶段用截断逆平方根加权。留出集默认按卡组组合切分而非按样本，避免同一套卡组同时进训练与验证；卡片 token 默认不加槽位 embedding——卡组顺序在游戏里没有含义，而训练数据的槽位与卡片类型完全相关，加了会让模型记顺序而非读属性

## 2026-08-29
- **拉面 NN policy 格位表**：新增 policy_schema 模块，把动作映射到 234 维固定格位并由单一入口分派；吃面按地区 ID 而非槽位编码、吃面与万能风味用法合成联合格、地区选择纳入第一代。**格位表冻结**
- **格位表与规则层的耦合回归**：原有测试只拿本文件常量自洽验证，规则层一改不会变红；补测试用采样器把真实候选（含合并决策形态）与规则层用法表全过一遍格位映射。顺带修合并决策的「不吃面」落格失败——原先只接受空 targets，实际每个吃面决策点都会漏掉该候选
- **NN 管线计划文档同步现状**：Phase 1/2 标完成；教师数据预算改按 search_n=1024 重算；拉面 CRN 机制更正为共享 rule_master（阶段重播种仅 onsen），配套作废一处无效测量；补 Phase 3 开跑前待办
- **MCTS pprof-rs profiler bin 固化**：sim_profiler 模板的 MCTS 版（pprof-rs 用户态采样，输出 .pb 给 go tool pprof / inferno-flamegraph）
- **性能分析指南文档**：新增 perf_profiling.md，记录 cargo flamegraph / pprof-rs 工具选择准则与 MCTS hot path 数据
- **决策理由输出按分排序 + 分差着色**：移除险胜门限触发，每回合都输出决策理由；"中选"改"首选"并固定亮绿色，其余按评分降序编号 `#2` 起，颜色按与**首选**差距分档（`<30` 亮绿 / `<100` 绿 / `<300` 黄 / 其余真彩色灰，与文本内 `±分差` 同源）；`reason_gap_threshold` 字段保留兼容但不再用作触发器；`test_color_thresholds` 在 `--features no-color` 下自动跳过
- **决策理由模块索引入项目文档**：`project_context.md` 新增"输出与决策理由"节，记录 `reason_color` 阈值调整位置（`reason.rs:107`）与 no-color feature 兼容性
- **拉面在线对接计划**：新增 `.trae/documents/ramen_online_integration_plan.md`——文件通道 thisTurn.json + scenarioId 分发、两阶段决策吃面前/吃面后、C# 端先行冻结协议再 Rust 接入
- **用户配置调整**：`game_config.toml` 切马娘 101901（stamina build）+ 卡组/蓝因子/extra_count 微调；`gamedata/default_config.toml` 同思路调 102601 + `ramen_region_strategy` 由 `"fixed"` 改 `"all"`

## 2026-08-28
- **MCTS rollout 诊断日志运行时屏蔽**：diagnostic 加进程级开关（DiagGuard 挂 `search_with_terminal`），`diag!` 双门控 + 8 处 explain 块补 `if enabled()`，rollout 搜索静默、业务日志不受影响，顺带拿回加速收益
- **险胜决策理由输出**：新增 output/reason——险胜回合（门限默认 150）显中选内容 + 未中选 top-N 分差与五维/PT 子项；参数走完整覆盖链，终局差异日志 info 降 debug
- **诊断出口整理**：basic.rs 回合分隔线 println 并入 diag!、地区选择 diag 补"手写逻辑"注记、state.rs 补五维上限初始化契约注释
- **合入 ramen_workbench 主干修改**：squash 单提交；实验脚本 / workflow / 实验采集 bin / 过程文档不合入
- **tests_overview 按 master 口径全量重写**：159→330 个测试逐条一行描述并按模块重组；旧「未来缩减参考」表随 159 口径移除

## 2026-08-27
- **五维属性上限剧本化**：上限基值改为随构造参数传入（`Uma::new` / `BaseGame::new` 新增 `limit_base`），顺序固定为"先写剧本基值、再加继承"，三个剧本各自从自己的 `scenario_*.json` 取值，`constants.json` 同名字段降级为 basic 与缺字段兜底。原先"先写全局值、再由各剧本事后修正"的打补丁式设计全部删除——拉面的整体赋值发生在累加开局继承之后，会把继承增量擦掉；温泉的 `min(2800)` 是速度基值 2600 时代的防御值，基值提高后变成硬截断，且在继承事件后还会再截一次。补丁写法本身就是这两个缺陷的来源，新剧本照抄必然复现。温泉基值补入 `scenario_onsen.json`（此前无该字段，一直吃全局值再被截断）。**改变拉面与温泉模拟数值，基线作废**
- **终局评分查表口径统一**：新增 `GameConstants::status_final_score`，越界一律饱和到表末。此前三处消费点行为各异——裸下标越界 panic、`unwrap_or(0)` 越界静默返回 0。后者最坏：属性增益按查表差分计算，返回 0 会让该维收益变成巨大负值，手写策略永久回避该维且不报错。评分表长度有限而上限＝剧本基值＋继承三次，蓝因子拉满即可越界。顺带修 `status_gain` 中负增量 `as usize` 回绕溢出（当前取值恒正打不到）
- **上限相关守门与契约测试**：新增跨三剧本的开局上限守门测试（期望值从各剧本 JSON 推导，故改代码会红、改数据不误报）、剧本基值字面量契约测试（守数据漂移，并锁两剧本基值必须不同——拉面与全局常量当前数值相同，误接全局的回归只有它能抓）、查表越界饱和测试。`expected_score_parts` 保持不调用生产查表函数，维持独立对照。修正 `eat_covered_train_gate_blocks_mismatched_ramen` 夹具写死旧上限当"满"的问题，改为从实际上限取值；三处硬守门快照基线随上限变化重抓
- **MCTS rollout 与 fallback 切到正式推荐策略**：原用机制残缺的策略核心评估局面；门控全关时逐位等价，rollout 档关掉观测开销
- **搜索掉分归因**：缺省 `radical_factor_max=50` 使有效样本量恒 3.9%，选择偏差压过搜索收益；rf=0 后方向反转，缺省值不动
- **硬守门快照重抓与收紧**：4 处基线随 trainer 切换重抓；合并搜索重搜断言改回逐位快照（先前放宽到搜 28/29 次也绿）；`for_rollout` 补决策等价守门
- **rollout 加速 −29% CPU**：编译期消掉 rollout 路径的屏幕输出，分数逐位一致；**仅关 diag 时生效**，umasim 自己的 bin 需显式关
- **perf 诊断工具与 Windows 可构建性**：新增 `sim_profiler`；pprof 编不过 Windows，收进可选 `profiler` feature；`microbench_top_fns` 改进程级 CWD，加 `#[ignore]`

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
