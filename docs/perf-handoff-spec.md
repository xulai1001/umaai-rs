# 性能补丁交接规格：打分链残余分配开销（供上游 / 本地合并）

> 作者：xf8410/umaai-rs workbench（2026-08-27）
> 背景：本文件作者通道无法整写 policy.rs / local_ramen_trainer.rs，以下改动需在源码侧落地。
> 硬约束：**任何改动必须保持同 seed 配对结果位级一致**（闸门体系依赖该性质验收）。

## 已在 workbench 完成的部分（无需重复）

1. `[profile.release]` 由体积档（opt-level='z'、lto=false）改为速度档（O3 + thin-LTO + codegen-units=1）；
   rust 无 fast-math，优化级别不影响数值语义，上游合并且无风险。
2. `ramen_metric_compare` 跑批改 Rayon 并行（per-index 种子确定性、collect 保序 ⇒ 输出逐位不变）。

## 建议清单（按收益/风险排序）

### P0 breakdown 键去堆分配
`RamenPolicyOutput.breakdown: Vec<(String, f32)>`
→ 每候选 5~6 个短命 String；键集合是闭集（attr/pt/vital_cost/shining/fail_adj/…）。
**方案**：键改 `&'static str`（调用点本来就是字面量）；Vec 可再换 SmallVec<[(K,f32);8]>。
**守门**：`test_breakdown_sums_to_score` 不变即证兼容。

### P1 reason 惰性构造
`score_train_action` 等处的 `format!(...)` 在 release 且无日志消费者时白付，
约每决策点 × 每候选一次。**方案**：`reason` 改 `OnceLock<String>` 或搬进
diag feature（沿用 33071358939 同款的 cfg 手法，预期再砍两位数百分比 CPU）。

### P2 吃面预演深拷贝
`pre_eat_action` / `eat_covered_train_passes` / `post_ramen_vital_transition`
每个候选面 clone 整个 RamenGame（人头/卡组 Vec 全量）。
**方案**：预演专用轻量快照（只含 decide_train 读到的字段：uma 五维与体力、
ramen 当前状态、distribution 只读借用即可——预演不消费策略流 RNG）。
风险最高，务必配「同 seed 300 局逐位一致」回归后再合入。

## 验收流程建议

1. `cargo bench` 微基准（上游 `microbench_top_fns` 已就绪）取基线；
2. 合入后先跑 `init_global` 单局 verbose 对拍决策序列哈希；
3. 300 局同种子配对四元组（分数/五维/skill_pt/scenario_pt）必须逐位相等；
4. 用上游 `sim_profiler` 出前后火焰图作 PR 附录。
