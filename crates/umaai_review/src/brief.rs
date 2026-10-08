//! brief.md 渲染（LLM 复盘简报：六问事实预答）
//!
//! 定位：`digest.json` 是完整数据（200KB 级，`timeline` + `decisions` 占八成），
//! 但复盘真正要用的**结论**只有一成。本模块把结论渲染成一份 markdown 简报，
//! 让 SKILL 层**一次 Read** 即可动笔，不必再派子代理逐块取数。
//!
//! 与 [`crate::report`] 同构——**计算与排版分离**：
//! - 跨块关联与判定（`top_gain` × `decisions` × `flagged_turns` 三表关联、
//!   分段边界、存疑判定、排序）在本模块算成预格式化的 [`BriefView`]；
//! - 排版与文案交给外置模板 `templates/brief.md.j2`（改措辞/表头**不重编译**）。
//!
//! **只给事实与预格式化表格，不下叙事结论**（「运气阴跌」「程序性波动降级」
//! 这类措辞仍由 SKILL 层写）。口径常量一律复用 [`crate::checks`] /
//! [`crate::execution`]，不在此复制。

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow};
use minijinja::{Environment, context};
use serde::Serialize;

use crate::checks::{INHERIT_TURNS, LOW_VITAL, MOTIVATION_WINDOW, SUPER_RAMEN_START, YEAR_BOUNDARIES};
use crate::clones::CloneDetailRow;
use crate::decisions::LuckPoint;
use crate::digest::Digest;
use crate::execution::{ExecRow, REST_MIN_VITAL};
use crate::report::find_template;
use crate::timeline::TimelineRow;

/// 模板名（查找顺序同 `report.html.j2`）
const TEMPLATE: &str = "brief.md.j2";

/// 五维属性名（训练位索引同序）
const ATTR_NAMES: [&str; 5] = ["速", "耐", "力", "根", "智"];

/// 判「这局运气差」的运气分阈值（§6.2；方差量级估计，用户拍板）
const LUCK_BAD_THRESHOLD: f64 = -2000.0;

/// 运气走势分段（按年份边界切分，末段 = 超级拉面期）
const LUCK_SEGMENTS: [(&str, u32, u32); 4] = [
    ("第1年", 0, 23),
    ("第2年", 24, 47),
    ("第3年", 48, 71),
    ("超拉期", SUPER_RAMEN_START, u32::MAX),
];

/// 五维逐年节点（年份边界 = 下一年的起点，故「年末」取边界 − 1）
const FIVE_NODES: [(&str, u32); 3] = [("第1年末", 23), ("第2年末", 47), ("第3年末", 71)];

// —————————————————————————— 渲染入口 ——————————————————————————

/// 渲染并落盘 brief.md → 返回文件路径
pub fn render(d: &Digest, out_dir: &Path) -> Result<PathBuf> {
    let tpl = find_template(TEMPLATE).ok_or_else(|| {
        anyhow!("找不到 templates/{TEMPLATE}（查找顺序：exe 同级/上级 templates、cwd 及其祖先的 templates 与 crates/umaai_review/templates）")
    })?;
    render_with_template(d, out_dir, &tpl)
}

/// 用指定模板渲染（测试入口；正常路径走 [`render`]）
pub fn render_with_template(d: &Digest, out_dir: &Path, tpl_path: &Path) -> Result<PathBuf> {
    let source = fs::read_to_string(tpl_path)
        .with_context(|| format!("读取模板失败: {}", tpl_path.display()))?;
    let view = build_view(d);
    // 模板名以 .j2 结尾 → minijinja 不开 HTML autoescape（markdown 原样输出）
    let mut env = Environment::new();
    env.add_template(TEMPLATE, &source)?;
    let md = env
        .get_template(TEMPLATE)?
        .render(context! { v => view })?;

    fs::create_dir_all(out_dir)
        .with_context(|| format!("创建输出目录失败: {}", out_dir.display()))?;
    let out = out_dir.join("brief.md");
    fs::write(&out, md).with_context(|| format!("写 brief.md 失败: {}", out.display()))?;
    Ok(out)
}

// —————————————————————————— 视图模型（全部已预格式化） ——————————————————————————

/// brief.md 渲染视图：**数值与判定已在 Rust 侧算好并格式化**，模板只负责铺排
#[derive(Debug, Default, Serialize)]
pub struct BriefView {
    pub game: u64,
    // §0 口径
    pub luck_formula: String,
    pub criteria: Vec<String>,
    // §1 总体
    pub uma_line: String,
    pub deck_line: String,
    pub score_line: String,
    pub match_line: String,
    pub luck_line: String,
    pub health_line: String,
    // §1.1 五维
    pub five_rows: Vec<FiveRow>,
    pub five_limit: String,
    pub five_capped: String,
    // §1.2 训练画像（增长顺序与运气归属）
    pub training_rows: Vec<String>,
    pub training_luck: String,
    // §2 运气走势
    pub series_count: usize,
    pub year_boundaries: String,
    pub super_ramen_start: u32,
    pub segments: Vec<SegmentRow>,
    pub luck_last: String,
    // §3 极值回合
    pub extremes: Vec<ExtremeBlock>,
    // §4 继承
    pub inherit_available: bool,
    pub inherit_ref: String,
    pub inherit_rows: Vec<InheritRow>,
    pub inherit_note: String,
    // §5 其他波动来源
    pub vital_lows: String,
    pub vital_min: String,
    pub motivation_rows: Vec<String>,
    pub clones_available: bool,
    pub clone_a: String,
    pub clone_b: String,
    /// 地区分身逐次彩圈明细（只含产生了彩圈的；report.html 共用同一构造）
    pub region_rows: Vec<CloneDetailRow>,
    pub mandatory: String,
    pub free_race_rows: Vec<String>,
    pub schedule_notes: Vec<String>,
    // §6 建议执行
    pub deviations: Vec<DeviationRow>,
    // §7 检查项
    pub findings: Vec<FindingRow>,
}

/// 五维一行（节点 + 五维显示值）
#[derive(Debug, Default, Serialize)]
pub struct FiveRow {
    pub label: String,
    pub v: Vec<String>,
}

/// 运气分段一行
#[derive(Debug, Default, Serialize)]
pub struct SegmentRow {
    pub name: String,
    pub turns: String,
    pub before: String,
    pub after: String,
    pub net: String,
    pub minmax: String,
}

/// 一个极值回合块
#[derive(Debug, Default, Serialize)]
pub struct ExtremeBlock {
    pub label: String,
    pub turn: u32,
    pub total: String,
    /// 「分段（N 段）：[..]」
    pub seg_note: String,
    /// 程序性波动原因（无则「无」）
    pub flagged: String,
    /// 该回合各决策行（已格式化）
    pub decisions: Vec<String>,
    pub rainbow: String,
}

/// 继承一行
#[derive(Debug, Default, Serialize)]
pub struct InheritRow {
    pub turn: u32,
    pub contrib: String,
    pub deviation: String,
    pub verdict: String,
}

/// 偏离一行（含存疑判定）
#[derive(Debug, Default, Serialize)]
pub struct DeviationRow {
    pub turn: u32,
    pub stage: String,
    pub ai: String,
    pub actual: String,
    /// 实际动作对应的候选（`#2 力训练`；偏离主要是选了 2-3 选——`—` = 候选外）
    pub alt: String,
    pub five: String,
    pub vital: String,
    pub doubt: String,
}

/// 检查项一行
#[derive(Debug, Default, Serialize)]
pub struct FindingRow {
    pub severity: String,
    pub kind: String,
    pub turn: u32,
    pub evidence: String,
}

/// 组装视图（纯函数：无 IO，便于单测）
pub fn build_view(d: &Digest) -> BriefView {
    let states = turn_states(&d.timeline);
    let mut v = BriefView {
        game: d.meta.game,
        luck_formula: squeeze(&d.context.luck_formula),
        criteria: d.context.criteria.iter().map(|c| squeeze(c)).collect(),
        series_count: d.luck.series.len(),
        year_boundaries: format!("{YEAR_BOUNDARIES:?}"),
        super_ramen_start: SUPER_RAMEN_START,
        inherit_available: d.inherit.is_some(),
        clones_available: d.clones.is_some(),
        ..Default::default()
    };

    fill_overview(&mut v, d, &states);
    fill_five(&mut v, d, &states);
    fill_training(&mut v, d);
    fill_luck(&mut v, d);
    fill_extremes(&mut v, d);
    fill_inherit(&mut v, d);
    fill_other_sources(&mut v, d, &states);
    fill_deviations(&mut v, d, &states);
    fill_findings(&mut v, d);
    v
}

// —————————————————————————— §1 总体 ——————————————————————————

fn fill_overview(v: &mut BriefView, d: &Digest, states: &BTreeMap<u32, TurnState>) {
    v.uma_line = format!(
        "马娘：{}（id {}）  局号 {}  结束 {}  中途接续 {}  起始回合 {}",
        d.meta.uma_name,
        d.meta.uma_id,
        d.meta.game,
        d.meta.end_reason,
        if d.meta.mid_entry { "是" } else { "否" },
        d.meta.start_turn
    );
    let deck = d
        .meta
        .deck
        .iter()
        .map(|c| {
            // 卡名本身多带 `[速]` 前缀，避免重复；缺失时才补类型标签
            if c.name.starts_with('[') {
                format!("{} LB{}", c.name, c.limit_break)
            } else {
                format!("{}{} LB{}", type_tag(c.card_type), c.name, c.limit_break)
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");
    v.deck_line = if deck.is_empty() {
        "—".to_string()
    } else {
        format!("{} 张：{}", d.meta.deck.len(), deck)
    };
    v.score_line = match (&d.meta.final_score, &d.meta.rank) {
        // 数据来源分流：真机终局帧（含全部结局事件）vs 末快照估算（缺结局事件 ≈ -2700）。
        // 评分 = AI 端按实际终局五维 + 技能点的估算（PT 折算高于实际买分）→ 略高于小黑板实际
        (Some(score), Some(rank)) => {
            let state = match (d.meta.is_qiezhe, d.meta.is_xiao_qie) {
                (true, _) => "，已计切者 PT×1.1",
                (false, true) => "，已计小切 PT×1.04",
                _ => ""
            };
            format!(
                "终局评分：{score}（{rank}）  口径：AI 端按实际终局五维+技能点估算（略高于小黑板实际）{state}，不含已学技能分；来源 {}",
                if d.meta.final_source == "final_frame" {
                    "真机终局帧"
                } else {
                    "末快照（另缺结局事件 ≈ -2700，相对再偏低）"
                }
            )
        }
        _ => "终局评分：不可用（gamedata 缺失，纯 ID 口径）".to_string(),
    };
    let (comparable, matched, mismatch) = exec_counts(&d.execution);
    v.match_line = if comparable > 0 {
        format!(
            "执行一致率：{:.1}%（可比 {comparable} / 一致 {matched} / 偏离 {mismatch}）",
            matched as f64 / comparable as f64 * 100.0
        )
    } else {
        "执行一致率：无可比行（执行推断不可用）".to_string()
    };
    v.luck_line = match d.meta.total_luck_end {
        Some(x) => {
            let verdict = if x < LUCK_BAD_THRESHOLD {
                format!("判「这局运气差」（阈值 {LUCK_BAD_THRESHOLD:.0}）")
            } else {
                format!("未达判「运气差」阈值（{LUCK_BAD_THRESHOLD:.0}）")
            };
            format!("终局运气分：{}  → {verdict}", signed_f(x))
        }
        None => "终局运气分：不可用".to_string(),
    };
    let coverage = match (d.timeline.first(), d.timeline.last()) {
        (Some(a), Some(b)) => format!("{}..={}", a.turn, b.turn),
        _ => "—".to_string(),
    };
    let skips = d
        .coverage
        .skip
        .by_reason
        .iter()
        .map(|(k, n)| format!("{k}:{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    v.health_line = format!(
        "数据健康度：快照 {} 份  决策行 {}  回合覆盖 {coverage}  快照解析失败 {}  未识别 {}  未派发 {}  skip 分原因 [{}]",
        d.meta.snapshots,
        d.meta.decision_rows,
        d.coverage.parse_error,
        d.coverage.unparsed,
        d.coverage.no_emit,
        if skips.is_empty() { "无".to_string() } else { skips }
    );
    let _ = states; // 供后续小节复用，本节不取
}

// —————————————————————————— §1.1 五维 ——————————————————————————

fn fill_five(v: &mut BriefView, d: &Digest, states: &BTreeMap<u32, TurnState>) {
    let last_turn = d.timeline.last().map(|r| r.turn);
    let mut nodes: Vec<(&str, Option<u32>)> =
        FIVE_NODES.iter().map(|(label, t)| (*label, Some(*t))).collect();
    if let Some(t) = last_turn {
        nodes.push(("终局", Some(t)));
    }
    for (label, turn) in nodes {
        let Some(st) = turn.and_then(|t| states.get(&t)) else {
            continue;
        };
        v.five_rows.push(FiveRow {
            label: label.to_string(),
            v: st.display.iter().map(|x| x.to_string()).collect(),
        });
    }
    let Some(last) = last_turn.and_then(|t| states.get(&t)) else {
        v.five_limit = "—".to_string();
        v.five_capped = "五维：无 timeline 数据".to_string();
        return;
    };
    v.five_limit = last.limit.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" | ");
    let capped: Vec<String> = (0..5)
        .filter(|&i| last.real[i] >= last.limit[i])
        .map(|i| format!("{}（{}/{}）", ATTR_NAMES[i], last.real[i], last.limit[i]))
        .collect();
    v.five_capped = if capped.is_empty() {
        "触顶维：无".to_string()
    } else {
        format!("触顶维（真实值达上限）：{}", capped.join("、"))
    };
}

// —————————————————————————— §1.2 训练画像 ——————————————————————————

/// 训练画像（逐年「训练次数 / 五维净增」一行 + 运气归属行）
///
/// 增长先后顺序（如先耐后速）与「涨运气的训练类型」由 SKILL 层结合卡组判读；
/// 本模块只做预格式化数字行（口径见 `profile` 模块头）。
fn fill_training(v: &mut BriefView, d: &Digest) {
    for y in &d.training.years {
        let counts = y
            .train_counts
            .iter()
            .zip(ATTR_NAMES.iter())
            .map(|(&c, n)| format!("{n}{c}"))
            .collect::<Vec<_>>()
            .join("/");
        let gains = y
            .gains
            .iter()
            .zip(ATTR_NAMES.iter())
            .map(|(&g, n)| format!("{n}{g:+}"))
            .collect::<Vec<_>>()
            .join("/");
        v.training_rows.push(format!(
            "{}：训练 {} 次；净增 {}（含事件/继承/比赛）",
            y.label, counts, gains
        ));
    }
    let luck = ATTR_NAMES
        .iter()
        .zip(d.training.luck_by_attr.iter())
        .map(|(n, &l)| format!("{n}{}", signed_f(l)))
        .collect::<Vec<_>>()
        .join(" / ");
    v.training_luck = format!(
        "运气归属（回合合计 Δ 计入当回合实际训练维）：{luck} / 其他 {} / 程序性大波动回合 {}（自选 swing 与学技能，已剔除）",
        signed_f(d.training.luck_other),
        signed_f(d.training.luck_program)
    );
}

// —————————————————————————— §2 运气走势 ——————————————————————————

fn fill_luck(v: &mut BriefView, d: &Digest) {
    let last_turn = d.timeline.last().map(|r| r.turn).unwrap_or(77);
    for (name, lo, hi) in LUCK_SEGMENTS {
        let hi_shown = if hi == u32::MAX { last_turn } else { hi };
        let in_seg = |p: &&LuckPoint| p.turn >= lo && p.turn <= hi;
        // 段末 = 段内最后一个点；段前 = 段前最后一个点（无则退化为段内首点）。
        // 「段前」口径与 checks::super_ramen_stats 的累加边界一致——该函数累加
        // turn>=72 的行，而行的 turn_delta 语义是「变化到达本行」，故实际覆盖
        // 的是「段前最后一点 → 段末」，两处数字必须同源。
        let end = d.luck.series.iter().filter(in_seg).next_back();
        let start = d
            .luck
            .series
            .iter()
            .filter(|p| p.turn < lo)
            .next_back()
            .or_else(|| d.luck.series.iter().filter(in_seg).next());
        let Some(end) = end else {
            v.segments.push(SegmentRow {
                name: name.to_string(),
                turns: format!("{lo}..={hi_shown}"),
                before: "—".to_string(),
                after: "—".to_string(),
                net: "—".to_string(),
                minmax: "无数据".to_string(),
            });
            continue;
        };
        let start_v = start.map(|p| p.total_luck).unwrap_or(0.0);
        let pts: Vec<f64> = d.luck.series.iter().filter(in_seg).map(|p| p.total_luck).collect();
        let lo_v = pts.iter().copied().fold(f64::MAX, f64::min);
        let hi_v = pts.iter().copied().fold(f64::MIN, f64::max);
        v.segments.push(SegmentRow {
            name: name.to_string(),
            turns: format!("{lo}..={hi_shown}"),
            before: signed_f(start_v),
            after: signed_f(end.total_luck),
            net: signed_f(end.total_luck - start_v),
            minmax: format!("{} / {}", signed_f(lo_v), signed_f(hi_v)),
        });
    }
    v.luck_last = d
        .luck
        .series
        .last()
        .map(|p| signed_f(p.total_luck))
        .unwrap_or_else(|| "—".to_string());
}

// —————————————————————————— §3 极值回合 ——————————————————————————

fn fill_extremes(v: &mut BriefView, d: &Digest) {
    for (label, list) in [("正跳", &d.luck.top_gain), ("负跳", &d.luck.top_loss)] {
        for t in list {
            let segs = t.segments.iter().map(|x| signed_f(*x)).collect::<Vec<_>>().join(", ");
            let flags = flag_reasons(d, t.turn);
            let mut decisions = Vec::new();
            for r in d.decisions.iter().filter(|r| r.turn == t.turn) {
                let gap = r
                    .candidates
                    .iter()
                    .find(|c| c.rank == 2)
                    .and_then(|c| c.gap_to_best);
                decisions.push(format!(
                    "决策 {}/{}：{}（领先 #2 {} 分，action_luck {}）",
                    r.stage,
                    r.decision_kind,
                    if r.chosen.desc.is_empty() { "—" } else { &r.chosen.desc },
                    gap.map(|g| format!("{g:.1}")).unwrap_or_else(|| "—".to_string()),
                    r.chosen
                        .action_luck
                        .map(signed_f)
                        .unwrap_or_else(|| "—".to_string())
                ));
            }
            let rainbow = rainbow_at(d, t.turn);
            v.extremes.push(ExtremeBlock {
                label: label.to_string(),
                turn: t.turn,
                total: signed_f(t.delta),
                seg_note: format!("分段（{} 段）：[{}]", t.segments.len(), segs),
                flagged: if flags.is_empty() {
                    "无".to_string()
                } else {
                    flags.join(" / ")
                },
                decisions,
                rainbow: if rainbow.is_empty() {
                    "无".to_string()
                } else {
                    rainbow.join(" / ")
                },
            });
        }
    }
}

// —————————————————————————— §4 继承 ——————————————————————————

fn fill_inherit(v: &mut BriefView, d: &Digest) {
    let Some(inh) = &d.inherit else {
        return;
    };
    v.inherit_ref = format!(
        "{}（回合 {:?}）",
        inh.reference_value
            .map(|x| x.to_string())
            .unwrap_or_else(|| "—（配置不可用）".to_string()),
        INHERIT_TURNS
    );
    for c in &inh.contributions {
        let verdict = match c.deviation {
            None => "—",
            Some(x) if x > 0 => "偏优",
            Some(x) if x < 0 => "偏弱",
            Some(_) => "正常",
        };
        v.inherit_rows.push(InheritRow {
            turn: c.turn,
            contrib: signed_i(c.five_status_sum),
            deviation: c.deviation.map(signed_i).unwrap_or_else(|| "—".to_string()),
            verdict: verdict.to_string(),
        });
    }
    v.inherit_note = squeeze(&inh.dev_note);
}

// —————————————————————————— §5 其他波动来源 ——————————————————————————

fn fill_other_sources(v: &mut BriefView, d: &Digest, states: &BTreeMap<u32, TurnState>) {
    // 5.1 体力低点
    let lows: Vec<String> = states
        .iter()
        .filter(|(_, st)| st.vital < LOW_VITAL)
        .map(|(t, st)| format!("t{}({}/{})", t, st.vital, st.max_vital))
        .collect();
    v.vital_lows = if lows.is_empty() {
        format!("无（全部回合体力 >= {LOW_VITAL}）")
    } else {
        format!("{} 个回合：{}", lows.len(), lows.join(", "))
    };
    v.vital_min = states
        .iter()
        .min_by_key(|(_, st)| st.vital)
        .map(|(t, st)| format!("全局最低：t{} = {}/{}", t, st.vital, st.max_vital))
        .unwrap_or_else(|| "全局最低：无数据".to_string());

    // 5.2 干劲掉落**未及时恢复**（及时处理的不提——用户拍板）
    let turns: Vec<u32> = states.keys().copied().collect();
    for pair in turns.windows(2) {
        let (t0, t1) = (pair[0], pair[1]);
        let (before, after) = (states[&t0].motivation, states[&t1].motivation);
        if after >= before {
            continue;
        }
        // 判据与 checks::bad_habits 同口径：窗口内既无恢复动作、干劲也未回升
        let acted = has_recovery_action(&d.execution, t1);
        let rose = (1..=MOTIVATION_WINDOW)
            .any(|k| states.get(&(t1 + k)).is_some_and(|st| st.motivation > after));
        if acted || rose {
            continue; // 及时处理 / 已回升 → 不提
        }
        v.motivation_rows.push(format!(
            "回合 {}：{}→{}，{} 回合内既无出行/休息恢复动作、干劲也未回升",
            t1, before, after, MOTIVATION_WINDOW
        ));
    }
    if v.motivation_rows.is_empty() {
        v.motivation_rows.push("无未及时处理的掉干劲".to_string());
    }

    // 5.3 分身彩圈
    if let Some(cl) = &d.clones {
        let a = &cl.region;
        v.clone_a = format!(
            "地区分身（turn < {}）：新增 {} / 落得意位 {} / 被训练 {} / 随机 {} / 规则 {}",
            SUPER_RAMEN_START, a.new_clones, a.rainbow_clones, a.trained_clones, a.rainbow_luck, a.rainbow_strategy
        );
        let b = &cl.super_ramen_clones;
        v.clone_b = format!(
            "超级拉面分身（turn >= {}）：新增 {} / 落得意位 {} / 被训练 {} / 随机 {} / 规则 {}",
            SUPER_RAMEN_START, b.new_clones, b.rainbow_clones, b.trained_clones, b.rainbow_luck, b.rainbow_strategy
        );
        // 地区分身逐次彩圈明细（只含彩圈；构造在 clones.rs，report.html 共用）
        let deck_names: Vec<String> = d.meta.deck.iter().map(|c| c.name.clone()).collect();
        v.region_rows = cl.region_detail_rows(&deck_names);
    }

    // 5.4 赛程
    let mandatory: Vec<String> = d.schedule.mandatory_turns.iter().map(|t| t.to_string()).collect();
    v.mandatory = if mandatory.is_empty() {
        "—".to_string()
    } else {
        mandatory.join(", ")
    };
    for f in &d.schedule.free_races {
        let picked: Vec<String> = f.picked_turns.iter().map(|t| t.to_string()).collect();
        v.free_race_rows.push(format!(
            "{}..={}：要求 {} 次，区间内实跑 {}",
            f.start_turn,
            f.end_turn,
            f.required,
            if picked.is_empty() { "无".to_string() } else { picked.join(", ") }
        ));
    }
    if v.free_race_rows.is_empty() {
        v.free_race_rows
            .push("无（该马娘没有自由比赛区间数据）".to_string());
    }
    v.schedule_notes = d.schedule.notes.iter().map(|n| squeeze(n)).collect();
}

// —————————————————————————— §6 建议执行 ——————————————————————————

fn fill_deviations(v: &mut BriefView, d: &Digest, states: &BTreeMap<u32, TurnState>) {
    for r in d.execution.iter().filter(|r| r.matches == Some(false)) {
        v.deviations.push(DeviationRow {
            turn: r.turn,
            stage: r.stage.clone(),
            ai: r.ai_choice.clone(),
            actual: r.actual_action.clone(),
            alt: r
                .alt_candidate
                .clone()
                .unwrap_or_else(|| "—".to_string()),
            five: r
                .evidence
                .five_status_delta
                .iter()
                .map(|x| signed_i(*x))
                .collect::<Vec<_>>()
                .join(", "),
            vital: signed_i(r.evidence.vital_delta),
            doubt: doubt_reason(r, states.get(&r.turn)).unwrap_or_else(|| "—".to_string()),
        });
    }
}

// —————————————————————————— §7 检查项 ——————————————————————————

fn fill_findings(v: &mut BriefView, d: &Digest) {
    // warn 优先，其次按回合升序
    let mut list: Vec<_> = d.findings.iter().collect();
    list.sort_by(|a, b| {
        let rank = |sev: &str| if sev == "warn" { 0 } else { 1 };
        rank(&a.severity)
            .cmp(&rank(&b.severity))
            .then(a.turn.cmp(&b.turn))
    });
    for f in list {
        v.findings.push(FindingRow {
            severity: f.severity.clone(),
            kind: f.kind.clone(),
            turn: f.turn,
            evidence: squeeze(&f.evidence),
        });
    }
}

// —————————————————————————— 取数与判定辅助 ——————————————————————————

/// 回合末状态（timeline 升序 → 后写覆盖 = 回合末）
#[derive(Debug, Default, Clone, Copy)]
struct TurnState {
    vital: i32,
    max_vital: i32,
    motivation: i32,
    /// 真实五维（触顶判定用）
    real: [i32; 5],
    /// 显示值五维（叙事口径）
    display: [i32; 5],
    limit: [i32; 5],
}

fn turn_states(tl: &[TimelineRow]) -> BTreeMap<u32, TurnState> {
    let mut m: BTreeMap<u32, TurnState> = BTreeMap::new();
    for r in tl {
        m.insert(
            r.turn,
            TurnState {
                vital: r.vital,
                max_vital: r.max_vital,
                motivation: r.motivation,
                real: r.five_status,
                display: r.five_status_display,
                limit: r.five_status_limit,
            },
        );
    }
    m
}

/// 执行一致率三元组（可比 / 一致 / 偏离）
fn exec_counts(rows: &[ExecRow]) -> (usize, usize, usize) {
    let comparable = rows.iter().filter(|r| r.matches.is_some()).count();
    let matched = rows.iter().filter(|r| r.matches == Some(true)).count();
    (comparable, matched, comparable - matched)
}

/// 该回合的伪波动标记原因
fn flag_reasons(d: &Digest, turn: u32) -> Vec<String> {
    d.luck
        .flagged_turns
        .iter()
        .filter(|f| f.turn == turn)
        .map(|f| f.reason.clone())
        .collect()
}

/// 该回合的 A 类彩圈（逐卡描述）
fn rainbow_at(d: &Digest, turn: u32) -> Vec<String> {
    let Some(cl) = &d.clones else {
        return Vec::new();
    };
    cl.region_per_turn
        .iter()
        .filter(|ct| ct.turn == turn)
        .flat_map(|ct| {
            ct.cards
                .iter()
                .filter(|c| !c.rainbow_positions.is_empty())
                .map(|c| {
                    format!(
                        "card{} 位[{}] 来源={} 吃到={}",
                        c.card,
                        c.rainbow_positions
                            .iter()
                            .map(|p| p.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                        c.origin.clone().unwrap_or_else(|| "—".to_string()),
                        c.used.map(|u| if u { "是" } else { "否" }).unwrap_or("—")
                    )
                })
        })
        .collect()
}

/// 观察窗口内是否有恢复动作（出行 / 友人出行 / 休息）
fn has_recovery_action(exec: &[ExecRow], turn: u32) -> bool {
    exec.iter()
        .filter(|r| r.turn >= turn && r.turn <= turn + MOTIVATION_WINDOW)
        .any(|r| ["出行", "友人出行", "休息"].iter().any(|a| r.actual_action.starts_with(a)))
}

/// 偏离是否存疑（推断失准的已知情形）→ 返回原因
fn doubt_reason(r: &ExecRow, st: Option<&TurnState>) -> Option<String> {
    // ① 体力卡在休息判定线（休息 + 事件小额属性 与 智训练大回复 难分）
    if r.actual_action.starts_with("休息") && (r.evidence.vital_delta - REST_MIN_VITAL).abs() <= 3 {
        return Some(format!(
            "体力 {} 卡休息判定线 {}",
            signed_i(r.evidence.vital_delta),
            REST_MIN_VITAL
        ));
    }
    // 建议不是训练 → 无法用属性上限口径判读
    let ai = attr_idx(&r.ai_choice)?;
    // ② 建议维已触顶：溢出使相邻维增量反超，被误判成该相邻维训练
    if let Some(st) = st {
        if st.real[ai] >= st.limit[ai] {
            return Some(format!(
                "建议维{}已触顶（{}/{}），溢出使相邻维反超",
                ATTR_NAMES[ai], st.real[ai], st.limit[ai]
            ));
        }
    }
    // ③ 推断维与建议维相邻：主增量维可能误判
    if let Some(b) = attr_idx(&r.actual_action) {
        if b != ai && (b as i32 - ai as i32).abs() == 1 {
            return Some(format!(
                "推断维{}与建议维{}相邻，主增量维可能误判",
                ATTR_NAMES[b], ATTR_NAMES[ai]
            ));
        }
    }
    None
}

/// 「速训练」→ 0..4；非训练动作 → None
fn attr_idx(action: &str) -> Option<usize> {
    ATTR_NAMES
        .iter()
        .position(|n| action.starts_with(&format!("{n}训练")))
}

/// 卡类型标签（0速 1耐 2力 3根 4智 5友人 6团队）
fn type_tag(t: i32) -> &'static str {
    match t {
        0 => "[速]",
        1 => "[耐]",
        2 => "[力]",
        3 => "[根]",
        4 => "[智]",
        5 => "[友]",
        6 => "[团]",
        _ => "[?]",
    }
}

/// 带符号整数（f64 四舍五入）
fn signed_f(x: f64) -> String {
    format!("{:+}", x.round() as i64)
}

/// 带符号整数
fn signed_i(x: i32) -> String {
    format!("{x:+}")
}

/// 压平换行与续行缩进（criteria / evidence 里带 Rust 字面量换行）
fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decisions::{Cand, Chosen, LuckBlock},
        digest::{DigestContext, DeckCard, Meta},
        execution::{Evidence, Finding},
        schedule::Schedule,
        timeline::TimelineRow,
    };

    fn meta() -> Meta {
        Meta {
            game: 1,
            uma_id: 103202,
            uma_name: "测试马".to_string(),
            deck: vec![DeckCard {
                card_id: 303114,
                name: "[速]测试卡".to_string(),
                card_type: 0,
                limit_break: 4,
            }],
            start_turn: 0,
            mid_entry: false,
            end_reason: "game_end".to_string(),
            snapshots: 2,
            decision_rows: 1,
            total_luck_end: Some(-2976.73),
            final_score: Some(67962),
            rank: Some("US4".to_string()),
            final_source: "last_snapshot".to_string(),
            is_qiezhe: false,
            is_xiao_qie: false,
        }
    }

    fn empty_digest() -> Digest {
        Digest {
            meta: meta(),
            timeline: vec![],
            decisions: vec![],
            execution: vec![],
            luck: LuckBlock::default(),
            schedule: Schedule::default(),
            inherit: None,
            clones: None,
            training: Default::default(),
            coverage: Default::default(),
            findings: vec![],
            context: DigestContext {
                region_names: Default::default(),
                luck_formula: "total_luck = T(n) − T(1)".to_string(),
                criteria: vec!["YEAR_BOUNDARIES=[24,48,72]".to_string()],
            },
        }
    }

    /// 构造 timeline 行（只填简报相关字段；显示值 = 真实值）
    fn tl_row(turn: u32, vital: i32, motivation: i32, five: [i32; 5], limit: [i32; 5]) -> TimelineRow {
        TimelineRow {
            turn,
            seq: 0,
            stage: "Train".to_string(),
            reason: None,
            source: "command".to_string(),
            playing_state: 1,
            vital,
            max_vital: 100,
            motivation,
            five_status: five,
            five_status_display: five,
            five_status_limit: limit,
            skill_pt: 0,
            train_level_count: [1; 5],
            friend_outgoing_used: 0,
            selected_regions: vec![],
            scenario_pt: 0,
            feeling_stock: vec![],
            super_ramen: -1,
            is_ill: false,
            is_qiezhe: false,
            is_xiao_qie: false,
            race_count: 0,
            absent_persons: vec![],
        }
    }

    /// 视图骨架：关键数字与降级文案
    #[test]
    fn test_view_skeleton() {
        let v = build_view(&empty_digest());
        println!("{v:#?}");
        assert_eq!(v.game, 1);
        assert!(v.uma_line.contains("测试马"));
        assert!(v.score_line.contains("67962"));
        assert!(v.score_line.contains("不含已学技能分"), "点技能前口径注记");
        assert!(!v.score_line.contains("切者"), "无切者不打状态括注");
        let mut dq = empty_digest();
        dq.meta.is_qiezhe = true;
        let vq = build_view(&dq);
        println!("切者局 score_line: {}", vq.score_line);
        assert!(vq.score_line.contains("已计切者 PT×1.1"), "切者局应打状态括注");
        assert!(v.luck_line.contains("-2977"), "终局运气分四舍五入");
        assert!(v.luck_line.contains("判「这局运气差」"));
        assert!(v.health_line.contains("数据健康度"));
        assert!(v.deck_line.contains("[速]测试卡"), "卡名带前缀时不重复补标签");
        assert!(!v.deck_line.contains("[速][速]"), "不应重复标签");
        assert!(!v.inherit_available && v.inherit_rows.is_empty());
        assert!(!v.clones_available);
        assert_eq!(v.segments.len(), 4, "四段运气走势恒定输出");
        assert!(v.five_capped.contains("五维：无 timeline 数据"), "无 timeline 应降级");
    }

    /// 五维：逐年节点 + 终局 + 上限 + 触顶维
    #[test]
    fn test_five_rows_and_capped() {
        let mut d = empty_digest();
        d.timeline = vec![
            tl_row(23, 80, 4, [500, 400, 300, 300, 600], [1200; 5]),
            tl_row(77, 80, 4, [1200, 900, 700, 700, 1100], [1200; 5]),
        ];
        let v = build_view(&d);
        println!("{v:#?}");
        assert_eq!(v.five_rows.len(), 2, "有数据的节点：第1年末 + 终局");
        assert_eq!(v.five_rows[0].label, "第1年末");
        assert_eq!(v.five_rows[0].v, vec!["500", "400", "300", "300", "600"]);
        assert_eq!(v.five_rows[1].label, "终局");
        assert_eq!(v.five_limit, "1200 | 1200 | 1200 | 1200 | 1200");
        assert!(v.five_capped.contains("速（1200/1200）"), "速达上限应列出");
        assert!(!v.five_capped.contains("耐"), "耐 900 未触顶不应列出");
    }

    /// 分段语义：段前取「段前最后一个点」，相邻段首尾相接
    #[test]
    fn test_luck_segments() {
        let mut d = empty_digest();
        d.luck.series = vec![
            LuckPoint { turn: 0, seq: 0, total_luck: 0.0 },
            LuckPoint { turn: 23, seq: 0, total_luck: -540.0 },
            LuckPoint { turn: 47, seq: 0, total_luck: -707.0 },
            LuckPoint { turn: 60, seq: 0, total_luck: -1292.0 },
            LuckPoint { turn: 72, seq: 0, total_luck: -2197.0 },
            LuckPoint { turn: 76, seq: 0, total_luck: -2589.2 },
        ];
        d.timeline = vec![tl_row(76, 80, 4, [0; 5], [1200; 5])];
        let v = build_view(&d);
        let seg = |name: &str| v.segments.iter().find(|s| s.name == name).unwrap();
        assert_eq!((seg("第1年").before.as_str(), seg("第1年").after.as_str()), ("+0", "-540"));
        assert_eq!((seg("第2年").before.as_str(), seg("第2年").net.as_str()), ("-540", "-167"));
        assert_eq!(
            (seg("超拉期").before.as_str(), seg("超拉期").net.as_str()),
            ("-1292", "-1297"),
            "超拉期段前 = turn 72 之前的最后一点（与 §7 super_ramen_luck 同源）"
        );
        assert_eq!(v.luck_last, "-2589");
    }

    /// 极值回合：合计 Δ + flagged + 决策 + 彩圈
    #[test]
    fn test_extremes() {
        let mut d = empty_digest();
        d.luck.top_gain = vec![crate::decisions::TurnDelta {
            turn: 47,
            delta: 2491.0,
            segments: vec![252.0, 31.0, 2208.0],
        }];
        d.luck.flagged_turns = vec![crate::decisions::FlaggedTurn {
            turn: 47,
            reason: "year_boundary(48)".to_string(),
        }];
        d.decisions = vec![crate::decisions::DecRow {
            file: "f47.json".to_string(),
            turn: 47,
            seq: 0,
            stage: "Train".to_string(),
            decision_kind: "train".to_string(),
            candidates: vec![
                Cand { rank: 1, desc: "智训练".to_string(), score: Some(67000.0), n: Some(8192), gap_to_best: Some(0.0) },
                Cand { rank: 2, desc: "速训练".to_string(), score: Some(66900.0), n: Some(7168), gap_to_best: Some(100.0) },
            ],
            chosen: Chosen { idx: Some(0), desc: "智训练".to_string(), action_luck: Some(81.04) },
            t_n_raw: None,
            t_n_display: None,
            total_luck: None,
            turn_delta: None,
            chain_len: 1,
            outcome: "calc".to_string(),
            reason: String::new(),
            step: 0,
        }];
        let v = build_view(&d);
        let e = &v.extremes[0];
        assert_eq!((e.label.as_str(), e.turn, e.total.as_str()), ("正跳", 47, "+2491"));
        assert_eq!(e.seg_note, "分段（3 段）：[+252, +31, +2208]");
        assert_eq!(e.flagged, "year_boundary(48)");
        assert_eq!(e.decisions.len(), 1);
        assert!(e.decisions[0].contains("智训练（领先 #2 100.0 分"));
        assert_eq!(e.rainbow, "无");
    }

    /// 偏离表：存疑三口径（触顶 / 相邻维 / 休息判定线）
    #[test]
    fn test_deviation_doubt() {
        let mut d = empty_digest();
        // 速已触顶 1200/1200；建议速训练、推断力训练（相邻维）
        d.timeline = vec![tl_row(70, 50, 4, [1200, 800, 700, 700, 900], [1200; 5])];
        d.execution = vec![
            ExecRow {
                turn: 70,
                stage: "Train".to_string(),
                ai_choice: "速训练".to_string(),
                actual_action: "力训练".to_string(),
                matches: Some(false),
                evidence: Evidence { five_status_delta: [0, 0, 99, 0, 0], vital_delta: -20, ..Default::default() },
                alt_candidate: Some("#2 力训练".to_string()),
            },
            ExecRow {
                turn: 70,
                stage: "Train".to_string(),
                ai_choice: "智训练".to_string(),
                actual_action: "休息".to_string(),
                matches: Some(false),
                evidence: Evidence { five_status_delta: [0, 0, 0, 0, 0], vital_delta: 25, ..Default::default() },
                alt_candidate: None,
            },
        ];
        let v = build_view(&d);
        println!("{v:#?}");
        assert_eq!(v.deviations.len(), 2);
        assert_eq!(v.deviations[0].alt, "#2 力训练", "对应候选列");
        assert_eq!(v.deviations[1].alt, "—", "无对应候选 → 占位符");
        assert!(v.deviations[0].doubt.contains("建议维速已触顶（1200/1200）"));
        assert!(v.deviations[1].doubt.contains("卡休息判定线 25"));
        assert!(v.match_line.contains("0.0%（可比 2 / 一致 0 / 偏离 2）"));
    }

    /// findings 排序：warn 在前
    #[test]
    fn test_findings_order() {
        let mut d = empty_digest();
        d.findings = vec![
            Finding { kind: "execution_mismatch".to_string(), turn: 15, file: None, evidence: "偏离".to_string(), severity: "info".to_string() },
            Finding { kind: "super_ramen_luck".to_string(), turn: 72, file: None, evidence: "判亏".to_string(), severity: "warn".to_string() },
        ];
        let v = build_view(&d);
        assert_eq!(v.findings[0].severity, "warn");
        assert_eq!(v.findings[1].severity, "info");
    }

    /// 模板端到端：渲染 → 校验章节与关键内容（模板经 find_template 定位，
    /// 测试 cwd 应为 workspace 根）
    #[test]
    fn test_render_brief() -> Result<()> {
        let mut d = empty_digest();
        d.timeline = vec![tl_row(77, 80, 4, [1200, 900, 700, 700, 1100], [1200; 5])];
        d.luck.series = vec![
            LuckPoint { turn: 0, seq: 0, total_luck: 0.0 },
            LuckPoint { turn: 76, seq: 0, total_luck: -2589.2 },
        ];
        let tpl = find_template(TEMPLATE)
            .ok_or_else(|| anyhow!("测试环境找不到模板（cwd 应为 workspace 根）"))?;
        println!("模板: {}", tpl.display());
        let out_dir = std::env::temp_dir().join(format!("brief_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&out_dir);
        let out = render_with_template(&d, &out_dir, &tpl)?;
        let md = fs::read_to_string(&out)?;
        println!("brief.md {} 字节\n{md}", md.len());
        assert!(md.contains("# 复盘简报 · game1"));
        for section in [
            "## 0 口径",
            "## 1 总体",
            "### 1.1 五维",
            "## 2 运气走势",
            "## 3 极值回合",
            "## 4 继承质量",
            "## 5 其他波动来源",
            "## 6 建议执行",
            "## 7 检查项",
        ] {
            assert!(md.contains(section), "缺章节 {section}");
        }
        assert!(md.contains("| 终局 | 1200 | 900 | 700 | 700 | 1100 |"), "五维终局行应入表");
        assert!(md.contains("触顶维（真实值达上限）：速（1200/1200）"));
        assert!(md.contains("继承块不可用"), "无继承块应降级说明");
        assert!(md.contains("自由比赛区间：无"), "无自由比赛应如实说明");
        assert!(md.contains("无偏离"), "无偏离应说明");
        assert!(md.contains("无命中"), "无 findings 应说明");
        assert!(!md.contains("{{"), "不应残留未渲染的模板标记");
        // 表格行数应等于分段数（4 段 + 终局行），不因模板循环产生多余空行
        let seg_rows = md.lines().filter(|l| l.starts_with("| 第") || l.starts_with("| 超拉期")).count();
        assert_eq!(seg_rows, 4, "分段表应恰有 4 行");
        let _ = fs::remove_dir_all(&out_dir);
        Ok(())
    }
}
