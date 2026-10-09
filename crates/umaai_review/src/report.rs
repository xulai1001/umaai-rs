//! report.html 渲染（文档 §2 架构、§8 报告结构、§11 步骤 7）
//!
//! - **四图**：自绘 SVG（复用 `umaai::plot::svg::Svg` 构建器，零 JS、零第三方
//!   图表库），作为模板变量经 `|safe` 注入
//!   - 图1 五维属性（终局真实值横条 + 上限竖线，技能 PT 加粗列出；**置于头部
//!     「终局估分」卡内**——用户 2026-10-09 拍板）
//!   - 图2 训练分布（年份纵轴 3 横条分段堆叠，比赛与其他合并）
//!   - 图3 运气走势（累计折线左轴 + 回合合计Δ柱右轴 + 关注点编号与图下注记，
//!     年度底色区，伪波动柱置灰）
//!   - 图4 心情走势（干劲 1-5 回合末阶梯折线 + 掉落 / 回升圆点标记，图下 `.note`
//!     给掉干劲区间的文字解释）
//! - **单一两栏网格**（叙述卡化 + 双卡合一 + 图表合并卡；奇数末格跨栏对齐；
//!   口径速览已删——判据全文在 digest.context.criteria）：minijinja 外置模板
//!   （`templates/report.html.j2`，运行时从文件加载，**改样式不重编译**）；
//!   附录数据（赛程 / 覆盖）不在报告落表——交给 SKILL 层处理成描述性文字后
//!   再考虑显示形式（用户拍板）
//! - 模板查找顺序：exe 同级 `templates/` → exe 上级（skill 布局 `bin/` +
//!   `templates/`）→ cwd 及其祖先的 `templates/` 与
//!   `crates/umaai_review/templates/`（开发期）

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf}
};

use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, context};
use serde::Serialize;
use umaai::plot::svg::Svg;

use crate::{
    decisions::{DecRow, LuckBlock},
    digest::Digest,
    execution::ExecRow,
    timeline::TimelineRow
};

/// 五维名与折线配色（图1 / 图4 训练位一致）
const ATTRS: [(&str, &str); 5] = [
    ("速", "#1f77b4"),
    ("耐", "#ff7f0e"),
    ("力", "#2ca02c"),
    ("根", "#d62728"),
    ("智", "#9467bd")
];

/// 四图 SVG 产物（图4 另带图下注记文本）
#[derive(Debug, Default)]
pub struct Charts {
    pub status: String,
    pub luck: String,
    pub actions: String,
    /// 图4 心情走势 SVG（`|safe` 注入）
    pub motivation: String,
    /// 图4 图下注记：掉干劲区间的文字解释（纯文本，模板不走 `|safe`）
    pub motivation_note: String
}

/// skill 写的 4 段叙述（`--narrative <md>` 注入；缺省则模板保留 NARRATIVE 占位）
///
/// 文件格式：`<!-- overview -->` 等标记分隔的四段 HTML 片段，顺序不限、可只给部分。
/// **Rust 不解析叙述内容**（原样注入 `|safe`），只负责分段与拼接——
/// 这样 skill 无需回写 39KB 的 report.html，只写 2KB 叙述即可。
#[derive(Debug, Default, Clone, Serialize)]
pub struct Narrative {
    pub overview: String,
    pub luck_trend: String,
    pub findings: String,
    pub summary: String,
}

impl Narrative {
    /// 四段是否全空（全空 → 模板保留占位）
    pub fn is_empty(&self) -> bool {
        self.overview.is_empty()
            && self.luck_trend.is_empty()
            && self.findings.is_empty()
            && self.summary.is_empty()
    }
}

/// 叙述段标记（与 `report.html.j2` 的占位同名）
const NARRATIVE_KEYS: [&str; 4] = ["overview", "luck_trend", "findings", "summary"];

/// 解析 narrative.md → [`Narrative`]（按 `<!-- key -->` 分段；缺失的段留空）
pub fn parse_narrative(text: &str) -> Narrative {
    let mut out = Narrative::default();
    for key in NARRATIVE_KEYS.iter() {
        let marker = format!("<!-- {key} -->");
        let Some(start) = text.find(&marker) else { continue };
        let body_start = start + marker.len();
        // 段末 = 其后最近的**其它**标记起点（不依赖文件内块的书写顺序），否则文件末尾
        let end = NARRATIVE_KEYS
            .iter()
            .filter(|k| **k != *key)
            .filter_map(|k| text.find(&format!("<!-- {k} -->")))
            .filter(|p| *p > body_start)
            .min()
            .unwrap_or(text.len());
        let body = text[body_start..end].trim().to_string();
        match *key {
            "overview" => out.overview = body,
            "luck_trend" => out.luck_trend = body,
            "findings" => out.findings = body,
            "summary" => out.summary = body,
            _ => {}
        }
    }
    out
}

/// 生成四图（纯函数，无 IO）
pub fn build_charts(digest: &Digest) -> Charts {
    let mood = motivation_states(&digest.timeline);
    Charts {
        status: chart_status(&digest.timeline),
        luck: chart_luck(&digest.luck, &digest.decisions),
        actions: chart_actions(&digest.execution, &digest.decisions),
        motivation: chart_motivation(&mood),
        motivation_note: motivation_note(&mood)
    }
}

/// 渲染 report.html（四图 + 三表 + 概览卡 → `out_dir/report.html`）
///
/// 模板按默认查找顺序定位（见模块头）；找不到报错并列出查找位置。
/// `narrative` 非空时注入 4 段叙述，否则模板保留 `NARRATIVE:` 占位（供 skill 二次回填）。
pub fn render(digest: &Digest, out_dir: &Path, narrative: &Narrative) -> Result<PathBuf> {
    let tpl = find_template("report.html.j2").ok_or_else(|| {
        anyhow!(
            "找不到 templates/report.html.j2（查找顺序：exe 同级/上级 templates、\
             cwd 及其祖先的 templates 与 crates/umaai_review/templates）"
        )
    })?;
    render_with_template(digest, out_dir, &tpl, narrative)
}

/// 用指定模板渲染（测试入口；正常路径走 [`render`]）
pub fn render_with_template(
    digest: &Digest,
    out_dir: &Path,
    tpl_path: &Path,
    narrative: &Narrative,
) -> Result<PathBuf> {
    let source = fs::read_to_string(tpl_path)
        .with_context(|| format!("读取模板失败: {}", tpl_path.display()))?;
    let charts = build_charts(digest);
    let match_rate = if digest.execution.is_empty() {
        0.0
    } else {
        let matched = digest.execution.iter().filter(|r| r.matches == Some(true)).count();
        let comparable = digest.execution.iter().filter(|r| r.matches.is_some()).count();
        if comparable == 0 { 0.0 } else { matched as f64 / comparable as f64 * 100.0 }
    };
    let overview = context! {
        match_rate => format!("{match_rate:.1}")
    };
    // 背景装饰开关：输出目录里有 yayoi.png 才加背景 CSS（skill 只需复制图片，不必改 HTML）
    let has_bg = out_dir.join("yayoi.png").is_file();

    // 模板名以 .html 结尾 → minijinja 默认开启 HTML autoescape；
    // SVG 通过 |safe 注入（模板内 {{ chart_xxx | safe }}）
    let mut env = Environment::new();
    env.add_template("report.html", &source)?;
    let html = env
        .get_template("report.html")?
        .render(context! {
            digest => digest,
            overview => overview,
            narrative => narrative,
            has_bg => has_bg,
            chart_status => charts.status,
            chart_luck => charts.luck,
            chart_actions => charts.actions,
            chart_motivation => charts.motivation,
            motivation_note => charts.motivation_note
        })?;

    fs::create_dir_all(out_dir)
        .with_context(|| format!("创建输出目录失败: {}", out_dir.display()))?;
    let out = out_dir.join("report.html");
    fs::write(&out, html).with_context(|| format!("写入 report.html 失败: {}", out.display()))?;
    Ok(out)
}

/// 模板查找（无编译期嵌入路径，规避绝对路径泄漏）
///
/// `name` 如 `report.html.j2` / `brief.md.j2`；查找顺序：exe 同级 `templates/` →
/// exe 上级（skill 布局 `bin/` + `templates/`）→ cwd 及其祖先的 `templates/` 与
/// `crates/umaai_review/templates/`（开发期）。brief 复用本函数。
pub(crate) fn find_template(name: &str) -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            cands.push(dir.join("templates").join(name));
            cands.push(dir.join("..").join("templates").join(name));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for anc in cwd.ancestors() {
            cands.push(anc.join("templates").join(name));
            cands.push(anc.join("crates/umaai_review/templates").join(name));
        }
    }
    cands.into_iter().find(|p| p.is_file())
}

// ======================= 图1：属性-上限（终局） =======================

/// 图1：五维属性与上限的关系（横条与刻度 = 终局真实值，轨道末竖线 = 上限）+ 技能 PT 值
///
/// 用户 2026-10-08 拍板：由「属性成长曲线」改为属性-上限图——只列五维与对应上限
/// 的关系，技能 PT 只**列出**数值（独立行，不与五维共刻度）。数值文本**只列显示值**
/// （真实值 > 1200 部分减半，小黑板口径；`timeline.five_status_display`）；触顶
/// （真实值 ≥ 上限）在数值后标注；底部图例说明行已删。
fn chart_status(tl: &[TimelineRow]) -> String {
    let Some(last) = tl.last() else {
        return note_svg("图1 五维属性：无 timeline 数据");
    };
    let (w, h) = (470.0, 256.0);
    let (x0, x1) = (56.0, 368.0);
    let top = 40.0;
    let row_h = 37.0;
    // 刻度 = 五维值与上限的最大值（技能 PT 不参与）
    let mut amax = 1.0f64;
    for i in 0..5 {
        amax = amax.max(last.five_status[i] as f64).max(last.five_status_limit[i] as f64);
    }
    let ax = |v: f64| x0 + (v / amax).max(0.0) * (x1 - x0);
    let mut svg = Svg::new(w, h);
    svg.text(w / 2.0, 18.0, "五维属性（显示值）", 13.0, "middle");
    for (i, (name, clr)) in ATTRS.iter().enumerate() {
        let bar_y = top + i as f64 * row_h + 8.0;
        let bar_h = 14.0;
        let (v, lim) = (last.five_status[i] as f64, last.five_status_limit[i] as f64);
        // 底轨（浅灰，到上限为止）+ 当前值条（条与刻度 = 真实值，视觉量级不变）
        svg.rect(x0, bar_y, (ax(lim) - x0).max(0.5), bar_h, "#ececec", 1.0);
        svg.rect(x0, bar_y, (ax(v) - x0).max(0.5), bar_h, clr, 0.92);
        // 上限竖线
        svg.line(ax(lim), bar_y - 3.0, ax(lim), bar_y + bar_h + 3.0, "#333333", 1.3);
        svg.text(x0 - 8.0, bar_y + 12.0, name, 12.0, "end");
        // 数值只列显示值（真实值 > 1200 部分减半）；触顶按真实值判定
        let tag = if v >= lim { " 触顶" } else { "" };
        svg.text(x1 + 10.0, bar_y + 12.0, &format!("{}{}", last.five_status_display[i], tag), 11.0, "start");
    }
    // 分隔线 + 技能 PT 值（仅列出，不与五维共刻度；加粗——用户 2026-10-08 拍板）
    let sep_y = top + 5.0 * row_h;
    svg.line(x0 - 26.0, sep_y, x1 + 76.0, sep_y, "#e5e7eb", 1.0);
    svg.push(format!(
        r#"<text x="{:.1}" y="{:.1}" font-size="11" text-anchor="end" font-weight="bold">技能PT</text>"#,
        x0 - 8.0,
        sep_y + 24.0
    ));
    svg.push(format!(
        r#"<text x="{:.1}" y="{:.1}" font-size="12" text-anchor="start" font-weight="bold">{}</text>"#,
        x0 + 4.0,
        sep_y + 24.0,
        last.skill_pt
    ));
    // 底部图例说明行已删（用户 2026-10-08 拍板）；h 收至 PT 行下方
    svg.render()
}

// ======================= 图3：运气分双轴（含关注点） =======================

/// 圆圈数字（关注点编号 ①-⑥；MS YaHei / Noto 均覆盖）
const CIRCLED: [&str; 6] = ["①", "②", "③", "④", "⑤", "⑥"];

/// 图3：累计运气分折线（左轴）+ 回合合计Δ柱（右轴）+ **关注点编号** + 正 / 负底色带
///
/// 用户 2026-10-08 拍板：图表改两栏宽度（470）；图上标「关注点」编号（top_gain /
/// top_loss 各取前 3、按回合去重，≤6 个）——编号的解释（回合 + Δ + 性质）由
/// 运气走势文字承担，图内不再逐条注记；底色带按累计运气符号分 run（正浅绿 /
/// 负浅红，零点插值切开、同色合并）。伪波动柱置灰（§8、§6.2）。
fn chart_luck(luck: &LuckBlock, dec: &[DecRow]) -> String {
    if luck.series.is_empty() {
        return note_svg("图3 运气走势：无决策数据");
    }
    let mut delta_by_turn: BTreeMap<u32, f64> = BTreeMap::new();
    for r in dec {
        if let Some(d) = r.turn_delta {
            *delta_by_turn.entry(r.turn).or_default() += d; // 回合合计口径（§6.2）
        }
    }
    let flagged: HashSet<u32> = luck.flagged_turns.iter().map(|f| f.turn).collect();
    // 关注点：正 / 负极值各取前 3，按回合去重后按回合排序（≤6 个）
    let mut points: Vec<(u32, f64)> = Vec::new();
    for list in [&luck.top_gain, &luck.top_loss] {
        for t in list.iter().take(3) {
            if !points.iter().any(|(turn, _)| *turn == t.turn) {
                points.push((t.turn, t.delta));
            }
        }
    }
    points.sort_by_key(|(t, _)| *t);
    points.truncate(CIRCLED.len());

    let t1 = (luck.series.last().map(|p| p.turn).unwrap_or(77) as f64 + 1.0).max(24.0);
    // 两栏宽度（图下仅 1 行图例；关注点编号的解释由走势文字承担——用户口径）
    let (w, h) = (470.0, 308.0);
    let (x0, x1, top, ph) = (44.0, 434.0, 32.0, 226.0);
    let x = |t: f64| x0 + t / t1 * (x1 - x0);
    // 左轴：累计运气分
    let (mut llo, mut lhi) = (0.0f64, 0.0f64);
    for p in &luck.series {
        llo = llo.min(p.total_luck);
        lhi = lhi.max(p.total_luck);
    }
    let lpad = ((lhi - llo) * 0.08).max(10.0);
    (llo, lhi) = (llo - lpad, lhi + lpad);
    let yl = y_map_fn(top, ph, llo, lhi);
    // 右轴：回合 Δ（不画右刻度——关注点注记已带数值，避免半宽拥挤）
    let (mut dlo, mut dhi) = (0.0f64, 0.0f64);
    for &d in delta_by_turn.values() {
        dlo = dlo.min(d);
        dhi = dhi.max(d);
    }
    let dpad = ((dhi - dlo) * 0.12).max(60.0);
    (dlo, dhi) = (dlo - dpad, dhi + dpad);
    let yr = y_map_fn(top, ph, dlo, dhi);

    let mut svg = Svg::new(w, h);
    svg.text(w / 2.0, 16.0, "运气走势（折线 = 累计运气分·左轴；柱 = 回合合计Δ；灰 = 伪波动）", 11.5, "middle");
    // 正 / 负运气底色带：按累计运气分符号分 run（零点处线性插值切开、同色连续段
    // 合并；正段浅绿、负段浅红——与走势文字的「阴跌 / 强势段」描述对应，用户口径）
    {
        let pos = "#ecfdf5";
        let neg = "#fef2f2";
        let mut segs: Vec<(f64, f64, &str)> = Vec::new();
        for w2 in luck.series.windows(2) {
            let (p0, p1) = (&w2[0], &w2[1]);
            let (v0, v1) = (p0.total_luck, p1.total_luck);
            let (ta, tb) = (p0.turn as f64, p1.turn as f64);
            if (v0 >= 0.0) == (v1 >= 0.0) {
                segs.push((ta, tb, if v1 >= 0.0 { pos } else { neg }));
            } else {
                // 零点跨界：按插值位置切成两段
                let tc = if (v1 - v0).abs() < 1e-9 {
                    (ta + tb) / 2.0
                } else {
                    ta + (0.0 - v0) / (v1 - v0) * (tb - ta)
                };
                segs.push((ta, tc, if v0 >= 0.0 { pos } else { neg }));
                segs.push((tc, tb, if v1 >= 0.0 { pos } else { neg }));
            }
        }
        // 同色相邻段合并（series 逐点相连，段首尾相接）
        let mut i = 0;
        while i + 1 < segs.len() {
            if segs[i].2 == segs[i + 1].2 && (segs[i].1 - segs[i + 1].0).abs() < 1e-9 {
                segs[i].1 = segs[i + 1].1;
                segs.remove(i + 1);
            } else {
                i += 1;
            }
        }
        for (ta, tb, clr) in segs {
            svg.rect(x(ta), top, (x(tb) - x(ta)).max(0.5), ph, clr, 1.0);
        }
    }
    // 柱（先画，折线覆盖其上）
    let bw = (x1 - x0) / (t1 + 1.0) * 0.62;
    for (&t, &d) in &delta_by_turn {
        let color = if flagged.contains(&t) {
            "#9e9e9e"
        } else if d >= 0.0 {
            "#2ca02c"
        } else {
            "#d62728"
        };
        let (yt, hh) = if d >= 0.0 {
            (yr(d), yr(0.0) - yr(d))
        } else {
            (yr(0.0), yr(d) - yr(0.0))
        };
        svg.rect(x(t as f64) - bw / 2.0, yt, bw, hh.max(0.5), color, 0.7);
    }
    // 零线 + 折线
    svg.line(x0, yr(0.0), x1, yr(0.0), "#999999", 0.8);
    let pts: Vec<(f64, f64)> = luck
        .series
        .iter()
        .map(|p| (x(p.turn as f64), yl(p.total_luck)))
        .collect();
    svg.polyline(&pts, "#c44e52", 1.6, false);
    // 左轴刻度（紧凑）
    for &v in &nice_ticks(llo, lhi) {
        svg.text(x0 - 5.0, yl(v) + 3.5, &fmt_tick(v), 9.0, "end");
    }
    // 横轴刻度（每 12 回合）
    let mut t = 0.0;
    while t <= t1 {
        svg.line(x(t), top + ph, x(t), top + ph + 4.0, "#555", 0.8);
        svg.text(x(t), top + ph + 15.0, &format!("{}", t as i32), 9.0, "middle");
        t += 12.0;
    }
    // 关注点编号（柱端白底圆 + 数字；正跳在柱顶上方、负跳在柱底下方）
    for (i, &(turn, d)) in points.iter().enumerate() {
        let mx = x(turn as f64);
        let my = if d >= 0.0 {
            (yr(d) - 10.0).max(top + 8.0)
        } else {
            (yr(d) + 11.0).min(top + ph - 8.0)
        };
        svg.push(format!(
            r##"<circle cx="{mx:.1}" cy="{my:.1}" r="7" fill="#ffffff" stroke="#374151" stroke-width="1.2"/>"##
        ));
        svg.text(mx, my + 3.5, CIRCLED[i], 9.5, "middle");
    }
    svg.frame(x0, top, x1 - x0, ph, "#333333", 1.0);
    // 图下一行图例（正 / 负底色与关注点编号不进图例——解释由走势文字承担，用户口径）
    let ny = top + ph + 32.0;
    svg.line(x0, ny - 3.0, x0 + 18.0, ny - 3.0, "#c44e52", 1.6);
    svg.text(x0 + 22.0, ny, "累计运气分（左轴）", 9.5, "start");
    svg.rect(x0 + 120.0, ny - 8.0, 9.0, 9.0, "#2ca02c", 0.85);
    svg.text(x0 + 133.0, ny, "Δ 正", 9.5, "start");
    svg.rect(x0 + 170.0, ny - 8.0, 9.0, 9.0, "#d62728", 0.85);
    svg.text(x0 + 183.0, ny, "Δ 负", 9.5, "start");
    svg.rect(x0 + 220.0, ny - 8.0, 9.0, 9.0, "#9e9e9e", 0.85);
    svg.text(x0 + 233.0, ny, "伪波动", 9.5, "start");
    svg.text(x0 + 285.0, ny, "回合", 9.5, "start");
    svg.render()
}

// ======================= 图2：行动分年累积 =======================

/// 行动分段配色（训练按维展开；「X训练·继承混合」并入对应训练；友人出行并入
/// 出行；**比赛与治病 / 剧本 / 未知合并**——用户 2026-10-08 拍板）
const SEGMENTS: [(&str, &str); 8] = [
    ("速训练", "#1f77b4"),
    ("耐训练", "#ff7f0e"),
    ("力训练", "#2ca02c"),
    ("根训练", "#d62728"),
    ("智训练", "#9467bd"),
    ("休息", "#8c8c8c"),
    ("出行", "#f59e0b"),
    // 青色：与智训练的紫色拉开距离（用户 2026-10-08 拍板「颜色过于接近」）
    ("比赛/其他", "#06b6d4")
];

/// 实际动作 → 分段下标（`starts_with` 覆盖「X训练·继承混合」后缀）
fn seg_idx(actual: &str) -> usize {
    if actual == "友人出行" {
        return 6; // 并入「出行」
    }
    for (i, (name, _)) in SEGMENTS.iter().enumerate() {
        if *name == "比赛/其他" {
            continue; // 非前缀名，走兜底
        }
        if actual.starts_with(name) {
            return i;
        }
    }
    7 // 比赛 / 治病 / 剧本 / 未知 → 比赛与其他
}

/// 年段下标（0/1/2；**第三年含超拉期与后续回合**——用户口径）
fn year_idx(turn: u32) -> usize {
    if turn < 24 {
        0
    } else if turn < 48 {
        1
    } else {
        2
    }
}

const YEAR_LABELS: [&str; 3] = ["第1年", "第2年", "第3年(含超拉)"];

/// 图2：实际执行行动分年累积（年份纵轴 3 横条分段堆叠）+ 第 4 行「吃面后训练选择」
/// （全程含超拉期：当回合 ramen_select 决策选了「吃面」且实际训练了；只看选了什么
/// 训练、不看吃了什么面——用户 2026-10-08 拍板）
fn chart_actions(exec: &[ExecRow], dec: &[DecRow]) -> String {
    if exec.is_empty() {
        return note_svg("图2 训练分布：无执行推断数据");
    }
    let mut counts = [[0u32; 3]; SEGMENTS.len()];
    for r in exec {
        counts[seg_idx(&r.actual_action)][year_idx(r.turn)] += 1;
    }
    // 吃面回合 = calc 行中 kind=ramen_select 且选中描述以「吃面」开头（不吃面路径为「不吃面」）
    let eat_turns: HashSet<u32> = dec
        .iter()
        .filter(|r| r.decision_kind == "ramen_select" && r.chosen.desc.starts_with("吃面"))
        .map(|r| r.turn)
        .collect();
    // 吃面后训练选择（全程）：训练维计数（含「·继承混合」；非训练回合不计）
    let mut eat_counts = [0u32; 5];
    for r in exec {
        if eat_turns.contains(&r.turn) {
            for (i, (name, _)) in SEGMENTS.iter().enumerate().take(5) {
                if r.actual_action.starts_with(name) {
                    eat_counts[i] += 1;
                    break;
                }
            }
        }
    }
    let eat_total: u32 = eat_counts.iter().sum();
    let year_totals: [u32; 3] = std::array::from_fn(|yi| counts.iter().map(|c| c[yi]).sum());
    let max_total = year_totals.iter().copied().chain([eat_total]).max().unwrap_or(1).max(1) as f64;
    let (w, h) = (470.0, 262.0);
    let (x0, x1) = (74.0, 404.0);
    let top = 42.0;
    let row_h = 40.0;
    let bar_h = 15.0;
    let x = |v: f64| x0 + v / max_total * (x1 - x0);
    let mut svg = Svg::new(w, h);
    svg.text(w / 2.0, 16.0, "训练分布（分年累积）", 13.0, "middle");
    // 画单行累积条（分段 + 总数标签）
    let draw_row = |svg: &mut Svg, ry: f64, segs: &[u32]| {
        let mut acc = 0u32;
        for (si, (_, clr)) in SEGMENTS.iter().enumerate() {
            let n = segs.get(si).copied().unwrap_or(0);
            if n == 0 {
                continue;
            }
            let xa = x(acc as f64);
            let xb = x((acc + n) as f64);
            svg.rect(xa, ry, (xb - xa).max(1.2), bar_h, clr, 0.95);
            if xb - xa >= 19.0 {
                svg.text((xa + xb) / 2.0, ry + bar_h - 2.5, &n.to_string(), 9.0, "middle");
            }
            acc += n;
        }
        svg.text(x(acc as f64) + 6.0, ry + 12.0, &format!("{acc}"), 10.0, "start");
    };
    for (yi, lab) in YEAR_LABELS.iter().enumerate() {
        let ry = top + yi as f64 * row_h;
        svg.text(x0 - 8.0, ry + 12.0, lab, 11.0, "end");
        draw_row(&mut svg, ry, &counts.iter().map(|c| c[yi]).collect::<Vec<_>>());
    }
    // 第 4 行：吃面后训练选择（全程含超拉；分隔线区分年段行）
    let ry = top + 3.0 * row_h;
    svg.line(x0 - 40.0, ry - 6.0, x1 + 26.0, ry - 6.0, "#e5e7eb", 1.0);
    svg.text(x0 - 8.0, ry + 12.0, "吃面后训练", 11.0, "end");
    draw_row(&mut svg, ry, &eat_counts.to_vec());
    // 图例（8 项分两行）
    for (si, (name, clr)) in SEGMENTS.iter().enumerate() {
        let col = si % 4;
        let line = si / 4;
        let lx = x0 + col as f64 * 88.0;
        let ly = top + 4.0 * row_h + 10.0 + line as f64 * 17.0;
        svg.rect(lx, ly, 10.0, 8.0, clr, 0.95);
        svg.text(lx + 14.0, ly + 8.0, name, 9.5, "start");
    }
    // 底部「execution 推断口径」说明行已删（用户 2026-10-08 拍板）
    svg.render()
}

// ======================= 图4：心情走势（干劲） =======================

/// 掉干劲区间（回合末口径）：一次下降开始，到回到下降前水平的前一回合为止
///
/// 区间内继续下降只加深 `floor`、不另起区间；干劲回到 ≥ 起始水平即闭合
/// （`recover` = 恢复回合）。`end` = 区间内最后一个仍低于起始水平的回合。
#[derive(Debug)]
struct MoodEpisode {
    /// 掉落回合（区间起点）
    start: u32,
    /// 区间末（含；仍低于起始水平的最后一个回合）
    end: u32,
    /// 下降前水平
    from: i32,
    /// 区间内最低值
    floor: i32,
    /// 恢复回合（回到 ≥ `from`）；None = 至终局未恢复
    recover: Option<u32>
}

/// 回合末干劲序列（timeline 升序 → 后写覆盖，与 brief::turn_states 同口径）
fn motivation_states(tl: &[TimelineRow]) -> BTreeMap<u32, i32> {
    let mut m: BTreeMap<u32, i32> = BTreeMap::new();
    for r in tl {
        m.insert(r.turn, r.motivation);
    }
    m
}

/// 掉干劲区间切分（状态机：低于区间起始水平则延续，回到即闭合）
fn mood_episodes(states: &BTreeMap<u32, i32>) -> Vec<MoodEpisode> {
    let pts: Vec<(u32, i32)> = states.iter().map(|(&t, &m)| (t, m)).collect();
    let mut eps: Vec<MoodEpisode> = Vec::new();
    let mut cur: Option<MoodEpisode> = None;
    let mut prev: Option<i32> = None;
    for &(t, m) in &pts {
        if let Some(ep) = cur.as_mut() {
            if m >= ep.from {
                // 回到下降前水平 → 闭合（end 停在上一个仍偏低的回合）
                ep.recover = Some(t);
                eps.push(cur.take().expect("闭合时 cur 必在"));
            } else {
                ep.floor = ep.floor.min(m);
                ep.end = t;
            }
        } else if prev.is_some_and(|p| m < p) {
            // 新掉落：起始水平 = 上一数据点的干劲
            cur = Some(MoodEpisode {
                start: t,
                end: t,
                from: prev.expect("is_some_and 已保证"),
                floor: m,
                recover: None
            });
        }
        prev = Some(m);
    }
    if let Some(ep) = cur {
        eps.push(ep);
    }
    eps
}

/// 图4：干劲随回合变化（1-5 固定档位、回合末阶梯折线）+ 掉落 / 回升圆点标记
///
/// 掉干劲本身归运气（数据外事件，输赛不掉干劲——SKILL 层口径），图只呈现事实；
/// 掉干劲区间的文字解释由 [`motivation_note`] 生成、经模板 `.note` 注入。
fn chart_motivation(states: &BTreeMap<u32, i32>) -> String {
    if states.is_empty() {
        return note_svg("图4 心情走势：无 timeline 数据");
    }
    let pts: Vec<(u32, i32)> = states.iter().map(|(&t, &m)| (t, m)).collect();
    let t1 = (pts.last().map(|&(t, _)| t).unwrap_or(77) as f64 + 1.0).max(24.0);
    // 干劲 1-5 固定五档（timeline 字段注释口径），纵轴不按数据伸缩
    let (lo, hi) = (1.0, 5.0);
    let (w, h) = (470.0, 146.0);
    let (x0, x1, top, ph) = (34.0, 440.0, 26.0, 96.0);
    let x = |t: f64| x0 + t / t1 * (x1 - x0);
    let y = |v: f64| top + (1.0 - (v - lo) / (hi - lo)) * ph;
    let mut svg = Svg::new(w, h);
    svg.text(w / 2.0, 14.0, "心情走势（干劲 1-5，回合末）", 11.5, "middle");
    // 绝好调（5 档）浅绿底带
    svg.rect(x0, y(hi), x1 - x0, y(hi - 1.0) - y(hi), "#ecfdf5", 1.0);
    // 档位网格线 + 左侧档位数字
    for v in 1..=5 {
        svg.line(x0, y(v as f64), x1, y(v as f64), "#e5e7eb", 0.8);
        svg.text(x0 - 6.0, y(v as f64) + 3.5, &v.to_string(), 9.0, "end");
    }
    // 阶梯折线：水平延伸到变化回合、变化回合处垂直跳变
    let mut pl: Vec<(f64, f64)> = Vec::with_capacity(pts.len() * 2);
    let mut prev_m: Option<i32> = None;
    for &(t, m) in &pts {
        if let Some(pm) = prev_m {
            if m != pm {
                pl.push((x(t as f64), y(pm as f64)));
            }
        }
        pl.push((x(t as f64), y(m as f64)));
        prev_m = Some(m);
    }
    svg.polyline(&pl, "#7c3aed", 1.8, false);
    // 掉落（红）/ 回升（绿）标记打在变化回合的新值处
    for pair in pts.windows(2) {
        let (_, m_prev) = pair[0];
        let (t_cur, m_cur) = pair[1];
        if m_cur < m_prev {
            svg.circle(x(t_cur as f64), y(m_cur as f64), 3.5, "#d62728", 1.0);
        } else if m_cur > m_prev {
            svg.circle(x(t_cur as f64), y(m_cur as f64), 3.5, "#16a34a", 1.0);
        }
    }
    // 横轴刻度（每 12 回合，与图3 一致）
    let mut t = 0.0;
    while t <= t1 {
        svg.line(x(t), top + ph, x(t), top + ph + 4.0, "#555", 0.8);
        svg.text(x(t), top + ph + 15.0, &format!("{}", t as i32), 9.0, "middle");
        t += 12.0;
    }
    svg.frame(x0, top, x1 - x0, ph, "#333333", 1.0);
    svg.render()
}

/// 图4 图下注记：掉干劲区间的文字解释（「t12..t14（5→4，t15 恢复）」式一行）
///
/// 区间语义见 [`MoodEpisode`]；无掉落输出「全程干劲无下降」。
fn motivation_note(states: &BTreeMap<u32, i32>) -> String {
    if states.is_empty() {
        return "无 timeline 数据".to_string();
    }
    let eps = mood_episodes(states);
    if eps.is_empty() {
        return "全程干劲无下降（回合末口径）".to_string();
    }
    let parts: Vec<String> = eps
        .iter()
        .map(|ep| {
            let range = if ep.start == ep.end {
                format!("t{}", ep.start)
            } else {
                format!("t{}..t{}", ep.start, ep.end)
            };
            let rec = match ep.recover {
                Some(r) => format!("t{r} 恢复"),
                None => "终局未恢复".to_string()
            };
            format!("{range}（{}→{}，{rec}）", ep.from, ep.floor)
        })
        .collect();
    format!("掉干劲区间：{}", parts.join("；"))
}

// ======================= 小工具 =======================

/// 数据值 → 像素 y（区间 [lo, hi] 映射到 [top, top+ph]，上大下小）
fn y_map_fn(top: f64, ph: f64, lo: f64, hi: f64) -> impl Fn(f64) -> f64 {
    let span = (hi - lo).max(1e-9);
    move |v| top + (1.0 - (v - lo) / span) * ph
}

/// 「好看」的纵轴刻度（1/2/5×10^n 步长；与 luck_trend 同算法的紧凑版）
fn nice_ticks(lo: f64, hi: f64) -> Vec<f64> {
    let span = (hi - lo).max(1e-9);
    let raw = span / 5.0;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = mag
        * if norm <= 1.0 {
            1.0
        } else if norm <= 2.0 {
            2.0
        } else if norm <= 5.0 {
            5.0
        } else {
            10.0
        };
    let mut out = Vec::new();
    let mut v = (lo / step).ceil() * step;
    while v <= hi + step * 1e-9 {
        if v >= lo {
            out.push(v);
        }
        v += step;
    }
    out
}

/// 刻度数字格式（大数取整、小数留 1-2 位）
fn fmt_tick(v: f64) -> String {
    let a = v.abs();
    if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.1}")
    } else if a > 0.0 {
        format!("{v:.2}")
    } else {
        "0".to_string()
    }
}

/// 无数据占位图
fn note_svg(msg: &str) -> String {
    let mut svg = Svg::new(900.0, 60.0);
    svg.text(20.0, 34.0, msg, 13.0, "start");
    svg.render()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decisions::{Cand, Chosen, LuckPoint},
        digest::{DeckCard, Digest, DigestContext, Meta},
        execution::Finding,
        schedule::Schedule,
        timeline::TimelineRow
    };

    /// 构造最小可用 Digest（图表与模板端到端用）
    fn test_digest() -> Digest {
        // 干劲：t0 = 5，t30 掉到 4，t77 恢复 5（图4 掉落 / 回升标记与区间注记的用例）
        let tl = vec![
            tl_row(0, [200, 100, 100, 100, 100], [3000; 5], 5),
            tl_row(30, [800, 600, 400, 300, 500], [3000; 5], 4),
            tl_row(77, [3200, 1900, 1700, 1200, 2100], [3272, 2402, 2240, 2200, 2452], 5),
        ];
        let dec = vec![DecRow {
            file: "f0.json".to_string(),
            turn: 0,
            seq: 0,
            stage: "Train".to_string(),
            decision_kind: "train".to_string(),
            candidates: vec![
                Cand { rank: 1, desc: "速训练".into(), score: Some(63000.0), n: Some(8192), gap_to_best: Some(0.0) },
                Cand { rank: 2, desc: "智训练".into(), score: Some(62900.0), n: Some(7168), gap_to_best: Some(100.0) },
            ],
            chosen: Chosen { idx: Some(0), desc: "速训练".into(), action_luck: Some(73.0) },
            t_n_raw: Some(63000.0),
            t_n_display: Some(63050.0),
            total_luck: Some(0.0),
            turn_delta: None,
            chain_len: 1,
            outcome: "calc".into(),
            reason: String::new(),
            step: 0,
        }];
        let exec = vec![ExecRow {
            turn: 0,
            stage: "Train".into(),
            ai_choice: "速训练".into(),
            actual_action: "速训练".into(),
            matches: Some(true),
            evidence: Default::default(),
            alt_candidate: None,
        }];
        Digest {
            meta: Meta {
                game: 6234,
                uma_id: 112402,
                uma_name: "吹波糖".into(),
                deck: vec![DeckCard { card_id: 303051, name: "[友]骏川手纲".into(), card_type: 5, limit_break: 1 }],
                start_turn: 0,
                mid_entry: false,
                end_reason: "game_end".into(),
                snapshots: 161,
                decision_rows: 155,
                total_luck_end: Some(-3186.05),
                final_score: Some(62500),
                rank: Some("UA9".into()),
                final_source: "last_snapshot".into(),
                is_qiezhe: false,
                is_xiao_qie: false,
            },
            timeline: tl,
            decisions: dec,
            execution: exec,
            luck: crate::decisions::LuckBlock {
                series: vec![
                    LuckPoint { turn: 0, seq: 0, total_luck: 0.0 },
                    LuckPoint { turn: 30, seq: 1, total_luck: -500.0 },
                    LuckPoint { turn: 77, seq: 2, total_luck: -3186.05 },
                ],
                top_gain: vec![],
                top_loss: vec![crate::decisions::TurnDelta {
                    turn: 30,
                    delta: -500.0,
                    segments: vec![-500.0]
                }],
                raw_delta_stats: Default::default(),
                flagged_turns: vec![crate::decisions::FlaggedTurn { turn: 30, reason: "inherit".into() }],
            },
            schedule: Schedule::default(),
            inherit: None,
            clones: None,
            training: crate::profile::TrainingProfile {
                years: vec![
                    crate::profile::YearProfile {
                        label: "第1年".into(),
                        train_counts: [3, 1, 0, 0, 2],
                        gains: [500, 400, 300, 300, 600]
                    },
                    crate::profile::YearProfile {
                        label: "第2年".into(),
                        train_counts: [5, 6, 1, 0, 3],
                        gains: [690, 800, 441, 372, 580]
                    },
                    crate::profile::YearProfile {
                        label: "第3年".into(),
                        train_counts: [9, 4, 2, 0, 4],
                        gains: [901, 420, 589, 391, 407]
                    },
                    crate::profile::YearProfile {
                        label: "超拉期".into(),
                        train_counts: [4, 1, 1, 0, 0],
                        gains: [127, 74, 189, 60, 30]
                    }
                ],
                luck_by_attr: [1500.0, 400.0, 300.0, 0.0, 200.0],
                luck_other: 344.0,
                luck_program: 0.0
            },
            coverage: Default::default(),
            findings: vec![Finding {
                kind: "mandatory_race_not_won".into(),
                turn: 45,
                file: None,
                evidence: "必赛回合未跑赢".into(),
                severity: "warn".into(),
            }],
            context: DigestContext {
                region_names: Default::default(),
                luck_formula: "…".into(),
                criteria: vec!["测试口径".into()],
            },
        }
    }

    /// timeline 测试行（只填图表所需字段；motivation 单独传）
    fn tl_row(turn: u32, five: [i32; 5], limit: [i32; 5], motivation: i32) -> TimelineRow {
        TimelineRow {
            turn,
            seq: 0,
            stage: "Train".into(),
            reason: None,
            source: "command".into(),
            playing_state: 1,
            vital: 100,
            max_vital: 108,
            motivation,
            five_status: five,
            five_status_display: crate::score::display_status_array(five),
            five_status_limit: limit,
            skill_pt: 100,
            train_level_count: [1; 5],
            friend_outgoing_used: 5,
            selected_regions: vec![10, 11, 14],
            scenario_pt: 100,
            feeling_stock: vec![],
            super_ramen: 1,
            is_ill: false,
            is_qiezhe: false,
            is_xiao_qie: false,
            race_count: 10,
            absent_persons: vec![],
        }
    }

    /// 四图冒烟：SVG 结构与关键标记
    #[test]
    fn test_charts_markers() {
        let d = test_digest();
        let c = build_charts(&d);
        println!("图1 {} 字节 / 图2 {} / 图3 {} / 图4 {}",
            c.status.len(), c.actions.len(), c.luck.len(), c.motivation.len());
        // 图1 属性-上限：五维条 + 上限竖线 + 数值 + 技能PT 行
        assert!(c.status.contains("<svg") && c.status.contains("五维属性"));
        assert!(c.status.contains("2200") && !c.status.contains("3200/"), "数值只列显示值");
        assert!(c.status.contains("技能PT"));
        // 图2 行动分年累积：年段标签 + 分段图例 + 第 4 行「吃面后训练」
        assert!(c.actions.contains("训练分布"));
        assert!(c.actions.contains("第1年") && c.actions.contains("第3年(含超拉)"));
        assert!(c.actions.contains("速训练") && c.actions.contains("比赛/其他"));
        assert!(c.actions.contains("吃面后训练"), "第 4 行：吃面后训练选择");
        // 图3 运气分双轴：关注点编号圆圈 + 正 / 负底色带；图内不注记（解释归走势文字）
        assert!(c.luck.contains("运气走势") && c.luck.contains("<polyline"));
        assert!(c.luck.contains("①"), "关注点编号标记");
        assert!(!c.luck.contains("t30 -500"), "图内不再逐条注记");
        assert!(c.luck.contains("#ecfdf5") && c.luck.contains("#fef2f2"), "正 / 负底色带");
        // 图4 心情走势：阶梯折线 + 掉落（红）/ 回升（绿）标记 + 绝好调底带
        assert!(c.motivation.contains("心情走势") && c.motivation.contains("<polyline"));
        assert!(c.motivation.contains("#d62728"), "掉干劲标记");
        assert!(c.motivation.contains("#16a34a"), "回升标记");
        assert!(c.motivation.contains("#ecfdf5"), "绝好调底带");
        assert_eq!(c.motivation_note, "掉干劲区间：t30（5→4，t77 恢复）", "掉干劲区间注记");
    }

    /// 掉干劲区间切分：单区间 / 连降合并 / 未恢复 / 无掉落
    #[test]
    fn test_mood_episodes() {
        let states: BTreeMap<u32, i32> =
            [(0, 3), (1, 4), (5, 5), (12, 4), (15, 5), (77, 5)].into_iter().collect();
        let eps = mood_episodes(&states);
        println!("{eps:?}");
        assert_eq!(eps.len(), 1, "开局 3→4→5 爬坡不算掉落，只有 t12 一处");
        assert_eq!((eps[0].start, eps[0].end, eps[0].from, eps[0].floor), (12, 12, 5, 4));
        assert_eq!(eps[0].recover, Some(15));
        assert_eq!(motivation_note(&states), "掉干劲区间：t12（5→4，t15 恢复）");
        // 连降合并为一个区间（回到起始水平才闭合）+ 尾部未恢复
        let s2: BTreeMap<u32, i32> =
            [(0, 5), (10, 4), (20, 3), (30, 4), (40, 5), (50, 4)].into_iter().collect();
        let eps2 = mood_episodes(&s2);
        println!("{eps2:?}");
        assert_eq!(eps2.len(), 2, "t10..t30 连降合并 + t50 尾部掉落");
        assert_eq!((eps2[0].start, eps2[0].end, eps2[0].from, eps2[0].floor), (10, 30, 5, 3));
        assert_eq!(eps2[0].recover, Some(40));
        assert_eq!((eps2[1].start, eps2[1].end, eps2[1].from), (50, 50, 5));
        assert_eq!(eps2[1].recover, None, "尾部掉落至终局未恢复");
        assert_eq!(
            motivation_note(&s2),
            "掉干劲区间：t10..t30（5→3，t40 恢复）；t50（5→4，终局未恢复）"
        );
        // 无掉落（起始即低也不算——只看下降事件）
        let s3: BTreeMap<u32, i32> = [(0, 4), (5, 4), (77, 4)].into_iter().collect();
        assert!(mood_episodes(&s3).is_empty());
        assert_eq!(motivation_note(&s3), "全程干劲无下降（回合末口径）");
        // 区间连续有数据时 end 覆盖整个低位段（game6263 实测形态）
        let s4: BTreeMap<u32, i32> =
            [(0, 3), (1, 4), (5, 5), (12, 4), (13, 4), (14, 4), (15, 5), (16, 5)]
                .into_iter()
                .collect();
        let eps4 = mood_episodes(&s4);
        assert_eq!((eps4[0].start, eps4[0].end), (12, 14), "区间末 = 回升前最后回合");
        assert_eq!(motivation_note(&s4), "掉干劲区间：t12..t14（5→4，t15 恢复）");
    }

    /// 模板端到端：渲染 → 校验关键内容（模板经 find_template 定位，测试 cwd
    /// 应为 workspace 根）
    #[test]
    fn test_render_report() -> Result<()> {
        let d = test_digest();
        let tpl = find_template("report.html.j2")
            .ok_or_else(|| anyhow!("测试环境找不到模板（cwd 应为 workspace 根）"))?;
        println!("模板: {}", tpl.display());
        let out_dir = std::env::temp_dir().join(format!("report_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&out_dir);
        let out = render_with_template(&d, &out_dir, &tpl, &Narrative::default())?;
        let html = fs::read_to_string(&out)?;
        println!("report.html {} 字节", html.len());
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("单局复盘 · game6234"));
        assert!(html.contains("吹波糖"));
        assert!(html.contains("UA9"));
        assert!(html.contains("<svg"), "四图 SVG 应经 |safe 注入");
        assert!(html.contains("图1") && html.contains("图2") && html.contains("图3") && html.contains("图4"));
        // 图1 已移入头部「终局估分」卡（不再有独立 section；用户 2026-10-09 拍板）
        assert!(!html.contains("<section><h2>图1"), "图1 不再是独立 section");
        assert!(html.contains("class=\"k k2\">图1 · 五维属性"), "图1 标签在估分卡内");
        // 图4 心情走势 + 掉干劲区间注记
        assert!(html.contains("图4 · 心情走势"));
        assert!(html.contains("掉干劲区间：t30（5→4，t77 恢复）"), "心情降低区间文字解释");
        // 单一两栏网格 + 叙述卡化 + 终局估分措辞
        assert!(html.contains("cols"), "两栏布局 class");
        assert!(html.contains("终局估分 62500（UA9）"), "头部措辞＝终局估分");
        assert!(html.contains("总体") && html.contains("运气走势") && html.contains("检查项"));
        assert!(html.contains("五维属性") && html.contains("训练分布") && html.contains("运气走势"));
        assert!(!html.contains("略高于小黑板"), "评分对比措辞已删");
        // 口径速览卡片已删（用户 2026-10-08 拍板）；判据全文在 digest.context.criteria
        assert!(!html.contains("口径速览"), "口径速览卡片已删除");
        // 4 个叙述占位标记（skill 层回填）；检查项表与决策明细表已移除
        for marker in ["NARRATIVE:overview", "NARRATIVE:luck_trend", "NARRATIVE:findings", "NARRATIVE:summary"] {
            assert!(html.contains(marker), "占位标记 {marker} 应存在");
        }
        assert!(!html.contains("mandatory_race_not_won"), "检查项表已换叙述占位");
        assert!(!html.contains("决策明细"), "决策明细表已移除（明细见 decisions.csv）");
        assert!(!html.contains("测试口径"), "context.criteria 全文不再渲染");
        assert!(html.contains("100.0%") || html.contains("100%"), "执行一致率");
        let _ = fs::remove_dir_all(&out_dir);
        Ok(())
    }

    /// narrative 解析：按标记分段、**不依赖块顺序**、缺段留空、末段到文件尾
    #[test]
    fn test_parse_narrative() {
        // 故意乱序 + 缺 findings
        let text = concat!(
            "<!-- summary -->\n<section>总结段</section>\n\n",
            "<!-- overview -->\n<section>总体段\n多行</section>\n\n",
            "<!-- luck_trend -->\n<section>走势段</section>\n",
        );
        let n = parse_narrative(text);
        println!("{n:#?}");
        assert_eq!(n.overview, "<section>总体段\n多行</section>", "多行内容原样保留");
        assert_eq!(n.luck_trend, "<section>走势段</section>");
        assert_eq!(n.summary, "<section>总结段</section>", "乱序也能正确切段");
        assert!(n.findings.is_empty(), "缺段留空");
        assert!(!n.is_empty());
        assert!(parse_narrative("无关内容").is_empty(), "无标记 → 全空");
    }

    /// narrative 注入端到端：给了叙述 → 占位消失、内容进文；不给 → 保留占位
    #[test]
    fn test_render_report_with_narrative() -> Result<()> {
        let d = test_digest();
        let tpl = find_template("report.html.j2")
            .ok_or_else(|| anyhow!("测试环境找不到模板（cwd 应为 workspace 根）"))?;
        let out_dir = std::env::temp_dir().join(format!("report_narr_{}", std::process::id()));
        let _ = fs::remove_dir_all(&out_dir);

        let narr = parse_narrative(concat!(
            "<!-- overview -->\n<section><summary>总体</summary>终局评分 64589</section>\n\n",
            "<!-- luck_trend -->\n<section>运气走势叙事</section>\n\n",
            "<!-- findings -->\n<section>检查项叙事</section>\n\n",
            "<!-- summary -->\n<section>总结叙事</section>\n",
        ));
        let out = render_with_template(&d, &out_dir, &tpl, &narr)?;
        let html = fs::read_to_string(&out)?;
        println!("注入后 {} 字节", html.len());
        assert!(html.contains("终局评分 64589"), "overview 段应注入");
        assert!(html.contains("运气走势叙事"), "luck_trend 段应注入");
        assert!(html.contains("检查项叙事"), "findings 段应注入");
        assert!(html.contains("总结叙事"), "summary 段应注入");
        assert!(
            !html.contains("NARRATIVE:"),
            "四段齐全时不应残留占位标记"
        );
        // 只给部分段 → 其余段保留占位
        let partial = parse_narrative("<!-- overview --><section>只有总体</section>");
        let out2 = render_with_template(&d, &out_dir, &tpl, &partial)?;
        let html2 = fs::read_to_string(&out2)?;
        assert!(html2.contains("只有总体"));
        assert!(
            html2.contains("<!-- NARRATIVE:summary -->"),
            "未提供的段应保留占位"
        );
        let _ = fs::remove_dir_all(&out_dir);
        Ok(())
    }
}
