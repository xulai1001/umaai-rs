//! report.html 渲染（文档 §2 架构、§8 报告结构、§11 步骤 7）
//!
//! - **三图**：自绘 SVG（复用 `umaai::plot::svg::Svg` 构建器，零 JS、零第三方
//!   图表库），作为模板变量经 `|safe` 注入
//!   - 图1 属性成长曲线（五维实线 + 上限虚线，标年界与继承回合）
//!   - 图2 运气分双轴（累计折线左轴 + 回合合计Δ柱右轴，伪波动柱置灰）
//!   - 图3 实际执行行动总计（环形图，execution 推断口径）
//! - **三表 + 概览卡 + 口径说明**：minijinja 外置模板（`templates/report.html.j2`，
//!   运行时从文件加载，**改样式不重编译**）；附录数据（赛程 / 覆盖）不在报告
//!   落表——交给 SKILL 层处理成描述性文字后再考虑显示形式（用户拍板）
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

/// 三图 SVG 产物
#[derive(Debug, Default)]
pub struct Charts {
    pub status: String,
    pub luck: String,
    pub actions: String
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

/// 生成三图（纯函数，无 IO）
pub fn build_charts(digest: &Digest) -> Charts {
    Charts {
        status: chart_status(&digest.timeline),
        luck: chart_luck(&digest.luck, &digest.decisions),
        actions: chart_actions(&digest.execution)
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
    let last = digest.timeline.last();
    let match_rate = if digest.execution.is_empty() {
        0.0
    } else {
        let matched = digest.execution.iter().filter(|r| r.matches == Some(true)).count();
        let comparable = digest.execution.iter().filter(|r| r.matches.is_some()).count();
        if comparable == 0 { 0.0 } else { matched as f64 / comparable as f64 * 100.0 }
    };
    let overview = context! {
        // 小黑板口径显示值（真实值 > 1200 部分减半）；成长曲线仍用真实值
        five_status => last.map(|r| r.five_status_display).unwrap_or([0; 5]),
        vital => last.map(|r| r.vital).unwrap_or(0),
        motivation => last.map(|r| r.motivation).unwrap_or(0),
        match_rate => format!("{match_rate:.1}")
    };
    // 地区分身逐次彩圈明细（构造在 clones.rs，brief.md 共用同一份，避免两处格式化分叉）
    let deck_names: Vec<String> = digest.meta.deck.iter().map(|c| c.name.clone()).collect();
    let region_rows = digest
        .clones
        .as_ref()
        .map(|c| c.region_detail_rows(&deck_names))
        .unwrap_or_default();
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
            region_rows => region_rows,
            narrative => narrative,
            has_bg => has_bg,
            chart_status => charts.status,
            chart_luck => charts.luck,
            chart_actions => charts.actions
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

// ======================= 图1：属性成长曲线 =======================

/// 图1：五维成长（实线）+ 上限（虚线）；年界竖线、继承点线标记（§8）
fn chart_status(tl: &[TimelineRow]) -> String {
    let mut by_turn: BTreeMap<u32, &TimelineRow> = BTreeMap::new();
    for r in tl {
        by_turn.insert(r.turn, r); // 升序 → 后写覆盖 = 回合末
    }
    if by_turn.is_empty() {
        return note_svg("图1 属性成长曲线：无 timeline 数据");
    }
    let t1 = (*by_turn.keys().max().unwrap_or(&77) as f64) + 1.0;
    let mut ymax: f64 = 100.0;
    for r in by_turn.values() {
        for i in 0..5 {
            ymax = ymax.max(r.five_status[i] as f64).max(r.five_status_limit[i] as f64);
        }
    }
    let (w, h) = (900.0, 380.0);
    let (x0, x1, top, ph) = (56.0, 706.0, 40.0, 280.0);
    let x = |t: f64| x0 + t / t1 * (x1 - x0);
    let y = y_map_fn(top, ph, 0.0, ymax);
    let mut svg = Svg::new(w, h);
    svg.text(w / 2.0, 20.0, "属性成长曲线（实线 = 当前值 / 虚线 = 上限）", 13.5, "middle");
    // 纵轴网格与刻度
    for &v in &nice_ticks(0.0, ymax) {
        svg.text(x0 - 6.0, y(v) + 4.0, &fmt_tick(v), 10.0, "end");
        svg.line(x0, y(v), x1, y(v), "#e8e8e8", 0.5);
    }
    // 横轴刻度（每 8 回合）
    let mut t = 0.0;
    while t < t1 {
        svg.line(x(t), top + ph, x(t), top + ph + 5.0, "#555", 0.8);
        svg.text(x(t), top + ph + 18.0, &format!("{}", t as i32), 10.0, "middle");
        t += 8.0;
    }
    svg.text((x0 + x1) / 2.0, top + ph + 36.0, "回合", 11.0, "middle");
    // 年界竖线（浅灰实线）
    for &b in &[24.0, 48.0, 72.0] {
        if b < t1 {
            svg.line(x(b), top, x(b), top + ph, "#b8b8b8", 1.0);
            svg.text(x(b) + 3.0, top + 11.0, &format!("年界{b:.0}"), 9.0, "start");
        }
    }
    // 继承回合（紫色点线）
    for &b in &[30.0, 54.0] {
        if b < t1 {
            svg.polyline(&[(x(b), top), (x(b), top + ph)], "#9333ea", 1.2, true);
            svg.text(x(b) + 3.0, top + 24.0, &format!("继承{b:.0}"), 9.0, "start");
        }
    }
    // 五维曲线 + 上限虚线
    for (i, (_, clr)) in ATTRS.iter().enumerate() {
        let cur: Vec<(f64, f64)> = by_turn
            .values()
            .map(|r| (x(r.turn as f64), y(r.five_status[i] as f64)))
            .collect();
        let lim: Vec<(f64, f64)> = by_turn
            .values()
            .map(|r| (x(r.turn as f64), y(r.five_status_limit[i] as f64)))
            .collect();
        svg.polyline(&cur, clr, 1.8, false);
        svg.polyline(&lim, clr, 1.0, true);
    }
    // 图例（右侧）
    let mut ly = top + 6.0;
    for (name, clr) in ATTRS {
        svg.line(x1 + 18.0, ly + 6.0, x1 + 40.0, ly + 6.0, clr, 1.8);
        svg.text(x1 + 46.0, ly + 10.0, name, 11.0, "start");
        ly += 20.0;
    }
    svg.polyline(&[(x1 + 18.0, ly + 6.0), (x1 + 40.0, ly + 6.0)], "#888888", 1.0, true);
    svg.text(x1 + 46.0, ly + 10.0, "上限", 11.0, "start");
    ly += 20.0;
    svg.line(x1 + 18.0, ly + 6.0, x1 + 40.0, ly + 6.0, "#b8b8b8", 1.0);
    svg.text(x1 + 46.0, ly + 10.0, "年界", 11.0, "start");
    ly += 20.0;
    svg.polyline(&[(x1 + 18.0, ly + 6.0), (x1 + 40.0, ly + 6.0)], "#9333ea", 1.2, true);
    svg.text(x1 + 46.0, ly + 10.0, "继承", 11.0, "start");
    svg.frame(x0, top, x1 - x0, ph, "#333333", 1.0);
    svg.render()
}

// ======================= 图2：运气分双轴 =======================

/// 图2：累计运气分折线（左轴）+ 回合合计Δ柱（右轴）；伪波动柱置灰（§8、§6.2）
fn chart_luck(luck: &LuckBlock, dec: &[DecRow]) -> String {
    if luck.series.is_empty() {
        return note_svg("图2 运气分：无决策数据");
    }
    let mut delta_by_turn: BTreeMap<u32, f64> = BTreeMap::new();
    for r in dec {
        if let Some(d) = r.turn_delta {
            *delta_by_turn.entry(r.turn).or_default() += d; // 回合合计口径（§6.2）
        }
    }
    let flagged: HashSet<u32> = luck.flagged_turns.iter().map(|f| f.turn).collect();
    let t1 = (luck.series.last().map(|p| p.turn).unwrap_or(77) as f64 + 1.0).max(24.0);
    let (w, h) = (900.0, 370.0);
    let (x0, x1, top, ph) = (64.0, 760.0, 40.0, 250.0);
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
    // 右轴：回合 Δ
    let (mut dlo, mut dhi) = (0.0f64, 0.0f64);
    for &d in delta_by_turn.values() {
        dlo = dlo.min(d);
        dhi = dhi.max(d);
    }
    let dpad = ((dhi - dlo) * 0.08).max(50.0);
    (dlo, dhi) = (dlo - dpad, dhi + dpad);
    let yr = y_map_fn(top, ph, dlo, dhi);

    let mut svg = Svg::new(w, h);
    svg.text(
        w / 2.0,
        20.0,
        "运气分双轴（折线 = 累计运气分·左轴；柱 = 回合合计Δ·右轴；灰 = 伪波动）",
        13.5,
        "middle"
    );
    // 柱（先画，折线覆盖其上）
    let bw = (x1 - x0) / (t1 + 1.0) * 0.7;
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
    svg.polyline(&pts, "#c44e52", 1.8, false);
    // 轴刻度（左红右黑）
    for &v in &nice_ticks(llo, lhi) {
        svg.text(x0 - 6.0, yl(v) + 4.0, &fmt_tick(v), 10.0, "end");
    }
    for &v in &nice_ticks(dlo, dhi) {
        svg.text(x1 + 6.0, yr(v) + 4.0, &fmt_tick(v), 10.0, "start");
    }
    // 横轴刻度
    let mut t = 0.0;
    while t <= t1 {
        svg.line(x(t), top + ph, x(t), top + ph + 5.0, "#555", 0.8);
        svg.text(x(t), top + ph + 18.0, &format!("{}", t as i32), 10.0, "middle");
        t += 8.0;
    }
    svg.text((x0 + x1) / 2.0, top + ph + 34.0, "回合", 11.0, "middle");
    // 图例（横排）
    let ly = top + ph + 54.0;
    svg.line(x0, ly, x0 + 24.0, ly, "#c44e52", 1.8);
    svg.text(x0 + 30.0, ly + 4.0, "累计运气分（左轴）", 10.5, "start");
    svg.rect(x0 + 170.0, ly - 7.0, 12.0, 12.0, "#2ca02c", 0.85);
    svg.text(x0 + 188.0, ly + 4.0, "回合Δ 正", 10.5, "start");
    svg.rect(x0 + 270.0, ly - 7.0, 12.0, 12.0, "#d62728", 0.85);
    svg.text(x0 + 288.0, ly + 4.0, "回合Δ 负", 10.5, "start");
    svg.rect(x0 + 370.0, ly - 7.0, 12.0, 12.0, "#9e9e9e", 0.85);
    svg.text(x0 + 388.0, ly + 4.0, "伪波动（年界/继承/RMJ）", 10.5, "start");
    svg.frame(x0, top, x1 - x0, ph, "#333333", 1.0);
    svg.render()
}

// ======================= 图3：行动总计（环形图） =======================

/// 行动类别与配色（训练位与图1 五维一致）
const ACTION_CATS: [(&str, &str); 12] = [
    ("速训练", "#1f77b4"),
    ("耐训练", "#ff7f0e"),
    ("力训练", "#2ca02c"),
    ("根训练", "#d62728"),
    ("智训练", "#9467bd"),
    ("休息", "#8c8c8c"),
    ("出行", "#f59e0b"),
    ("友人出行", "#ec4899"),
    ("比赛", "#7b6cb5"),
    ("治病", "#38bdf8"),
    ("剧本", "#a8a29e"),
    ("未知", "#57534e")
];

/// 极坐标 → 直角坐标（deg 以正上方为 0°，顺时针）
fn pt(cx: f64, cy: f64, r: f64, deg: f64) -> (f64, f64) {
    let rad = deg.to_radians();
    (cx + r * rad.cos(), cy + r * rad.sin())
}

/// 图3：实际执行行动总计（环形图，execution 推断口径；扇形走 `Svg::push`
/// 原生 path——构建器只有基元方法，圆弧需手拼）
fn chart_actions(exec: &[ExecRow]) -> String {
    if exec.is_empty() {
        return note_svg("图3 行动总计：无执行推断数据");
    }
    let mut counts = vec![0u32; ACTION_CATS.len()];
    for r in exec {
        if let Some(ci) = ACTION_CATS.iter().position(|(c, _)| *c == r.actual_action) {
            counts[ci] += 1;
        }
    }
    let total: u32 = counts.iter().sum();
    let present: Vec<usize> = (0..ACTION_CATS.len()).filter(|&i| counts[i] > 0).collect();
    let (w, h) = (900.0, 330.0);
    let (cx, cy, r_out, r_in) = (200.0, 185.0, 118.0, 62.0);
    let mut svg = Svg::new(w, h);
    svg.text(w / 2.0, 20.0, "实际执行行动总计（execution 推断口径）", 13.5, "middle");
    // 中心总数
    svg.text(cx, cy - 2.0, &total.to_string(), 24.0, "middle");
    svg.text(cx, cy + 22.0, "回合", 11.0, "middle");
    if present.len() == 1 {
        // 单一类别：整环（360° 圆弧是退化 path，直接画圆环）
        let (_, clr) = ACTION_CATS[present[0]];
        svg.push(format!(
            r#"<circle cx="{cx}" cy="{cy}" r="{}" fill="none" stroke="{clr}" stroke-width="{}" stroke-opacity="0.9"/>"#,
            (r_out + r_in) / 2.0,
            r_out - r_in
        ));
    } else {
        let mut acc = 0.0f64; // 累计占比（起点 = 正上方，顺时针）
        for &i in &present {
            let n = counts[i];
            let (name, clr) = ACTION_CATS[i];
            let frac = n as f64 / total as f64;
            let a0 = acc * 360.0;
            let a1 = (acc + frac) * 360.0;
            acc += frac;
            let (p0x, p0y) = pt(cx, cy, r_out, a0);
            let (p1x, p1y) = pt(cx, cy, r_out, a1);
            let (q0x, q0y) = pt(cx, cy, r_in, a0);
            let (q1x, q1y) = pt(cx, cy, r_in, a1);
            let large = u8::from(a1 - a0 > 180.0);
            let d = format!(
                "M {q0x:.2} {q0y:.2} L {p0x:.2} {p0y:.2} \
                 A {r_out:.2} {r_out:.2} 0 {large} 1 {p1x:.2} {p1y:.2} \
                 L {q1x:.2} {q1y:.2} A {r_in:.2} {r_in:.2} 0 {large} 0 {q0x:.2} {q0y:.2} Z"
            );
            svg.push(format!(
                r##"<path d="{d}" fill="{clr}" fill-opacity="0.9" stroke="#ffffff" stroke-width="1.5"/>"##
            ));
            // 大扇区（≥4%）外圈标签
            if frac >= 0.04 {
                let mid = (a0 + a1) / 2.0;
                let (lx, ly) = pt(cx, cy, r_out + 30.0, mid);
                svg.text(lx, ly + 4.0, &format!("{name} {n}"), 10.5, "middle");
            }
        }
    }
    // 图例（右侧两列：色块 + 名称 + 次数 + 占比）
    let lx = 470.0;
    let mut row = 0usize;
    for &i in &present {
        let (name, clr) = ACTION_CATS[i];
        let n = counts[i];
        let col = row % 2;
        let line = row / 2;
        let xx = lx + col as f64 * 215.0;
        let yy = 70.0 + line as f64 * 24.0;
        svg.rect(xx, yy, 12.0, 12.0, clr, 0.9);
        svg.text(
            xx + 17.0,
            yy + 11.0,
            &format!("{name} {n}（{:.1}%）", n as f64 / total as f64 * 100.0),
            11.0,
            "start"
        );
        row += 1;
    }
    svg.render()
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
        let tl = vec![
            tl_row(0, [200, 100, 100, 100, 100], [3000; 5]),
            tl_row(30, [800, 600, 400, 300, 500], [3000; 5]),
            tl_row(77, [3200, 1900, 1700, 1200, 2100], [3272, 2402, 2240, 2200, 2452]),
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
                top_loss: vec![],
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

    /// timeline 测试行（只填图表所需字段）
    fn tl_row(turn: u32, five: [i32; 5], limit: [i32; 5]) -> TimelineRow {
        TimelineRow {
            turn,
            seq: 0,
            stage: "Train".into(),
            reason: None,
            source: "command".into(),
            playing_state: 1,
            vital: 100,
            max_vital: 108,
            motivation: 5,
            five_status: five,
            five_status_display: five,
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

    /// 三图冒烟：SVG 结构与关键标记
    #[test]
    fn test_charts_markers() {
        let d = test_digest();
        let c = build_charts(&d);
        println!("图1 {} 字节 / 图2 {} / 图3 {}",
            c.status.len(), c.luck.len(), c.actions.len());
        assert!(c.status.contains("<svg") && c.status.contains("属性成长曲线"));
        assert!(c.status.contains("年界") && c.status.contains("继承30"));
        assert!(c.luck.contains("运气分双轴") && c.luck.contains("<polyline"));
        assert!(c.actions.contains("实际执行行动总计"));
        // 单一类别（测试数据只有速训练）→ 整环 circle 而非 path 扇形
        assert!(c.actions.contains("<circle") && c.actions.contains("77"));
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
        assert!(html.contains("<svg"), "三图 SVG 应经 |safe 注入");
        assert!(html.contains("图1") && html.contains("图3"));
        assert!(!html.contains("图4"), "行动图已改为总计环形（原图4 删除）");
        // 4 个叙述占位标记（skill 层回填）；检查项表与决策明细表已移除
        for marker in ["NARRATIVE:overview", "NARRATIVE:luck_trend", "NARRATIVE:findings", "NARRATIVE:summary"] {
            assert!(html.contains(marker), "占位标记 {marker} 应存在");
        }
        assert!(!html.contains("mandatory_race_not_won"), "检查项表已换叙述占位");
        assert!(!html.contains("决策明细"), "决策明细表已移除（明细见 decisions.csv）");
        assert!(html.contains("口径速览"), "口径说明已简化为速览");
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
