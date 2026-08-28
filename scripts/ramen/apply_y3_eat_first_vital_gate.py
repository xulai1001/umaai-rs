#!/usr/bin/env python3
"""应用第三年“先判断吃面、再判断体力”的最小策略改动。

该脚本只负责可复验地修改正式 Rust 源码；实验工作流留在 ramen workbench
体系，后续向 master 发布时只移植生成后的正式逻辑、回归测试和确认报告。
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "crates/umasim/src/game/ramen/policy.rs"
TRAINER = ROOT / "crates/umasim/src/trainer/local_ramen_trainer.rs"


def replace_once(path: Path, old: str, new: str) -> None:
    """在文件中精确替换一次，避免上游变化时静默打错补丁。"""
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: 期望匹配 1 次，实际 {count} 次")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    """修改训练体力门、吃面前判断顺序以及推荐 preset。"""
    replace_once(
        POLICY,
        """        // 守门 2：体力低 → 休息（防失败率崩盘；优先于心情、训练）
        if uma.vital < self.config.vital_rest {
""",
        """        // 守门 2：体力低 → 休息（防失败率崩盘；优先于心情、训练）。
        // 第三年已经吃面时失败率下降 100%，应先兑现本回合拉面训练窗口，
        // 不再由吃面前使用的普通体力门把低体力高价值训练挡掉。
        let vital_rest = if game.current_year() == 3 && game.ramen.current_ramen.is_some() {
            0
        } else {
            self.config.vital_rest
        };
        if uma.vital < vital_rest {
""",
    )
    replace_once(
        POLICY,
        '                    reason: format!("守门: 体力{}<{}休息", uma.vital, self.config.vital_rest),',
        '                    reason: format!("守门: 体力{}<{}休息", uma.vital, vital_rest),',
    )
    replace_once(
        TRAINER,
        """        if self.config.eat_requires_training && !matches!(pre_action, Operation::Train(_)) && !deadline_exception {
""",
        """        // 第三年先比较吃面候选：不吃面路径仍使用普通体力门恢复；若选择吃面，
        // Train 阶段因 current_ramen 已落地而跳过体力硬门，直接兑现 100% 减失败率。
        // `vital_rest == 0` 保留为配对实验的旧基准，不启用此顺序修正。
        let y3_eat_before_vital_gate = g.current_year() == 3
            && self.policy.config.vital_rest > 0
            && matches!(pre_action, Operation::Rest | Operation::FriendOuting);
        if self.config.eat_requires_training
            && !matches!(pre_action, Operation::Train(_))
            && !deadline_exception
            && !y3_eat_before_vital_gate
        {
""",
    )
    replace_once(
        TRAINER,
        """    /// 构造当前正式推荐 preset。
    pub fn new() -> Self {
""",
        """    /// 构造旧版第三年无体力门基准，仅供同种子配对实验。
    pub fn y3_vital_legacy_baseline() -> Self {
        let mut trainer = Self::new();
        trainer.years[2].policy.config.vital_rest = 0;
        trainer
    }

    /// 构造当前正式推荐 preset。
    pub fn new() -> Self {
""",
    )
    replace_once(
        TRAINER,
        """            years: [make(16.0, 30), make(64.0, 30), make(64.0, 0)],
""",
        """            years: [make(16.0, 30), make(64.0, 30), make(64.0, 30)],
""",
    )
    replace_once(
        TRAINER,
        """    fn recommended_ramen_uses_025_friend_pacing() {
""",
        """    fn recommended_ramen_uses_025_friend_pacing() {
""",
    )
    test_anchor = """    fn recommended_ramen_uses_025_friend_pacing() {
        let trainer = RecommendedRamenTrainer::new();
"""
    text = TRAINER.read_text(encoding="utf-8")
    if test_anchor not in text:
        raise RuntimeError(f"{TRAINER}: 找不到推荐 preset 测试锚点")
    test = """    /// 正式 preset 第三年不吃面时保留 30 体力门；旧基准显式关闭，供配对实验。
    #[test]
    #[allow(clippy::panic)]
    fn recommended_ramen_uses_eat_first_y3_vital_gate() {
        let current = RecommendedRamenTrainer::new();
        let baseline = RecommendedRamenTrainer::y3_vital_legacy_baseline();
        let actual = current.years.each_ref().map(|year| year.policy.config.vital_rest);
        let legacy = baseline.years.each_ref().map(|year| year.policy.config.vital_rest);
        println!("正式分年体力门: {actual:?}；旧第三年基准: {legacy:?}");
        if actual != [30, 30, 30] || legacy != [30, 30, 0] {
            panic!("第三年先吃面体力门配置错误: current={actual:?}, legacy={legacy:?}");
        }
    }

"""
    TRAINER.write_text(text.replace(test_anchor, test + test_anchor, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
