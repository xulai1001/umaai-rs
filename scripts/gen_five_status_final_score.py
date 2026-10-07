#!/usr/bin/env python3
"""从 UmamusumeResponseAnalyzer 的 StatusToPoint 表生成 umaai 的 five_status_final_score。

## 背景

两边表的索引口径不同：

- URA（`Database.cs` 的 `StatusToPoint`）以**显示值** d 为索引，共 2501 项（0..2500）。
- umaai（`gamedata/constants.json` 的 `five_status_final_score`）以**原始属性** r 为索引
  （1200 以上未减半），历史上长度 3399。

显示空间截断（`explain.rs::five_status_cutted`）：

    cut(r) = r                    (r <= 1200)
    cut(r) = 1200 + (r-1200)//2   (r >  1200)

因此把 URA 表展开回未减半索引的映射是：

    raw[r] = ura[cut(r)]

最大 r 满足 cut(r) <= 2500 的是 r = 3801，故生成长度 3802（末尾两项 3800/3801 都映射到
ura[2500]，`status_final_score` 对更长的 raw 越界饱和到该项）。

## 一致性

显示值 0..2000（raw 0..2800）两表逐位相等，脚本会断言这一点；差异仅出现在
raw >= 2802（显示值 >= 2001）。

用法：

    python scripts/gen_five_status_final_score.py            # 写入 constants.json
    python scripts/gen_five_status_final_score.py --dry-run  # 只打印，不写入
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # f:\UmaAI_Active\umaai-rs
URA_DB = Path(
    r"f:\UmaAI_Active\UmamusumeResponseAnalyzer\UmamusumeResponseAnalyzer\Database.cs"
)
CONSTANTS = REPO_ROOT / "gamedata" / "constants.json"

# URA 表覆盖的最大显示值（含）
MAX_DISPLAY = 2500
# 显示空间截断起点
CUT_POINT = 1200
# 两表在 raw 0..该值（含）逐位相等
EQUAL_UP_TO_RAW = 2800


def load_ura_table() -> list[int]:
    """抽取 Database.cs 中 `StatusToPoint` 的 int 列表"""
    text = URA_DB.read_text(encoding="utf-8")
    m = re.search(r"StatusToPoint\s*\{[^=]*=\s*\[([^\]]*)\]", text, re.DOTALL)
    if m is None:
        raise SystemExit(f"在 {URA_DB} 中找不到 StatusToPoint 列表")
    body = m.group(1)
    values = [int(tok) for tok in (t.strip() for t in body.split(",")) if tok]
    if len(values) != MAX_DISPLAY + 1:
        raise SystemExit(
            f"URA 表长度异常：期望 {MAX_DISPLAY + 1}，实际 {len(values)}"
        )
    return values


def cut(r: int) -> int:
    """显示空间截断"""
    return r if r <= CUT_POINT else CUT_POINT + (r - CUT_POINT) // 2


def build_raw_table(ura: list[int]) -> list[int]:
    """把显示值索引表展开回未减半的 raw 索引表"""
    max_raw = CUT_POINT + 2 * (MAX_DISPLAY - CUT_POINT) + 1  # = 3801
    return [ura[cut(r)] for r in range(max_raw + 1)]


def read_constants_raw() -> str:
    """按字节读入，不做换行翻译（保留原 CRLF）"""
    with CONSTANTS.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def parse_current_table(text: str) -> list[int]:
    m = re.search(r'"five_status_final_score"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m is None:
        raise SystemExit("在 constants.json 中找不到 five_status_final_score 数组")
    body = m.group(1)
    return [int(tok) for tok in (tok.strip() for tok in body.split(",")) if tok]


def format_and_write(text: str, raw: list[int], dry_run: bool) -> None:
    """定向替换数组体内的数字，其余字节（含换行风格）原样保留"""
    eol = "\r\n" if "\r\n" in text else "\n"
    body = ",".join(str(v) for v in raw)
    replacement = '"five_status_final_score": [' + eol + "    " + body + eol + "  ]"
    m = re.search(r'"five_status_final_score"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    assert m is not None
    new_text = text[: m.start()] + replacement + text[m.end() :]
    if not dry_run:
        with CONSTANTS.open("w", encoding="utf-8", newline="") as f:
            f.write(new_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写入文件")
    args = ap.parse_args()

    ura = load_ura_table()
    raw = build_raw_table(ura)

    old_text = read_constants_raw()
    old = parse_current_table(old_text)

    # 断言：raw 0..2800 与旧表逐位相等（显示值 0..2000 区间两表本就一致）
    n = EQUAL_UP_TO_RAW + 1
    assert len(old) >= n, f"旧表长度 {len(old)} 不足以比对前 {n} 项"
    for i in range(n):
        if old[i] != raw[i]:
            raise SystemExit(f"一致性断言失败：raw[{i}] 旧={old[i]} 新={raw[i]}")

    diffs = [i for i in range(min(len(old), len(raw))) if old[i] != raw[i]]
    print(f"URA 表项数        : {len(ura)}")
    print(f"旧 raw 表长度     : {len(old)}")
    print(f"新 raw 表长度     : {len(raw)}")
    print(f"raw 0..{EQUAL_UP_TO_RAW} 逐位相等 : OK")
    print(f"差异项数(共同区间): {len(diffs)}")
    if diffs:
        print(f"首个差异 raw 索引 : {diffs[0]} (显示值 {cut(diffs[0])})")
        print(f"末尾差异 raw 索引 : {diffs[-1]} (显示值 {cut(diffs[-1])})")
    print(f"新表表尾 8 项     : {raw[-8:]}")
    print(f"raw=2802/2900/3000/3100/3200/3300 -> {[raw[i] for i in (2802,2900,3000,3100,3200,3300)]}")

    format_and_write(old_text, raw, args.dry_run)
    print("已写入" if not args.dry_run else "（dry-run，未写入）", CONSTANTS)


if __name__ == "__main__":
    main()