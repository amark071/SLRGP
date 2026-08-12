#!/usr/bin/env python3
"""Fetch arXiv LaTeX sources for the review corpus (one-click, resumable).

按 data/common/arxiv_review_manifest.jsonl 中的 arXiv ID 逐篇拉取 LaTeX 源码,
供 code/experiments/ 的解析与实验脚本使用(E1 解析 \\section/\\cite 层级)。

- 仅下载 parse_status == "ok" 的 2,223 篇(失败记录保留在清单中供抽样框架核算)
- 礼貌限速:每次请求间隔 4 秒(arXiv 官方对 export 单篇接口的建议),全程约 2.5 小时
- 断点续传:已存在且非空的文件直接跳过,可反复安全重跑

输出: data/common/arxiv_latex/<discipline>/<arxiv_id>.tar.gz
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = pathlib_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data/common/arxiv_review_manifest.jsonl")
OUT_DIR = os.path.join(ROOT, "data/common/arxiv_latex")
SLEEP_SECONDS = 4
ARXIV_EPRINT = "https://export.arxiv.org/e-print/{arxiv_id}"
ID_SAFE = re.compile(r"^[a-z0-9.\-/]+$", re.IGNORECASE)


def main():
    todo = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("parse_status") != "ok":
                continue
            aid, disc = rec["arxiv_id"], rec["discipline"]
            if not ID_SAFE.match(aid) or not ID_SAFE.match(disc):
                print(f"!! 非法 ID,跳过: {aid}", file=sys.stderr)
                continue
            target = os.path.join(OUT_DIR, disc, f"{aid}.tar.gz")
            if os.path.exists(target) and os.path.getsize(target) > 0:
                continue
            todo.append((aid, disc, target))
    print(f"待下载 {len(todo)} 篇(已完成的自动跳过),预计 {len(todo) * SLEEP_SECONDS / 3600:.1f} 小时")
    ok = fail = 0
    for i, (aid, disc, target) in enumerate(todo, 1):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        rc = subprocess.call(
            ["curl", "-sL", "--retry", "3", "--retry-delay", "5",
             "-o", target, ARXIV_EPRINT.format(arxiv_id=aid)]
        )
        if rc == 0 and os.path.exists(target) and os.path.getsize(target) > 0:
            ok += 1
        else:
            fail += 1
            if os.path.exists(target):
                os.remove(target)
            print(f"[{i}/{len(todo)}] !! 失败: {aid}", file=sys.stderr)
        if i % 50 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] 成功 {ok} 失败 {fail}")
        time.sleep(SLEEP_SECONDS)
    print(f"完成: 成功 {ok},失败 {fail}(失败的可重跑本脚本续传)")


if __name__ == "__main__":
    main()
