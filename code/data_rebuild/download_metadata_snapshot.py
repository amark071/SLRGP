#!/usr/bin/env python3
"""下载 librarian-bots/arxiv-metadata-snapshot 全部10个分片（HuggingFace 官方源）。"""
import os
import subprocess
import sys

OUT_DIR = "data/common/arxiv_metadata_snapshot"
BASE_URL = "https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot/resolve/main/data"
FILES = [f"train-{i:05d}-of-00010.parquet" for i in range(10)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, fname in enumerate(FILES, 1):
        target = os.path.join(OUT_DIR, fname)
        if os.path.exists(target) and os.path.getsize(target) > 1_000_000:
            print(f"[{i}/10] 跳过(已存在): {fname}")
            continue
        print(f"[{i}/10] 下载: {fname}")
        url = f"{BASE_URL}/{fname}"
        rc = subprocess.call(["curl", "-sL", "--retry", "5", "--retry-delay", "3", "-o", target, url])
        if rc != 0 or not os.path.exists(target):
            print(f"!! 下载失败: {fname}", file=sys.stderr)
            sys.exit(1)
    print("全部10个分片下载完成")


if __name__ == "__main__":
    main()
