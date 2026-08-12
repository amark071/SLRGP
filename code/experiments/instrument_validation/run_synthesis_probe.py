"""
单独补跑 synthesis_flattened 探针（critical_synthesis 维度的直接操纵检查），
合并进 phase0_full_scores.json/csv，不重跑其余 7 个变体。
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "slrgp"))
from llm_client import LLMClient  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from extract_review import load_review  # noqa: E402
from make_variants import build_all_variants  # noqa: E402
from instrument_j import (  # noqa: E402
    DIMENSIONS, JUDGES, PRICING, N_REPEATS, score_one, OUT_DIR,
)

OFOX_API_KEY = os.environ.get("OFOX_API_KEY", "")
PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
SOURCES = {
    "2306.01660": os.path.join(PKG_ROOT, "data", "instrument_validation", "sources", "2306.01660.tar.gz"),
    "1809.00057": os.path.join(PKG_ROOT, "data", "instrument_validation", "sources", "1809.00057.tar.gz"),
}
FROZEN_SCORES = os.path.join(PKG_ROOT, "data", "instrument_validation", "scores", "phase0_full_scores.json")


def main():
    if not OFOX_API_KEY:
        print("[FATAL] 请先 export OFOX_API_KEY=...", file=sys.stderr)
        sys.exit(1)

    base = os.path.join(PKG_ROOT, "work")
    texts = {}
    hits_report = {}
    for doc_id, tarball in SOURCES.items():
        work_dir = os.path.join(base, "instrument_validation", doc_id)
        doc = load_review(tarball, work_dir)
        variants = build_all_variants(doc)
        hits_report[doc_id] = variants.pop("_synthesis_flattened_hits", 0)
        texts[doc_id] = variants["synthesis_flattened"]

    print(f"[synthesis_probe] 替换命中次数: {hits_report}")
    print(f"[synthesis_probe] {len(texts)} 篇 x {len(JUDGES)} 裁判 x {N_REPEATS} 次重复 "
          f"= {len(texts) * len(JUDGES) * N_REPEATS} 次调用")

    rows = []
    total_cost = 0.0
    for judge_tag, model_id in JUDGES.items():
        llm = LLMClient(base_url="https://api.ofox.ai/v1", model=model_id, api_key=OFOX_API_KEY,
                         send_thinking_kwarg=False, default_timeout=180, verbose=False)
        for doc_id, text in texts.items():
            for rep in range(1, N_REPEATS + 1):
                print(f"[{judge_tag}][{doc_id}][synthesis_flattened][rep{rep}] 打分中 ...", flush=True)
                try:
                    r = score_one(llm, text)
                    print(f"  -> {[r.get(f'{d}_score') for d in DIMENSIONS]}", flush=True)
                except Exception as e:
                    r = {"error": f"{type(e).__name__}: {e}"}
                    print(f"  -> FAIL: {r['error']}", flush=True)
                r.update({"judge": judge_tag, "model_id": model_id, "doc_id": doc_id,
                          "variant": "synthesis_flattened", "repeat": rep})
                rows.append(r)
        in_p, out_p = PRICING.get(model_id, (0, 0))
        cost = llm.total_prompt_tokens / 1e6 * in_p + llm.total_completion_tokens / 1e6 * out_p
        total_cost += cost
        print(f"[{judge_tag}] calls={llm.n_calls} cost=${cost:.3f}")

    out_json = os.path.join(OUT_DIR, "phase0_full_scores.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(out_json) and os.path.exists(FROZEN_SCORES):
        # 以随包的冻结评分为起点，在其上合并本次补跑结果。
        shutil.copy2(FROZEN_SCORES, out_json)
    with open(out_json, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing = [r for r in existing if r.get("variant") != "synthesis_flattened"]  # 幂等：重跑覆盖旧的
    existing.extend(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    fields = ["judge", "model_id", "doc_id", "variant", "repeat", "n_words", "truncated", "elapsed_sec"] + \
             [f"{d}_score" for d in DIMENSIONS] + [f"{d}_rationale" for d in DIMENSIONS] + ["error"]
    out_csv = os.path.join(OUT_DIR, "phase0_full_scores.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in existing:
            w.writerow(row)

    print(f"\n{'='*70}\n追加写入 {out_json} / {out_csv}\n本次预估花费: ${total_cost:.3f}\n{'='*70}")


if __name__ == "__main__":
    main()
