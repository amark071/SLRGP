#!/usr/bin/env python3
"""
S2d 正例链接审计——MODEL-A 复核（调用 OpenAI 兼容端点）。

对 audit_items.json 每条，让 MODEL-A(gpt-5.4-mini) 判定"被引原始条目"与"语料库记录"是否同一工作
（预印本↔期刊版视为同一工作=正确链接；同名不同文/版本错配=错误链接）。
报链接精度 + Wilson 95% CI（整体 + 分层）。协议门：CI 下界 ≥0.95。

用法：export OFOX_API_KEY=...; python3 s2d_audit_verify.py
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "slrgp"))
from llm_client import LLMClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--tight", action="store_true", help="读 audit_items_tight.json（收紧标签重审）")
_ARGS, _ = _ap.parse_known_args()
_SUF = "_tight" if _ARGS.tight else ""
ITEMS = os.path.join(HERE, "work", f"audit_items{_SUF}.json")
OUT = os.path.join(HERE, "work", f"audit_verdicts{_SUF}.json")
MODEL = "openai/gpt-5.4-mini"
OFOX = os.environ.get("OFOX_API_KEY", "")

PROMPT = """You are auditing a bibliographic linkage. We linked a citation from a review's reference list \
to a record in our corpus by normalized-title matching. Decide whether the two describe THE SAME scholarly work.

Rules:
- A preprint (e.g. arXiv) and its published journal/conference version ARE the same work → SAME.
- Two genuinely different papers that happen to share a similar/identical title → DIFFERENT.
- If authors clearly mismatch, or the titles refer to different topics, → DIFFERENT.
- If you cannot tell, → UNCERTAIN.

Cited reference (raw, from the review's bibliography):
{cited}

Corpus record we linked it to:
title: {ct}
authors: {ca}
year: {cy}
venue: {cv}

Respond ONLY as JSON: {{"verdict": "SAME|DIFFERENT|UNCERTAIN", "reason": "one sentence"}}"""


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, center - half, center + half)


def verify_one(item):
    llm = LLMClient(base_url="https://api.ofox.ai/v1", model=MODEL, api_key=OFOX,
                    send_thinking_kwarg=False, default_timeout=90)
    cited = item.get("cited_entry_raw") or {}
    cr = item["corpus_record"]
    msg = [{"role": "user", "content": PROMPT.format(
        cited=(cited.get("raw") or cited.get("title") or "(not recovered)"),
        ct=cr.get("title"), ca=str(cr.get("authors"))[:300], cy=cr.get("year"), cv=cr.get("venue"))}]
    try:
        r = llm.chat_json(msg, max_tokens=2000, temperature=0.0)
        v = str(r.get("verdict", "UNCERTAIN")).upper()
        if v not in ("SAME", "DIFFERENT", "UNCERTAIN"):
            v = "UNCERTAIN"
        return {**item, "verdict": v, "reason": r.get("reason", "")}
    except Exception as e:
        return {**item, "verdict": "ERROR", "reason": f"{type(e).__name__}:{e}"}


def main():
    if not OFOX:
        print("请先 export OFOX_API_KEY=...", file=sys.stderr); sys.exit(1)
    data = json.load(open(ITEMS, encoding="utf-8"))
    items = data["items"]
    print(f"审计 {len(items)} 条，MODEL-A={MODEL}")
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, res in enumerate(ex.map(verify_one, items)):
            results[i] = res
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(items)}")

    same = sum(1 for r in results if r["verdict"] == "SAME")
    diff = sum(1 for r in results if r["verdict"] == "DIFFERENT")
    unc = sum(1 for r in results if r["verdict"] == "UNCERTAIN")
    err = sum(1 for r in results if r["verdict"] == "ERROR")
    n_dec = same + diff  # 有明确裁决的
    p, lo, hi = wilson(same, n_dec)
    p_c, lo_c, hi_c = wilson(same, same + diff + unc)  # 保守：uncertain 计入分母

    # 分层
    strata_stats = {}
    for s in ["no_doi", "short_generic_title", "low_reliability_bbl", "preprint_source", "clean"]:
        sub = [r for r in results if s in r.get("strata", [])]
        if not sub:
            continue
        sm = sum(1 for r in sub if r["verdict"] == "SAME")
        df = sum(1 for r in sub if r["verdict"] == "DIFFERENT")
        pp, ll, hh = wilson(sm, sm + df)
        strata_stats[s] = {"n": len(sub), "same": sm, "different": df,
                           "uncertain": sum(1 for r in sub if r["verdict"] == "UNCERTAIN"),
                           "precision": pp, "wilson95": [ll, hh]}

    out = {"model": MODEL, "n": len(items),
           "counts": {"SAME": same, "DIFFERENT": diff, "UNCERTAIN": unc, "ERROR": err},
           "precision_decided": {"point": p, "wilson95": [lo, hi], "n_denominator": n_dec},
           "precision_conservative_unc_in_denom": {"point": p_c, "wilson95": [lo_c, hi_c]},
           "gate_ci_lower_ge_0.95": lo >= 0.95,
           "per_stratum": strata_stats,
           "verdicts": results}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSAME={same} DIFFERENT={diff} UNCERTAIN={unc} ERROR={err}")
    print(f"链接精度(有裁决) = {p:.4f}  Wilson95 [{lo:.4f}, {hi:.4f}]  (n={n_dec})")
    print(f"保守精度(unc计入) = {p_c:.4f}  Wilson95 [{lo_c:.4f}, {hi_c:.4f}]")
    print(f"协议门 CI下界≥0.95: {lo >= 0.95}")
    for s, st in strata_stats.items():
        print(f"  [{s}] n={st['n']} same={st['same']} diff={st['different']} unc={st['uncertain']} prec={st['precision']:.3f}")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
