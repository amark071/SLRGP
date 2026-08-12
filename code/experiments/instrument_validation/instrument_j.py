"""
仪器 J 三探针的完整实现（对应论文 Methods「Instrument validation」）。

设计要点：
  - 引用腐蚀覆盖删除 / 替换 / 打乱三档；
  - 每个 (文档, 条件, 裁判) 独立重复 2 次，报 within-judge 方差；
  - 使用两篇不同学科、非紧凑截断的真实综述，结构扰动效应
    不受"来源文本本身就被截断"干扰；
  - 盲化：裁判只看到一份文献综述正文，不知道它属于哪个条件/哪篇源文档；
  - 要求裁判先给每个维度写 rationale 再给分（而不是先给一个总分再补理由）。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "slrgp"))
from llm_client import LLMClient  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from extract_review import load_review  # noqa: E402
from make_variants import build_all_variants  # noqa: E402

OFOX_API_KEY = os.environ.get("OFOX_API_KEY", "")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "work", "instrument_validation", "scores")

JUDGES = {
    "gpt55": "openai/gpt-5.5",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "sonnet5": "anthropic/claude-sonnet-5",
}
PRICING = {  # ($/M input, $/M output) 仅用于估算花费
    "openai/gpt-5.5": (5.0, 30.0),
    "google/gemini-3.1-pro-preview": (2.0, 12.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
}
JUDGE_MAX_TOKENS = 3000
N_REPEATS = 2
MAX_WORDS = 15000  # 防止 lengthened_redundant 变体把 prompt 撑爆

DIMENSIONS = ["organizational_quality", "critical_synthesis", "global_coherence", "citation_plausibility"]

RUBRIC = """You are an expert academic reviewer. You will be shown the full text of a literature review article. \
Evaluate it strictly on its own merits — you have no information about how it was produced or whether it has been \
modified in any way; judge only the text in front of you.

Score the review on these 4 dimensions, each on a 1-5 integer scale (1=poor, 3=acceptable, 5=excellent). \
For EACH dimension, first write a 2-3 sentence rationale citing concrete evidence from the text (section names, \
specific claims, or citation markers like [author2020]), THEN give the integer score. Do not let the score for one \
dimension influence your rationale for another.

1. organizational_quality: Is the review well-structured — does the sequence of sections form a coherent logical \
progression (not just a list of topics in arbitrary order)?
2. critical_synthesis: Does the text compare, contrast, or synthesize across cited works (rather than describing \
one paper at a time with no connective analysis)?
3. global_coherence: Is terminology consistent and is there a coherent narrative thread with no internal \
contradictions or non-sequiturs across sections?
4. citation_plausibility: For each in-text citation marker like [author2020], does it plausibly support the claim \
it is attached to (topically and logically), rather than appearing random, mismatched, or decorative?

Review text (may be truncated if very long):
---
{text}
---

Respond with ONLY a JSON object, no markdown fences, no other text, in exactly this shape:
{{"organizational_quality": {{"rationale": "...", "score": <1-5>}}, "critical_synthesis": {{"rationale": "...", "score": <1-5>}}, "global_coherence": {{"rationale": "...", "score": <1-5>}}, "citation_plausibility": {{"rationale": "...", "score": <1-5>}}}}
"""


def truncate(text, max_words=MAX_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]) + "\n\n[... truncated due to context limit ...]", True


def build_all_probe_texts():
    """返回 {doc_id: {variant_name: text}}。"""
    pkg_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    sources = {
        "2306.01660": os.path.join(pkg_root, "data", "instrument_validation", "sources", "2306.01660.tar.gz"),
        "1809.00057": os.path.join(pkg_root, "data", "instrument_validation", "sources", "1809.00057.tar.gz"),
    }
    base = os.path.join(pkg_root, "work")
    out = {}
    for doc_id, tarball in sources.items():
        work_dir = os.path.join(base, "instrument_validation", doc_id)
        doc = load_review(tarball, work_dir)
        variants = build_all_variants(doc)
        n_hits = variants.pop("_synthesis_flattened_hits", None)
        if n_hits is not None:
            print(f"[phase0] {doc_id} synthesis_flattened: {n_hits} 处话语标记词被替换", flush=True)
        out[doc_id] = variants
    return out


def score_one(llm: LLMClient, text: str) -> dict:
    text_trunc, truncated = truncate(text)
    prompt = RUBRIC.format(text=text_trunc)
    t0 = time.time()
    result = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=JUDGE_MAX_TOKENS,
                            temperature=0.2, retries=2)
    elapsed = round(time.time() - t0, 1)
    flat = {"truncated": truncated, "elapsed_sec": elapsed, "n_words": len(text.split())}
    for dim in DIMENSIONS:
        d = result.get(dim, {})
        flat[f"{dim}_score"] = d.get("score")
        flat[f"{dim}_rationale"] = d.get("rationale")
    return flat


def main():
    if not OFOX_API_KEY:
        print("[FATAL] 请先 export OFOX_API_KEY=...", file=sys.stderr)
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    probes = build_all_probe_texts()
    variant_names = list(next(iter(probes.values())).keys())
    print(f"[phase0] {len(probes)} 篇源文档 x {len(variant_names)} 个条件 x {len(JUDGES)} 个裁判 x {N_REPEATS} 次重复 "
          f"= {len(probes) * len(variant_names) * len(JUDGES) * N_REPEATS} 次调用")

    rows = []
    total_cost = 0.0
    for judge_tag, model_id in JUDGES.items():
        llm = LLMClient(base_url="https://api.ofox.ai/v1", model=model_id, api_key=OFOX_API_KEY,
                         send_thinking_kwarg=False, default_timeout=180, verbose=False)
        for doc_id, variants in probes.items():
            for variant_name, text in variants.items():
                for rep in range(1, N_REPEATS + 1):
                    print(f"[{judge_tag}][{doc_id}][{variant_name}][rep{rep}] 打分中 ...", flush=True)
                    try:
                        r = score_one(llm, text)
                        print(f"  -> {[r.get(f'{d}_score') for d in DIMENSIONS]}", flush=True)
                    except Exception as e:
                        r = {"error": f"{type(e).__name__}: {e}"}
                        print(f"  -> FAIL: {r['error']}", flush=True)
                    r.update({"judge": judge_tag, "model_id": model_id, "doc_id": doc_id,
                              "variant": variant_name, "repeat": rep})
                    rows.append(r)
        in_p, out_p = PRICING.get(model_id, (0, 0))
        cost = llm.total_prompt_tokens / 1e6 * in_p + llm.total_completion_tokens / 1e6 * out_p
        total_cost += cost
        print(f"[{judge_tag}] calls={llm.n_calls} prompt_tok={llm.total_prompt_tokens} "
              f"completion_tok={llm.total_completion_tokens} cost=${cost:.3f}")

    out_json = os.path.join(OUT_DIR, "phase0_full_scores.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    fields = ["judge", "model_id", "doc_id", "variant", "repeat", "n_words", "truncated", "elapsed_sec"] + \
             [f"{d}_score" for d in DIMENSIONS] + [f"{d}_rationale" for d in DIMENSIONS] + ["error"]
    out_csv = os.path.join(OUT_DIR, "phase0_full_scores.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n{'='*70}\n写入 {out_json} / {out_csv}\n总预估花费: ${total_cost:.3f}\n{'='*70}")


if __name__ == "__main__":
    main()
