#!/usr/bin/env python3
"""
S2e/O Phase 2 — checkpointed MODEL-A two-axis annotation.

Each organization node is labelled independently on two compositional axes:
  facet: what property partitions evidence
  relation: how sibling sections unfold

The input is the deterministic depth-aware corpus from s2e_build_corpus.py.
Old Qwen labels/clusters are never read. Output is JSONL and safely resumable:
only records with a valid label pair count as complete.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "code/slrgp")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from split_utils import stable_seed_for  # noqa: E402
from slrgp.llm_client import LLMClient  # noqa: E402

IN_DIR = os.environ.get("S2E_OUT_DIR", "data/exp2_operator_learnability/organization/confirmatory")
MODEL = "openai/gpt-5.4-mini"
OFOX = os.environ.get("OFOX_API_KEY", "")
FACETS = ["topic", "method", "theory", "object", "level", "controversy", "application_task", "other"]
RELATIONS = [
    "chronological", "foundational_to_applied", "simple_to_complex",
    "consensus_to_controversy", "general_to_specific", "enumerative_none", "other",
]
FACET_ALIASES = {
    "model": "method",
    "model_family": "method",
    "technique": "method",
    "task": "application_task",
    "application": "application_task",
}
RELATION_ALIASES = {
    "foundation_to_applied": "foundational_to_applied",
    "foundational-applied": "foundational_to_applied",
    "problem_solution": "other",
    "topic": "enumerative_none",
    "method": "enumerative_none",
}


def load_nodes() -> list[dict]:
    out = []
    for split in ["train", "heldout"]:
        p = Path(IN_DIR) / f"nodes_{split}.jsonl"
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def node_id(n: dict) -> str:
    return f"{n['arxiv_id']}::{n['node_path']}"


def valid(r: dict) -> bool:
    return r.get("facet") in FACETS and r.get("relation") in RELATIONS


def canonical_label(value, allowed: list[str], aliases: dict[str, str]) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in allowed:
        return raw
    return aliases.get(raw, "other")


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if valid(r):
                    done.add(r["node_id"])
            except Exception:
                pass
    return done


def compact(n: dict) -> dict:
    titles = [(x or "")[:180] for x in n["child_titles"][:12]]
    return {
        "node_id": node_id(n),
        "depth": n["node_depth"],
        "review_length": n["review_length_bucket"],
        "parent": (n.get("node_title") or "")[:180],
        "children": titles,
    }


def prompt(batch: list[dict]) -> str:
    return f"""You are labelling the organizational logic of sibling sections in academic literature reviews.
For EACH node, assign exactly one value on each independent axis.

PARTITION FACET (what property separates the siblings):
topic | method | theory | object | level | controversy | application_task | other

ORDERING RELATION (how sibling sections unfold):
chronological | foundational_to_applied | simple_to_complex | consensus_to_controversy | general_to_specific | enumerative_none | other

Rules:
- `facet` and `relation` are compositional: do not collapse them into one category.
- A review root is a valid organization node; depth and length are context only, not labels.
- Use `enumerative_none` only when no meaningful sequence is evident.
- Do not infer citation status or any hidden model label.

Nodes:
{json.dumps(batch, ensure_ascii=False)}

Return ONLY JSON: {{"annotations":[{{"node_id":"...","facet":"...","relation":"...","confidence":0.0,"rationale":"<=20 words"}}]}}"""


def annotate_batch(nodes: list[dict], pass_id: int) -> list[dict]:
    client = LLMClient(
        base_url="https://api.ofox.ai/v1", model=MODEL, api_key=OFOX,
        send_thinking_kwarg=False, default_timeout=90,
    )
    seed = stable_seed_for("s2e", str(pass_id), *(node_id(n) for n in nodes))
    try:
        result = client.chat_json(
            [{"role": "user", "content": prompt([compact(n) for n in nodes])}],
            max_tokens=1600, temperature=0.15 if pass_id == 1 else 0.35, seed=seed,
        )
        lookup = {x.get("node_id"): x for x in result.get("annotations", []) if isinstance(x, dict)}
        out = []
        for n in nodes:
            a = lookup.get(node_id(n), {})
            raw_facet = a.get("facet")
            raw_relation = a.get("relation")
            r = {
                "node_id": node_id(n), "arxiv_id": n["arxiv_id"], "node_path": n["node_path"],
                "split": n["split"], "node_depth": n["node_depth"],
                "review_length_bucket": n["review_length_bucket"], "discipline": n["discipline"],
                "facet": canonical_label(raw_facet, FACETS, FACET_ALIASES),
                "relation": canonical_label(raw_relation, RELATIONS, RELATION_ALIASES),
                "raw_facet": raw_facet, "raw_relation": raw_relation,
                "confidence": a.get("confidence"), "rationale": a.get("rationale", ""),
                "pass_id": pass_id,
            }
            out.append(r)
        return out
    except Exception as e:
        return [{"node_id": node_id(n), "arxiv_id": n["arxiv_id"], "node_path": n["node_path"],
                 "split": n["split"], "node_depth": n["node_depth"],
                 "review_length_bucket": n["review_length_bucket"], "discipline": n["discipline"],
                 "facet": None, "relation": None, "confidence": 0.0,
                 "rationale": f"ERROR:{type(e).__name__}:{e}", "pass_id": pass_id}
                for n in nodes]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-id", type=int, choices=[1, 2], required=True)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not OFOX:
        raise SystemExit("OFOX_API_KEY missing")

    out_path = Path(IN_DIR) / f"annotations_pass{args.pass_id}.jsonl"
    done = completed(out_path)
    nodes = [n for n in load_nodes() if node_id(n) not in done]
    if args.limit:
        nodes = nodes[:args.limit]
    batches = [nodes[i:i + args.batch_size] for i in range(0, len(nodes), args.batch_size)]
    print(f"pass={args.pass_id} pending_nodes={len(nodes)} batches={len(batches)} done={len(done)}")

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(annotate_batch, b, args.pass_id) for b in batches]
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(batches):
                print(f"  batches {i}/{len(batches)}", flush=True)


if __name__ == "__main__":
    main()
