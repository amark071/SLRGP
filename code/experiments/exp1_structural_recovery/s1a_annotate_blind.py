#!/usr/bin/env python3
"""Blind MODEL-A annotation for S1a hard-synthetic discrimination.

The prompt never exposes item_id values that encode `authentic`, `tier1`, or
`tier2`. Each batch uses opaque IDs and maps them back after the model returns
JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "code"))
sys.path.insert(0, str(WORKSPACE / "code" / "experiments"))
from slrgp.llm_client import LLMClient  # noqa: E402
from split_utils import stable_seed_for  # noqa: E402

DEFAULT_IN = WORKSPACE / "data" / "exp1_structural_recovery" / "confirmatory"
IN = Path(os.environ.get("S1A_IN_DIR", DEFAULT_IN))
MODEL = os.environ.get("S1A_MODEL_A", "openai/gpt-5.4-mini")
OFOX = os.environ.get("OFOX_API_KEY", "")
FACETS = ["topic", "method", "theory", "object", "level", "controversy", "application_task", "other"]
RELATIONS = [
    "chronological", "foundational_to_applied", "simple_to_complex",
    "consensus_to_controversy", "general_to_specific", "enumerative_none", "other",
]


def load_items(limit=None):
    rows = []
    with open(IN / "s1a_items.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def done_ids(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (
                        isinstance(r.get("expressibility_score"), (int, float))
                        and r.get("facet") in FACETS
                        and r.get("relation") in RELATIONS
                    ):
                        done.add(r["item_id"])
                except Exception:
                    pass
    return done


def compact(batch):
    prompt_rows = []
    mapping = {}
    for i, x in enumerate(batch):
        oid = f"item_{i:03d}"
        mapping[oid] = x
        prompt_rows.append({
            "opaque_id": oid,
            "discipline": x.get("discipline"),
            "depth": x.get("node_depth"),
            "parent_title": (x.get("parent_title") or "")[:180],
            "child_titles": [(c or "")[:180] for c in (x.get("child_titles") or [])[:12]],
        })
    return prompt_rows, mapping


def make_prompt(prompt_rows):
    payload = json.dumps(prompt_rows, ensure_ascii=False)
    return f"""You are evaluating whether a sibling group of section titles from an academic literature review is naturally expressible by a fixed operator algebra for review organization.

For each item, judge only the visible parent title, sibling section titles, discipline and depth. Some groups are authentic; some are hard synthetic regroupings. You are blind to condition.

Return for EACH item:
- opaque_id: copy exactly from the input
- facet: one of {FACETS}
- relation: one of {RELATIONS}
- expressibility_score: 0.0 to 1.0, how naturally the siblings instantiate a coherent review-organization operation under the fixed inventory
- faithfulness_rating: integer 1-5, whether the grouping would be faithful expert organization rather than post-hoc rationalization
- confidence: 0.0 to 1.0
- rationale: <=18 words

Give lower scores to locally plausible but incoherent mixtures; do not reward generic labels if the siblings do not belong together.

Items:
{payload}

Return ONLY JSON with key annotations, one object per item."""


def annotate_batch(batch, pass_id):
    client = LLMClient(
        base_url="https://api.ofox.ai/v1",
        model=MODEL,
        api_key=OFOX,
        send_thinking_kwarg=False,
        default_timeout=90,
    )
    prompt_rows, mapping = compact(batch)
    seed = stable_seed_for("s1a_blind", str(pass_id), *(x["item_id"] for x in batch))
    try:
        res = client.chat_json(
            [{"role": "user", "content": make_prompt(prompt_rows)}],
            max_tokens=2200,
            temperature=0.10 if pass_id == 1 else 0.25,
            seed=seed,
        )
        annotations = res.get("annotations", []) if isinstance(res, dict) else res
        lookup = {x.get("opaque_id"): x for x in annotations if isinstance(x, dict)}
        out = []
        for oid, x in mapping.items():
            a = lookup.get(oid, {})
            row = {
                k: x.get(k)
                for k in [
                    "item_id", "pair_id", "tier", "condition", "target_node_id",
                    "donor_node_id", "discipline", "node_depth", "n_children",
                ]
            }
            row.update({
                "facet": a.get("facet"),
                "relation": a.get("relation"),
                "expressibility_score": a.get("expressibility_score"),
                "faithfulness_rating": a.get("faithfulness_rating"),
                "confidence": a.get("confidence"),
                "rationale": a.get("rationale", ""),
                "pass_id": pass_id,
                "blind_prompt": True,
            })
            out.append(row)
        return out
    except Exception as e:
        return [
            {
                **{
                    k: x.get(k)
                    for k in [
                        "item_id", "pair_id", "tier", "condition", "target_node_id",
                        "donor_node_id", "discipline", "node_depth", "n_children",
                    ]
                },
                "facet": None,
                "relation": None,
                "expressibility_score": None,
                "faithfulness_rating": None,
                "confidence": 0.0,
                "rationale": f"ERROR:{type(e).__name__}:{e}",
                "pass_id": pass_id,
                "blind_prompt": True,
            }
            for x in batch
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-id", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not OFOX:
        raise SystemExit("OFOX_API_KEY missing")
    out_path = IN / f"s1a_annotations_blind_pass{args.pass_id}.jsonl"
    done = done_ids(out_path)
    items = [x for x in load_items(args.limit) if x["item_id"] not in done]
    batches = [items[i:i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    print(f"model={MODEL} blind_pass={args.pass_id} pending_items={len(items)} batches={len(batches)} done={len(done)}")
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(annotate_batch, b, args.pass_id) for b in batches]
        for i, future in enumerate(as_completed(futures), 1):
            for row in future.result():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(futures):
                print(f"batches {i}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
