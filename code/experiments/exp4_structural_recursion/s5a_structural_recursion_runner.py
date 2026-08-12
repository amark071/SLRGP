#!/usr/bin/env python3
"""Qualitative S5a structural-recursion diagnostic.

This runner intentionally differs from ``s5a_length_runner.py``.  The earlier
runner tested naive recursive chunking: fixed top-level groups, deterministic
binary splits, leaf-only writing, and concatenation.  This diagnostic tests the
actual SLRGP recursion semantics at small scale:

    O(node) -> child LOTCF-LR tree
    Descend(parent, child) -> child state with inherited evidence
    W(leaf) -> bounded local subsection
    Merge(parent) -> parent-level synthesis and transitions

It is qualitative by design: one long-topic run with three arms is enough to
verify whether the experimental object matches the theory before any larger
confirmatory rerun.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from s5a_length_runner import (  # noqa: E402
    DEFAULT_TIERS,
    OFOXClient,
    card_lines,
    cards_for_ids,
    citation_ids,
    expand_if_short,
    load_topic_assets,
    render_flat,
    render_recursive,
    safe_name,
    sha_json,
    usage,
    usage_delta,
    word_count,
)


ARMS = ("structural_lotcf_recursive", "naive_recursive_chunking", "flat_single_pass")


def parse_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"No JSON object in response: {raw[:500]!r}")


def call_json(client: OFOXClient, prompt: str, max_tokens: int, seed: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = client.chat(prompt, max_tokens=max_tokens, temperature=0.15, seed=seed + attempt)
            return parse_json_object(raw)
        except Exception as exc:  # noqa: BLE001 - preserve LLM response debugging context
            last_error = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"JSON call failed: {last_error!r}")


def normalize_children(raw_children: list[dict[str, Any]], valid_ids: list[str]) -> list[dict[str, Any]]:
    """Keep child assignments valid, unique and complete."""
    valid = set(valid_ids)
    seen: set[str] = set()
    children: list[dict[str, Any]] = []
    for idx, child in enumerate(raw_children):
        title = str(child.get("title") or child.get("name") or f"Subproblem {idx + 1}").strip()
        facet = str(child.get("facet") or child.get("partition_facet") or "topic").strip()
        relation = str(child.get("relation") or child.get("ordering_relation") or "enumerative").strip()
        rationale = str(child.get("rationale") or "").strip()
        doc_ids = []
        for doc_id in child.get("doc_ids", []):
            if doc_id in valid and doc_id not in seen:
                doc_ids.append(doc_id)
                seen.add(doc_id)
        if doc_ids:
            children.append({"title": title, "facet": facet, "relation": relation, "rationale": rationale, "doc_ids": doc_ids})

    missing = [doc_id for doc_id in valid_ids if doc_id not in seen]
    if not children:
        children = [{"title": "Evidence cluster", "facet": "topic", "relation": "enumerative", "rationale": "", "doc_ids": []}]
    for idx, doc_id in enumerate(missing):
        children[idx % len(children)]["doc_ids"].append(doc_id)
    return [child for child in children if child["doc_ids"]]


def organize_node(
    topic: str,
    node_title: str,
    cards: list[dict],
    depth: int,
    d_max: int,
    leaf_size: int,
    client: OFOXClient,
    trace: dict,
    seed: int,
) -> dict[str, Any]:
    doc_ids = [card["doc_id"] for card in cards]
    node = {
        "title": node_title,
        "depth": depth,
        "doc_ids": doc_ids,
        "facet": "root" if depth == 0 else "topic",
        "relation": "enumerative",
        "children": [],
    }
    trace["descend_events"].append({"node_path": node_title, "depth": depth, "n_docs": len(doc_ids), "event": "enter_node"})
    if depth >= d_max or len(cards) <= leaf_size:
        trace["descend_events"].append({"node_path": node_title, "depth": depth, "n_docs": len(doc_ids), "event": "leaf_stop"})
        return node

    max_children = min(4, max(2, math.ceil(len(cards) / leaf_size)))
    prompt = f"""You are instantiating the SLRGP organization operator O for one node of a literature-review recursion tree.

Topic: {topic}
Current node: {node_title}
Depth: {depth}; maximum depth: {d_max}

Task:
Partition the evidence cards into 2 to {max_children} intellectually meaningful child subproblems.
This is NOT a flat outline. Choose a partition facet and an ordering relation for this node.
Every doc_id below must be assigned to exactly one child. Do not invent doc_ids.

Return ONLY JSON:
{{
  "facet": "topic|method|theory|object-of-study|level-of-analysis|controversy|application|dataset-task|other",
  "relation": "chronological|foundational-to-applied|simple-to-complex|consensus-to-controversy|general-to-specific|enumerative|other",
  "rationale": "one sentence",
  "children": [
    {{"title": "child title", "facet": "facet used inside child if later expanded", "relation": "expected unfolding relation", "rationale": "why these papers belong together", "doc_ids": ["..."]}}
  ]
}}

EVIDENCE CARDS:
{card_lines(cards, max_abs_chars=650)}
"""
    value = call_json(client, prompt, max_tokens=3500, seed=seed)
    node["facet"] = str(value.get("facet") or "topic")
    node["relation"] = str(value.get("relation") or "enumerative")
    node["rationale"] = str(value.get("rationale") or "")
    children = normalize_children(list(value.get("children") or []), doc_ids)
    trace["o_calls"].append({
        "node_title": node_title,
        "depth": depth,
        "n_docs": len(doc_ids),
        "facet": node["facet"],
        "relation": node["relation"],
        "n_children": len(children),
    })
    by_id = {card["doc_id"]: card for card in cards}
    for child_idx, child in enumerate(children):
        child_cards = [by_id[doc_id] for doc_id in child["doc_ids"] if doc_id in by_id]
        child_node = organize_node(
            topic=topic,
            node_title=f"{node_title}/{child['title']}",
            cards=child_cards,
            depth=depth + 1,
            d_max=d_max,
            leaf_size=leaf_size,
            client=client,
            trace=trace,
            seed=seed + 1000 + child_idx * 97,
        )
        child_node.update({key: child.get(key, child_node.get(key)) for key in ("title", "facet", "relation", "rationale")})
        node["children"].append(child_node)
    return node


def collect_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    if not node.get("children"):
        return [node]
    leaves: list[dict[str, Any]] = []
    for child in node["children"]:
        leaves.extend(collect_leaves(child))
    return leaves


def write_structural_leaf(topic: str, node: dict[str, Any], cards: list[dict], target_words: int, client: OFOXClient, trace: dict, seed: int) -> str:
    path = node["title"]
    prompt = f"""Write a literature review leaf subsection of about {target_words} English words.

Topic: {topic}
LOTCF-LR node path: {path}
Partition facet: {node.get('facet')}
Ordering relation: {node.get('relation')}
Node rationale: {node.get('rationale', '')}

Use ONLY the evidence cards below. Do not invent papers or cite IDs not listed.
Write coherent scholarly prose, not bullets. Do not include Markdown headings.
Compare papers when the evidence supports comparison. Cite inline with exact bracket IDs.

EVIDENCE CARDS:
{card_lines(cards)}
"""
    text = client.chat(prompt, max_tokens=max(1600, int(target_words * 2.5)), temperature=0.3, seed=seed).strip()
    text = re.sub(r"^#{1,6}\s+.*(?:\n|$)", "", text, flags=re.M).strip()
    trace["leaf_writes"].append({
        "node_path": path,
        "candidate_ids": [card["doc_id"] for card in cards],
        "written_citation_ids": citation_ids(text),
        "word_count": word_count(text),
    })
    return text


def summarize_text(text: str, limit: int = 900) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def merge_node(topic: str, node: dict[str, Any], child_blocks: list[tuple[dict[str, Any], str]], client: OFOXClient, trace: dict, seed: int) -> str:
    if not child_blocks:
        return ""
    child_summaries = "\n".join(
        f"- {child.get('title')}: {summarize_text(block, 550)}" for child, block in child_blocks
    )
    prompt = f"""Write a parent-level synthesis bridge for a recursive literature review node.

Topic: {topic}
Node title: {node.get('title')}
Partition facet: {node.get('facet')}
Ordering relation: {node.get('relation')}
Node rationale: {node.get('rationale', '')}

Write 180-260 words that explain why the child sections belong together and how they should be read in sequence.
Do not add citations unless already present in the child summaries. Do not invent claims.

CHILD SECTION SUMMARIES:
{child_summaries}
"""
    bridge = client.chat(prompt, max_tokens=900, temperature=0.25, seed=seed).strip()
    trace["merge_events"].append({"node_title": node.get("title"), "n_children": len(child_blocks), "bridge_words": word_count(bridge)})
    rendered = [f"{bridge.strip()}"]
    for child, block in child_blocks:
        heading_level = min(5, 2 + int(child.get("depth", 1)))
        rendered.append(f"{'#' * heading_level} {child.get('title')}\n\n{block.strip()}")
    return "\n\n".join(rendered)


def render_structural_tree(topic: str, node: dict[str, Any], evidence_by_id: dict[str, dict], leaf_texts: dict[str, str], client: OFOXClient, trace: dict, seed: int) -> str:
    if not node.get("children"):
        return leaf_texts[node["title"]]
    child_blocks = []
    for idx, child in enumerate(node["children"]):
        block = render_structural_tree(topic, child, evidence_by_id, leaf_texts, client, trace, seed + idx * 131)
        child_blocks.append((child, block))
    return merge_node(topic, node, child_blocks, client, trace, seed)


def write_global_frame(topic: str, tree: dict[str, Any], body: str, client: OFOXClient, trace: dict, seed: int) -> str:
    leaves = collect_leaves(tree)
    leaf_titles = "\n".join(f"- {leaf['title']} ({len(leaf.get('doc_ids', []))} docs)" for leaf in leaves)
    prompt = f"""Write an introduction and conclusion for a long recursive literature review.

Topic: {topic}
Root facet: {tree.get('facet')}
Root ordering relation: {tree.get('relation')}

Leaf structure:
{leaf_titles}

Return ONLY JSON with:
{{"introduction": "350-500 words", "conclusion": "250-400 words"}}
Do not invent citations. You may cite only if a citation appears in the structure summaries, but citations are not required.
"""
    value = call_json(client, prompt, max_tokens=2200, seed=seed)
    intro = str(value.get("introduction") or "").strip()
    conclusion = str(value.get("conclusion") or "").strip()
    trace["merge_events"].append({"node_title": "GLOBAL", "n_children": len(tree.get("children", [])), "intro_words": word_count(intro), "conclusion_words": word_count(conclusion)})
    return f"## Introduction\n\n{intro}\n\n{body.strip()}\n\n## Conclusion\n\n{conclusion}"


def render_structural_recursive(topic: str, evidence: list[dict], client: OFOXClient, target_words: int, d_max: int, trace: dict, seed: int) -> tuple[str, dict[str, Any]]:
    cards = evidence[:32]
    leaf_size = 4
    tree = organize_node(topic, topic, cards, depth=0, d_max=d_max, leaf_size=leaf_size, client=client, trace=trace, seed=seed)
    leaves = collect_leaves(tree)
    evidence_by_id = {card["doc_id"]: card for card in cards}
    per_leaf = max(650, int(target_words * 0.72 / max(1, len(leaves))))
    leaf_texts = {}
    for idx, leaf in enumerate(leaves):
        leaf_cards = cards_for_ids(evidence_by_id, leaf["doc_ids"])
        leaf_texts[leaf["title"]] = write_structural_leaf(topic, leaf, leaf_cards, per_leaf, client, trace, seed + 5000 + idx * 37)
    body = render_structural_tree(topic, tree, evidence_by_id, leaf_texts, client, trace, seed + 9000)
    text = write_global_frame(topic, tree, body, client, trace, seed + 12000)
    return text, tree


def run_case(topic: str, arm: str, evidence: list[dict], bundle: dict, out_dir: Path, client: OFOXClient) -> dict[str, Any]:
    tier_cfg = dict(DEFAULT_TIERS["long"])
    started = time.time()
    before = usage(client)
    trace = {"topic": topic, "tier": "long", "arm": arm, "descend_events": [], "o_calls": [], "leaf_writes": [], "merge_events": []}
    result: dict[str, Any] = {
        "topic": topic,
        "tier": "long",
        "arm": arm,
        "ok": False,
        "target_words": tier_cfg["target_words"],
        "d_max": tier_cfg["d_max"],
        "model_g": client.model,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tree = None
    try:
        evidence_by_id = {row["doc_id"]: row for row in evidence}
        seed = int(sha_json([topic, arm])[:8], 16) % 1_000_000
        groups = []
        if arm == "structural_lotcf_recursive":
            text, tree = render_structural_recursive(topic, evidence, client, tier_cfg["target_words"], tier_cfg["d_max"], trace, seed)
        else:
            from s5a_length_runner import make_groups

            groups = make_groups(bundle, evidence, tier_cfg)
            if arm == "naive_recursive_chunking":
                text = render_recursive(topic, groups, evidence_by_id, client, tier_cfg["target_words"], tier_cfg["d_max"], trace, seed)
            elif arm == "flat_single_pass":
                text = render_flat(topic, groups, evidence_by_id, client, tier_cfg["target_words"], trace, seed)
            else:
                raise ValueError(f"unknown arm: {arm}")
        text = expand_if_short(topic, text, evidence, tier_cfg["target_words"], tier_cfg["min_words"], client, seed + 999)
        valid_ids = {card["doc_id"] for card in evidence}
        cites = citation_ids(text)
        invalid = [doc_id for doc_id in cites if doc_id not in valid_ids]
        result.update({
            "ok": True,
            "text": text,
            "word_count": word_count(text),
            "refs": [doc_id for doc_id in cites if doc_id in valid_ids],
            "n_refs": len([doc_id for doc_id in cites if doc_id in valid_ids]),
            "invalid_citations": invalid,
            "n_invalid_citations": len(invalid),
            "n_o_calls": len(trace["o_calls"]),
            "n_descend_events": len(trace["descend_events"]),
            "n_leaf_writes": len(trace["leaf_writes"]),
            "n_merge_events": len(trace["merge_events"]),
            "evidence_pool_ids": [card["doc_id"] for card in evidence],
            "evidence_pool_hash": sha_json([card["doc_id"] for card in evidence]),
        })
        if groups:
            result["groups"] = [{"name": group["name"], "n": len(group["doc_ids"])} for group in groups]
        if tree:
            result["tree"] = tree
    except Exception as exc:  # noqa: BLE001 - write failed case for diagnosis
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["elapsed_sec"] = round(time.time() - started, 1)
    result["llm_usage_delta"] = usage_delta(before, usage(client))
    write_result(out_dir, topic, arm, result, evidence, trace)
    return {key: value for key, value in result.items() if key not in {"text", "tree"}}


def write_result(out_dir: Path, topic: str, arm: str, result: dict[str, Any], evidence: list[dict], trace: dict) -> None:
    tdir = out_dir / safe_name(topic) / arm
    tdir.mkdir(parents=True, exist_ok=True)
    if result.get("ok"):
        (tdir / "survey.md").write_text(f"# {topic} — {arm}\n\n{result['text']}", encoding="utf-8")
        (tdir / "evidence_package.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        (tdir / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        if "tree" in result:
            (tdir / "lotcf_tree.json").write_text(json.dumps(result["tree"], ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {key: value for key, value in result.items() if key not in {"text", "tree"}}
    (tdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    pkg_root = here.parents[2]
    exp3_data = pkg_root / "data/exp3_interface_substitution"
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="Retrieval-Augmented Generation for Large Language Models")
    parser.add_argument("--topics", default="", help="Comma-separated topics. Overrides --topic when provided.")
    parser.add_argument("--source-roots", default=f"{exp3_data}/s4a_main8_20260711_173157,{exp3_data}/s4a_matched_main8_20260713_1548,{exp3_data}/s4b_qwen4_20260713_1745")
    parser.add_argument("--out-dir", type=Path, default=pkg_root / "work/results/s5/s5a_structural_qual_20260714")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api_key = __import__("os").environ.get("OFOX_API_KEY", "")
    if not api_key:
        raise SystemExit("OFOX_API_KEY missing")
    source_roots = [Path(part.strip()) for part in args.source_roots.split(",") if part.strip()]
    topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()] or [args.topic]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    bad = [arm for arm in arms if arm not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms: {bad}; allowed={ARMS}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = OFOXClient(api_key, args.model, verbose=args.verbose)
    rows = []
    for topic in topics:
        bundle, evidence = load_topic_assets(topic, source_roots)
        for arm in arms:
            print(f"[S5a-structural] RUN {topic} | {arm}", flush=True)
            row = run_case(topic, arm, evidence, bundle, args.out_dir, client)
            rows.append(row)
            print(f"[S5a-structural] {topic} | {arm} ok={row.get('ok')} words={row.get('word_count')} refs={row.get('n_refs')} invalid={row.get('n_invalid_citations')} o_calls={row.get('n_o_calls')} merges={row.get('n_merge_events')}", flush=True)
    summary = {"topics": topics, "model": args.model, "arms": arms, "n_ok": sum(1 for row in rows if row.get("ok")), "n_total": len(rows), "llm_usage": usage(client), "rows": rows}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S5a-structural] summary n_ok={summary['n_ok']}/{summary['n_total']} usage={summary['llm_usage']}", flush=True)


if __name__ == "__main__":
    main()
