#!/usr/bin/env python3
"""Matched S4a confirmatory rerun.

This runner freezes each topic's root candidate order and semantic grouping
once, then applies interface-compatible substitutes over that same scaffold.
It deliberately avoids the main8 confound where each arm re-solved retrieval,
organization depth and writing length independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
import traceback
from pathlib import Path

from s4a_runner import (
    LLMClient,
    PKG_ROOT,
    LearnedHybridLIndex,
    LazyCorpusById,
    SLRGPState,
    citation_ids,
    descend_s4,
    evidence_cards,
    get_s2d_model,
    is_valid,
    merge_s4,
    op_C,
    op_E,
    op_F,
    op_O,
    op_P,
    op_V,
    op_W_leaf_s3,
    ranker_s2d,
    safe_name,
    solve_for_mode_s4a,
    word_count,
    _usage,
    _usage_delta,
)
from slrgp.state import Paper


ARMS = (
    "intact",
    "o_rank_slab_matched",
    "flat_no_reentry",
    "v_guarded_stress",
    "v_unguarded_stress",
)


def sha_json(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def papers_from_ids(index, doc_ids: list[str]):
    by_id = index._fetch_papers(doc_ids)
    return [by_id[d] for d in doc_ids if d in by_id]


class SQLiteMetadataIndex:
    """Metadata-only paper fetcher for bundle-source reruns.

    S4b Qwen robustness reuses frozen root bundles, so it does not need BM25,
    dense retrieval, S2b embeddings, or the S2d ranker. Avoiding those loads
    keeps GPU memory exclusively for vLLM.
    """

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def _fetch_papers(self, doc_ids: list[str]) -> dict[str, Paper]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self.conn.execute(
            f"SELECT doc_id, title, abstract, authors, year, tier, venue FROM papers WHERE doc_id IN ({placeholders})",
            doc_ids,
        )
        out = {}
        for row in rows:
            try:
                authors = json.loads(row["authors"] or "[]")
            except Exception:
                authors = []
            out[row["doc_id"]] = Paper(
                doc_id=row["doc_id"],
                title=row["title"] or "",
                abstract=row["abstract"] or "",
                authors=authors,
                year=row["year"],
                tier=row["tier"] or "UNRANKED",
                venue=row["venue"] or "",
            )
        return out


def freeze_bundle(topic: str, args, index, llm) -> dict:
    trace = {"solve_events": [], "backtrack_events": []}
    state = SLRGPState(
        q={"seed": topic, "terms": []},
        D=[],
        Gamma={},
        kappa={},
        x={},
        meta={"top_n": args.top_n, "min_abs_len": args.min_abs_len, "arm": "freeze"},
    )
    state = op_E(state, llm)
    terms = [state.q["seed"]] + state.q.get("terms", [])
    query_text = " ".join(terms)
    state.D = index.search(query_text, top_k_each=3000, top_n=args.top_n)
    retrieved_ids = [p.doc_id for p in state.D]
    state = op_F(state)
    filtered_ids = [p.doc_id for p in state.D]
    state = ranker_s2d(state, index)
    ranked_ids = [p.doc_id for p in state.D]
    state = op_O(state, llm)
    state = op_V(state)
    groups_all = list(state.Gamma.get("groups", []))
    groups = groups_all[: args.root_groups]
    if len(groups) < 2:
        # Deterministic fallback keeps the topic usable while recording it.
        sizes = [args.group_size] * args.root_groups
        groups = rank_slab_groups(ranked_ids, sizes, prefix="Fallback semantic slab")
    else:
        fixed = []
        used = set()
        cursor = 0
        for g in groups:
            doc_ids = [d for d in g.get("doc_ids", []) if d in ranked_ids and d not in used][: args.group_size]
            used.update(doc_ids)
            while len(doc_ids) < args.group_size and cursor < len(ranked_ids):
                d = ranked_ids[cursor]
                cursor += 1
                if d not in used:
                    doc_ids.append(d)
                    used.add(d)
            fixed.append({"name": g.get("name", "Semantic group"), "doc_ids": doc_ids})
        groups = fixed
    bundle = {
        "topic": topic,
        "expanded_terms": state.q.get("terms", []),
        "query": query_text,
        "retrieved_ids": retrieved_ids,
        "filtered_ids": filtered_ids,
        "ranked_ids": ranked_ids,
        "root_doc_ids": [d for g in groups for d in g["doc_ids"]],
        "semantic_groups": groups,
        "group_size_vector": [len(g["doc_ids"]) for g in groups],
        "gamma_dimension": state.Gamma.get("dimension"),
        "kappa": state.kappa,
        "freeze_valid": is_valid(state),
        "freeze_trace": trace,
        "controls": {
            "top_n": args.top_n,
            "root_groups": args.root_groups,
            "group_size": args.group_size,
            "total_words": args.total_words,
            "d_max": args.d_max,
            "theta_leaf": args.theta_leaf,
            "model": args.model,
            "temperature": 0.35,
        },
    }
    bundle["bundle_hash"] = sha_json({k: v for k, v in bundle.items() if k != "bundle_hash"})
    return bundle


def load_bundle_from_main8(topic: str, source_root: Path, args) -> dict:
    source_dir = source_root / safe_name(topic) / "intact"
    meta = json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))
    trace = json.loads((source_dir / "trace.json").read_text(encoding="utf-8"))
    root_event = next((e for e in trace.get("solve_events", []) if e.get("node_path") == "root"), {})
    ranked_ids = list(root_event.get("ranked_ids") or meta.get("leaf_evidence_ids") or [])
    descend_groups = [
        {"name": e.get("group_name", "Semantic group"), "doc_ids": e.get("seed_doc_ids", [])}
        for e in trace.get("descend_events", [])
        if str(e.get("node_path", "")).count("/") == 1
    ]
    groups_raw = descend_groups or list(meta.get("groups") or [])
    groups = []
    used = set()
    cursor = 0
    for group in groups_raw[: args.root_groups]:
        doc_ids = []
        for doc_id in ranked_ids:
            if doc_id in used:
                continue
            # Prefer IDs already in the semantic group; top up deterministically if needed.
            if doc_id in set(group.get("doc_ids", [])):
                doc_ids.append(doc_id)
                used.add(doc_id)
            if len(doc_ids) >= args.group_size:
                break
        while len(doc_ids) < args.group_size and cursor < len(ranked_ids):
            doc_id = ranked_ids[cursor]
            cursor += 1
            if doc_id not in used:
                doc_ids.append(doc_id)
                used.add(doc_id)
        groups.append({"name": group.get("name", "Semantic group"), "doc_ids": doc_ids})
    if len(groups) < args.root_groups:
        sizes = [args.group_size] * args.root_groups
        groups = rank_slab_groups(ranked_ids, sizes, prefix="Fallback semantic slab")
    bundle = {
        "topic": topic,
        "expanded_terms": meta.get("expanded_terms", []),
        "query": root_event.get("query", topic),
        "retrieved_ids": root_event.get("retrieved_ids", []),
        "filtered_ids": root_event.get("filtered_ids", []),
        "ranked_ids": ranked_ids,
        "root_doc_ids": [d for g in groups for d in g["doc_ids"]],
        "semantic_groups": groups,
        "group_size_vector": [len(g["doc_ids"]) for g in groups],
        "gamma_dimension": meta.get("gamma_dimension"),
        "kappa": meta.get("kappa", {}),
        "freeze_valid": True,
        "freeze_source": str(source_dir),
        "controls": {
            "top_n": args.top_n,
            "root_groups": args.root_groups,
            "group_size": args.group_size,
            "total_words": args.total_words,
            "d_max": args.d_max,
            "theta_leaf": args.theta_leaf,
            "model": args.model,
            "temperature": 0.35,
        },
    }
    bundle["bundle_hash"] = sha_json({k: v for k, v in bundle.items() if k != "bundle_hash"})
    return bundle


def rank_slab_groups(ranked_ids: list[str], sizes: list[int], prefix: str = "Rank slab") -> list[dict]:
    out = []
    cursor = 0
    for idx, size in enumerate(sizes, 1):
        chunk = ranked_ids[cursor: cursor + size]
        cursor += size
        lo = cursor - len(chunk) + 1
        hi = cursor
        out.append({"name": f"{prefix} {idx}: positions {lo}-{hi}", "doc_ids": chunk})
    return out


def invalid_stress_group(bundle: dict) -> list[dict]:
    return [{"name": "Injected invalid unbalanced organization", "doc_ids": list(bundle["root_doc_ids"])}]


def state_for_groups(topic: str, terms: list[str], groups: list[dict], index, meta: dict | None = None) -> SLRGPState:
    doc_ids = [d for g in groups for d in g["doc_ids"]]
    return SLRGPState(
        q={"seed": topic, "terms": terms},
        D=papers_from_ids(index, doc_ids),
        Gamma={"dimension": "frozen_scaffold", "groups": groups},
        kappa={},
        x={},
        meta=meta or {},
    )


def render_direct(topic: str, terms: list[str], groups: list[dict], index, llm, target_leaf_words: int, trace: dict) -> str:
    corpus_by_id = LazyCorpusById(index)
    section_texts = []
    for idx, group in enumerate(groups):
        child_path = f"root/{idx}"
        state = state_for_groups(f"{topic} — {group['name']}", terms, [group], index, {"mode": "direct"})
        state = op_P(state, corpus_by_id)
        state = op_W_leaf_s3(state, llm, target_words=target_leaf_words, trace=trace, node_path=child_path)
        section_texts.append((group["name"], state.x["text"]))
    return merge_s4(section_texts, depth=0)


def render_recursive(topic: str, terms: list[str], groups: list[dict], index, llm, target_leaf_words: int, args, trace: dict) -> str:
    corpus_by_id = LazyCorpusById(index)
    parent = state_for_groups(topic, terms, groups, index, {"mode": "root", "top_n": args.top_n, "min_abs_len": args.min_abs_len})
    section_texts = []
    for idx, group in enumerate(groups):
        child_path = f"root/{idx}"
        sub = descend_s4(parent, group, corpus_by_id, min_seed=args.min_seed, trace=trace, node_path=child_path)
        sub = solve_for_mode_s4a(sub, llm, index, "intact", trace=trace, node_path=child_path)
        # One child solve is the recursive treatment; writing then uses the solved child context.
        sub = op_P(sub, corpus_by_id)
        sub = op_W_leaf_s3(sub, llm, target_words=target_leaf_words, trace=trace, node_path=child_path)
        section_texts.append((group["name"], sub.x["text"]))
    return merge_s4(section_texts, depth=0)


def run_arm(topic: str, arm: str, bundle: dict, args, index, llm) -> dict:
    t0 = time.time()
    u0 = _usage(llm)
    trace = {
        "arm": arm,
        "bundle_hash": bundle["bundle_hash"],
        "descend_events": [],
        "solve_events": [],
        "backtrack_events": [],
        "leaf_writes": [],
        "validation_events": [],
    }
    result = {
        "topic": topic,
        "arm": arm,
        "ok": False,
        "model_g": args.model,
        "bundle_hash": bundle["bundle_hash"],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_total_words": args.total_words,
    }
    try:
        sizes = list(bundle["group_size_vector"])
        per_leaf = max(150, int(args.total_words / max(1, len(sizes))))
        terms = list(bundle.get("expanded_terms", []))
        if arm == "intact":
            groups = list(bundle["semantic_groups"])
            text = render_recursive(topic, terms, groups, index, llm, per_leaf, args, trace)
        elif arm == "o_rank_slab_matched":
            groups = rank_slab_groups(list(bundle["ranked_ids"]), sizes)
            text = render_recursive(topic, terms, groups, index, llm, per_leaf, args, trace)
        elif arm == "flat_no_reentry":
            groups = list(bundle["semantic_groups"])
            text = render_direct(topic, terms, groups, index, llm, per_leaf, trace)
        elif arm in {"v_guarded_stress", "v_unguarded_stress"}:
            invalid = invalid_stress_group(bundle)
            stress_state = state_for_groups(topic, terms, invalid, index, {"mode": "v_stress"})
            stress_state = op_V(stress_state)
            trace["validation_events"].append({
                "stage": "injected_stress",
                "kappa": stress_state.kappa,
                "valid": is_valid(stress_state),
            })
            if arm == "v_guarded_stress":
                groups = list(bundle["semantic_groups"])
                trace["validation_events"].append({"stage": "guarded_repair", "action": "restore_frozen_semantic_groups"})
            else:
                groups = invalid
                trace["validation_events"].append({"stage": "unguarded_continue", "action": "write_invalid_grouping"})
            text = render_direct(topic, terms, groups, index, llm, per_leaf if arm == "v_guarded_stress" else args.total_words, trace)
        else:
            raise ValueError(f"unknown arm: {arm}")

        final_state = state_for_groups(topic, terms, groups, index, {"mode": arm})
        final_state.x["text"] = text
        final_state = op_C(final_state, LazyCorpusById(index), min_keep=5)
        result.update({
            "groups": [{"name": g["name"], "n": len(g["doc_ids"])} for g in groups],
            "group_size_vector": [len(g["doc_ids"]) for g in groups],
            "refs": final_state.x.get("refs", []),
            "n_refs": len(final_state.x.get("refs", [])),
            "word_count": word_count(text),
            "text": text,
            "evidence_cards": evidence_cards(final_state, LazyCorpusById(index), n=args.evidence_cards),
            "trace": trace,
            "written_citation_ids": citation_ids(text),
            "leaf_evidence_ids": sorted({doc_id for leaf in trace["leaf_writes"] for doc_id in leaf["candidate_ids"]}),
            "primary_length_eligible": args.min_words <= word_count(text) <= args.max_words,
            "ok": True,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["llm_usage_delta"] = _usage_delta(u0, _usage(llm))
    return result


def write_result(out_dir: Path, topic: str, arm: str, res: dict, index) -> None:
    tdir = out_dir / safe_name(topic) / arm
    tdir.mkdir(parents=True, exist_ok=True)
    if res.get("ok"):
        (tdir / "survey.md").write_text(f"# {topic}\n\n{res['text']}", encoding="utf-8")
        corpus_by_id = LazyCorpusById(index)
        refs = {}
        for doc_id in res.get("refs", []):
            paper = corpus_by_id.get(doc_id)
            if paper:
                refs[doc_id] = {"doc_id": doc_id, "title": paper.title, "year": paper.year, "venue": paper.venue}
        (tdir / "ref.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
        (tdir / "evidence_package.json").write_text(json.dumps(res["evidence_cards"], ensure_ascii=False, indent=2), encoding="utf-8")
        leaf = {}
        for doc_id in res.get("leaf_evidence_ids", []):
            paper = corpus_by_id.get(doc_id)
            if paper:
                leaf[doc_id] = {"doc_id": doc_id, "title": paper.title, "year": paper.year, "venue": paper.venue, "abstract": (paper.abstract or "")[:1200]}
        (tdir / "leaf_evidence_provenance.json").write_text(json.dumps(leaf, ensure_ascii=False, indent=2), encoding="utf-8")
        (tdir / "trace.json").write_text(json.dumps(res["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {k: v for k, v in res.items() if k not in {"text", "evidence_cards", "trace"}}
    (tdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-file", default=str(Path(__file__).resolve().parent / "topics_main8.txt"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default=str(PKG_ROOT / "data/exp3_interface_substitution/s4a_matched_20260713"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--min-abs-len", type=int, default=100)
    parser.add_argument("--root-groups", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--total-words", type=int, default=1000)
    parser.add_argument("--min-words", type=int, default=850)
    parser.add_argument("--max-words", type=int, default=1150)
    parser.add_argument("--d-max", type=int, default=1)
    parser.add_argument("--theta-leaf", type=int, default=8)
    parser.add_argument("--min-seed", type=int, default=4)
    parser.add_argument("--evidence-cards", type=int, default=60)
    parser.add_argument("--resume-skip-ok", action="store_true")
    parser.add_argument("--bundle-source-root", type=Path, default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--corpus-db", type=Path, default=Path("data/common/unified_corpus/unified_corpus.db"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = [line.strip() for line in Path(args.topics_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        topics = topics[: args.limit]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    bad = [arm for arm in arms if arm not in ARMS]
    if bad:
        raise SystemExit(f"Unknown arms: {bad}; allowed={ARMS}")

    print(f"[S4a matched] topics={topics}", flush=True)
    print(f"[S4a matched] arms={arms}", flush=True)
    if args.metadata_only:
        if not args.bundle_source_root:
            raise SystemExit("--metadata-only requires --bundle-source-root")
        learned = SQLiteMetadataIndex(args.corpus_db)
    else:
        learned = LearnedHybridLIndex(device="cuda")
        get_s2d_model()
    llm = LLMClient(base_url=args.base_url, model=args.model, api_key="dummy", send_thinking_kwarg=False, default_timeout=240, verbose=True)
    all_meta = []
    for topic in topics:
        bundle_path = out_dir / safe_name(topic) / "frozen_bundle.json"
        if bundle_path.exists():
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        else:
            print(f"[S4a matched] freeze bundle: {topic}", flush=True)
            if args.bundle_source_root:
                bundle = load_bundle_from_main8(topic, args.bundle_source_root, args)
            else:
                bundle = freeze_bundle(topic, args, learned, llm)
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        for arm in arms:
            meta_path = out_dir / safe_name(topic) / arm / "meta.json"
            if args.resume_skip_ok and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("ok"):
                    print(f"[S4a matched] SKIP {topic} | {arm}", flush=True)
                    all_meta.append(meta)
                    continue
            print(f"[S4a matched] RUN {topic} | {arm}", flush=True)
            res = run_arm(topic, arm, bundle, args, learned, llm)
            write_result(out_dir, topic, arm, res, learned)
            meta = {k: v for k, v in res.items() if k not in {"text", "evidence_cards", "trace"}}
            all_meta.append(meta)
            print(f"[S4a matched] {topic} | {arm} ok={res.get('ok')} words={res.get('word_count')} refs={res.get('n_refs')} usage={res.get('llm_usage_delta')}", flush=True)

    summary = {
        "runner": "s4a_matched_runner.py",
        "model": args.model,
        "arms": arms,
        "topics": topics,
        "controls": {
            "top_n": args.top_n,
            "root_groups": args.root_groups,
            "group_size": args.group_size,
            "total_words": args.total_words,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "d_max": args.d_max,
            "theta_leaf": args.theta_leaf,
            "min_seed": args.min_seed,
        },
        "n_ok": sum(1 for row in all_meta if row.get("ok")),
        "n_total": len(all_meta),
        "llm_usage": _usage(llm),
        "rows": all_meta,
    }
    (out_dir / "_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S4a matched] summary n_ok={summary['n_ok']}/{summary['n_total']} usage={summary['llm_usage']}", flush=True)


if __name__ == "__main__":
    main()
