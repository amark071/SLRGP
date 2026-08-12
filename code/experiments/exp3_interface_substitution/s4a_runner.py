#!/usr/bin/env python3
"""S4 runner gate for provenance-safe SLRGP full deployment.

Runner-gate frontend: E_default -> L_bounded learned hybrid -> F_default ->
R_learned -> O_learned/reference-compatible -> shared V/P/W/C.

It uses the unified corpus and the S2b fine-tuned hybrid L index; trained
artifact locations follow models/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG_ROOT / "code"))

from slrgp.llm_client import LLMClient
from slrgp.operators import op_C, op_E, op_F, op_O, op_P, op_V, is_valid
from slrgp.pipeline_real import LazyCorpusById, backtrack_real
import lightgbm as lgb
from slrgp.retrieval import UnifiedIndex
from slrgp.state import SLRGPState

ROOT = PKG_ROOT
UNIFIED_DIR = ROOT / "data/common/unified_corpus"
S2B_EMB_DIR = ROOT / "models/bge_small_L_ft_corpus_emb"
FT_MODEL_PATH = ROOT / "models/bge_small_L_ft"
S2D_R_MODEL = ROOT / "models/s2d_tight/lgb_main.txt"
QPREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60


class LearnedHybridLIndex:
    """Unified corpus index with S2b hybrid_ft retrieval.

    BM25 side reuses UnifiedIndex; dense side uses the S2b fine-tuned BGE model
    and its precomputed corpus embeddings. The returned .search() interface
    matches UnifiedIndex and stores scores for the S2d ranker.
    """

    def __init__(self, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        print("[S3-L] loading base UnifiedIndex for BM25/fetch ...", flush=True)
        self.base = UnifiedIndex(device="cpu")
        self.conn = self.base.conn
        print("[S3-L] loading S2b fine-tuned embeddings ...", flush=True)
        self.ft_docids = pickle.load(open(S2B_EMB_DIR / "ft_docids.pkl", "rb"))
        self.ft_emb = np.load(S2B_EMB_DIR / "ft_emb.npy", mmap_mode="r")
        print(f"[S3-L] ft_emb={self.ft_emb.shape}", flush=True)
        print(f"[S3-L] loading fine-tuned query encoder on {device} ...", flush=True)
        self.ft_model = SentenceTransformer(str(FT_MODEL_PATH), device=device)
        self.ft_model.max_seq_length = 256
        self._last_search_scores = {}

    def _fetch_papers(self, doc_ids):
        return self.base._fetch_papers(doc_ids)

    def citation_count(self, doc_id: str) -> int:
        row = self.conn.execute("SELECT cited_by_count FROM papers WHERE doc_id=?", (doc_id,)).fetchone()
        return int(row[0] or 0) if row else 0

    def _dense_ft_search(self, query_text: str, k: int):
        qvec = self.ft_model.encode([QPREFIX + query_text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        scores = self.ft_emb @ qvec
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            idx = idx[np.argsort(-scores[idx])]
        return [(self.ft_docids[i], float(scores[i])) for i in idx]

    def search(self, query_text: str, top_k_each: int = 3000, top_n: int = 300):
        bm25_hits = self.base._bm25_search(query_text, top_k_each)
        dense_hits = self._dense_ft_search(query_text, top_k_each)
        bm25_rank = {d: r for r, (d, _) in enumerate(bm25_hits)}
        bm25_score = {d: s for d, s in bm25_hits}
        dense_rank = {d: r for r, (d, _) in enumerate(dense_hits)}
        dense_score = {d: s for d, s in dense_hits}
        rrf_score = {}
        for d, r in bm25_rank.items():
            rrf_score[d] = rrf_score.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
        for d, r in dense_rank.items():
            rrf_score[d] = rrf_score.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
        ranked_ids = sorted(rrf_score.keys(), key=lambda d: -rrf_score[d])[:top_n]
        self._last_search_scores = {
            d: {
                "bm25_score": bm25_score.get(d, 0.0),
                "bm25_rank": bm25_rank.get(d, -1),
                "dense_score": dense_score.get(d, 0.0),
                "dense_rank": dense_rank.get(d, -1),
                "rrf_score": rrf_score.get(d, 0.0),
            }
            for d in ranked_ids
        }
        papers_by_id = self._fetch_papers(ranked_ids)
        out = []
        for d in ranked_ids:
            p = papers_by_id.get(d)
            if p is not None:
                p.score = rrf_score.get(d, 0.0)
                out.append(p)
        return out


S2D_FEATURES = [
    "bm25_score", "bm25_rank_recip", "dense_score", "dense_rank_recip", "rrf_score",
    "candidate_year", "year_diff", "year_missing", "tier_ord", "same_discipline",
    "abstract_len_words", "title_len_words", "is_self_citation", "self_citation_reliable", "n_authors",
]
_TIER_ORD = {"T1": 4, "T2": 3, "T3": 2, "PREPRINT": 1, "UNRANKED": 0}
_R_MODEL = None


def get_s2d_model():
    global _R_MODEL
    if _R_MODEL is None:
        _R_MODEL = lgb.Booster(model_file=str(S2D_R_MODEL))
    return _R_MODEL


def _s2d_features_for_paper(p, index):
    scores = getattr(index, "_last_search_scores", {}).get(p.doc_id, {})
    year_diff = (2026 - p.year) if p.year else -1
    bm25_rank = scores.get("bm25_rank", -1)
    dense_rank = scores.get("dense_rank", -1)
    vals = {
        "bm25_score": scores.get("bm25_score", 0.0),
        "bm25_rank_recip": 1.0 / (bm25_rank + 1) if bm25_rank >= 0 else 0.0,
        "dense_score": scores.get("dense_score", 0.0),
        "dense_rank_recip": 1.0 / (dense_rank + 1) if dense_rank >= 0 else 0.0,
        "rrf_score": scores.get("rrf_score", 0.0),
        "candidate_year": p.year if p.year else 0,
        "year_diff": year_diff,
        "year_missing": int(not p.year),
        "tier_ord": _TIER_ORD.get(p.tier or "UNRANKED", 0),
        "same_discipline": 1,  # S3 topic-level generation has no historical review discipline; use deployment convention.
        "abstract_len_words": len((p.abstract or "").split()),
        "title_len_words": len((p.title or "").split()),
        "is_self_citation": 0,
        "self_citation_reliable": 0,
        "n_authors": len(p.authors) if p.authors else 0,
    }
    return [vals[c] for c in S2D_FEATURES]


def ranker_s2d(state, index):
    if not state.D:
        return state
    X = np.array([_s2d_features_for_paper(p, index) for p in state.D], dtype=float)
    scores = get_s2d_model().predict(X)
    for p, sc in zip(state.D, scores):
        p.score = float(sc)
    state.D = [p for _, p in sorted(zip(scores, state.D), key=lambda x: -x[0])]
    return state


def citation_ids(text: str) -> list[str]:
    found = []
    for match in re.finditer(r"\[([^\]]+)\]", text):
        for value in re.split(r"[,;\s]+", match.group(1)):
            value = value.strip().rstrip(".")
            if value.startswith(("arxiv_", "ss_", "oa_")) and value not in found:
                found.append(value)
    return found


def op_W_leaf_s3(state, llm, target_words=320, trace=None, node_path=""):
    prompt = f"""Write a literature review subsection of about {target_words} English words on:
"{state.q['seed']}"

Use ONLY the evidence cards below. Do not invent facts or cite papers not listed.

Rules:
- Write coherent scholarly prose, not bullet points.
- Do not include Markdown headings or section titles; the runner will add headings.
- Compare papers where the evidence supports comparison.
- Cite papers inline using bracket IDs exactly as given, e.g. [2301.03516].
- Do not cite any paper ID not present in the evidence cards.

EVIDENCE CARDS:
{state.x['context']}
"""
    text = llm.chat([{"role": "user", "content": prompt}], max_tokens=max(900, int(target_words * 2.8)), temperature=0.35)
    text = re.sub(r"^#{1,6}\s+.*(?:\n|$)", "", text, flags=re.M).strip()
    state.x["text"] = text
    if trace is not None:
        trace["leaf_writes"].append({
            "node_path": node_path,
            "seed": state.q["seed"],
            "candidate_ids": [p.doc_id for p in state.D],
            "written_citation_ids": citation_ids(state.x["text"]),
            "word_count": word_count(state.x["text"]),
        })
    return state


def solve_real_s3(state, llm, index, t_max=3, trace=None, node_path="root"):
    for i in range(t_max):
        # L_bounded learned hybrid
        terms = [state.q["seed"]] + state.q.get("terms", [])
        query_text = " ".join(terms)
        state.D = index.search(query_text, top_k_each=3000, top_n=state.meta.get("top_n", 300))
        retrieved_ids = [p.doc_id for p in state.D]
        state = op_F(state)  # default/threshold F
        filtered_ids = [p.doc_id for p in state.D]
        state = ranker_s2d(state, index)
        ranked_ids = [p.doc_id for p in state.D]
        state = op_O(state, llm)
        state = op_V(state)
        if trace is not None:
            trace["solve_events"].append({
                "node_path": node_path, "round": i + 1, "mode": "full",
                "query": query_text, "retrieved_ids": retrieved_ids,
                "filtered_ids": filtered_ids, "ranked_ids": ranked_ids,
                "group_names": [g["name"] for g in state.Gamma.get("groups", [])],
                "kappa": state.kappa, "valid": is_valid(state),
            })
        if is_valid(state):
            state.meta["solve_rounds"] = i + 1
            return state
        state = backtrack_real(state, i)
        if trace is not None:
            trace["backtrack_events"].append({"node_path": node_path, "round": i + 1, "mode": "full"})
    state.meta["solve_rounds"] = t_max
    state.meta["solve_gave_up"] = True
    return state


def solve_for_mode_s3(state, llm, index, t_max=3, trace=None, node_path="root"):
    if state.meta.get("mode") == "partial":
        for i in range(t_max):
            state = op_O(state, llm)
            state = op_V(state)
            if trace is not None:
                trace["solve_events"].append({
                    "node_path": node_path, "round": i + 1, "mode": "partial",
                    "query": state.q["seed"], "retrieved_ids": [p.doc_id for p in state.D],
                    "filtered_ids": [p.doc_id for p in state.D], "ranked_ids": [p.doc_id for p in state.D],
                    "group_names": [g["name"] for g in state.Gamma.get("groups", [])],
                    "kappa": state.kappa, "valid": is_valid(state),
                })
            if is_valid(state):
                state.meta["solve_rounds"] = i + 1
                return state
            state.meta["_relaxed"] = state.meta.get("_relaxed", 0) + 1
        return state
    state = op_E(state, llm)
    return solve_real_s3(state, llm, index, t_max=t_max, trace=trace, node_path=node_path)


def descend_s4(parent, group, corpus_by_id, min_seed=4, trace=None, node_path="root"):
    """Provenance-safe descent: child topic keeps the parent topic context.

    The archived S3 full-deployment batch exposed a bug where full re-entry for
    small child groups could lose the parent topic and drift off-topic. This
    gate runner makes the child query explicitly parent-aware while preserving
    the typed partial/full mode interface.
    """
    d_seed = [corpus_by_id.get(d) for d in group.get("doc_ids", []) if corpus_by_id.get(d)]
    parent_seed = parent.q.get("seed", "")
    group_name = group.get("name", "subtopic")
    child_seed = f"{parent_seed} — {group_name}" if parent_seed and group_name not in parent_seed else group_name
    child_terms = list(dict.fromkeys(parent.q.get("terms", []) + [group_name, parent_seed]))
    if trace is not None:
        trace.setdefault("descend_events", []).append({
            "node_path": node_path,
            "parent_seed": parent_seed,
            "group_name": group_name,
            "child_seed": child_seed,
            "mode": "partial" if len(d_seed) >= min_seed else "full",
            "n_seed_docs": len(d_seed),
            "seed_doc_ids": [getattr(x, "doc_id", None) for x in d_seed],
        })
    if len(d_seed) >= min_seed:
        return SLRGPState(q={"seed": child_seed, "terms": child_terms}, D=d_seed, Gamma={}, kappa={}, x={}, meta={"mode": "partial"})
    return SLRGPState(q={"seed": child_seed, "terms": child_terms}, D=[], Gamma={}, kappa={}, x={}, meta={"mode": "full", "top_n": parent.meta.get("top_n", 50), "min_abs_len": parent.meta.get("min_abs_len", 200)})


def merge_s4(section_texts, depth=0):
    hlevel = "#" * min(depth + 3, 6)
    seen = {}
    parts = []
    for name, text in section_texts:
        base = (name or "Section").strip()
        key = base.lower()
        seen[key] = seen.get(key, 0) + 1
        final = base if seen[key] == 1 else f"{base} ({seen[key]})"
        parts.append(f"{hlevel} {final}\n\n{text}")
    return "\n\n".join(parts)


def render_s3(state, index, llm, depth=0, d_max=2, theta_leaf=8, corpus_by_id=None,
              target_leaf_words=320, trace=None, node_path="root", max_groups_per_node=3):
    corpus_by_id = corpus_by_id or LazyCorpusById(index)
    if depth >= d_max or len(state.D) <= theta_leaf or not state.Gamma.get("groups"):
        state = op_P(state, corpus_by_id)
        state = op_W_leaf_s3(state, llm, target_words=target_leaf_words, trace=trace, node_path=node_path)
        return state.x["text"]
    section_texts = []
    groups_all = list(state.Gamma.get("groups", []))
    groups = groups_all[:max_groups_per_node] if max_groups_per_node and max_groups_per_node > 0 else groups_all
    if trace is not None:
        trace.setdefault("group_truncation_events", []).append({
            "node_path": node_path,
            "depth": depth,
            "n_groups_original": len(groups_all),
            "n_groups_used": len(groups),
            "used_group_names": [g.get("name") for g in groups],
        })
    for g in groups:
        child_path = f"{node_path}/{len(section_texts)}"
        sub = descend_s4(state, g, corpus_by_id, trace=trace, node_path=child_path)
        sub = solve_for_mode_s3(sub, llm, index, trace=trace, node_path=child_path)
        sub_text = render_s3(sub, index, llm, depth=depth + 1, d_max=d_max, theta_leaf=theta_leaf,
                             corpus_by_id=corpus_by_id, target_leaf_words=target_leaf_words,
                             trace=trace, node_path=child_path, max_groups_per_node=max_groups_per_node)
        section_texts.append((g["name"], sub_text))
    return merge_s4(section_texts, depth=depth)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:[-']\w+)?\b", text))


def evidence_cards(state, corpus_by_id, n=120):
    out = []
    for p in state.D[:n]:
        out.append({
            "doc_id": p.doc_id,
            "title": p.title,
            "year": p.year,
            "venue": p.venue,
            "tier": p.tier,
            "score": getattr(p, "score", None),
            "abstract": (p.abstract or "")[:1200],
        })
    return out




class DefaultHybridLIndex:
    """Production/default L: base BM25 + base dense RRF with S2d-compatible score traces."""

    def __init__(self, base=None):
        print("[S4a-L-default] using production/default UnifiedIndex RRF", flush=True)
        self.base = base or UnifiedIndex(device="cpu")
        self.conn = self.base.conn
        self._last_search_scores = {}

    def _fetch_papers(self, doc_ids):
        return self.base._fetch_papers(doc_ids)

    def citation_count(self, doc_id: str) -> int:
        row = self.conn.execute("SELECT cited_by_count FROM papers WHERE doc_id=?", (doc_id,)).fetchone()
        return int(row[0] or 0) if row else 0

    def search(self, query_text: str, top_k_each: int = 3000, top_n: int = 300):
        bm25_hits = self.base._bm25_search(query_text, top_k_each)
        dense_hits = self.base._dense_search(query_text, top_k_each)
        bm25_rank = {d: r for r, (d, _) in enumerate(bm25_hits)}
        bm25_score = {d: s for d, s in bm25_hits}
        dense_rank = {d: r for r, (d, _) in enumerate(dense_hits)}
        dense_score = {d: s for d, s in dense_hits}
        rrf_score = {}
        for d, r in bm25_rank.items():
            rrf_score[d] = rrf_score.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
        for d, r in dense_rank.items():
            rrf_score[d] = rrf_score.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
        ranked_ids = sorted(rrf_score.keys(), key=lambda d: -rrf_score[d])[:top_n]
        self._last_search_scores = {
            d: {
                "bm25_score": bm25_score.get(d, 0.0),
                "bm25_rank": bm25_rank.get(d, -1),
                "dense_score": dense_score.get(d, 0.0),
                "dense_rank": dense_rank.get(d, -1),
                "rrf_score": rrf_score.get(d, 0.0),
            }
            for d in ranked_ids
        }
        papers_by_id = self._fetch_papers(ranked_ids)
        out = []
        for d in ranked_ids:
            p = papers_by_id.get(d)
            if p is not None:
                p.score = rrf_score.get(d, 0.0)
                out.append(p)
        return out


def op_O_rank_slabs(state, n_groups=2, top_k=15):
    """Ablate learned/semantic O by preserving rank order and splitting into slabs."""
    papers = state.D[:top_k]
    if not papers:
        state.Gamma = {"dimension": "rank_slab", "groups": []}
        return state
    n_groups = max(1, min(int(n_groups or 2), len(papers)))
    size = int(np.ceil(len(papers) / n_groups))
    groups = []
    for gi in range(n_groups):
        chunk = papers[gi * size:(gi + 1) * size]
        if not chunk:
            continue
        lo = gi * size + 1
        hi = gi * size + len(chunk)
        groups.append({
            "name": f"Rank slab {gi + 1}: positions {lo}-{hi}",
            "doc_ids": [p.doc_id for p in chunk],
        })
    state.Gamma = {"dimension": "rank_slab_order", "groups": groups}
    return state


def apply_O_s4a(state, llm, arm, n_groups=2):
    if arm == "o_rank_slab":
        return op_O_rank_slabs(state, n_groups=n_groups, top_k=15)
    return op_O(state, llm)


def solve_real_s4a(state, llm, index, arm, t_max=3, trace=None, node_path="root"):
    for i in range(t_max):
        terms = [state.q["seed"]] + state.q.get("terms", [])
        query_text = " ".join(terms)
        state.D = index.search(query_text, top_k_each=3000, top_n=state.meta.get("top_n", 300))
        retrieved_ids = [p.doc_id for p in state.D]
        state = op_F(state)
        filtered_ids = [p.doc_id for p in state.D]
        if arm != "r_pass":
            state = ranker_s2d(state, index)
        ranked_ids = [p.doc_id for p in state.D]
        state = apply_O_s4a(state, llm, arm, n_groups=state.meta.get("rank_slab_groups", 2))
        state = op_V(state)
        if trace is not None:
            trace["solve_events"].append({
                "node_path": node_path,
                "round": i + 1,
                "mode": "full",
                "arm": arm,
                "query": query_text,
                "retrieved_ids": retrieved_ids,
                "filtered_ids": filtered_ids,
                "ranked_ids": ranked_ids,
                "group_names": [g["name"] for g in state.Gamma.get("groups", [])],
                "kappa": state.kappa,
                "valid": is_valid(state),
            })
        if is_valid(state):
            state.meta["solve_rounds"] = i + 1
            return state
        state = backtrack_real(state, i)
        if trace is not None:
            trace["backtrack_events"].append({"node_path": node_path, "round": i + 1, "mode": "full", "arm": arm})
    state.meta["solve_rounds"] = t_max
    state.meta["solve_gave_up"] = True
    return state


def solve_for_mode_s4a(state, llm, index, arm, t_max=3, trace=None, node_path="root"):
    if state.meta.get("mode") == "partial":
        for i in range(t_max):
            state = apply_O_s4a(state, llm, arm, n_groups=state.meta.get("rank_slab_groups", 2))
            state = op_V(state)
            if trace is not None:
                trace["solve_events"].append({
                    "node_path": node_path,
                    "round": i + 1,
                    "mode": "partial",
                    "arm": arm,
                    "query": state.q["seed"],
                    "retrieved_ids": [p.doc_id for p in state.D],
                    "filtered_ids": [p.doc_id for p in state.D],
                    "ranked_ids": [p.doc_id for p in state.D],
                    "group_names": [g["name"] for g in state.Gamma.get("groups", [])],
                    "kappa": state.kappa,
                    "valid": is_valid(state),
                })
            if is_valid(state):
                state.meta["solve_rounds"] = i + 1
                return state
            state.meta["_relaxed"] = state.meta.get("_relaxed", 0) + 1
        return state
    state = op_E(state, llm)
    return solve_real_s4a(state, llm, index, arm, t_max=t_max, trace=trace, node_path=node_path)


def render_s4a(state, index, llm, arm, depth=0, d_max=2, theta_leaf=8, corpus_by_id=None,
               target_leaf_words=320, trace=None, node_path="root", max_groups_per_node=3):
    corpus_by_id = corpus_by_id or LazyCorpusById(index)
    if depth >= d_max or len(state.D) <= theta_leaf or not state.Gamma.get("groups"):
        state = op_P(state, corpus_by_id)
        state = op_W_leaf_s3(state, llm, target_words=target_leaf_words, trace=trace, node_path=node_path)
        return state.x["text"]
    section_texts = []
    groups_all = list(state.Gamma.get("groups", []))
    groups = groups_all[:max_groups_per_node] if max_groups_per_node and max_groups_per_node > 0 else groups_all
    if trace is not None:
        trace.setdefault("group_truncation_events", []).append({
            "node_path": node_path,
            "depth": depth,
            "arm": arm,
            "n_groups_original": len(groups_all),
            "n_groups_used": len(groups),
            "used_group_names": [g.get("name") for g in groups],
        })
    for g in groups:
        child_path = f"{node_path}/{len(section_texts)}"
        sub = descend_s4(state, g, corpus_by_id, trace=trace, node_path=child_path)
        sub.meta["rank_slab_groups"] = state.meta.get("rank_slab_groups", 2)
        sub = solve_for_mode_s4a(sub, llm, index, arm, trace=trace, node_path=child_path)
        sub_text = render_s4a(sub, index, llm, arm, depth=depth + 1, d_max=d_max, theta_leaf=theta_leaf,
                              corpus_by_id=corpus_by_id, target_leaf_words=target_leaf_words,
                              trace=trace, node_path=child_path, max_groups_per_node=max_groups_per_node)
        section_texts.append((g["name"], sub_text))
    return merge_s4(section_texts, depth=depth)


def _usage(llm):
    return {"n_calls": llm.n_calls, "prompt_tokens": llm.total_prompt_tokens, "completion_tokens": llm.total_completion_tokens}


def _usage_delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b}


def run_topic_s4a(topic, arm, args, indexes, llm):
    t0 = time.time()
    u0 = _usage(llm)
    index = indexes["default"] if arm == "l_default" else indexes["learned"]
    d_max_eff = min(args.d_max, 1) if arm == "no_recursion" else args.d_max
    result = {
        "topic": topic,
        "arm": arm,
        "ok": False,
        "model_g": args.model,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "d_max_effective": d_max_eff,
    }
    corpus_by_id = LazyCorpusById(index)
    trace = {"solve_events": [], "backtrack_events": [], "descend_events": [], "group_truncation_events": [], "leaf_writes": []}
    try:
        state = SLRGPState(q={"seed": topic, "terms": []}, D=[], Gamma={}, kappa={}, x={},
                           meta={"top_n": args.top_n, "min_abs_len": args.min_abs_len,
                                 "arm": arm, "rank_slab_groups": args.rank_slab_groups})
        state = op_E(state, llm)
        result["expanded_terms"] = state.q.get("terms", [])
        state = solve_real_s4a(state, llm, index, arm, trace=trace)
        result.update({
            "n_candidates": len(state.D),
            "solve_rounds": state.meta.get("solve_rounds"),
            "solve_gave_up": state.meta.get("solve_gave_up", False),
            "gamma_dimension": state.Gamma.get("dimension"),
            "groups": [{"name": g["name"], "n": len(g["doc_ids"])} for g in state.Gamma.get("groups", [])],
            "kappa": state.kappa,
        })
        cards = evidence_cards(state, corpus_by_id, n=args.evidence_cards)
        text = render_s4a(state, index, llm, arm, depth=0, d_max=d_max_eff, theta_leaf=args.theta_leaf,
                          corpus_by_id=corpus_by_id, target_leaf_words=args.target_leaf_words,
                          trace=trace, max_groups_per_node=args.max_groups_per_node)
        state.x["text"] = text
        state = op_C(state, corpus_by_id, min_keep=5)
        result["refs"] = state.x.get("refs", [])
        result["n_refs"] = len(result["refs"])
        result["word_count"] = word_count(text)
        result["text"] = text
        result["evidence_cards"] = cards
        result["trace"] = trace
        result["written_citation_ids"] = citation_ids(text)
        result["leaf_evidence_ids"] = sorted({doc_id for leaf in trace["leaf_writes"] for doc_id in leaf["candidate_ids"]})
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["llm_usage_delta"] = _usage_delta(u0, _usage(llm))
    result["llm_usage_so_far"] = _usage(llm)
    return result




def safe_name(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", topic).strip("_")

def write_result_s4a(out_dir, topic, arm, res, index):
    tdir = Path(out_dir) / safe_name(topic) / arm
    tdir.mkdir(parents=True, exist_ok=True)
    corpus_by_id = LazyCorpusById(index)
    if res.get("ok"):
        (tdir / "survey.md").write_text(f"# {topic}\n\n" + res["text"], encoding="utf-8")
        ref_json = {}
        for doc_id in res.get("refs", []):
            p = corpus_by_id.get(doc_id)
            if p:
                ref_json[doc_id] = {"doc_id": doc_id, "title": p.title, "year": p.year, "venue": p.venue}
        (tdir / "ref.json").write_text(json.dumps(ref_json, ensure_ascii=False, indent=2), encoding="utf-8")
        (tdir / "evidence_package.json").write_text(json.dumps(res["evidence_cards"], ensure_ascii=False, indent=2), encoding="utf-8")
        leaf_evidence = {}
        for doc_id in res.get("leaf_evidence_ids", []):
            p = corpus_by_id.get(doc_id)
            if p:
                leaf_evidence[doc_id] = {"doc_id": doc_id, "title": p.title, "year": p.year,
                                         "venue": p.venue, "abstract": (p.abstract or "")[:1200]}
        (tdir / "leaf_evidence_provenance.json").write_text(json.dumps(leaf_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        (tdir / "trace.json").write_text(json.dumps(res["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {k: v for k, v in res.items() if k not in {"text", "evidence_cards", "trace"}}
    (tdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics-file", default=str(PKG_ROOT / "code/experiments/exp3_interface_substitution/topics_pilot.txt"))
    ap.add_argument("--topic", default="")
    ap.add_argument("--out-dir", default=str(PKG_ROOT / "work/results/s4/s4a_pilot"))
    ap.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="intact,l_default,r_pass,o_rank_slab,no_recursion")
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--min-abs-len", type=int, default=100)
    ap.add_argument("--d-max", type=int, default=2)
    ap.add_argument("--theta-leaf", type=int, default=8)
    ap.add_argument("--target-leaf-words", type=int, default=400)
    ap.add_argument("--evidence-cards", type=int, default=60)
    ap.add_argument("--max-groups-per-node", type=int, default=2)
    ap.add_argument("--rank-slab-groups", type=int, default=2)
    ap.add_argument("--resume-skip-ok", action="store_true", help="Skip topic/arm outputs whose meta.json already has ok=true")
    ap.add_argument("--stop-on-connection-error", action="store_true", default=True, help="Stop batch if relay/tunnel connection is refused")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = [x.strip() for x in Path(args.topics_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.topic:
        topics = [t for t in topics if args.topic.lower() in t.lower()]
    if args.limit:
        topics = topics[: args.limit]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    allowed = {"intact", "l_default", "r_pass", "o_rank_slab", "no_recursion"}
    bad = [a for a in arms if a not in allowed]
    if bad:
        raise SystemExit(f"Unknown arms: {bad}; allowed={sorted(allowed)}")

    print(f"[S4a] topics={topics}")
    print(f"[S4a] arms={arms}")
    print(f"[S4a] model={args.model} top_n={args.top_n} d_max={args.d_max} leaf_words={args.target_leaf_words} max_groups={args.max_groups_per_node}")

    learned = LearnedHybridLIndex(device="cuda")
    indexes = {"learned": learned}
    if "l_default" in arms:
        indexes["default"] = DefaultHybridLIndex(base=learned.base)
    get_s2d_model()
    llm = LLMClient(base_url=args.base_url, model=args.model, api_key="dummy", send_thinking_kwarg=False,
                    default_timeout=240, verbose=True)
    all_meta = []
    for topic in topics:
        for arm in arms:
            existing_meta_path = out_dir / safe_name(topic) / arm / "meta.json"
            if args.resume_skip_ok and existing_meta_path.exists():
                try:
                    existing_meta = json.loads(existing_meta_path.read_text(encoding="utf-8"))
                    if existing_meta.get("ok") is True:
                        print(f"[S4a] SKIP existing OK: {topic} | {arm}", flush=True)
                        all_meta.append(existing_meta)
                        continue
                except Exception:
                    pass
            print("\n" + "=" * 80)
            print(f"[S4a] topic: {topic} | arm: {arm}")
            print("=" * 80, flush=True)
            res = run_topic_s4a(topic, arm, args, indexes, llm)
            idx_for_write = indexes["default"] if arm == "l_default" else indexes["learned"]
            write_result_s4a(out_dir, topic, arm, res, idx_for_write)
            meta = {k: v for k, v in res.items() if k not in {"text", "evidence_cards", "trace"}}
            all_meta.append(meta)
            print(f"[S4a] {topic} | {arm} -> {'OK' if res.get('ok') else 'FAIL'} words={res.get('word_count')} refs={res.get('n_refs')} elapsed={res.get('elapsed_sec')}s usage={res.get('llm_usage_delta')}", flush=True)
            if not res.get("ok"):
                print(res.get("error"), flush=True)
                err = str(res.get("error", "")) + "\n" + str(res.get("traceback", ""))
                if args.stop_on_connection_error and ("Connection refused" in err or "Errno 111" in err or "Payment Required" in err or "HTTP Error 402" in err):
                    print("[S4a] STOP: relay/tunnel connection refused; preserve completed outputs and resume after tunnel is restored.", flush=True)
                    summary = {
                        "runner": "s4a_runner.py",
                        "model": args.model,
                        "base_url": args.base_url,
                        "arms": arms,
                        "top_n": args.top_n,
                        "d_max": args.d_max,
                        "theta_leaf": args.theta_leaf,
                        "target_leaf_words": args.target_leaf_words,
                        "max_groups_per_node": args.max_groups_per_node,
                        "rank_slab_groups": args.rank_slab_groups,
                        "n_ok": sum(1 for x in all_meta if x.get("ok")),
                        "n_total": len(all_meta),
                        "llm_usage": _usage(llm),
                        "topics": all_meta,
                        "stopped_on_connection_error": True,
                    }
                    (out_dir / "_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    raise SystemExit(111)
    summary = {
        "runner": "s4a_runner.py",
        "model": args.model,
        "base_url": args.base_url,
        "arms": arms,
        "top_n": args.top_n,
        "d_max": args.d_max,
        "theta_leaf": args.theta_leaf,
        "target_leaf_words": args.target_leaf_words,
        "max_groups_per_node": args.max_groups_per_node,
        "rank_slab_groups": args.rank_slab_groups,
        "n_ok": sum(1 for x in all_meta if x.get("ok")),
        "n_total": len(all_meta),
        "llm_usage": _usage(llm),
        "topics": all_meta,
    }
    (out_dir / "_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[S4a] summary", json.dumps({"n_ok": summary["n_ok"], "n_total": summary["n_total"], "llm_usage": summary["llm_usage"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
