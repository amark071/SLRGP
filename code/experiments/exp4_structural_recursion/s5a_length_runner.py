#!/usr/bin/env python3
"""S5a length/depth generalization runner.

The runner is intentionally self-contained: it consumes frozen S4 bundles and
downloaded evidence packages, then calls an OpenAI-compatible LLM endpoint.
It does not require the local unified-corpus SQLite database.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


ARMS = ("recursive_slrgp", "fixed_outline_no_reentry", "flat_single_pass")
DEFAULT_TIERS = {
    "short": {"target_words": 1200, "min_words": 950, "max_words": 1550, "d_max": 1, "n_groups": 2, "group_size": 4},
    "medium": {"target_words": 3600, "min_words": 3000, "max_words": 4500, "d_max": 2, "n_groups": 3, "group_size": 6},
    "long": {"target_words": 9000, "min_words": 7600, "max_words": 11250, "d_max": 3, "n_groups": 4, "group_size": 8},
}
API_URL = "https://api.ofox.ai/v1/chat/completions"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def citation_ids(text: str) -> list[str]:
    found = []
    for match in re.finditer(r"\[([^\]]+)\]", text):
        for value in re.split(r"[,;\s]+", match.group(1)):
            value = value.strip().rstrip(".")
            if value.startswith(("arxiv_", "ss_", "oa_")) and value not in found:
                found.append(value)
    return found


def sha_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class OFOXClient:
    def __init__(self, api_key: str, model: str, timeout: int = 360, verbose: bool = False):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.verbose = verbose
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, prompt: str, max_tokens: int, temperature: float = 0.35, seed: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(4):
            req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
            started = time.time()
            try:
                if self.verbose:
                    print(f"    [llm] call #{self.n_calls + 1} chars={len(prompt)} max_tokens={max_tokens} attempt={attempt}", flush=True)
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                usage = body.get("usage", {})
                self.n_calls += 1
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                if self.verbose:
                    print(f"    [llm] done {time.time() - started:.1f}s usage={usage}", flush=True)
                return body["choices"][0]["message"]["content"]
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if self.verbose:
                    print(f"    [llm] error {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(45, 2 ** (attempt + 2)))
        raise RuntimeError(repr(last_error))


def usage(client: OFOXClient) -> dict:
    return {"n_calls": client.n_calls, "prompt_tokens": client.prompt_tokens, "completion_tokens": client.completion_tokens}


def usage_delta(before: dict, after: dict) -> dict:
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_topic_assets(topic: str, source_roots: list[Path]) -> tuple[dict, list[dict]]:
    rel = safe_name(topic)
    bundle = None
    evidence_candidates = []
    for root in source_roots:
        tdir = root / rel
        if bundle is None and (tdir / "frozen_bundle.json").exists():
            bundle = load_json(tdir / "frozen_bundle.json")
        for arm in ("intact", "recursive_slrgp"):
            ep = tdir / arm / "evidence_package.json"
            if ep.exists():
                evidence_candidates.append(load_json(ep))
                break
        if bundle and evidence_candidates and len(evidence_candidates[-1]) >= 32:
            break
    if bundle is None:
        raise FileNotFoundError(f"frozen_bundle.json not found for {topic}")
    if not evidence_candidates:
        raise FileNotFoundError(f"evidence_package.json not found for {topic}")
    evidence = max(evidence_candidates, key=len)
    by_id = {row["doc_id"]: row for row in evidence}
    ranked = [doc_id for doc_id in bundle.get("ranked_ids", []) if doc_id in by_id]
    if not ranked:
        ranked = [row["doc_id"] for row in evidence]
    evidence_sorted = [by_id[doc_id] for doc_id in ranked if doc_id in by_id]
    return bundle, evidence_sorted


def make_groups(bundle: dict, evidence: list[dict], tier_cfg: dict) -> list[dict]:
    n_groups = tier_cfg["n_groups"]
    group_size = tier_cfg["group_size"]
    ranked_ids = [row["doc_id"] for row in evidence]
    semantic = list(bundle.get("semantic_groups") or [])
    groups = []
    used = set()
    cursor = 0
    for idx in range(n_groups):
        name = semantic[idx].get("name", f"Evidence theme {idx + 1}") if idx < len(semantic) else f"Evidence theme {idx + 1}"
        preferred = set(semantic[idx].get("doc_ids", [])) if idx < len(semantic) else set()
        doc_ids = []
        for doc_id in ranked_ids:
            if doc_id in used:
                continue
            if doc_id in preferred:
                doc_ids.append(doc_id)
                used.add(doc_id)
            if len(doc_ids) >= group_size:
                break
        while len(doc_ids) < group_size and cursor < len(ranked_ids):
            doc_id = ranked_ids[cursor]
            cursor += 1
            if doc_id not in used:
                doc_ids.append(doc_id)
                used.add(doc_id)
        groups.append({"name": name, "doc_ids": doc_ids})
    return groups


def card_lines(cards: list[dict], max_abs_chars: int = 900) -> str:
    lines = []
    for card in cards:
        abstract = (card.get("abstract") or "")[:max_abs_chars].replace("\n", " ")
        lines.append(f"- [{card['doc_id']}] {card.get('title','')} ({card.get('year','NA')}, {card.get('venue','')}). {abstract}")
    return "\n".join(lines)


def cards_for_ids(evidence_by_id: dict[str, dict], doc_ids: list[str]) -> list[dict]:
    return [evidence_by_id[doc_id] for doc_id in doc_ids if doc_id in evidence_by_id]


def write_leaf(topic: str, path: str, cards: list[dict], target_words: int, client: OFOXClient, trace: dict, seed: int) -> str:
    if not cards:
        raise ValueError(f"empty evidence cards at {topic} / {path}")
    prompt = f"""Write a literature review subsection of about {target_words} English words.

Topic context: {topic}
Subproblem path: {path}

Use ONLY the evidence cards below. Do not invent papers or cite IDs not listed.
Write coherent scholarly prose, not bullets. Do not include Markdown headings.
Compare papers when the evidence supports comparison. Cite inline with exact bracket IDs, e.g. [arxiv_2308.04079].

EVIDENCE CARDS:
{card_lines(cards)}
"""
    text = client.chat(prompt, max_tokens=max(1200, int(target_words * 2.5)), temperature=0.35, seed=seed).strip()
    text = re.sub(r"^#{1,6}\s+.*(?:\n|$)", "", text, flags=re.M).strip()
    trace["leaf_writes"].append({
        "node_path": path,
        "candidate_ids": [card["doc_id"] for card in cards],
        "written_citation_ids": citation_ids(text),
        "word_count": word_count(text),
    })
    return text


def expand_if_short(topic: str, text: str, evidence: list[dict], target_words: int, min_words: int,
                    client: OFOXClient, seed: int) -> str:
    if word_count(text) >= min_words:
        return text
    allowed = ", ".join(card["doc_id"] for card in evidence[:80])
    current = text
    for attempt in range(3):
        current_words = word_count(current)
        if current_words >= min_words:
            return current
        need = max(700, min_words - current_words + 600)
        prompt = f"""The draft literature review below is too short ({current_words} words). Write additional scholarly paragraphs of about {need} English words to append to it.

Rules:
- Do not invent papers.
- Cite only these allowed IDs: {allowed}
- Keep coherent scholarly prose.
- Add substantive synthesis, comparisons, caveats, and transitions where appropriate.
- Do not repeat the same sentences from the draft.
- Return ONLY the additional paragraphs to append, not a full rewrite.

TOPIC: {topic}

SELECTED EVIDENCE CARDS:
{card_lines(evidence[:40], max_abs_chars=500)}

CURRENT DRAFT EXCERPT:
---
{current[-6000:]}
---
"""
        addition = client.chat(prompt, max_tokens=max(1800, int(need * 2.4)), temperature=0.25, seed=seed + attempt).strip()
        if word_count(addition) > 100:
            current = current.rstrip() + "\n\n" + addition
        else:
            break
    return current


def split_group(group: dict, depth: int) -> list[dict]:
    doc_ids = list(group["doc_ids"])
    if len(doc_ids) <= 4:
        return [group]
    mid = max(2, len(doc_ids) // 2)
    return [
        {"name": f"{group['name']} — evidence cluster A", "doc_ids": doc_ids[:mid]},
        {"name": f"{group['name']} — evidence cluster B", "doc_ids": doc_ids[mid:]},
    ]


def render_recursive(topic: str, groups: list[dict], evidence_by_id: dict[str, dict], client: OFOXClient,
                     target_words: int, d_max: int, trace: dict, seed_base: int) -> str:
    leaves: list[tuple[str, list[dict], str]] = []

    def descend(group: dict, depth: int, path: str) -> None:
        trace["descend_events"].append({"node_path": path, "depth": depth, "group_name": group["name"], "n_docs": len(group["doc_ids"])})
        if depth >= d_max or len(group["doc_ids"]) <= 4:
            leaves.append((group["name"], cards_for_ids(evidence_by_id, group["doc_ids"]), path))
            return
        for idx, child in enumerate(split_group(group, depth)):
            descend(child, depth + 1, f"{path}/{idx}")

    for idx, group in enumerate(groups):
        descend(group, 1, f"root/{idx}")

    per_leaf = max(350, int(target_words / max(1, len(leaves))))
    sections = []
    for idx, (name, cards, path) in enumerate(leaves):
        text = write_leaf(topic, path, cards, per_leaf, client, trace, seed_base + idx)
        sections.append(f"### {name}\n\n{text}")
    return "\n\n".join(sections)


def render_fixed_outline(topic: str, groups: list[dict], evidence_by_id: dict[str, dict], client: OFOXClient,
                         target_words: int, trace: dict, seed_base: int) -> str:
    per_group = max(400, int(target_words / max(1, len(groups))))
    sections = []
    for idx, group in enumerate(groups):
        path = f"root/{idx}"
        cards = cards_for_ids(evidence_by_id, group["doc_ids"])
        text = write_leaf(topic, path, cards, per_group, client, trace, seed_base + idx)
        sections.append(f"### {group['name']}\n\n{text}")
    return "\n\n".join(sections)


def render_flat(topic: str, groups: list[dict], evidence_by_id: dict[str, dict], client: OFOXClient,
                target_words: int, trace: dict, seed: int) -> str:
    doc_ids = [doc_id for group in groups for doc_id in group["doc_ids"]]
    cards = cards_for_ids(evidence_by_id, doc_ids)
    return write_leaf(topic, "root/flat", cards, target_words, client, trace, seed)


def run_case(topic: str, tier: str, tier_cfg: dict, arm: str, bundle: dict, evidence: list[dict], client: OFOXClient) -> dict:
    started = time.time()
    before = usage(client)
    trace = {"topic": topic, "tier": tier, "arm": arm, "descend_events": [], "leaf_writes": []}
    result = {
        "topic": topic,
        "tier": tier,
        "arm": arm,
        "ok": False,
        "target_words": tier_cfg["target_words"],
        "d_max": tier_cfg["d_max"],
        "model_g": client.model,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        groups = make_groups(bundle, evidence, tier_cfg)
        evidence_by_id = {row["doc_id"]: row for row in evidence}
        seed_base = int(hashlib.sha256(f"{topic}|{tier}|{arm}".encode()).hexdigest()[:8], 16) % 1_000_000
        if arm == "recursive_slrgp":
            text = render_recursive(topic, groups, evidence_by_id, client, tier_cfg["target_words"], tier_cfg["d_max"], trace, seed_base)
        elif arm == "fixed_outline_no_reentry":
            text = render_fixed_outline(topic, groups, evidence_by_id, client, tier_cfg["target_words"], trace, seed_base)
        elif arm == "flat_single_pass":
            text = render_flat(topic, groups, evidence_by_id, client, tier_cfg["target_words"], trace, seed_base)
        else:
            raise ValueError(f"unknown arm: {arm}")
        text = expand_if_short(topic, text, evidence, tier_cfg["target_words"], tier_cfg["min_words"], client, seed_base + 999)
        valid_ids = {card["doc_id"] for card in evidence}
        cites = citation_ids(text)
        invalid = [doc_id for doc_id in cites if doc_id not in valid_ids]
        result.update({
            "ok": True,
            "text": text,
            "word_count": word_count(text),
            "groups": [{"name": group["name"], "n": len(group["doc_ids"])} for group in groups],
            "group_size_vector": [len(group["doc_ids"]) for group in groups],
            "refs": [doc_id for doc_id in cites if doc_id in valid_ids],
            "n_refs": len([doc_id for doc_id in cites if doc_id in valid_ids]),
            "invalid_citations": invalid,
            "n_invalid_citations": len(invalid),
            "leaf_evidence_ids": sorted({doc_id for leaf in trace["leaf_writes"] for doc_id in leaf["candidate_ids"]}),
            "evidence_pool_ids": [card["doc_id"] for card in evidence],
            "evidence_pool_hash": sha_json([card["doc_id"] for card in evidence]),
            "trace": trace,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["elapsed_sec"] = round(time.time() - started, 1)
    result["llm_usage_delta"] = usage_delta(before, usage(client))
    return result


def write_result(out_dir: Path, tier: str, topic: str, arm: str, result: dict, evidence: list[dict]) -> None:
    tdir = out_dir / tier / safe_name(topic) / arm
    tdir.mkdir(parents=True, exist_ok=True)
    if result.get("ok"):
        (tdir / "survey.md").write_text(f"# {topic} ({tier})\n\n{result['text']}", encoding="utf-8")
        (tdir / "evidence_package.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        (tdir / "trace.json").write_text(json.dumps(result["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {k: v for k, v in result.items() if k not in {"text", "trace"}}
    (tdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_tier_spec(spec: str) -> dict[str, dict]:
    tiers = {}
    for name in [part.strip() for part in spec.split(",") if part.strip()]:
        if name not in DEFAULT_TIERS:
            raise ValueError(f"unknown tier {name}; allowed={sorted(DEFAULT_TIERS)}")
        tiers[name] = dict(DEFAULT_TIERS[name])
    return tiers


def main() -> None:
    here = Path(__file__).resolve().parent
    pkg_root = here.parents[2]
    exp3_data = pkg_root / "data/exp3_interface_substitution"
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-file", type=Path, default=here / "topics_s5a.txt")
    parser.add_argument("--source-roots", default=f"{exp3_data}/s4a_main8_20260711_173157,{exp3_data}/s4a_matched_main8_20260713_1548,{exp3_data}/s4b_qwen4_20260713_1745")
    parser.add_argument("--out-dir", type=Path, default=pkg_root / "work/results/s5/s5a_length_20260713")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--tiers", default="short,medium,long")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--limit-topics", type=int, default=0)
    parser.add_argument("--resume-skip-ok", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OFOX_API_KEY", "")
    if not api_key:
        raise SystemExit("OFOX_API_KEY missing")
    topics = [line.strip() for line in args.topics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit_topics:
        topics = topics[: args.limit_topics]
    source_roots = [Path(part.strip()) for part in args.source_roots.split(",") if part.strip()]
    tiers = parse_tier_spec(args.tiers)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    bad = [arm for arm in arms if arm not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms: {bad}; allowed={ARMS}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = OFOXClient(api_key, args.model, verbose=args.verbose)
    rows = []
    for topic in topics:
        bundle, evidence = load_topic_assets(topic, source_roots)
        (args.out_dir / safe_name(topic) / "source_bundle.json").parent.mkdir(parents=True, exist_ok=True)
        for tier_name, tier_cfg in tiers.items():
            for arm in arms:
                meta_path = args.out_dir / tier_name / safe_name(topic) / arm / "meta.json"
                if args.resume_skip_ok and meta_path.exists():
                    meta = load_json(meta_path)
                    length_ok = tier_cfg["min_words"] <= int(meta.get("word_count") or 0) <= tier_cfg["max_words"]
                    cite_ok = int(meta.get("n_invalid_citations") or 0) == 0
                    if meta.get("ok") and length_ok and cite_ok:
                        print(f"[S5a] SKIP {topic} | {tier_name} | {arm}", flush=True)
                        rows.append(meta)
                        continue
                print(f"[S5a] RUN {topic} | {tier_name} | {arm}", flush=True)
                result = run_case(topic, tier_name, tier_cfg, arm, bundle, evidence, client)
                write_result(args.out_dir, tier_name, topic, arm, result, evidence)
                meta = {k: v for k, v in result.items() if k not in {"text", "trace"}}
                rows.append(meta)
                print(f"[S5a] {topic} | {tier_name} | {arm} ok={meta.get('ok')} words={meta.get('word_count')} refs={meta.get('n_refs')}", flush=True)
    summary = {
        "runner": "s5a_length_runner.py",
        "model": args.model,
        "topics": topics,
        "tiers": tiers,
        "arms": arms,
        "n_ok": sum(1 for row in rows if row.get("ok")),
        "n_total": len(rows),
        "llm_usage": usage(client),
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S5a] summary n_ok={summary['n_ok']}/{summary['n_total']} usage={summary['llm_usage']}", flush=True)


if __name__ == "__main__":
    main()
