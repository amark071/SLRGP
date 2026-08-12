#!/usr/bin/env python3
"""Freeze reference-derived core-paper sets from pre-cutoff OpenAlex reviews."""
from __future__ import annotations

import argparse
import collections
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://api.openalex.org"
REVIEW_PATTERN = re.compile(r"\b(survey|review|overview|tutorial)\b", re.I)
REQUIRED_TITLE_GROUPS = {
    "mature_neural_machine_translation": (("machine translation",),),
    "mature_deep_reinforcement_learning": (("reinforcement",), ("learning",)),
    "mature_adversarial_robustness": (
        ("adversarial",),
        ("robust", "attack", "defense", "example"),
    ),
    "fast_text_to_image_diffusion": (
        ("diffusion",),
    ),
    "fast_vision_language_pretraining": (
        ("vision-language", "vision language", "image-text", "image text"),
    ),
    "fast_large_language_model_agents": (
        ("agent",),
        ("large language model", "llm"),
    ),
    "method_self_supervised_learning": (
        ("self-supervised", "self supervised"),
    ),
    "method_continual_learning": (("continual", "lifelong"), ("learning",)),
    "method_uncertainty_quantification": (
        ("uncertainty",),
        ("deep learning", "neural network", "machine learning"),
    ),
    "cross_medical_image_analysis": (("medical",), ("image", "imaging")),
    "cross_ml_drug_discovery": (
        ("drug", "molecular"),
        ("discover", "design"),
    ),
    "cross_efficient_deep_learning": (
        ("efficient", "compression", "pruning", "quantization"),
        ("deep", "neural"),
    ),
}
TITLE_SEARCH = {
    "mature_neural_machine_translation": "machine translation",
    "mature_deep_reinforcement_learning": "reinforcement learning",
    "mature_adversarial_robustness": "adversarial",
    "fast_text_to_image_diffusion": "diffusion",
    "fast_vision_language_pretraining": "vision language",
    "fast_large_language_model_agents": "language model agents",
    "method_self_supervised_learning": "self supervised",
    "method_continual_learning": "continual learning",
    "method_uncertainty_quantification": "uncertainty deep learning",
    "cross_medical_image_analysis": "medical image",
    "cross_ml_drug_discovery": "drug discovery machine learning",
    "cross_efficient_deep_learning": "efficient deep learning",
}


def get_json(path: str, params: dict, retries: int = 6) -> dict:
    params = dict(params)
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "SLRGP-S3-core-set/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read())
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.IncompleteRead,
        ) as exc:
            error = exc
            time.sleep(min(60, 3 * (2**attempt)))
    raise RuntimeError(f"OpenAlex request failed: {error!r}")


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def select_source_reviews(
    topic: dict, cutoff: str, source_registry: dict | None = None
) -> list[dict]:
    if source_registry:
        selected = []
        for registered in source_registry[topic["topic_id"]]:
            identifier = registered.get("doi") or (
                f"10.48550/arXiv.{registered['arxiv_id']}"
                if registered.get("arxiv_id")
                else ""
            )
            if not identifier:
                raise RuntimeError(
                    f"{topic['topic_id']}: registered review has no persistent ID"
                )
            row = get_json(
                f"/works/https://doi.org/{identifier}",
                {
                    "select": (
                        "id,doi,display_name,publication_date,cited_by_count,"
                        "referenced_works,authorships,primary_location,type"
                    )
                },
            )
            if normalize_title(str(row.get("display_name") or "")) != normalize_title(
                registered["title"]
            ):
                raise RuntimeError(
                    f"{topic['topic_id']}: identifier-title mismatch: "
                    f"{registered['title']} != {row.get('display_name')}"
                )
            selected.append(row)
        return selected

    body = get_json(
        "/works",
        {
            "filter": (
                "from_publication_date:2000-01-01,"
                f"to_publication_date:{cutoff},"
                f"title.search:{TITLE_SEARCH[topic['topic_id']]}"
            ),
            "per-page": 200,
            "sort": "cited_by_count:desc",
            "select": (
                "id,doi,display_name,publication_date,cited_by_count,"
                "authorships,primary_location,type"
            ),
        },
    )
    groups = REQUIRED_TITLE_GROUPS[topic["topic_id"]]
    candidates = []
    for row in body.get("results") or []:
        title = str(row.get("display_name") or "").lower()
        if (
            REVIEW_PATTERN.search(title)
            and all(any(term in title for term in group) for group in groups)
        ):
            candidates.append(row)
    selected = []
    seen_doi = set()
    for row in candidates:
        doi = str(row.get("doi") or row.get("id"))
        if doi in seen_doi:
            continue
        full = get_json(
            f"/works/{str(row['id']).rsplit('/', 1)[-1]}",
            {
                "select": (
                    "id,doi,display_name,publication_date,cited_by_count,"
                    "referenced_works,authorships,primary_location,type"
                )
            },
        )
        if len(full.get("referenced_works") or []) < 20:
            continue
        selected.append(full)
        seen_doi.add(doi)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        raise RuntimeError(
            f"{topic['topic_id']}: only {len(selected)} eligible source reviews"
        )
    return selected


def fetch_works(ids: list[str]) -> list[dict]:
    rows = []
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        body = get_json(
            "/works",
            {
                "filter": "openalex_id:"
                + "|".join(value.rsplit("/", 1)[-1] for value in chunk),
                "per-page": 50,
                "select": (
                    "id,doi,display_name,publication_date,cited_by_count,"
                    "type,primary_location"
                ),
            },
        )
        rows.extend(body.get("results") or [])
        time.sleep(0.2)
    return rows


def resolve_fallback_arxiv_ids(arxiv_ids: list[str], cutoff: str) -> list[str]:
    resolved = []
    for start in range(0, len(arxiv_ids), 50):
        chunk = arxiv_ids[start : start + 50]
        body = get_json(
            "/works",
            {
                "filter": "doi:"
                + "|".join(f"10.48550/arxiv.{value}" for value in chunk),
                "per-page": 50,
                "select": "id,publication_date",
            },
        )
        resolved.extend(
            row["id"]
            for row in body.get("results") or []
            if row.get("id")
            and str(row.get("publication_date") or "9999-12-31") <= cutoff
        )
        time.sleep(0.2)
    return resolved


def build_topic(
    topic: dict,
    cutoff: str,
    size: int,
    source_registry: dict | None = None,
    fallback_arxiv_ids: list[str] | None = None,
) -> dict:
    reviews = select_source_reviews(topic, cutoff, source_registry)
    frequency = collections.Counter(
        reference
        for review in reviews
        for reference in set(review.get("referenced_works") or [])
    )
    frequency.update(
        resolve_fallback_arxiv_ids(fallback_arxiv_ids or [], cutoff)
    )
    co_cited_ids = [
        openalex_id for openalex_id, count in frequency.items() if count >= 2
    ]
    metadata = fetch_works(co_cited_ids)
    central_body = get_json(
        "/works",
        {
            "search": topic["title"],
            "filter": (
                "from_publication_date:2000-01-01,"
                f"to_publication_date:{cutoff}"
            ),
            "per-page": 200,
            "sort": "cited_by_count:desc",
            "select": (
                "id,doi,display_name,publication_date,cited_by_count,"
                "type,primary_location"
            ),
        },
    )
    seen = {row["id"] for row in metadata}
    metadata.extend(
        row
        for row in central_body.get("results") or []
        if row.get("id") in frequency and row.get("id") not in seen
    )
    seen = {row["id"] for row in metadata}
    supplemental_ids = []
    for review in reviews:
        for openalex_id in (review.get("referenced_works") or [])[:200]:
            if openalex_id not in seen and openalex_id not in supplemental_ids:
                supplemental_ids.append(openalex_id)
    for openalex_id in frequency:
        if openalex_id not in seen and openalex_id not in supplemental_ids:
            supplemental_ids.append(openalex_id)
    metadata.extend(fetch_works(supplemental_ids))
    eligible = [
        row
        for row in metadata
        if str(row.get("publication_date") or "9999-12-31") <= cutoff
    ]
    eligible.sort(
        key=lambda row: (
            frequency.get(row["id"], 0) >= 2,
            frequency.get(row["id"], 0),
            int(row.get("cited_by_count") or 0),
        ),
        reverse=True,
    )
    core = eligible[:size]
    if len(core) < size:
        raise RuntimeError(f"{topic['topic_id']}: only {len(core)} core candidates")
    return {
        "topic_id": topic["topic_id"],
        "title": topic["title"],
        "publication_cutoff": cutoff,
        "construction": {
            "source_review_count": 3,
            "priority": "cited by at least two source reviews, then cutoff-safe citation centrality",
            "target_size": size,
            "deduplication": "OpenAlex work ID with DOI retained when available",
            "fallback": (
                "Registered arXiv identifiers deterministically extracted from "
                "frozen review source when OpenAlex omitted its bibliography"
                if fallback_arxiv_ids
                else None
            ),
        },
        "source_reviews": [
            {
                "openalex_id": row["id"],
                "doi": row.get("doi"),
                "title": row.get("display_name"),
                "publication_date": row.get("publication_date"),
                "cited_by_count": row.get("cited_by_count"),
                "n_referenced_works": len(row.get("referenced_works") or []),
            }
            for row in reviews
        ],
        "fallback_reference_arxiv_ids": fallback_arxiv_ids or [],
        "core_references": [
            {
                "openalex_id": row["id"],
                "doi": row.get("doi"),
                "title": row.get("display_name"),
                "publication_date": row.get("publication_date"),
                "cited_by_count": row.get("cited_by_count"),
                "source_review_frequency": frequency.get(row["id"], 0),
            }
            for row in core
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--source-registry", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.topics.read_text(encoding="utf-8"))
    result = {
        "schema_version": "1.0",
        "status": "frozen_before_confirmatory_generation",
        "publication_cutoff": manifest["publication_cutoff"],
        "topics": {},
    }
    if args.output.is_file():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    source_registry_payload = (
        json.loads(args.source_registry.read_text(encoding="utf-8"))
        if args.source_registry
        else {}
    )
    source_registry = source_registry_payload.get("topics") or None
    fallback_registry = source_registry_payload.get(
        "fallback_reference_arxiv_ids", {}
    )
    for topic in manifest["topics"]:
        if topic["topic_id"] in result["topics"]:
            continue
        result["topics"][topic["topic_id"]] = build_topic(
            topic,
            manifest["publication_cutoff"],
            args.size,
            source_registry,
            fallback_registry.get(topic["topic_id"], []),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(topic["topic_id"], "frozen", flush=True)


if __name__ == "__main__":
    main()
