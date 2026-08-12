#!/usr/bin/env python3
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from slrgp.state import Paper, SLRGPState  # noqa: E402

fake_retrieval = types.ModuleType("slrgp.retrieval")
fake_retrieval.op_L_hybrid = lambda state, index: state
sys.modules["slrgp.retrieval"] = fake_retrieval

from slrgp.pipeline_real import descend_real, merge_real  # noqa: E402


def paper(doc_id: str) -> Paper:
    return Paper(
        doc_id=doc_id,
        title=doc_id,
        abstract="abstract",
        authors=[],
        year=2024,
    )


class ReentryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = SLRGPState(
            q={
                "seed": "Graph Neural Networks",
                "terms": ["message passing", "graph representation learning"],
            },
            D=[],
            Gamma={},
            kappa={},
            x={},
            meta={
                "root_topic": "Graph Neural Networks",
                "top_n": 300,
                "min_abs_len": 100,
                "publication_cutoff_year": 2024,
            },
        )

    def test_full_reentry_preserves_parent_context_and_meta(self) -> None:
        trace = {"descend_events": []}
        child = descend_real(
            self.parent,
            {"name": "Graph Pooling", "doc_ids": ["p1"]},
            {"p1": paper("p1")},
            min_seed=4,
            trace=trace,
            node_path="root/0",
        )
        self.assertEqual(child.meta["mode"], "full")
        self.assertIn("Graph Neural Networks", child.q["seed"])
        self.assertIn("Graph Pooling", child.q["seed"])
        self.assertIn("Graph Neural Networks", child.q["terms"])
        self.assertEqual(child.meta["top_n"], 300)
        self.assertEqual(child.meta["min_abs_len"], 100)
        self.assertEqual(child.meta["publication_cutoff_year"], 2024)
        self.assertEqual(trace["descend_events"][0]["mode"], "full")

    def test_partial_reentry_also_preserves_parent_context(self) -> None:
        corpus = {f"p{i}": paper(f"p{i}") for i in range(4)}
        child = descend_real(
            self.parent,
            {"name": "Graph Pooling", "doc_ids": list(corpus)},
            corpus,
            min_seed=4,
        )
        self.assertEqual(child.meta["mode"], "partial")
        self.assertEqual(len(child.D), 4)
        self.assertIn("Graph Neural Networks", child.q["seed"])
        self.assertEqual(child.meta["publication_cutoff_year"], 2024)

    def test_merge_disambiguates_duplicate_titles(self) -> None:
        rendered = merge_real(
            [
                ("Graph Pooling", "first"),
                ("Graph Pooling", "second"),
            ]
        )
        self.assertIn("### Graph Pooling\n", rendered)
        self.assertIn("### Graph Pooling (2)\n", rendered)


if __name__ == "__main__":
    unittest.main()
