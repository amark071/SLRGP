#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from s3_native_common import RunDirectory, RunSpec, TopicIntent, redact


def topic() -> TopicIntent:
    return TopicIntent(
        topic_id="pilot_topic",
        title="Pilot Topic",
        scope="A frozen pilot scope.",
        target_audience="researchers",
        language="English",
        publication_cutoff="2024-09-25",
        target_words=4000,
        min_words=3200,
        max_words=4800,
        target_references_min=40,
        target_references_max=60,
        seed=20260715,
        analysis_role="pilot",
    )


class NativeContractTest(unittest.TestCase):
    def test_redacts_keys(self) -> None:
        value = "api_key=secret-value Authorization: BearerSecret sk-example123456789"
        redacted = redact(value)
        self.assertNotIn("secret-value", redacted)
        self.assertNotIn("BearerSecret", redacted)
        self.assertNotIn("sk-example123456789", redacted)

    def test_completion_requires_artifacts_and_hash(self) -> None:
        spec = RunSpec(
            run_id="test",
            system_id="example",
            source_revision="abc123",
            model_policy="common_backbone",
            topic=topic(),
            budget_usd=1.0,
            timeout_seconds=60,
            protocol_hash="protocol",
        )
        with tempfile.TemporaryDirectory() as directory:
            run = RunDirectory(Path(directory), spec)
            run.initialize()
            self.assertFalse(run.completed_and_valid())
            for filename, value in {
                "preflight.json": "{}\n",
                "survey.md": "# Survey\n",
                "references.json": "[]\n",
                "retrieval_manifest.json": "{}\n",
                "trace_or_stage_log.json": "{}\n",
                "usage.json": "{}\n",
                "meta.json": "{}\n",
            }.items():
                (run.root / filename).write_text(value, encoding="utf-8")
            run.finalize_success()
            self.assertTrue(run.completed_and_valid())
            status = json.loads(run.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "ok")
            (run.root / "survey.md").write_text("# Tampered\n", encoding="utf-8")
            self.assertFalse(run.completed_and_valid())


if __name__ == "__main__":
    unittest.main()
