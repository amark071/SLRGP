#!/usr/bin/env python3
"""Submit acquired social-science review PDFs to a local GROBID service."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def call_grobid(pdf: Path, endpoint: str) -> bytes:
    boundary = "----SLRGPBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="input"; filename="{pdf.name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--tei-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8070/api/processFulltextDocument")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.tei_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        if not row.get("downloaded"):
            continue
        pdf = args.ledger.parent / row["discipline"] / f"{row['review_id']}.pdf"
        tei = args.tei_dir / row["discipline"] / f"{row['review_id']}.tei.xml"
        tei.parent.mkdir(parents=True, exist_ok=True)
        status = "ok"
        error = ""
        try:
            if not tei.exists() or tei.stat().st_size < 500:
                tei.write_bytes(call_grobid(pdf, args.endpoint))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        manifest.append({
            "review_id": row["review_id"],
            "discipline": row["discipline"],
            "tei_path": str(tei),
            "metadata": {"doi": row.get("doi"), "pdf_url": row.get("pdf_url"), "source_discipline": row.get("source_discipline")},
            "grobid_status": status,
            "grobid_error": error,
        })
        time.sleep(args.sleep)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"n_submitted": len(manifest), "n_ok": sum(row["grobid_status"] == "ok" for row in manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
