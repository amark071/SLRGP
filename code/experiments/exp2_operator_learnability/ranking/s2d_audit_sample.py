#!/usr/bin/env python3
"""
S2d 正例链接审计——清单生成（离线运行，无 LLM）。

审计对象：我们的"参考文献条目 → 语料库记录"链接（归一化标题精确匹配可能同形异义错链/版本混淆）。
不是论文引用规范本身。

步骤：
  1) 汇集 1,102 达标综述的全部正例 (arxiv_id, matched_doc_id)。
  2) 从语料库取每个正例记录的 title/doi/source/discipline，打 5 个风险层标记：
     低匹配可靠（此处以 bbl 源解析代理，因 bbl 精度低于 bib）、无 DOI、预印本源、
     短/通用标题(<=3 词)、学科。
  3) 按 5 层分层抽样固定 200 条（SHA-256 确定性）。
  4) 对被抽中的综述重解析 .bib/.bbl，按归一化标题找回"被引原始条目"(raw title + 原始条目串)。
  5) 落 audit_items.json：{arxiv_id, doc_id, strata, cited_entry_raw, corpus_record}。

用法：python3 s2d_audit_sample.py
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict

import bibtexparser

RESOLVED_DIR = "data/exp2_operator_learnability/ranking/resolved"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
LATEX_DIR = "data/common/arxiv_latex"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
OUT = "work/exp2_operator_learnability/ranking/audit_items.json"
N_SAMPLE = 200
MIN_POS = 5


def norm_title(t):
    t = (t or "").lower().strip().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def sha_u(*p):
    return int.from_bytes(hashlib.sha256("||".join(map(str, p)).encode()).digest()[:8], "big") / 2**64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tight", action="store_true", help="用 resolved_tight 收紧后正例集抽样，输出 audit_items_tight.json")
    args = ap.parse_args()
    global OUT
    tight_pos = None
    if args.tight:
        OUT = OUT.replace("audit_items.json", "audit_items_tight.json")
        tight_pos = {}
        for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
            r = json.load(open(jf, encoding="utf-8"))
            tight_pos[r["arxiv_id"]] = set(r["matched_doc_ids"])
        print(f"收紧审计模式：{len(tight_pos)} 篇收紧正例集")

    conn = sqlite3.connect(CORPUS_DB)
    # 1) 汇集正例（元数据取自原始 resolved；tight 模式下正例集用收紧后的）
    pos = []  # (arxiv_id, discipline, source_res, doc_id)
    review_meta = {}
    for jf in sorted(glob.glob(os.path.join(RESOLVED_DIR, "*", "*.json"))):
        r = json.load(open(jf, encoding="utf-8"))
        if r.get("n_matched_corpus", 0) < MIN_POS:
            continue
        arxiv_id = r["arxiv_id"]
        ids = r["matched_doc_ids"]
        if tight_pos is not None:
            ids = sorted(tight_pos.get(arxiv_id, set()))
            if len(ids) < MIN_POS:
                continue
        review_meta[arxiv_id] = {"discipline": r["discipline"], "resolution_source": r.get("resolution_source")}
        for d in ids:
            pos.append((arxiv_id, r["discipline"], r.get("resolution_source"), d))
    print(f"达标综述正例总数: {len(pos)}")

    # 2) 取正例记录元数据
    ids = sorted(set(d for _, _, _, d in pos))
    meta = {}
    for i in range(0, len(ids), 800):
        chunk = ids[i:i + 800]
        ph = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT doc_id, title, doi, source, discipline, year, authors, venue FROM papers WHERE doc_id IN ({ph})", chunk):
            meta[row[0]] = {"title": row[1], "doi": row[2], "source": row[3], "discipline": row[4],
                            "year": row[5], "authors": row[6], "venue": row[7]}

    # 3) 打风险层
    def strata_of(arxiv_id, res_src, d):
        m = meta.get(d, {})
        nt = norm_title(m.get("title"))
        s = []
        if res_src == "bbl":
            s.append("low_reliability_bbl")
        if not m.get("doi"):
            s.append("no_doi")
        src = (m.get("source") or "").lower()
        if src.startswith("arxiv") or "arxiv" in src:
            s.append("preprint_source")
        if nt and len(nt.split()) <= 3:
            s.append("short_generic_title")
        return s or ["clean"]

    strata_pool = defaultdict(list)
    for arxiv_id, disc, res_src, d in pos:
        for s in strata_of(arxiv_id, res_src, d):
            strata_pool[s].append((arxiv_id, disc, res_src, d))
    print("风险层规模:", {k: len(v) for k, v in strata_pool.items()})

    # 4) 分层抽样（5 高风险层各配额，学科均衡在层内用 sha 排序近似）
    target_strata = ["low_reliability_bbl", "no_doi", "preprint_source", "short_generic_title", "clean"]
    per = max(1, N_SAMPLE // len(target_strata))
    chosen = {}
    for s in target_strata:
        cand = strata_pool.get(s, [])
        cand_sorted = sorted(cand, key=lambda x: sha_u("audit_v1", s, x[0], x[3]))
        for item in cand_sorted:
            key = (item[0], item[3])
            if key in chosen:
                continue
            chosen[key] = {"arxiv_id": item[0], "discipline": item[1], "doc_id": item[3],
                           "strata": strata_of(item[0], item[2], item[3])}
            if sum(1 for v in chosen.values() if s in v["strata"]) >= per:
                break
    # 补足到 200
    if len(chosen) < N_SAMPLE:
        allc = sorted(pos, key=lambda x: sha_u("audit_fill", x[0], x[3]))
        for arxiv_id, disc, res_src, d in allc:
            key = (arxiv_id, d)
            if key in chosen:
                continue
            chosen[key] = {"arxiv_id": arxiv_id, "discipline": disc, "doc_id": d,
                           "strata": strata_of(arxiv_id, res_src, d)}
            if len(chosen) >= N_SAMPLE:
                break
    print(f"抽样条数: {len(chosen)}")

    # 5) 对被抽中的综述重解析 bib/bbl，找回被引原始条目
    by_review = defaultdict(list)
    for v in chosen.values():
        by_review[v["arxiv_id"]].append(v)

    def load_bibmap(arxiv_id, disc):
        import tarfile, gzip
        tar = os.path.join(LATEX_DIR, disc, f"{arxiv_id}.tar.gz")
        if not os.path.exists(tar):
            return {}
        members = {}
        try:
            with tarfile.open(tar, "r:*") as tf:
                for m in tf.getmembers():
                    if m.isfile() and m.name.lower().endswith((".bib", ".bbl", ".tex")):
                        try:
                            members[m.name] = tf.extractfile(m).read().decode("utf-8", "ignore")
                        except Exception:
                            pass
        except Exception:
            try:
                with gzip.open(tar, "rb") as gf:
                    members["main.tex"] = gf.read().decode("utf-8", "ignore")
            except Exception:
                return {}
        # bib
        norm2raw = {}
        for f, c in members.items():
            if f.lower().endswith(".bib"):
                try:
                    db = bibtexparser.loads(c)
                except Exception:
                    continue
                for e in db.entries:
                    ti = re.sub(r"[{}]", "", e.get("title", "")).strip()
                    if ti:
                        raw = "; ".join(f"{k}={v}" for k, v in e.items() if k in ("title", "author", "year", "journal", "booktitle"))
                        norm2raw.setdefault(norm_title(ti), {"title": ti, "raw": raw[:400]})
        if norm2raw:
            return norm2raw
        # bbl 兜底
        for f, c in members.items():
            if f.lower().endswith(".bbl") or "\\begin{thebibliography}" in c:
                for chunk in re.split(r"\\bibitem", c)[1:]:
                    mm = re.match(r"^(?:\[[^\]]*\])?\{([^}]*)\}", chunk, re.DOTALL)
                    if not mm:
                        continue
                    raw = chunk[mm.end():][:400]
                    raw = re.sub(r"\\newblock", " ", raw)
                    raw = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", raw)
                    raw = re.sub(r"[{}]", " ", raw)
                    raw = re.sub(r"\s+", " ", raw).strip()
                    norm2raw.setdefault(norm_title(raw), {"title": raw[:200], "raw": raw})
        return norm2raw

    items = []
    for arxiv_id, vs in by_review.items():
        disc = vs[0]["discipline"]
        norm2raw = load_bibmap(arxiv_id, disc)
        for v in vs:
            corp = meta.get(v["doc_id"], {})
            cited = norm2raw.get(norm_title(corp.get("title")), None)
            items.append({
                "arxiv_id": arxiv_id, "doc_id": v["doc_id"], "strata": v["strata"],
                "cited_entry_raw": cited,
                "corpus_record": {"title": corp.get("title"), "authors": corp.get("authors"),
                                  "year": corp.get("year"), "venue": corp.get("venue"),
                                  "doi": corp.get("doi"), "source": corp.get("source")},
            })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n": len(items), "items": items}, f, ensure_ascii=False, indent=2)
    n_recovered = sum(1 for it in items if it["cited_entry_raw"])
    print(f"审计清单 {len(items)} 条，其中恢复到被引原始条目 {n_recovered} 条 → {OUT}")


if __name__ == "__main__":
    main()
