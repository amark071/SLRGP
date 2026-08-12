#!/usr/bin/env python3
"""
S2d 标签收紧（作者/年份护栏，回应审计短标题层同形异义错链）。

对"归一化标题精确匹配"的初步链接结果追加一道护栏：
  - 仅对 corpus 标题 ≤3 个归一化词 的"短/通用标题"匹配施加护栏（长标题精度已 0.93-0.96，不动）；
  - 短标题匹配需满足 (作者姓氏token重叠 ≥1) 或 (被引年份与语料库年份相差 ≤1) 才保留为正例，
    否则判为同形异义错链、剔除（该候选回落为 observed negative）。

输出 resolved_tight/{disc}/{arxiv}.json：{arxiv_id, discipline, matched_doc_ids(收紧后),
  n_matched_before, n_matched_after, n_short_examined, n_short_dropped}
以及 _tighten_summary.json。

用法：python3 s2d_retighten.py
"""
import glob
import json
import os
import re
import sqlite3
import tarfile
import gzip

import bibtexparser
import logging
logging.getLogger("bibtexparser.bparser").setLevel(logging.ERROR)

PARSED_DIR = "data/exp1_structural_recovery/lotcf_trees"
LATEX_DIR = "data/common/arxiv_latex"
RESOLVED_DIR = "data/exp2_operator_learnability/ranking/resolved"
OUT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
DB = "data/common/unified_corpus/unified_corpus.db"
MIN_POS = 5
SHORT_MAXWORDS = 3
STOP = {"the", "and", "for", "with", "from", "into", "using", "based", "study", "analysis",
        "model", "models", "data", "learning", "networks", "network", "review", "survey"}


class TO(Exception):
    pass


def _h(s, f):
    raise TO()


def norm_title(t):
    t = (t or "").lower().strip().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def surname_tokens(s):
    """从作者字符串抽取小写姓氏候选 token（len>=3、非停用词）。"""
    toks = set()
    for part in re.split(r"\band\b|,|;|\n", (s or "").lower()):
        words = re.findall(r"[a-z]{3,}", part)
        for w in words:
            if w not in STOP:
                toks.add(w)
    return toks


def used_keys(tree):
    keys = set()

    def walk(n):
        keys.update(n.get("total_cite_keys", []))
        for c in n.get("children", []):
            walk(c)
    for t in tree:
        walk(t)
    return keys


def load_members(tar):
    if not os.path.exists(tar):
        return {}, ""
    with open(tar, "rb") as f:
        if f.read(4) == b"%PDF":
            return {}, ""
    members = {}
    try:
        with tarfile.open(tar, "r:*") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.lower().endswith((".tex", ".bib", ".bbl")):
                    try:
                        members[m.name] = tf.extractfile(m).read().decode("utf-8", "ignore")
                    except Exception:
                        pass
        return members, "\n".join(v for k, v in members.items() if k.lower().endswith(".tex"))
    except tarfile.ReadError:
        try:
            with gzip.open(tar, "rb") as gf:
                t = gf.read().decode("utf-8", "ignore")
            return {"main.tex": t}, t
        except OSError:
            return {}, ""


def bib_entries(members):
    """key -> (title, authors, year)。bib 优先；bbl 兜底只给 title。"""
    out = {}
    has_bib = False
    for f, c in members.items():
        if not f.lower().endswith(".bib"):
            continue
        has_bib = True
        try:
            db = bibtexparser.loads(c)
        except Exception:
            continue
        for e in db.entries:
            k = e.get("ID")
            ti = re.sub(r"[{}]", "", e.get("title", "")).strip()
            if k and ti:
                out[k] = (ti, e.get("author", ""), e.get("year", ""))
    if out:
        return out
    # bbl 兜底：整段作标题候选，无作者/年份 → 短标题护栏会因无作者信息而依赖年份（也无）→ 短标题匹配在 bbl 下直接判丢
    src = [c for f, c in members.items() if f.lower().endswith(".bbl")]
    if not src:
        merged = "\n".join(members.values())
        if "\\begin{thebibliography}" in merged:
            src = [merged]
    for s in src:
        for chunk in re.split(r"\\bibitem", s)[1:]:
            mm = re.match(r"^(?:\[[^\]]*\])?\{([^}]*)\}", chunk, re.DOTALL)
            if not mm:
                continue
            key = mm.group(1).strip()
            raw = chunk[mm.end():][:600]
            raw = re.sub(r"\\newblock", " ", raw)
            raw = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", raw)
            raw = re.sub(r"[{}]", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            yr = re.search(r"\b(19|20)\d{2}\b", raw)
            if key and raw:
                out.setdefault(key, (raw[:300], raw, yr.group(0) if yr else ""))
    return out


def main():
    conn = sqlite3.connect(DB)
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESOLVED_DIR, "*", "*.json")))
    qual = []
    for jf in files:
        r = json.load(open(jf, encoding="utf-8"))
        if r.get("n_matched_corpus", 0) >= MIN_POS:
            qual.append(r)
    print(f"达标综述: {len(qual)}")

    tot_before = tot_after = tot_short = tot_dropped = 0
    n_below_min = 0
    for i, r in enumerate(qual):
        arxiv_id, disc = r["arxiv_id"], r["discipline"]
        # 载入解析树拿 used_keys
        pjf = os.path.join(PARSED_DIR, disc, f"{arxiv_id}.json")
        if not os.path.exists(pjf):
            keep = r["matched_doc_ids"]
        else:
            tree = json.load(open(pjf, encoding="utf-8")).get("tree", [])
            uk = used_keys(tree)
            members, _ = load_members(os.path.join(LATEX_DIR, disc, f"{arxiv_id}.tar.gz"))
            bmap = bib_entries(members)
            # 归一化标题 -> (bib authors, bib year)
            norm2meta = {}
            for k in uk:
                if k in bmap:
                    ti, au, yr = bmap[k]
                    norm2meta.setdefault(norm_title(ti), (au, yr))
            # 对既有 matched_doc_ids 逐个决定是否保留
            ids = r["matched_doc_ids"]
            crec = {}
            if ids:
                ph = ",".join("?" for _ in ids)
                for row in conn.execute(f"SELECT doc_id,title,authors,year FROM papers WHERE doc_id IN ({ph})", ids):
                    crec[row[0]] = (row[1], row[2], row[3])
            keep = []
            for d in ids:
                title, authors, year = crec.get(d, (None, None, None))
                nt = norm_title(title)
                if not nt or len(nt.split()) > SHORT_MAXWORDS:
                    keep.append(d); continue  # 长标题不动
                tot_short += 1
                bib_au, bib_yr = norm2meta.get(nt, ("", ""))
                ok = False
                # 作者姓氏 token 重叠
                if surname_tokens(bib_au) & surname_tokens(authors):
                    ok = True
                # 年份相差 ≤1
                try:
                    if bib_yr and year and abs(int(bib_yr) - int(year)) <= 1:
                        ok = True
                except ValueError:
                    pass
                if ok:
                    keep.append(d)
                else:
                    tot_dropped += 1
        tot_before += len(r["matched_doc_ids"])
        tot_after += len(keep)
        if len(keep) < MIN_POS:
            n_below_min += 1
        out = {"arxiv_id": arxiv_id, "discipline": disc, "matched_doc_ids": sorted(keep),
               "n_matched_before": len(r["matched_doc_ids"]), "n_matched_after": len(keep)}
        od = os.path.join(OUT_DIR, disc)
        os.makedirs(od, exist_ok=True)
        json.dump(out, open(os.path.join(od, f"{arxiv_id}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        if (i + 1) % 200 == 0:
            print(f"进度 {i+1}/{len(qual)}  short_examined={tot_short} dropped={tot_dropped}")

    summ = {"n_reviews": len(qual), "positives_before": tot_before, "positives_after": tot_after,
            "short_title_matches_examined": tot_short, "short_title_dropped": tot_dropped,
            "reviews_now_below_min5": n_below_min}
    json.dump(summ, open(os.path.join(OUT_DIR, "_tighten_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n=== 收紧汇总 ===")
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
