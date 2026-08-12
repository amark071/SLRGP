#!/usr/bin/env python3
"""
S2b (Learning L) Tier-2 阶段A：citation-supervised 微调 bi-encoder。

学习目标：让编码器把"真实被引文献"嵌入到离综述查询更近的位置，既改善并集内排序，
又能把当前 depth-3000 之外的正例（gap_retrieval≈0.45）拉进召回——这是仅靠重排检索
分（Tier-1，仅 +1 点）无法做到的。故正例取"全部时间安全收紧真引用"（含未被召回者，
从语料库取其文本），硬负取同综述并集内高 rrf 的未引用文献。

base = BAAI/bge-small-en-v1.5（与既有 dense 基线同架构，隔离"微调"这一唯一变量）。
query 加 bge 检索前缀，doc 不加（bge 非对称用法，与 retrieval.py 一致）。
loss = CachedMultipleNegativesRankingLoss（GradCache，支持大 batch；in-batch 负例 + 每正例一个显式硬负）。
只用 train 划分构造训练对；val/test 综述及其引用完全不进训练（防泄漏）。

输出模型：models/bge_small_L_ft
用法（服务器）：python3 s2b_tier2_train.py
"""
import json
import os
import random
import sqlite3

import pandas as pd

UNION_DIR = os.environ.get("S2B_UNION_DIR", "work/exp2_operator_learnability/ranking/union_L")
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
OUT_MODEL = os.environ.get("S2B_FT_MODEL", "models/bge_small_L_ft")

QPREFIX = "Represent this sentence for searching relevant passages: "
BASE = "BAAI/bge-small-en-v1.5"
HARD_NEG_PER_POS = 1
MAX_HARD_POOL = 50       # 每篇综述取 rrf 最高的前 N 个未引用并集 doc 作硬负池
MAX_POS_PER_REVIEW = 30  # 每综述正例上限（随机采样），抑制同 query in-batch 假负例 & 长尾主导
EPOCHS = int(os.environ.get("S2B_FT_EPOCHS", "1"))
BATCH = int(os.environ.get("S2B_FT_BATCH", "256"))      # 大 batch（cached MNR 支持）
MINI_BATCH = int(os.environ.get("S2B_FT_MINIBATCH", "32"))  # GradCache 分块，控显存
LR = float(os.environ.get("S2B_FT_LR", "1e-5"))
MAX_SEQ = int(os.environ.get("S2B_FT_MAXSEQ", "256"))
DOC_CHARS = 2000  # 截断文档文本，防个别超长 abstract 撑爆显存
SEED = 42


def load_tight():
    import glob
    tight = {}
    for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(jf, encoding="utf-8"))
        tight[r["arxiv_id"]] = set(r["matched_doc_ids"])
    return tight


def doc_text(title, abstract):
    return ((title or "") + ". " + (abstract or ""))[:DOC_CHARS]


def main():
    import glob
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    random.seed(SEED)
    tight = load_tight()
    train_union = pd.read_parquet(os.path.join(UNION_DIR, "train.parquet"))
    train_aids = set(train_union["arxiv_id"].unique())
    print(f"train 综述: {len(train_aids)}")

    conn = sqlite3.connect(CORPUS_DB)

    # 每篇 train 综述的 query 文本 + review_year
    cand_files = {}
    for jf in glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")):
        aid = os.path.basename(jf)[:-5]
        cand_files[aid] = jf

    # 硬负池：并集内未引用、按 rrf 降序前 MAX_HARD_POOL
    hard_pool = {}
    for aid, g in train_union.groupby("arxiv_id"):
        neg = g[g["label"] == 0].nlargest(MAX_HARD_POOL, "rrf_score")["doc_id"].tolist()
        hard_pool[aid] = neg

    # 需要取文本的 doc_id 集合：train 正例（全部时间安全收紧引用）+ 硬负
    need_docs = set()
    review_query = {}
    pos_of = {}
    for aid in train_aids:
        jf = cand_files.get(aid)
        if not jf:
            continue
        rec = json.load(open(jf, encoding="utf-8"))
        rdid = rec["review_doc_id"]
        ry = rec.get("review_year")
        row = conn.execute("SELECT title, abstract FROM papers WHERE doc_id=?", (rdid,)).fetchone()
        if not row or not row[0]:
            continue
        review_query[aid] = QPREFIX + (row[0] or "") + ". " + (row[1] or "")[:1000]
        # 正例：时间安全收紧引用（含不可达）
        pos = []
        for d in tight.get(aid, set()):
            pos.append(d)
        pos_of[aid] = (pos, ry)
        for d in pos:
            need_docs.add(d)
        for d in hard_pool.get(aid, []):
            need_docs.add(d)

    print(f"需取文本 doc: {len(need_docs)}")
    text_of = {}
    year_of = {}
    ids = list(need_docs)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        ph = ",".join("?" for _ in chunk)
        for d, t, a, y in conn.execute(
                f"SELECT doc_id, title, abstract, year FROM papers WHERE doc_id IN ({ph})", chunk):
            text_of[d] = doc_text(t, a)
            year_of[d] = y

    # 组装 InputExample(query, pos, hard_neg)；每综述正例封顶（去重 + 随机采样），
    # 抑制同 query 在同一 batch 内互为假负例，并防少数高引综述主导。
    examples = []
    dropped_cap = 0
    for aid in train_aids:
        if aid not in review_query:
            continue
        q = review_query[aid]
        pos_list, ry = pos_of[aid]
        negs = [d for d in hard_pool.get(aid, []) if d in text_of]
        # 过滤：有文本 + 时间安全
        valid_pos = []
        for d in set(pos_list):
            if d not in text_of:
                continue
            cy = year_of.get(d)
            if ry and cy and cy > ry:
                continue
            valid_pos.append(d)
        if len(valid_pos) > MAX_POS_PER_REVIEW:
            dropped_cap += len(valid_pos) - MAX_POS_PER_REVIEW
            valid_pos = random.sample(valid_pos, MAX_POS_PER_REVIEW)
        for d in valid_pos:
            ptext = text_of[d]
            if negs:
                nd = random.choice(negs)
                examples.append(InputExample(texts=[q, ptext, text_of[nd]]))
            else:
                examples.append(InputExample(texts=[q, ptext]))
    print(f"训练三元组: {len(examples)}（因每综述封顶丢弃正例 {dropped_cap}）")

    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = MAX_SEQ
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
    # CachedMultipleNegativesRankingLoss（GradCache）：低显存下支持大 batch，
    # 大 batch 是 MNR 有效性的关键（in-batch 负例数 = batch-1），并绕开常规 MNR 的显存爆点。
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH)
    warmup = max(1, int(len(loader) * EPOCHS * 0.1))
    print(f"开始微调: epochs={EPOCHS} batch={BATCH} mini={MINI_BATCH} lr={LR} "
          f"maxseq={MAX_SEQ} steps/epoch={len(loader)} warmup={warmup}")
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=warmup, show_progress_bar=True, use_amp=True,
              optimizer_params={"lr": LR})
    model.save(OUT_MODEL)
    with open(os.path.join(OUT_MODEL, "train_meta.json"), "w") as f:
        json.dump({"base": BASE, "n_examples": len(examples), "epochs": EPOCHS,
                   "batch": BATCH, "mini_batch": MINI_BATCH, "lr": LR,
                   "loss": "CachedMultipleNegativesRankingLoss",
                   "max_pos_per_review": MAX_POS_PER_REVIEW,
                   "hard_neg_per_pos": HARD_NEG_PER_POS,
                   "n_train_reviews": len(review_query)}, f, indent=2)
    print(f"已保存微调模型: {OUT_MODEL}")


if __name__ == "__main__":
    main()
