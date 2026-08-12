"""
真实混合检索:BM25(bm25s)+ 向量(bge-small-en-v1.5 余弦相似度)双路召回,
RRF (Reciprocal Rank Fusion) 融合排序——L 算子在真实语料上的实例化
(对应论文 Methods「Learned operator instantiations」中 L 的描述)。
"""
import json
import os
import pickle
import sqlite3

import bm25s
import numpy as np

from .state import Paper

# Corpus paths resolve relative to the package root (two levels above this
# file) and can be overridden with SLRGP_UNIFIED_DIR.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED_DIR = os.environ.get("SLRGP_UNIFIED_DIR", os.path.join(_PKG_ROOT, "data/common/unified_corpus"))
DB_PATH = f"{UNIFIED_DIR}/unified_corpus.db"
EMB_PATH = f"{UNIFIED_DIR}/unified_embeddings.npy"
DOCIDS_PATH = f"{UNIFIED_DIR}/unified_docids.pkl"
BM25_DIR = f"{UNIFIED_DIR}/bm25_index"

RRF_K = 60


class UnifiedIndex:
    """全量语料 + BM25索引 + 向量索引 + embedding模型，进程内常驻单例。"""

    def __init__(self, device=None):
        from sentence_transformers import SentenceTransformer

        print("[UnifiedIndex] 加载 doc_ids / embeddings ...")
        self.doc_ids = pickle.load(open(DOCIDS_PATH, "rb"))
        self.docid_to_row = {d: i for i, d in enumerate(self.doc_ids)}
        self.embeddings = np.load(EMB_PATH)  # (N, 384) float32, 已归一化
        print(f"[UnifiedIndex] 向量矩阵: {self.embeddings.shape}")

        print("[UnifiedIndex] 加载 BM25 索引 ...")
        self.bm25 = bm25s.BM25.load(BM25_DIR, load_corpus=True)

        # query embedding 默认用 CPU 跑(bge-small 很小,单条 query 编码几十毫秒级),
        # 避免与本地 vLLM 推理实例争用显存。
        device = device or "cpu"
        print(f"[UnifiedIndex] 加载 embedding 模型 (device={device}) ...")
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # 最近一次 search() 的三路分数(BM25/dense/RRF 及其名次),
        # 供 rank_model._features_for_paper() 构建检索特征。
        self._last_search_scores = {}

    def citation_count(self, doc_id):
        """语料自带被引数(近似 OpenAlex cited_by_count);列缺失时退回 0(按 rank_model 的缺失规则处理)。"""
        try:
            row = self.conn.execute("SELECT cited_by_count FROM papers WHERE doc_id=?", (doc_id,)).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0] or 0) if row else 0

    def _embed_query(self, text):
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        vec = self.embed_model.encode(
            [prefixed], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        return vec

    def _dense_search(self, query_text, k):
        qvec = self._embed_query(query_text)
        scores = self.embeddings @ qvec  # (N,) 余弦相似度（均已归一化）
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]

    def _bm25_search(self, query_text, k):
        tokens = bm25s.tokenize([query_text], stopwords="en", show_progress=False)
        results, scores = self.bm25.retrieve(tokens, k=k, show_progress=False)
        # bm25s 把非dict语料(纯字符串)自动包成 {"id":.., "text":..}，取回 text 才是真正的 doc_id
        out = []
        for j in range(results.shape[1]):
            item = results[0, j]
            doc_id = item["text"] if isinstance(item, dict) else item
            out.append((doc_id, float(scores[0, j])))
        return out

    def _fetch_papers(self, doc_ids):
        if not doc_ids:
            return {}
        out = {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self.conn.execute(
            f"SELECT doc_id, title, abstract, authors, year, tier, venue "
            f"FROM papers WHERE doc_id IN ({placeholders})",
            doc_ids,
        )
        for r in rows:
            try:
                authors = json.loads(r["authors"] or "[]")
            except Exception:
                authors = []
            out[r["doc_id"]] = Paper(
                doc_id=r["doc_id"],
                title=r["title"] or "",
                abstract=r["abstract"] or "",
                authors=authors,
                year=r["year"],
                tier=r["tier"] or "UNRANKED",
                venue=r["venue"] or "",
            )
        return out

    def search(self, query_text, top_k_each=200, top_n=50):
        bm25_hits = self._bm25_search(query_text, top_k_each)
        dense_hits = self._dense_search(query_text, top_k_each)

        bm25_rank = {d: r for r, (d, _) in enumerate(bm25_hits)}
        bm25_score = {d: s for d, s in bm25_hits}
        dense_rank = {d: r for r, (d, _) in enumerate(dense_hits)}
        dense_score = {d: s for d, s in dense_hits}

        rrf_score = {}
        for rank, (doc_id, _) in enumerate(bm25_hits):
            rrf_score[doc_id] = rrf_score.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (doc_id, _) in enumerate(dense_hits):
            rrf_score[doc_id] = rrf_score.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked_ids = sorted(rrf_score.keys(), key=lambda d: -rrf_score[d])[:top_n]
        # 记录本次检索的三路分数,供 rank_model 构建排序特征
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
        return [papers_by_id[d] for d in ranked_ids if d in papers_by_id]


_INDEX_SINGLETON = None


def get_index():
    global _INDEX_SINGLETON
    if _INDEX_SINGLETON is None:
        _INDEX_SINGLETON = UnifiedIndex()
    return _INDEX_SINGLETON


def op_L_hybrid(state, index=None, top_n=None):
    """真实检索版 L 算子：BM25 + 向量双路 RRF 融合，替代 operators.op_L 的词汇重叠打分。"""
    index = index or get_index()
    top_n = top_n if top_n is not None else state.meta.get("top_n", 50)
    terms = [state.q["seed"]] + state.q.get("terms", [])
    query_text = " ".join(terms)
    state.D = index.search(query_text, top_k_each=max(200, top_n * 4), top_n=top_n)
    return state
