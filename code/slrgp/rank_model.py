"""
学习式排序算子 R 的部署侧实现(论文实验 2-R / Methods「Learned operator instantiations」):
用引文监督训练的 LightGBM LambdaMART 模型替换 operators.op_R 的手写公式,
实例化"排序算子从经验启发式到可学习函数"这一主张。

部署期特征可得性处理(与训练时相同的缺失标记规则,不重训阉割版模型):
  - is_self_citation / self_citation_reliable:置 0。训练时该特征衡量"候选论文作者是否与
    真实历史综述的作者重合";从零生成新综述时没有"本篇综述作者"的概念,属概念不适用而非数据缺失。
  - max/mean_author_hindex:语料无 OpenAlex 作者 ID 时置 0,author_hindex_missing 置 1。
  - tier_ord:未分档语料统一按 PREPRINT(=1) 编码。
  - cited_by_count_openalex_log:无 OpenAlex 引用数时用语料自带 citation_count 近似。
  - same_discipline:语料与选题同学科时恒为 1。
  - retrieved_by_L:恒为 1(R 只能对 L 实际检索到的候选重排序,与训练评估的主口径一致)。
"""

import lightgbm as lgb
import numpy as np

CURRENT_YEAR = 2026  # 生成时点参考年,用于年份差特征(生成场景下没有"某篇历史综述发表年")

FEATURE_COLS = [
    "bm25_score", "bm25_rank_recip", "dense_score", "dense_rank_recip", "rrf_score",
    "candidate_year", "year_diff", "year_missing",
    "tier_ord", "cited_by_count_log", "cited_by_count_openalex_log",
    "is_self_citation", "self_citation_reliable", "max_author_hindex", "mean_author_hindex",
    "author_hindex_missing", "n_authors",
    "abstract_len_words", "title_len_words", "same_discipline",
    "retrieved_by_L",
]

_MODEL_SINGLETON = None


def get_model(model_path="models/lgb_lambdamart_model.txt"):
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = lgb.Booster(model_file=model_path)
    return _MODEL_SINGLETON


def _features_for_paper(p, index):
    scores = getattr(index, "_last_search_scores", {}).get(p.doc_id, {})
    year_diff = (CURRENT_YEAR - p.year) if p.year else None
    cbc = index.citation_count(p.doc_id) if hasattr(index, "citation_count") else 0
    cbc_log = float(np.log1p(cbc)) if cbc else 0.0
    return {
        "bm25_score": scores.get("bm25_score", 0.0),
        "bm25_rank_recip": 1.0 / (scores["bm25_rank"] + 1) if scores.get("bm25_rank", -1) >= 0 else 0.0,
        "dense_score": scores.get("dense_score", 0.0),
        "dense_rank_recip": 1.0 / (scores["dense_rank"] + 1) if scores.get("dense_rank", -1) >= 0 else 0.0,
        "rrf_score": scores.get("rrf_score", 0.0),
        "candidate_year": p.year if p.year else 0,
        "year_diff": year_diff if year_diff is not None else -1,
        "year_missing": int(year_diff is None),
        "tier_ord": 1,  # PREPRINT
        "cited_by_count_log": cbc_log,
        "cited_by_count_openalex_log": cbc_log,
        "is_self_citation": 0,
        "self_citation_reliable": 0,
        "max_author_hindex": 0.0,
        "mean_author_hindex": 0.0,
        "author_hindex_missing": 1,
        "n_authors": len(p.authors) if p.authors else 0,
        "abstract_len_words": len((p.abstract or "").split()),
        "title_len_words": len((p.title or "").split()),
        "same_discipline": 1,
        "retrieved_by_L": 1,
    }


def op_R_learned(state, index, model=None):
    """学习到的排序算子:用 LightGBM LambdaMART 模型给 state.D 里的候选打分排序,替代 operators.op_R。"""
    model = model or get_model()
    papers = state.D
    if not papers:
        return state
    X = np.array([[_features_for_paper(p, index)[c] for c in FEATURE_COLS] for p in papers])
    scores = model.predict(X)
    for p, s in zip(papers, scores):
        p.score = float(s)
    state.D = [p for _, p in sorted(zip(scores, papers), key=lambda x: -x[0])]
    return state
