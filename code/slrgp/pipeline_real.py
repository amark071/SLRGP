"""
真实语料版 Solve/Render:与 control.py 的演示版保持同一套算子代数,
只是把 op_L 换成 retrieval.op_L_hybrid(BM25+向量 RRF),corpus_by_id 换成
懒加载的 LazyCorpusById(按需从 unified_corpus.db 查询,无需把百万级语料
整个载入内存)。
"""
from .operators import op_F, op_O, op_P, op_R, op_V, op_W_leaf, is_valid
from .retrieval import op_L_hybrid
from .state import SLRGPState


class LazyCorpusById:
    """corpus_by_id 的懒加载替身：只在 op_P / op_C / descend 需要时才查库，带简单LRU。"""

    def __init__(self, index, cache_size=2000):
        self.index = index
        self._cache = {}
        self._cache_size = cache_size

    def get(self, doc_id, default=None):
        if doc_id in self._cache:
            return self._cache[doc_id]
        papers = self.index._fetch_papers([doc_id])
        p = papers.get(doc_id, default)
        if p is not None:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[doc_id] = p
        return p

    def __contains__(self, doc_id):
        return self.get(doc_id) is not None

    def __getitem__(self, doc_id):
        p = self.get(doc_id)
        if p is None:
            raise KeyError(doc_id)
        return p


def backtrack_real(state, round_idx):
    state.meta["top_n"] = state.meta.get("top_n", 50) + 20
    state.meta["min_abs_len"] = max(state.meta.get("min_abs_len", 200) - 50, 0)
    return state


def solve_real(state, llm, index, t_max=3, ranker=None, organizer=None, guarded=True):
    """ranker: 可选的排序算子替身,签名 ranker(state, index) -> state。
    默认 None 时用 operators.op_R(手写公式);传入 rank_model.op_R_learned 则换成
    学习式 LambdaMART 模型,控制流(E/L/F/O/V/回溯)完全不变,只替换 R 的实例化——
    这是论文实验 3(接口匹配替换)的注入点。

    organizer: 可选的组织算子替身,签名 organizer(state, llm) -> state,与 ranker 同一套
    "只换算子实例化"的注入模式;默认 None 时用 operators.op_O(LLM 语义分组)。实验 3 的
    "去 O"对照臂传入一个不做语义分组、把候选塞进单个 catch-all 分组的替身,测试"退化成罗列"的预测。

    guarded: 实验 3 的"去 V 守卫回溯"对照臂用;False 时仍会算 kappa(供日志观察)但不用它
    做是否重试的判据——O 只跑一轮就直接接受,不再触发 backtrack_real 的参数放宽重试。"""
    org_fn = organizer if organizer is not None else op_O
    for i in range(t_max):
        state = op_L_hybrid(state, index)
        state = op_F(state)
        state = ranker(state, index) if ranker is not None else op_R(state)
        state = org_fn(state, llm)
        state = op_V(state)
        if not guarded or is_valid(state):
            state.meta["solve_rounds"] = i + 1
            if not guarded:
                state.meta["guard_bypassed"] = True
            return state
        state = backtrack_real(state, i)
    state.meta["solve_rounds"] = t_max
    state.meta["solve_gave_up"] = True
    return state


def solve_for_mode_real(state, llm, index, t_max=3, ranker=None, organizer=None, guarded=True):
    org_fn = organizer if organizer is not None else op_O
    if state.meta.get("mode") == "partial":
        for i in range(t_max):
            state = org_fn(state, llm)
            state = op_V(state)
            if not guarded or is_valid(state):
                state.meta["solve_rounds"] = i + 1
                if not guarded:
                    state.meta["guard_bypassed"] = True
                return state
            state.meta["_relaxed"] = state.meta.get("_relaxed", 0) + 1
        return state
    else:
        from .operators import op_E
        state = op_E(state, llm)
        return solve_real(state, llm, index, t_max=t_max, ranker=ranker, organizer=organizer, guarded=guarded)


def descend_real(parent, group, corpus_by_id, min_seed=4, trace=None, node_path="root"):
    """Create a parent-aware child state for partial or full re-entry.

    Replacing the review topic with a short group label would let sparse child
    groups retrieve an unrelated literature, so both branches preserve the
    parent topic and deployment parameters.
    """
    d_seed = []
    for doc_id in group.get("doc_ids", []):
        paper = corpus_by_id.get(doc_id)
        if paper is not None:
            d_seed.append(paper)
    parent_seed = parent.q.get("seed", "")
    group_name = group.get("name", "subtopic")
    child_seed = (
        f"{parent_seed} — {group_name}"
        if parent_seed and group_name.lower() not in parent_seed.lower()
        else group_name
    )
    child_terms = list(
        dict.fromkeys(
            term
            for term in [*parent.q.get("terms", []), group_name, parent_seed]
            if term
        )
    )
    mode = "partial" if len(d_seed) >= min_seed else "full"
    if trace is not None:
        trace.setdefault("descend_events", []).append(
            {
                "node_path": node_path,
                "parent_seed": parent_seed,
                "group_name": group_name,
                "child_seed": child_seed,
                "mode": mode,
                "n_seed_docs": len(d_seed),
                "seed_doc_ids": [getattr(paper, "doc_id", None) for paper in d_seed],
            }
        )
    if len(d_seed) >= min_seed:
        return SLRGPState(
            q={"seed": child_seed, "terms": child_terms},
            D=d_seed, Gamma={}, kappa={}, x={},
            meta={
                "mode": "partial",
                "root_topic": parent.meta.get("root_topic", parent_seed),
                "parent_seed": parent_seed,
                "top_n": parent.meta.get("top_n", 50),
                "min_abs_len": parent.meta.get("min_abs_len", 200),
                "publication_cutoff_year": parent.meta.get("publication_cutoff_year"),
            },
        )
    else:
        return SLRGPState(
            q={"seed": child_seed, "terms": child_terms},
            D=[], Gamma={}, kappa={}, x={},
            meta={
                "mode": "full",
                "root_topic": parent.meta.get("root_topic", parent_seed),
                "parent_seed": parent_seed,
                "top_n": parent.meta.get("top_n", 50),
                "min_abs_len": parent.meta.get("min_abs_len", 200),
                "publication_cutoff_year": parent.meta.get("publication_cutoff_year"),
            },
        )


def merge_real(section_texts, depth=0):
    # depth->标题级别编码:depth=0 时用"### "(与历史输出向后兼容),更深层级依次降一级(####/#####/...),
    # 封顶到 markdown 最大的 6 级;让渲染出的标题树直接反映递归深度,供结构完整性度量从文本
    # 解析标题树深度,不改变任何算子/控制流逻辑。
    hlevel = "#" * min(depth + 3, 6)
    parts = []
    seen = {}
    for name, text in section_texts:
        base = (name or "Section").strip()
        key = base.casefold()
        seen[key] = seen.get(key, 0) + 1
        final_name = base if seen[key] == 1 else f"{base} ({seen[key]})"
        parts.append(f"{hlevel} {final_name}\n\n{text}")
    return "\n\n".join(parts)


def render_real(state, index, llm, depth=0, d_max=1, theta_leaf=6, corpus_by_id=None, ranker=None, organizer=None, guarded=True):
    corpus_by_id = corpus_by_id or LazyCorpusById(index)
    if depth >= d_max or len(state.D) <= theta_leaf or not state.Gamma.get("groups"):
        state = op_P(state, corpus_by_id)
        state = op_W_leaf(state, llm)
        return state.x["text"]

    section_texts = []
    for g in state.Gamma["groups"]:
        sub = descend_real(state, g, corpus_by_id)
        sub = solve_for_mode_real(sub, llm, index, ranker=ranker, organizer=organizer, guarded=guarded)
        sub_text = render_real(sub, index, llm, depth=depth + 1, d_max=d_max, theta_leaf=theta_leaf, corpus_by_id=corpus_by_id, ranker=ranker, organizer=organizer, guarded=guarded)
        section_texts.append((g["name"], sub_text))
    return merge_real(section_texts, depth=depth)
