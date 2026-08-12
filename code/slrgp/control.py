"""
控制结构(对应论文 Methods「Guarded validation and termination」与「Structural recursion」)。
"""
from .state import SLRGPState
from .operators import op_E, op_L, op_F, op_R, op_O, op_V, op_P, op_W_leaf, is_valid


# ---------------- Solve：带守卫的组合 + 单调回溯 ----------------

def backtrack(state, round_idx):
    """单调放松：先扩大候选池 top_n，再放宽摘要长度门槛。"""
    state.meta["top_n"] = state.meta.get("top_n", 20) + 10
    state.meta["min_abs_len"] = max(state.meta.get("min_abs_len", 200) - 50, 0)
    return state


def solve(state, corpus, llm, t_max=3):
    """全链路 Solve：(O∘R∘F∘L)(s)，V 守卫，不满足则 Backtrack 重跑，直到 valid 或到达迭代上限。"""
    for i in range(t_max):
        state = op_L(state, corpus)
        state = op_F(state)
        state = op_R(state)
        state = op_O(state, llm)
        state = op_V(state)
        if is_valid(state):
            state.meta["solve_rounds"] = i + 1
            return state
        state = backtrack(state, i)
    state.meta["solve_rounds"] = t_max
    state.meta["solve_gave_up"] = True
    return state


def solve_for_mode(state, corpus, llm, t_max=3):
    """Descend 之后子问题的求解入口：按 meta['mode'] 区分部分/全递归（见 Methods「Structural recursion」）。"""
    if state.meta.get("mode") == "partial":
        # 部分递归：复用父层候选池，跳过 E/L/F/R，只重新按更细维度 O + 守卫 V
        for i in range(t_max):
            state = op_O(state, llm)
            state = op_V(state)
            if is_valid(state):
                state.meta["solve_rounds"] = i + 1
                return state
            # 分组太细导致不满足约束时，放宽 balance/min_total 要求而不是重新撒网
            state.meta["_relaxed"] = state.meta.get("_relaxed", 0) + 1
        return state
    else:
        # 全递归：重新撒网，先 E 再走完整 Solve
        state = op_E(state, llm)
        return solve(state, corpus, llm, t_max=t_max)


# ---------------- 递归展开：Render / Descend / Merge ----------------

def descend(parent, group, corpus_by_id, min_seed=4):
    """构造子问题状态：部分递归复用父层候选池 vs 全递归重新撒网。"""
    d_seed = [corpus_by_id[d] for d in group.get("doc_ids", []) if d in corpus_by_id]
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
    if len(d_seed) >= min_seed:
        return SLRGPState(
            q={"seed": child_seed, "terms": child_terms},
            D=d_seed, Gamma={}, kappa={}, x={},
            meta={
                "mode": "partial",
                "root_topic": parent.meta.get("root_topic", parent_seed),
                "parent_seed": parent_seed,
                "top_n": parent.meta.get("top_n", 20),
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
                "top_n": parent.meta.get("top_n", 20),
                "min_abs_len": parent.meta.get("min_abs_len", 200),
                "publication_cutoff_year": parent.meta.get("publication_cutoff_year"),
            },
        )


def merge(section_texts, parent_state):
    """自底向上合并 + 过渡衔接(参考实现用简单的标题拼装;学习式合并是对照实验的替换臂之一)。"""
    parts = []
    for name, text in section_texts:
        parts.append(f"### {name}\n\n{text}")
    return "\n\n".join(parts)


def render(state, corpus, corpus_by_id, llm, depth=0, d_max=1, theta_leaf=6):
    """
    Render(node, depth):
        if 停机条件: return LocalWrite(node)      # 叶子
        else: for each 子主题 g_j: Descend -> Solve -> Render(depth+1); return Merge(...)
    """
    if depth >= d_max or len(state.D) <= theta_leaf or not state.Gamma.get("groups"):
        state = op_P(state, corpus_by_id)
        state = op_W_leaf(state, llm)
        return state.x["text"]

    section_texts = []
    for g in state.Gamma["groups"]:
        sub = descend(state, g, corpus_by_id)
        sub = solve_for_mode(sub, corpus, llm)
        sub_text = render(sub, corpus, corpus_by_id, llm, depth=depth + 1, d_max=d_max, theta_leaf=theta_leaf)
        section_texts.append((g["name"], sub_text))
    return merge(section_texts, state)
