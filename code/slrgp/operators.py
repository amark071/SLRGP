"""
9 个异构认知算子的参考实现(对应论文 Methods「State representation and operator contracts」):

  E expand    语义   q -> q
  L locate    混合   q -> D
  F filter    确定   D -> D
  R rank      确定   D -> D
  O organize  语义   (D,mu) -> Gamma
  V validate  确定   (D,Gamma) -> kappa
  P format    确定   Gamma -> x
  W write     语义   x -> x   (此处只含叶子局部写作 LocalWrite,递归结构见 control.py)
  C reffilter 确定   x -> x

注:本参考实现仅依赖论文的形式化定义,不依赖任何外部生产系统代码。
"""
import re


# ---------- E: expand（语义算子，LLM） ----------

def op_E(state, llm):
    prompt = f"""You are helping construct a search strategy for an academic literature review.

Research topic: "{state.q['seed']}"

List 8-12 English search terms/concepts that expand this topic for retrieval: include synonyms,
closely related theoretical frameworks, key methodology terms, and important subtopics.

Output ONLY valid JSON, no other text, no markdown fences:
{{"terms": ["term1", "term2", ...]}}"""
    result = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=400, temperature=0.3)
    state.q["terms"] = list(result.get("terms", []))
    return state


# ---------- L: locate（混合算子：字面匹配 + 词汇重叠打分） ----------
# 参考实现的检索侧:真实语料上的双路 RRF 融合实现见 retrieval.py;
# 本演示语料很小(几十篇同主题论文),用词汇重叠打分近似排序效果,
# 目的是验证算子接口/控制流,而非验证检索质量本身。

def op_L(state, corpus, top_n=None):
    top_n = top_n if top_n is not None else state.meta.get("top_n", 20)
    terms = [state.q["seed"]] + state.q.get("terms", [])
    query_tokens = set(w.lower() for t in terms for w in re.findall(r"[a-zA-Z]+", t))
    scored = []
    for p in corpus:
        text = (p.title + " " + p.abstract).lower()
        text_tokens = set(re.findall(r"[a-zA-Z]+", text))
        overlap = len(query_tokens & text_tokens)
        if overlap > 0:
            scored.append((overlap, p))
    scored.sort(key=lambda x: -x[0])
    state.D = [p for _, p in scored[:top_n]]
    return state


# ---------- F: filter（确定性算子：资格门槛） ----------

def op_F(state, min_abs_len=None, allowed_tiers=("T1", "T2", "T3", "PREPRINT")):
    min_abs_len = min_abs_len if min_abs_len is not None else state.meta.get("min_abs_len", 200)
    state.D = [p for p in state.D if len(p.abstract) >= min_abs_len and p.tier in allowed_tiers]
    return state


# ---------- R: rank（确定性算子，纯公式复合打分） ----------

_TIER_WEIGHT = {"T1": 1.0, "T2": 0.7, "T3": 0.4, "PREPRINT": 0.25}


def op_R(state):
    terms = set(
        w.lower()
        for t in ([state.q["seed"]] + state.q.get("terms", []))
        for w in re.findall(r"[a-zA-Z]+", t)
    )
    years = [p.year for p in state.D if p.year]
    max_year = max(years) if years else 2026
    for p in state.D:
        text_tokens = set(re.findall(r"[a-zA-Z]+", (p.title + " " + p.abstract).lower()))
        kw_overlap = len(terms & text_tokens) / max(len(terms), 1)
        recency = 1.0 - min(max(max_year - (p.year or max_year), 0) / 20.0, 1.0)
        tier_score = _TIER_WEIGHT.get(p.tier, 0.4)
        p.score = 0.5 * kw_overlap + 0.25 * recency + 0.25 * tier_score
    state.D.sort(key=lambda p: -p.score)
    return state


# ---------- O: organize（语义算子，核心差异算子） ----------
# allowed_dimensions 的第 7 类 complexity_progression 由组织图式学习实验(论文实验 2-O)
# 数据驱动新增:在同粒度(k=6)强制对比下,68.2% 的训练节点被判定为专家原 6 类之外的
# "由浅入深/基础到应用"复杂度递进维度,是样本中最主要的未覆盖组织原则。

def op_O(state, llm, top_k=15):
    papers = state.D[:top_k]
    if not papers:
        state.Gamma = {"dimension": "none", "groups": []}
        return state
    listing = "\n".join(
        f"{i + 1}. {p.title}\n   Abstract: {p.abstract[:280]}"
        for i, p in enumerate(papers)
    )
    prompt = f"""You are organizing literature for a review section on: "{state.q['seed']}"

Given these {len(papers)} paper abstracts (numbered), choose the single best organizational
dimension for grouping them, then assign each paper to the group it is most topically
relevant to (4-6 groups). Every group must be a genuinely coherent, topically related
cluster of papers under the chosen dimension. If a paper is not clearly and substantively
related to "{state.q['seed']}" or does not fit any coherent group, LEAVE IT UNASSIGNED —
do not create a "catch-all"/"miscellaneous"/"other topics" group and do not force
off-topic or weakly-related papers into a group just to reach full coverage. It is
expected and fine for some papers to end up unassigned.

Allowed dimensions: "theme" (research sub-topic), "theoretical_perspective",
"methodology", "chronological_evolution", "debate_consensus", "analysis_level",
"complexity_progression" (foundational/simple to applied/complex exposition order,
independent of chronology or field history).

Papers:
{listing}

Output ONLY valid JSON, no other text, no markdown fences:
{{"dimension_chosen": "...", "groups": [{{"group_name": "...", "paper_indices": [1,3,5]}}]}}"""
    result = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=700, temperature=0.3)
    groups = []
    for g in result.get("groups", []):
        doc_ids = [papers[i - 1].doc_id for i in g.get("paper_indices", []) if 1 <= i <= len(papers)]
        if doc_ids:
            groups.append({"name": g.get("group_name", "Untitled"), "doc_ids": doc_ids})
    state.Gamma = {"dimension": result.get("dimension_chosen", "theme"), "groups": groups}
    return state


# ---------- V: validate（确定性算子，守卫谓词） ----------

def op_V(state, min_total=6, min_groups=2, min_balance=0.15):
    groups = state.Gamma.get("groups", [])
    sizes = [len(g["doc_ids"]) for g in groups]
    total_assigned = sum(sizes)
    balance = (min(sizes) / max(sizes)) if sizes and max(sizes) > 0 else 0.0
    state.kappa = {
        "min_total": total_assigned >= min_total,
        "min_groups": len(groups) >= min_groups,
        "balance": balance >= min_balance,
    }
    return state


def is_valid(state):
    return bool(state.kappa) and all(state.kappa.values())


# ---------- P: format（确定性算子，只读素材卡组装） ----------

def op_P(state, corpus_by_id, classics=None):
    parts = []
    for g in state.Gamma.get("groups", []):
        parts.append(f"## {g['name']}")
        for doc_id in g["doc_ids"]:
            p = corpus_by_id.get(doc_id)
            if not p:
                continue
            authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
            year_str = f" ({p.year})" if p.year else ""
            parts.append(f"[{doc_id}] {authors}{year_str} — \"{p.title}\": {p.abstract[:400]}")
    if classics:
        parts.append("## Foundational works (theoretical grounding)")
        parts.extend(classics)
    state.x["context"] = "\n".join(parts)
    return state


# ---------- W: write（语义算子，叶子局部写作 LocalWrite） ----------
# 递归展开（Render/Descend/Merge）见 control.py；这里只是叶子节点的"就地写作"。

def op_W_leaf(state, llm, target_words=500):
    prompt = f"""Write a literature review section (~{target_words} words) on the topic
"{state.q['seed']}" using ONLY the material below (read-only source, do not invent content
or cite anything not listed here).

Rules:
- Organize the prose following the group structure given in the material.
- Cite papers inline using their bracket IDs exactly as given, e.g. [2301.03516].
- Be comparative/critical where the material supports it, not just a list of summaries.
- Do not cite any paper ID not present in the material below.

MATERIAL (read-only):
{state.x['context']}
"""
    text = llm.chat([{"role": "user", "content": prompt}], max_tokens=1200, temperature=0.4)
    state.x["text"] = text.strip()
    return state


# ---------- C: reffilter（确定性算子，引用一致性核对） ----------

def op_C(state, corpus_by_id, min_keep=5):
    text = state.x.get("text", "")
    cited = set(re.findall(r"\[([\w\.\-/]+)\]", text))
    valid_ids = sorted(d for d in cited if d in corpus_by_id)
    if len(valid_ids) < min_keep:
        valid_ids = sorted(set(g_id for g in state.Gamma.get("groups", []) for g_id in g["doc_ids"]))
    state.x["refs"] = valid_ids
    return state
