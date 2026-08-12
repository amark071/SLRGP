"""
按论文 Methods「Instrument validation」构造仪器 J 的全部扰动变体。

每篇源文档产出 8 个变体：
  control                原文（section 顺序、引用、篇幅均不变）
  structure_shuffled      顶层 section 顺序打乱，每个 section 内部文字/引用不变
  citation_shuffled       引用标记的位置不变，但把各处引用的 key 列表在文档内互相交换
  citation_deleted        删除一部分"承重"引用标记（整段唯一引用 或 高频关键 key）
  citation_replaced       把被选中的引用替换成文档里其它、真实存在但语境不相关的 key
  shortened               每个 section 保留约 50% 文字（掐头去尾各段落等比截断）
  lengthened_redundant    每个 section 的段落机械重复一次，制造"注水"但不增加信息
  synthesis_flattened     固定词典替换比较/对比/批判类话语标记词为中性等长词，
                          结构、引用、总篇幅基本不变，只移除"跨文献比较/评判"的表层信号，
                          用于直接验证 critical_synthesis 维度（其余变体均未触及该维度）

每个 (文档, 条件, 裁判) 组合独立重复 2 次，由调用方负责。
"""
from __future__ import annotations

import copy
import random
import re

from extract_review import CiteToken, render_document


def structure_shuffled(sections: list, seed: int) -> list:
    rng = random.Random(seed)
    shuffled = sections[:]
    while len(shuffled) > 1:
        order = list(range(len(shuffled)))
        rng.shuffle(order)
        if order != list(range(len(shuffled))):
            break
    else:
        order = list(range(len(shuffled)))
    return [shuffled[i] for i in order]


def _citation_positions(sections: list):
    """返回 [(section_idx, token_idx, keys)]，仅含非空 key 列表的引用 token。"""
    positions = []
    for si, s in enumerate(sections):
        for ti, t in enumerate(s.tokens):
            if isinstance(t, CiteToken) and t.keys:
                positions.append((si, ti, t.keys))
    return positions


def citation_shuffled(sections: list, seed: int) -> list:
    sections = copy.deepcopy(sections)
    positions = _citation_positions(sections)
    if len(positions) < 2:
        return sections
    rng = random.Random(seed)
    key_lists = [p[2] for p in positions]
    shuffled_key_lists = key_lists[:]
    rng.shuffle(shuffled_key_lists)
    # 避免退化成恒等置换
    if shuffled_key_lists == key_lists and len(key_lists) > 1:
        shuffled_key_lists[0], shuffled_key_lists[1] = shuffled_key_lists[1], shuffled_key_lists[0]
    for (si, ti, _), new_keys in zip(positions, shuffled_key_lists):
        sections[si].tokens[ti] = CiteToken(new_keys)
    return sections


def citation_deleted(sections: list, seed: int, frac: float = 0.4) -> list:
    """删除一部分引用标记（直接替换为空文本token，即从正文中消失），偏向删"孤立/低频"引用
    （更像是删掉了某个具体claim唯一支撑的文献,即"load-bearing")。"""
    sections = copy.deepcopy(sections)
    positions = _citation_positions(sections)
    if not positions:
        return sections
    key_freq = {}
    for _, _, keys in positions:
        for k in keys:
            key_freq[k] = key_freq.get(k, 0) + 1
    # 优先删除仅出现一次的 key 所在的引用 token（更可能是某个具体claim的唯一支撑）
    scored = sorted(positions, key=lambda p: min(key_freq[k] for k in p[2]))
    n_delete = max(1, int(len(scored) * frac))
    rng = random.Random(seed)
    to_delete = scored[:n_delete]
    rng.shuffle(to_delete)
    for si, ti, _ in to_delete:
        sections[si].tokens[ti] = CiteToken([])  # render() 对空 keys 会渲染为 "[]",空 CiteToken 在下一步统一移除
    for s in sections:
        s.tokens = [t for t in s.tokens if not (isinstance(t, CiteToken) and not t.keys)]
    return sections


def citation_replaced(sections: list, seed: int, frac: float = 0.4) -> list:
    """把选中引用替换为文档中"距离较远"章节的真实 key（保证是bib里真实存在的文献,
    但大概率与当前claim的语境不相关）。"""
    sections = copy.deepcopy(sections)
    positions = _citation_positions(sections)
    if len(positions) < 2:
        return sections
    rng = random.Random(seed)
    n_replace = max(1, int(len(positions) * frac))
    targets = rng.sample(positions, n_replace)
    all_section_keys = {si: sections[si].all_keys() for si in range(len(sections))}
    for si, ti, orig_keys in targets:
        # 优先从"非相邻"的其它 section 取 key，制造语境不相关的替换
        far_sections = [j for j in range(len(sections)) if abs(j - si) >= max(2, len(sections) // 3) and all_section_keys[j]]
        pool_sections = far_sections if far_sections else [j for j in range(len(sections)) if j != si and all_section_keys[j]]
        if not pool_sections:
            continue
        donor = rng.choice(pool_sections)
        donor_keys = all_section_keys[donor]
        new_keys = rng.sample(donor_keys, k=min(len(orig_keys), len(donor_keys))) or [rng.choice(donor_keys)]
        sections[si].tokens[ti] = CiteToken(new_keys)
    return sections


_PARA_SPLIT_RE = re.compile(r"(\n\s*\n)")


def shortened(sections: list, keep_frac: float = 0.5) -> list:
    sections = copy.deepcopy(sections)
    for s in sections:
        text_tokens = [t for t in s.tokens if isinstance(t, str)]
        n_keep = max(1, int(len(s.tokens) * keep_frac))
        # 按 token 顺序保留前 n_keep 个（含穿插的引用），近似"从中间截断到后半部分被砍掉"
        s.tokens = s.tokens[:n_keep] if n_keep < len(s.tokens) else s.tokens
    return sections


def lengthened_redundant(sections: list) -> list:
    """机械复读:每个非引用文本token紧跟着重复自身一次(不引入新信息的"注水"),
    用于测试裁判是否奖励纯粹的字数堆砌。"""
    sections = copy.deepcopy(sections)
    for s in sections:
        new_tokens = []
        for t in s.tokens:
            new_tokens.append(t)
            if isinstance(t, str) and len(t.split()) > 8:
                new_tokens.append(" " + t.strip())
        s.tokens = new_tokens
    return sections


# 固定词典：比较/对比/批判类话语标记 -> 中性等长（尽量同词性、同数量音节）替代词。
# 全部为规则替换（非 LLM 改写），保证操纵可审计、可复现；只针对"跨文献比较评判"的
# 表层连接词/形容词，不触碰引用标记、专有名词或数值。
_SYNTHESIS_MARKERS = [
    (r"\bhowever,\s*", "additionally, "),
    (r"\bHowever,\s*", "Additionally, "),
    (r"\bhowever\b", "additionally"),
    (r"\bHowever\b", "Additionally"),
    (r"\bin contrast,\s*", "similarly, "),
    (r"\bIn contrast,\s*", "Similarly, "),
    (r"\bin contrast to\b", "along with"),
    (r"\bIn contrast to\b", "Along with"),
    (r"\bby contrast,\s*", "similarly, "),
    (r"\bBy contrast,\s*", "Similarly, "),
    (r"\bon the other hand,\s*", "in addition, "),
    (r"\bOn the other hand,\s*", "In addition, "),
    (r"\bunlike\b", "similar to"),
    (r"\bUnlike\b", "Similar to"),
    (r"\bwhereas\b", "and"),
    (r"\bWhereas\b", "And"),
    (r"\bconversely,\s*", "similarly, "),
    (r"\bConversely,\s*", "Similarly, "),
    (r"\bnevertheless,\s*", "additionally, "),
    (r"\bNevertheless,\s*", "Additionally, "),
    (r"\bnonetheless,\s*", "additionally, "),
    (r"\bNonetheless,\s*", "Additionally, "),
    (r"\bdespite\b", "alongside"),
    (r"\bDespite\b", "Alongside"),
    (r"\balthough\b", "and"),
    (r"\bAlthough\b", "And"),
    (r"\boutperforms?\b", "reports results alongside"),
    (r"\boutperforming\b", "reporting results alongside"),
    (r"\bsurpasses?\b", "reports results alongside"),
    (r"\bsuperior to\b", "presented alongside"),
    (r"\binferior to\b", "presented alongside"),
    (r"\badvantage(s)? over\b", r"difference\1 from"),
    (r"\bfails? to\b", "does not"),
    (r"\bfailing to\b", "not"),
    (r"\blimitations?\b", "aspects"),
    (r"\bLimitations?\b", "Aspects"),
    (r"\bdrawbacks?\b", "aspects"),
    (r"\bshortcomings?\b", "aspects"),
    (r"\bweaknesses?\b", "aspects"),
    (r"\btrade-offs?\b", "aspects"),
    (r"\bcontradicts?\b", "differs from"),
    (r"\bcontradicting\b", "differing from"),
    (r"\bcritically,\s*", ""),
    (r"\bCritically,\s*", ""),
    (r"\bimportantly,\s*", ""),
    (r"\bImportantly,\s*", ""),
    (r"\bnotably,\s*", ""),
    (r"\bNotably,\s*", ""),
    (r"\bcrucially,\s*", ""),
    (r"\bCrucially,\s*", ""),
    (r"\bwe argue that\b", "one view holds that"),
    (r"\bWe argue that\b", "One view holds that"),
    (r"\bwe contend\b", "one view holds"),
    (r"\bWe contend\b", "One view holds"),
    (r"\bcompared (to|with)\b", r"presented alongside \1"),
    (r"\bCompared (to|with)\b", r"Presented alongside \1"),
    (r"\bin comparison (to|with)\b", r"alongside \1"),
    (r"\bIn comparison (to|with)\b", r"Alongside \1"),
    (r"\bas opposed to\b", "along with"),
    (r"\bAs opposed to\b", "Along with"),
    (r"\bat odds with\b", "alongside"),
    (r"\bconflicts? with\b", "differs from"),
    (r"\bbetter than\b", "presented alongside"),
    (r"\bBetter than\b", "Presented alongside"),
    (r"\bworse than\b", "presented alongside"),
    (r"\bWorse than\b", "Presented alongside"),
    (r"\bmore effective than\b", "presented alongside"),
    (r"\bless effective than\b", "presented alongside"),
    (r"\bmore accurate than\b", "presented alongside"),
    (r"\bless accurate than\b", "presented alongside"),
    (r"\bmore robust than\b", "presented alongside"),
    (r"\bdiffers? significantly from\b", "is presented alongside"),
    (r"\bdiffers? from\b", "is presented alongside"),
    (r"\bdiffering from\b", "presented alongside"),
]


def synthesis_flattened_text(text: str) -> tuple:
    """对已渲染的正文做固定词典替换，返回 (新文本, 命中次数)。"""
    n_hits = 0
    for pattern, repl in _SYNTHESIS_MARKERS:
        text, k = re.subn(pattern, repl, text)
        n_hits += k
    return text, n_hits


VARIANT_BUILDERS = {
    "control": lambda sections, seed: sections,
    "structure_shuffled": structure_shuffled,
    "citation_shuffled": citation_shuffled,
    "citation_deleted": lambda sections, seed: citation_deleted(sections, seed),
    "citation_replaced": lambda sections, seed: citation_replaced(sections, seed),
    "shortened": lambda sections, seed: shortened(sections),
    "lengthened_redundant": lambda sections, seed: lengthened_redundant(sections),
}


def build_all_variants(doc: dict, seed: int = 42) -> dict:
    out = {}
    for name, builder in VARIANT_BUILDERS.items():
        sections = builder(doc["sections"], seed)
        used_keys = set()
        for s in doc["sections"]:
            used_keys.update(s.all_keys())  # 参考文献列表固定用原文引用集合，保证跨变体一致
        rendered = render_document(doc["title"], sections, doc["bib"], used_keys=used_keys)
        out[name] = rendered
    # synthesis_flattened 是对 control 渲染结果做词典级替换的后处理变体，不改变 section 结构
    flattened, n_hits = synthesis_flattened_text(out["control"])
    out["synthesis_flattened"] = flattened
    out["_synthesis_flattened_hits"] = n_hits  # 供调用方记录操纵检查统计，不作为文档送去打分
    return out


if __name__ == "__main__":
    import sys
    from extract_review import load_review
    doc = load_review(sys.argv[1], sys.argv[2])
    variants = build_all_variants(doc)
    for name, text in variants.items():
        print(f"{name}: {len(text.split())} words")
