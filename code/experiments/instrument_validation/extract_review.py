"""
从 arXiv LaTeX 源码抽取"分节结构化"的综述文本，供仪器 J 探针的
扰动变体构造使用。

抽取逻辑为一套确定性的正则规则（章节/引用/浮动体处理），输出结构化形式：
每个顶层 section 保留为 (title, tokens) 的列表，tokens 是文本片段和引用
标记交替的序列，这样才能精确做"引用打乱/删除/替换"而不破坏其余文字。
"""
from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CiteToken:
    keys: list


@dataclass
class Section:
    title: str
    level: int  # 1=section, 2=subsection, 3=subsubsection
    tokens: list  # 交替的 str（正文片段）与 CiteToken

    def all_keys(self) -> list:
        keys = []
        for t in self.tokens:
            if isinstance(t, CiteToken):
                keys.extend(t.keys)
        return keys

    def render(self) -> str:
        parts = []
        for t in self.tokens:
            if isinstance(t, CiteToken):
                parts.append("[" + "; ".join(t.keys) + "]")
            else:
                parts.append(t)
        return "".join(parts)


CITE_CMD_RE = re.compile(r"\\(?:cite[tp]?|citealp|citeauthor|citeyear|Cite)\*?(?:\[[^\]]*\])?\{([^}]+)\}")


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def _strip_floats(tex: str) -> str:
    for env in ("figure", "table", "algorithm", "table*", "figure*"):
        tex = re.sub(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", "\n", tex, flags=re.S)
    return tex


def _clean_inline_commands(text: str) -> str:
    text = re.sub(r"\\(?:textbf|emph|textit|texttt)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:ref|label|url|href)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = text.replace("{", "").replace("}", "").replace("$", "").replace("~", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _tokenize_with_citations(raw: str) -> list:
    tokens = []
    pos = 0
    for m in CITE_CMD_RE.finditer(raw):
        if m.start() > pos:
            chunk = _clean_inline_commands(raw[pos:m.start()])
            if chunk:
                tokens.append(chunk)
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        tokens.append(CiteToken(keys))
        pos = m.end()
    tail = _clean_inline_commands(raw[pos:])
    if tail:
        tokens.append(tail)
    return tokens


SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^{}]+)\}")


def extract_sections(tex: str) -> list:
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, flags=re.S)
    body = m.group(1) if m else tex
    body = _strip_comments(body)
    body = _strip_floats(body)

    level_map = {"section": 1, "subsection": 2, "subsubsection": 3}
    matches = list(SECTION_RE.finditer(body))
    sections = []
    for i, m in enumerate(matches):
        level = level_map[m.group(1)]
        if level != 1:
            continue  # Phase 0 探针只用顶层 section 做结构操作，足够构造 sibling 扰动
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[start:end]
        tokens = _tokenize_with_citations(raw)
        sections.append(Section(title=title, level=level, tokens=tokens))
    return sections


def find_main_tex(extract_dir: Path) -> Path:
    candidates = [p for p in extract_dir.rglob("*.tex")]
    scored = []
    for p in candidates:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "\\begin{document}" in txt and "\\documentclass" in txt:
            scored.append((len(txt), p))
    if not scored:
        raise FileNotFoundError(f"未找到含 \\begin{{document}} 的主 tex 文件于 {extract_dir}")
    scored.sort(reverse=True)
    return scored[0][1]


def find_bbl(extract_dir: Path, main_tex: Path) -> Optional[Path]:
    candidate = main_tex.with_suffix(".bbl")
    if candidate.exists():
        return candidate
    bbls = list(extract_dir.rglob("*.bbl"))
    return bbls[0] if bbls else None


def extract_bib_snippets(bbl_path: Optional[Path]) -> dict:
    """从 .bbl 提取 {key: 一句话摘要文本}，用于渲染参考文献列表和"引用替换"时挑选真实存在的 key。"""
    if bbl_path is None or not bbl_path.exists():
        return {}
    raw = bbl_path.read_text(encoding="utf-8", errors="ignore")
    entries = {}
    parts = re.split(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", raw)
    # re.split 交替产出 [pre, key1, body1, key2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip()
        body = parts[i + 1]
        body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda m: m.group(1) or " ", body)
        body = body.replace("{", "").replace("}", "")
        body = re.sub(r"\s+", " ", body).strip()
        entries[key] = body[:220]
    return entries


def load_review(tarball_path: str, work_dir: str) -> dict:
    work = Path(work_dir)
    extract_dir = work / "extract"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with tarfile.open(tarball_path, "r:*") as tar:
            tar.extractall(extract_dir)
    main_tex_path = find_main_tex(extract_dir)
    tex = main_tex_path.read_text(encoding="utf-8", errors="ignore")
    tex_no_comments = _strip_comments(tex)
    sections = extract_sections(tex)
    bbl_path = find_bbl(extract_dir, main_tex_path)
    bib = extract_bib_snippets(bbl_path)
    title_m = re.search(r"\\title\{([^{}]+)\}", tex_no_comments)
    title = _clean_inline_commands(title_m.group(1)) if title_m else Path(tarball_path).stem
    return {"title": title.strip(), "sections": sections, "bib": bib, "source": tarball_path}


def render_document(title: str, sections: list, bib: dict, used_keys: Optional[set] = None) -> str:
    parts = [f"# {title}\n"]
    for s in sections:
        parts.append(f"\n## {s.title}\n\n{s.render()}\n")
    if used_keys is None:
        used_keys = set()
        for s in sections:
            used_keys.update(s.all_keys())
    if bib:
        parts.append("\n## References\n")
        for k in sorted(used_keys):
            if k in bib:
                parts.append(f"- [{k}] {bib[k]}")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    doc = load_review(sys.argv[1], sys.argv[2])
    print("title:", doc["title"])
    print("n_sections:", len(doc["sections"]))
    for s in doc["sections"]:
        print(f"  - {s.title} ({len(s.all_keys())} cite keys, {len(s.render().split())} words)")
    rendered = render_document(doc["title"], doc["sections"], doc["bib"])
    print("total words:", len(rendered.split()))
