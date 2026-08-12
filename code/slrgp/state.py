"""
状态空间 s = <q, D, Gamma, kappa, x>(对应论文 Methods「State representation and operator contracts」)。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    doc_id: str
    title: str
    abstract: str
    authors: list
    year: Optional[int] = None
    tier: str = "T3"          # T1/T2/T3，来自SJR/OpenAlex分级（arXiv预印本无tier时用中性档占位）
    venue: str = ""
    score: float = 0.0        # 由算子R写入


@dataclass
class SLRGPState:
    q: dict                              # {"seed": str, "terms": [str,...]}
    D: list                              # list[Paper]
    Gamma: dict                          # {"dimension": str, "groups": [{"name":str,"doc_ids":[str,...]}]}
    kappa: dict                          # {constraint_name: bool}
    x: dict                              # {"context": str, "text": str, "refs": [str,...]}
    meta: dict = field(default_factory=dict)   # 控制结构用的运行参数（top_n, min_abs_len, mode, depth...)
