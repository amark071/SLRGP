"""
切分与种子基础设施（对应论文 Methods「Seeds and splits」）。

说明：Python 内建 `hash()` 对 str 默认加盐、每次进程重启不一致，不能用于
需要跨机器/跨运行复现的切分判定。这里全部改用 SHA-256 派生的稳定整数/分桶，
并提供留痕（split manifest）落盘的标准写法。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence


def stable_hash_int(*parts: str) -> int:
    """把任意数量的字符串字段拼接后做 SHA-256，取前 8 字节转成非负整数。

    用法示例：stable_hash_int(article_id, node_path) 作为该节点的采样/切分种子。
    """
    key = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def stable_unit_interval(*parts: str) -> float:
    """把 stable_hash_int 映射到 [0, 1) 区间，用于按比例切分（如 80/20）。"""
    return stable_hash_int(*parts) / 2**64


def assign_split(key: str, ratios: dict, salt: str = "") -> str:
    """按累计区间稳定分配到某个切分桶。

    ratios: 例如 {"train": 0.8, "heldout": 0.2}（必须求和为 1，允许 >2 个桶）。
    salt: 同一批 key 需要用于不同切分目的时（如先按 review 切 train/test，
          再在 train 内部切 train/val），必须传不同 salt，否则会重复复用同一个随机数
          导致隐蔽的切分相关性。
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios 必须求和为 1，得到 {total}: {ratios}")
    u = stable_unit_interval(key, salt)
    cum = 0.0
    for bucket, r in ratios.items():
        cum += r
        if u < cum:
            return bucket
    return list(ratios.keys())[-1]  # 浮点边界兜底


def stable_seed_for(*parts: str, modulus: int = 2**31 - 1) -> int:
    """派生一个可安全传给 numpy/random 的 32 位种子。"""
    return stable_hash_int(*parts) % modulus


@dataclass
class SplitManifest:
    """切分清单：记录每个 unit 的最终归属，供入版本控制、审计和复现。"""
    experiment: str
    salt: str
    ratios: dict
    assignments: dict  # key -> bucket

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        payload = {
            "experiment": self.experiment,
            "salt": self.salt,
            "ratios": self.ratios,
            "n_units": len(self.assignments),
            "bucket_counts": {b: sum(1 for v in self.assignments.values() if v == b) for b in self.ratios},
            "assignments": self.assignments,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def build(cls, experiment: str, keys: Iterable[str], ratios: dict, salt: str = "") -> "SplitManifest":
        assignments = {k: assign_split(k, ratios, salt=salt or experiment) for k in keys}
        return cls(experiment=experiment, salt=salt or experiment, ratios=ratios, assignments=assignments)

    @classmethod
    def from_json(cls, path: str) -> "SplitManifest":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(experiment=payload["experiment"], salt=payload["salt"], ratios=payload["ratios"],
                    assignments=payload["assignments"])


def randomized_presentation_order(items: Sequence[str], *context_parts: str) -> list:
    """给定一组 item（如 blinded 系统标签），按 context（如 topic_id + judge_id）
    生成确定性但看似随机的呈现顺序，用于仪器 J 的呈现顺序随机化（§0.3 Blinding）。

    确定性的意义：同一 (topic, judge) 组合每次重跑得到同一个顺序,便于复现;
    但不同 (topic, judge) 组合之间顺序相互独立、不可预测。
    """
    seed = stable_seed_for(*context_parts)
    import random
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def self_check():
    # 稳定性：同样输入多次调用结果一致
    a1 = stable_hash_int("2306.01660", "node_0_1")
    a2 = stable_hash_int("2306.01660", "node_0_1")
    assert a1 == a2, "stable_hash_int 应当确定性"

    # 不同 salt 产生独立切分（用于同一批 key 的多层切分，如 train/test 再 train/val）
    keys = [f"review_{i}" for i in range(2000)]
    split1 = SplitManifest.build("demo_outer", keys, {"train": 0.8, "heldout": 0.2}, salt="outer")
    train_keys = [k for k, v in split1.assignments.items() if v == "train"]
    split2 = SplitManifest.build("demo_inner", train_keys, {"train": 0.8, "val": 0.2}, salt="inner")
    counts1 = {b: sum(1 for v in split1.assignments.values() if v == b) for b in ["train", "heldout"]}
    counts2 = {b: sum(1 for v in split2.assignments.values() if v == b) for b in ["train", "val"]}
    print("outer split counts (expect ~1600/400):", counts1)
    print("inner split counts (expect ~1280/320):", counts2)

    order1 = randomized_presentation_order(["A", "B", "C", "D"], "topic_1", "judge_gpt55")
    order2 = randomized_presentation_order(["A", "B", "C", "D"], "topic_1", "judge_gpt55")
    order3 = randomized_presentation_order(["A", "B", "C", "D"], "topic_2", "judge_gpt55")
    assert order1 == order2, "同一 context 应当得到同一顺序"
    print("order for (topic_1, gpt55):", order1)
    print("order for (topic_2, gpt55):", order3, "(expect likely different from topic_1's order)")
    print("self_check passed.")


if __name__ == "__main__":
    self_check()
