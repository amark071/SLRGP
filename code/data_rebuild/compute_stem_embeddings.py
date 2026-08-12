#!/usr/bin/env python3
"""
用本机 GPU (RTX 4080 SUPER) 跑 BAAI/bge-small-en-v1.5，给11个学科候选池的摘要生成embedding，
与既有社科语料的 embeddings_en.npy 同一模型/同一384维空间，可直接合并检索。

输出（每个学科一组，与社科语料格式对齐）：
  data/stem_embeddings/<discipline>_embeddings.npy        (float32, N x 384)
  data/stem_embeddings/<discipline>_docids.pkl            (list[str]，与embeddings行对应的arxiv_id)
"""
import glob
import os
import pickle


import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

POOL_DIR = "data/common/stem_pools"
OUT_DIR = "work/corpus_rebuild/stem_embeddings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 256


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    model = SentenceTransformer(MODEL_NAME, device=device)

    for f in sorted(glob.glob(f"{POOL_DIR}/*.parquet")):
        discipline = os.path.basename(f).replace(".parquet", "")
        out_emb_path = f"{OUT_DIR}/{discipline}_embeddings.npy"
        out_ids_path = f"{OUT_DIR}/{discipline}_docids.pkl"
        if os.path.exists(out_emb_path) and os.path.exists(out_ids_path):
            print(f"{discipline}: 已存在，跳过")
            continue

        df = pd.read_parquet(f, columns=["arxiv_id", "title", "abstract"])
        texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()
        print(f"{discipline}: {len(texts)} 篇，开始编码...")

        embeddings = model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)

        np.save(out_emb_path, embeddings)
        with open(out_ids_path, "wb") as fh:
            pickle.dump(df["arxiv_id"].tolist(), fh)

        print(f"{discipline}: 完成，shape={embeddings.shape} -> {out_emb_path}")

    print("\n全部学科 embedding 计算完成")


if __name__ == "__main__":
    main()
