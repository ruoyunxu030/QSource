#!/usr/bin/env python3
"""Build Amazon Reviews 2023 category npz files (BGE embeddings + PCA).

Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
Encoder: https://huggingface.co/BAAI/bge-base-en-v1.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 20261017
N_SAMPLE = 20000
PCA_DIM = 32
CATEGORIES = [
    "All_Beauty", "Video_Games", "Musical_Instruments", "Software",
    "Industrial_and_Scientific", "Appliances", "Amazon_Fashion",
    "Baby_Products", "Handmade_Products", "Health_and_Personal_Care",
]
HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
ENCODER = "BAAI/bge-base-en-v1.5"


def _split(n, rng):
    idx = rng.permutation(n)
    n_sell = int(0.6 * n)
    n_query = int(0.2 * n)
    return idx[:n_sell], idx[n_sell:n_sell + n_query], idx[n_sell + n_query:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "amazon"))
    ap.add_argument("--raw", default="")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit("pip install 'qsource[data]' for Amazon preparation") from e

    raw = Path(args.raw) if args.raw else out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    texts, ratings, cats = [], [], []
    rng = np.random.default_rng(SEED)
    for name in CATEGORIES:
        rel = f"raw/review_categories/{name}.jsonl"
        path = hf_hub_download(HF_DATASET, rel, repo_type="dataset", local_dir=str(raw))
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                t = (rec.get("title") or "") + " " + (rec.get("text") or "")
                y = rec.get("rating")
                if not t.strip() or y is None:
                    continue
                rows.append((t.strip(), float(y)))
                if len(rows) >= N_SAMPLE * 4:
                    break
        pick = rng.choice(len(rows), size=min(N_SAMPLE, len(rows)), replace=False)
        for i in pick:
            texts.append(rows[i][0])
            ratings.append(rows[i][1])
            cats.append(name)
        print(name, "n", min(N_SAMPLE, len(rows)), flush=True)

    model = SentenceTransformer(ENCODER)
    emb = np.asarray(model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True), float)
    y = np.asarray(ratings, float)
    cat_arr = np.asarray(cats)
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    Z = StandardScaler().fit_transform(pca.fit_transform(emb))
    X = np.hstack([Z, np.ones((len(Z), 1))])
    per = {}
    for name in CATEGORIES:
        m = cat_arr == name
        Xi, yi = X[m], y[m]
        rng_c = np.random.default_rng([SEED, sum(ord(c) for c in name)])
        sell, query, test = _split(len(Xi), rng_c)
        np.savez_compressed(
            out / f"cat_bge-base-en-v1.5_{name}.npz",
            X_sell=Xi[sell], y_sell=yi[sell],
            X_query=Xi[query], y_query=yi[query],
            X_test=Xi[test], y_test=yi[test],
        )
        per[name] = dict(n=int(len(Xi)), n_sell=int(len(sell)), n_query=int(len(query)), n_test=int(len(test)))
    from sklearn.linear_model import Ridge

    ridge_lambda = 0.1
    Xs = np.vstack([np.load(out / f"cat_bge-base-en-v1.5_{n}.npz")["X_sell"] for n in CATEGORIES])
    ys = np.concatenate([np.load(out / f"cat_bge-base-en-v1.5_{n}.npz")["y_sell"] for n in CATEGORIES])
    pred = Ridge(alpha=ridge_lambda, fit_intercept=False).fit(Xs, ys).predict(Xs)
    sigma2 = float(np.mean((ys - pred) ** 2))
    meta = dict(
        dataset="amazon_reviews_2023", demand_categories=CATEGORIES,
        d=int(X.shape[1]), pca_dim=PCA_DIM, primary_encoder="bge-base-en-v1.5",
        n_sample_per_category=N_SAMPLE, seed=SEED, per_category=per,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
    )
    (out / "amazon_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        d=int(X.shape[1]), sigma2=sigma2, ridge_lambda=ridge_lambda,
        encoder="bge-base-en-v1.5", packet_rows=40,
    ), indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
