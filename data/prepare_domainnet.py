#!/usr/bin/env python3
"""Build DomainNet category npz files (CLIP embeddings + PCA).

Images: https://ai.bu.edu/M3SDA/  (DomainNet)
Encoder: https://huggingface.co/openai/clip-vit-large-patch14
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 20261017
PCA_DIM = 32
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
URLS = {
    "clipart": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
    "infograph": "https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
    "painting": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
    "quickdraw": "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
    "real": "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
    "sketch": "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
}
TXT_BASE = "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt"
ENCODER = "openai/clip-vit-large-patch14"
N_CLASSES = 40


def _split(n, rng):
    idx = rng.permutation(n)
    n_sell = int(0.6 * n)
    n_query = int(0.2 * n)
    return idx[:n_sell], idx[n_sell:n_sell + n_query], idx[n_sell + n_query:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "domainnet"))
    ap.add_argument("--raw", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import clip
        import torch
        from PIL import Image
    except ImportError as e:
        raise SystemExit("DomainNet prep needs CLIP (see README) and pillow") from e

    raw = Path(args.raw) if args.raw else out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    rng = np.random.default_rng(SEED)
    all_X, all_y, all_id, all_dom = [], [], [], []
    class_names = None
    for dom in DOMAINS:
        zpath = raw / f"{dom}.zip"
        if not zpath.exists():
            print("download", URLS[dom], flush=True)
            r = requests.get(URLS[dom], timeout=60)
            r.raise_for_status()
            zpath.write_bytes(r.content)
        dest = raw / dom
        if not dest.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(raw)
        txt = requests.get(f"{TXT_BASE}/{dom}_train.txt", timeout=60).text
        lines = [ln.split() for ln in txt.splitlines() if ln.strip()]
        if class_names is None:
            names = sorted({Path(a).parts[-2] for a, _ in lines})
            class_names = names[:N_CLASSES]
        keep = {c: i for i, c in enumerate(class_names)}
        paths, yid = [], []
        for rel, _lab in lines:
            c = Path(rel).parts[-2]
            if c in keep:
                paths.append(raw / rel)
                yid.append(keep[c])
        take = rng.choice(len(paths), size=min(4000, len(paths)), replace=False)
        embs, ids = [], []
        for i in take:
            img = preprocess(Image.open(paths[i]).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                e = model.encode_image(img)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().numpy()[0])
            ids.append(yid[i])
        Xd = np.asarray(embs, float)
        yid = np.asarray(ids, int)
        yoh = np.eye(N_CLASSES)[yid]
        all_X.append(Xd)
        all_y.append(yoh)
        all_id.append(yid)
        all_dom.append(np.array([dom] * len(Xd)))
        print(dom, len(Xd), flush=True)
    X_raw = np.vstack(all_X)
    y = np.vstack(all_y)
    yid = np.concatenate(all_id)
    dom = np.concatenate(all_dom)
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    Z = StandardScaler().fit_transform(pca.fit_transform(X_raw))
    X = np.hstack([Z, np.ones((len(Z), 1))])
    for name in DOMAINS:
        m = dom == name
        Xi, yi, idi = X[m], y[m], yid[m]
        rng_c = np.random.default_rng([SEED, sum(ord(c) for c in name)])
        sell, query, test = _split(len(Xi), rng_c)
        np.savez_compressed(
            out / f"cat_clip-vit-l14_{name}.npz",
            X_sell=Xi[sell], y_sell=yi[sell], y_id_sell=idi[sell],
            X_query=Xi[query], y_query=yi[query],
            X_test=Xi[test], y_test=yi[test], y_id_test=idi[test],
        )
    from sklearn.linear_model import Ridge

    ridge_lambda = 100.0
    pred = Ridge(alpha=ridge_lambda, fit_intercept=False).fit(X, y.astype(float)).predict(X)
    sigma2 = float(np.mean((y.astype(float) - pred) ** 2))
    meta = dict(
        dataset="domainnet", demand_categories=DOMAINS, domains=DOMAINS,
        n_classes=N_CLASSES, class_names=class_names,
        encoder_key="clip-vit-l14", d=int(X.shape[1]), pca_dim=PCA_DIM, seed=SEED,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
    )
    (out / "domainnet_meta_clip-vit-l14.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        d=int(X.shape[1]), sigma2=sigma2, ridge_lambda=ridge_lambda,
        encoder="clip-vit-l14", packet_rows=40,
    ), indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
