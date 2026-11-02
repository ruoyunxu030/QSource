# QSource

Code for the paper "Source-Aware Maintenance of Shared Quality Evidence in Data Markets" (QSource).

## Install

```bash
git clone https://github.com/ruoyunxu030/QSource.git
cd QSource
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

Dataset preparation extras:

```bash
pip install -e ".[data]"
```

DomainNet embeddings need PyTorch and OpenAI CLIP.

## Data

Processed files are not in the repo. Prepare them, or point `QSOURCE_DATA` at a directory that already has them:

```bash
python data/prepare_amazon.py
python data/prepare_domainnet.py
```

Layout: `$QSOURCE_DATA/amazon/` and `$QSOURCE_DATA/domainnet/`.

Sources: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), [DomainNet](https://ai.bu.edu/M3SDA/), [BGE](https://huggingface.co/BAAI/bge-base-en-v1.5), [CLIP](https://huggingface.co/openai/clip-vit-large-patch14). Check each project's license.

## Run one instance

Single-threaded; Amazon about 22 minutes, DomainNet about 13 minutes.

```bash
python experiments/run_market.py \
  --dataset amazon \
  --design buyer_level_floor \
  --method qsource \
  --config none \
  --instances 230 \
  --workers 1 \
  --data "$QSOURCE_DATA" \
  --out results/one
```

Optional flags:

- methods: `equal_weight`, `value_filter`, `median`, `trimmed_mean`, `purchase_weight`, `forced_trials`, `qsource`
- designs: `row_level`, `buyer_level`, `buyer_level_floor`, `no_sharing`, `known_reliability`
- configs: `none`, `low_honest`, `mediocre`, `fixed`, `extreme`, `extreme_one_account`, `raise_weak`
- timing: `early`, `late`, `before_first`
- other: `--skip-threshold`, `--delta`, `--w-max`

## Reproduce the paper

```bash
bash experiments/reproduce_paper.sh \
  --workers 20 \
  --instances 230-249 \
  --data "$QSOURCE_DATA" \
  --out results/paper
```

```bash
python experiments/make_tables.py --results results/paper --out results/paper
```

`reproduce_paper.sh` runs the market grid, sensitivity, and store benchmark, then calls `make_tables.py` and `plot_figures.py` to write Tables I–IV and the paper figures under `$OUT` / `$OUT/figures`.

## Value Filter threshold calibration

```bash
python experiments/calibrate_value_filter.py --results results/paper --out results/paper/thresholds.csv
python experiments/run_market.py --dataset amazon --design buyer_level_floor \
  --method value_filter --config extreme --thresholds results/paper/thresholds.csv
```

## Tests

```bash
pytest
```

## Citation

```bibtex
@misc{xu2027qsource,
  title        = {Source-Aware Maintenance of Shared Quality Evidence in Data Markets},
  author       = {Xu, Ruoyun and Zhu, Tianze},
  year         = {2026},
  howpublished = {\url{https://github.com/ruoyunxu030/QSource}}
}
```
