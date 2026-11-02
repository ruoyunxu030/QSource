#!/usr/bin/env bash
# Reproduce paper Tables I–IV, the sensitivity grid, figures, and the store benchmark.
# Usage: experiments/reproduce_paper.sh [--workers N] [--instances A-B] [--data DIR] [--out DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
for k in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
  export "$k"=1
done
WORKERS=20
INSTANCES=230-249
DATA="${QSOURCE_DATA:-$ROOT/data}"
OUT="$ROOT/results/paper"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --instances) INSTANCES="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT/logs"
LOG="$OUT/logs/reproduce.log"
run() {
  echo "+" "$@" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

market() {
  run "$PY" experiments/run_market.py --data "$DATA" --out "$OUT" \
    --workers "$WORKERS" --instances "$INSTANCES" "$@"
}

# --- Table I: Equal Weight, all designs ---
DESIGNS_I=(row_level buyer_level buyer_level_floor no_sharing known_quality)
CONFIGS_SHARE=(none low_honest mediocre fixed extreme)
for ds in amazon domainnet; do
  for design in "${DESIGNS_I[@]}"; do
    if [[ "$design" == "no_sharing" || "$design" == "known_quality" ]]; then
      market --dataset "$ds" --design "$design" --method equal_weight --config none
      continue
    fi
    for cfg in "${CONFIGS_SHARE[@]}"; do
      market --dataset "$ds" --design "$design" --method equal_weight --config "$cfg"
    done
    if [[ "$ds" == "amazon" ]]; then
      market --dataset "$ds" --design "$design" --method equal_weight --config extreme_one_account
    fi
  done
done

# Table I Extreme late (equal weight; row / buyer / buyer+floor)
for ds in amazon domainnet; do
  for design in row_level buyer_level buyer_level_floor; do
    market --dataset "$ds" --design "$design" --method equal_weight --config extreme --timing late
  done
done

# --- Value Filter thresholds from the same none runs ---
run "$PY" experiments/calibrate_value_filter.py --results "$OUT" --out "$OUT/thresholds.csv" --instances "$INSTANCES"

# --- Table II: methods on buyer_level and buyer_level_floor ---
DESIGNS_II=(buyer_level buyer_level_floor)
METHODS_II=(value_filter forced_trials qsource)
for ds in amazon domainnet; do
  for design in "${DESIGNS_II[@]}"; do
    for method in "${METHODS_II[@]}"; do
      extra=()
      if [[ "$method" == "value_filter" ]]; then
        extra+=(--thresholds "$OUT/thresholds.csv")
      fi
      for cfg in none low_honest mediocre extreme; do
        market --dataset "$ds" --design "$design" --method "$method" --config "$cfg" "${extra[@]}"
      done
      if [[ "$ds" == "domainnet" ]]; then
        market --dataset "$ds" --design "$design" --method "$method" --config fixed "${extra[@]}"
      fi
      if [[ "$ds" == "amazon" && "$method" != "value_filter" ]]; then
        market --dataset "$ds" --design "$design" --method "$method" --config extreme_one_account "${extra[@]}"
      fi
    done
  done
done

# Robust aggregation baselines: buyer_level_floor only
for ds in amazon domainnet; do
  for method in median trimmed_mean; do
    for cfg in none low_honest mediocre extreme; do
      market --dataset "$ds" --design buyer_level_floor --method "$method" --config "$cfg"
    done
    if [[ "$ds" == "domainnet" ]]; then
      market --dataset "$ds" --design buyer_level_floor --method "$method" --config fixed
    fi
    if [[ "$ds" == "amazon" ]]; then
      market --dataset "$ds" --design buyer_level_floor --method "$method" --config extreme_one_account
    fi
  done
done

# --- Additional experiments ---
# Purchase Weight (Table II column): buyer_level and buyer_level_floor
for ds in amazon domainnet; do
  for design in buyer_level buyer_level_floor; do
    for cfg in none low_honest mediocre extreme; do
      market --dataset "$ds" --design "$design" --method purchase_weight --config "$cfg"
    done
    if [[ "$ds" == "domainnet" ]]; then
      market --dataset "$ds" --design "$design" --method purchase_weight --config fixed
    fi
    if [[ "$ds" == "amazon" ]]; then
      market --dataset "$ds" --design "$design" --method purchase_weight --config extreme_one_account
    fi
  done
done

# raise_weak (first π=0.5 seller) and none control with the same target
for ds in amazon domainnet; do
  for method in equal_weight qsource; do
    market --dataset "$ds" --design buyer_level_floor --method "$method" \
      --config none --target-goal raise
    market --dataset "$ds" --design buyer_level_floor --method "$method" \
      --config raise_weak
  done
done

# skip-threshold 2 on Equal Weight and QSource, buyer-level + floor
for ds in amazon domainnet; do
  for method in equal_weight qsource; do
    for cfg in none low_honest mediocre extreme; do
      market --dataset "$ds" --design buyer_level_floor --method "$method" \
        --config "$cfg" --skip-threshold 2
    done
    if [[ "$ds" == "domainnet" ]]; then
      market --dataset "$ds" --design buyer_level_floor --method "$method" \
        --config fixed --skip-threshold 2
    fi
    if [[ "$ds" == "amazon" ]]; then
      market --dataset "$ds" --design buyer_level_floor --method "$method" \
        --config extreme_one_account --skip-threshold 2
    fi
  done
done

# Fig. 1: three Mediocre accounts write before buyer 1
for method in equal_weight qsource; do
  market --dataset amazon --design buyer_level_floor --method "$method" \
    --config mediocre --timing before_first
done

# --- Sensitivity (main text; not Table III) ---
run "$PY" experiments/run_sensitivity.py --data "$DATA" --out "$OUT/sensitivity" \
  --workers "$WORKERS" --instances "$INSTANCES" \
  --dataset amazon --design buyer_level_floor --config low_honest

# --- Store benchmark from this repo's none forced_trials records ---
run "$PY" experiments/run_benchmark.py --records "$OUT" --out "$OUT/benchmark" \
  --n 1000,10000,100000

# --- Tables and figures ---
run "$PY" experiments/make_tables.py --results "$OUT" --out "$OUT"
run "$PY" experiments/plot_figures.py --results "$OUT" --out "$OUT/figures"

echo "done. tables in $OUT  figures in $OUT/figures" | tee -a "$LOG"
