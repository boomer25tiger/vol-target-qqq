#!/usr/bin/env bash
# reproduce.sh - regenerate every table, figure, and animation from a clean clone.
#
# Prerequisites:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.13.13
#   network access (Stage 1 downloads from Yahoo Finance and FRED)
#
# Determinism: every stage is seeded (config random_seed 20260806). All results are exact to
# machine precision EXCEPT the three EGARCH Monte-Carlo stages, whose 10,000-path simulation and
# maximum-likelihood fit are reproducible only to a tolerance (~1.3e-5 in variance on the h=21
# reconciliation). A fresh Yahoo download may also differ from the
# committed run if the vendor has revised historical prices or dividends since 2026-08-06.
#
# Runtime: roughly 45-60 minutes end to end, dominated by the three EGARCH Monte-Carlo stages
# (fit_models, layer1_egarch_daily, fit_term_structure - about 11 minutes each). Run single-threaded
# BLAS to avoid oversubscription: the export below is required, not optional.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PYTHON:-.venv/bin/python}"
cd "$(dirname "$0")"
mkdir -p data/raw data/processed figures gallery outputs/tables   # gitignored dirs a clone lacks

echo "== Stage 1/6  data acquisition (network: Yahoo Finance, FRED, Stooq) =="
$PY scripts/fetch_daily_data.py
$PY scripts/validate_daily_data.py

echo "== Stage 2/6  realized-measure panel (IXIC seed rescaled by c = 2.7812) =="
$PY scripts/seed_level_1987.py            # computes/verifies the seed scalar c
$PY scripts/build_rv_panel.py

echo "== Stage 3/6  model fitting  (EGARCH Monte Carlo - the cost, ~33 min of the total) =="
$PY scripts/fit_models.py                 # GARCH family -> param_paths, forecast_v21 (GARCH cols)
$PY scripts/fit_direct_h.py               # rv/har/rfsv -> forecast_v21 (direct-h cols), Hurst/HAR paths
$PY scripts/layer1_egarch_daily.py        # EGARCH daily forecasts for the Layer-1 secondary sample
$PY scripts/fit_term_structure.py         # V_t(h) h=1..63 -> forecast_vh (S3)

echo "== Stage 4/6  backtest =="
$PY scripts/run_backtest.py               # -> backtest_equity, backtest_metrics

echo "== Stage 5/6  evaluation and inference (deterministic; fixed bootstrap seeds) =="
$PY scripts/layer1_eval.py                # QLIKE/MSE losses, DM, MCS, loss concentration
$PY scripts/layer1_secondary.py           # overlapping-daily secondary sample
$PY scripts/layer2_eval.py                # adherence, risk/return, return gap, subperiods
$PY scripts/phase_d_audit.py              # Sharpe decomposition, RFSV retransform audit, h-stage spread
$PY scripts/phase_h.py                    # Sharpe inference (Ledoit-Wolf + deflated), cross-layer ranks
$PY scripts/phase_n_adherence.py          # adherence bootstrap intervals + pairwise
$PY scripts/phase_o_grouping.py           # adherence MCS
$PY scripts/phase_r_signed.py             # signed adherence deviation + dispersion
$PY scripts/phase_s_decompose.py          # level/concavity decomposition

echo "== Stage 6/6  figures and gallery =="
$PY scripts/build_figures.py              # -> figures/*.png
$PY scripts/build_readme_assets.py        # -> assets/performance.png, outputs/tables/portfolio_summary.csv
$PY scripts/subsample_robustness.py       # -> outputs/tables/portfolio_summary_from{2003,2010}.csv
$PY scripts/build_animations.py           # -> gallery/*.gif (+ .mp4 only if ffmpeg is installed)

echo "== done.  Tables in outputs/tables/, figures in figures/ and assets/, animations in gallery/. =="
echo "   Optional diagnostics (memos, not paper tables): mz_report.py, phase_e1_audit.py,"
echo "   phase_i.py, phase_p_power.py, report_models.py, report_direct_h.py, qqq_estimator_check.py."
