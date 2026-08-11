#!/usr/bin/env python3
"""Direct-h diagnostics: Hurst path (with SE band) and HAR coefficient paths."""
from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import repo_root                                       # noqa: E402

PROC = os.path.join(repo_root(), "data", "processed")
FIG = os.path.join(repo_root(), "outputs", "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    hp = pd.read_parquet(os.path.join(PROC, "rfsv_hurst_path.parquet"))
    hp["date"] = pd.to_datetime(hp["date"])
    cp = pd.read_parquet(os.path.join(PROC, "har_coef_path.parquet"))
    cp["date"] = pd.to_datetime(cp["date"])

    # Hurst path with +/-2 SE band
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(hp["date"], hp["H"], color="#6a3d9a", lw=1.4, label="Ĥ (variogram)")
    ax.fill_between(hp["date"], hp["H"] - 2 * hp["H_se"], hp["H"] + 2 * hp["H_se"],
                    color="#6a3d9a", alpha=0.2, label="±2 SE")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="H=0.5 (Brownian)")
    ax.set_title("RFSV Hurst estimate across the sample (Ĥ biased downward; noisy daily proxy)")
    ax.set_ylabel("Ĥ"); ax.set_ylim(0, 0.55); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hurst_path.png"), dpi=130)
    plt.close(fig)

    # HAR coefficient paths
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for col, lab, color in [("beta_d", "daily", "#1f77b4"),
                            ("beta_w", "weekly", "#ff7f0e"),
                            ("beta_m", "monthly", "#2ca02c")]:
        ax.plot(cp["date"], cp[col], label=lab, color=color, lw=1.3)
    ax.set_title("HAR-RV coefficient paths (direct-h regression, refit each rebalance date)")
    ax.set_ylabel("coefficient"); ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "har_coef_path.png"), dpi=130)
    plt.close(fig)

    print(f"figures -> {FIG}/hurst_path.png, har_coef_path.png")
    print(f"HAR coef means: d={cp.beta_d.mean():.3f} w={cp.beta_w.mean():.3f} "
          f"m={cp.beta_m.mean():.3f}  (sum {cp[['beta_d','beta_w','beta_m']].sum(axis=1).mean():.3f})")
    print(f"HAR retransform: mean {cp.retransform_pct.mean():.1%}; residual var (log RV) "
          f"mean {cp.sigma2_resid.mean():.3f}")
    print(f"RFSV Ĥ: {hp.H.iloc[0]:.3f} -> {hp.H.iloc[-1]:.3f}, mean SE {hp.H_se.mean():.4f}; "
          f"nu^2 mean {hp.nu2.mean():.3f}; retransform mean {hp.retransform_pct.mean():.1%}")


if __name__ == "__main__":
    main()
