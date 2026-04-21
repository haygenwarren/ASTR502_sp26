from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_age_comparison_plot(results_df: pd.DataFrame, out_path: Path) -> None:
    """Create one age-comparison plot using all available stars in results_df."""
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))

    x = results_df["st_age_gyr_measured"].to_numpy(dtype=float)
    y = results_df["age_yr_model"].to_numpy(dtype=float) / 1e9
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)

    if np.any(good):
        ax.scatter(x[good], y[good], alpha=0.85)
        lo = min(np.nanmin(x[good]), np.nanmin(y[good]))
        hi = max(np.nanmax(x[good]), np.nanmax(y[good]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured Age (Gyr)")
    ax.set_ylabel("Model Age (Gyr)")
    ax.set_title("Age Comparison")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_age_comparison_plot_from_csv(results_csv: Path, out_path: Path) -> None:
    results_df = pd.read_csv(results_csv)
    make_age_comparison_plot(results_df, out_path)
    print(f"Saved: {out_path}")
