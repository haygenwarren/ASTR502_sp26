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


def make_lee_comparison_plot_from_csv(
    results_csv: Path, lee_results_csv: Path, out_path: Path
) -> None:
    """Compare model ages in two CSV outputs for overlapping stars."""
    model_df = pd.read_csv(results_csv)
    lee_df = pd.read_csv(lee_results_csv)

    model_ages = (
        model_df[["hostname", "age_yr_model"]]
        .rename(columns={"hostname": "star", "age_yr_model": "age_yr_ours"})
        .copy()
    )
    lee_ages = (
        lee_df[["starname", "age_gyr"]]
        .rename(columns={"starname": "star", "age_gyr": "age_gyr_lee"})
        .copy()
    )
    model_ages["star"] = model_ages["star"].astype(str).str.strip()
    lee_ages["star"] = lee_ages["star"].astype(str).str.strip()

    merged = model_ages.merge(lee_ages, on="star", how="inner")
    x = merged["age_gyr_lee"].to_numpy(dtype=float)
    y = (merged["age_yr_ours"].to_numpy(dtype=float) / 1e9)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    if np.any(good):
        ax.scatter(x[good], y[good], alpha=0.85)
        lo = min(np.nanmin(x[good]), np.nanmin(y[good]))
        hi = max(np.nanmax(x[good]), np.nanmax(y[good]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Lee Results Age (Gyr)")
    ax.set_ylabel("Interpolator Age (Gyr)")
    ax.set_title("Age Comparison: Lee Results vs Interpolator")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path} (matched stars: {np.count_nonzero(good)})")
