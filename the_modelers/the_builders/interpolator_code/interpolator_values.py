from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from the_interpolator import fit_best_params, get_model_mag, load_catalogs

# Observed photometry columns in ASTR502_Master_Photometry_List.csv
OBS_MAP = {
    "G": "gaia_Gmag",
    "BP": "gaia_BPmag",
    "RP": "gaia_RPmag",
    "J": "Jmag",
    "H": "Hmag",
    "K": "Kmag",
    "W1": "w1mag",
    "W2": "w2mag",
    "W3": "w3mag",
    "W4": "w4mag",
    "g": "gmag",
    "r": "rmag",
    "i": "imag",
    "z": "zmag",
}


def apparent_to_absolute(m_app: float, d_pc: float) -> float:
    return float(m_app - 5.0 * np.log10(d_pc / 10.0))


def get_youngest_stars(mega_df: pd.DataFrame, n_stars: int = 10) -> list[str]:
    """Return hostnames sorted by youngest st_age (Gyr), skipping missing ages."""
    valid = mega_df[np.isfinite(mega_df["st_age"])].copy()
    return valid.sort_values("st_age", ascending=True)["hostname"].head(n_stars).tolist()


def build_row(hostname: str, mega_df: pd.DataFrame, phot_df: pd.DataFrame, fit_result: dict) -> dict:
    mrow = mega_df.loc[mega_df["hostname"] == hostname].iloc[0]
    prow = phot_df.loc[phot_df["hostname"] == hostname].iloc[0]

    d_pc = float(mrow["bj_dist_pc"])
    row = {
        "hostname": hostname,
        "distance_pc": d_pc,
        "mass_model": fit_result["mass"],
        "mass_err": fit_result["mass_err"],
        "mass_err_minus": fit_result["mass_err_minus"],
        "mass_err_plus": fit_result["mass_err_plus"],
        "age_yr_model": fit_result["age_yr"],
        "age_yr_err": fit_result["age_yr_err"],
        "age_yr_err_minus": fit_result["age_yr_err_minus"],
        "age_yr_err_plus": fit_result["age_yr_err_plus"],
        "feh_model": fit_result["feh"],
        "feh_err": fit_result["feh_err"],
        "feh_err_minus": fit_result["feh_err_minus"],
        "feh_err_plus": fit_result["feh_err_plus"],
        "av_model": fit_result["av"],
        "chi2_total": fit_result["chi2_total"],
        "st_mass_measured": float(mrow["st_mass"]) if np.isfinite(mrow["st_mass"]) else np.nan,
        "st_age_gyr_measured": float(mrow["st_age"]) if np.isfinite(mrow["st_age"]) else np.nan,
        "st_met_measured": float(mrow["st_met"]) if np.isfinite(mrow["st_met"]) else np.nan,
    }

    # Save measured absolute magnitudes and model absolute magnitudes
    for band, col in OBS_MAP.items():
        app_val = prow[col] if col in prow.index else np.nan
        if np.isfinite(app_val):
            row[f"{band}_mag_abs_measured"] = apparent_to_absolute(float(app_val), d_pc)
        else:
            row[f"{band}_mag_abs_measured"] = np.nan
        row[f"{band}_mag_abs_model"] = fit_result["model_mags"].get(band, np.nan)

    return row


def _run_fit_for_hostname(
    hostname: str,
    mega_csv: Path,
    phot_csv: Path,
    sigma_phot: float,
    fallback_sigma_param: float,
    nwalkers: int,
    nsteps: int,
    burn_in: int,
) -> dict:
    mega_df = pd.read_csv(mega_csv)
    phot_df = pd.read_csv(phot_csv)
    load_catalogs(str(mega_csv), str(phot_csv))

    mass, age_yr, feh, av, res = fit_best_params(
        hostname=hostname,
        sigma_phot=sigma_phot,
        fallback_sigma_param=fallback_sigma_param,
        run_emcee=True,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn_in=burn_in,
        make_walker_plots=False,
        verbose=True,
    )

    mcmc = res.mcmc_summary
    if mcmc is not None:
        mass_err_minus = mcmc["mass"]["err_minus"]
        mass_err_plus = mcmc["mass"]["err_plus"]
        feh_err_minus = mcmc["feh"]["err_minus"]
        feh_err_plus = mcmc["feh"]["err_plus"]

        age_med_log = mcmc["log10_age"]["median"]
        age_err_minus_log = mcmc["log10_age"]["err_minus"]
        age_err_plus_log = mcmc["log10_age"]["err_plus"]
        age_med_yr = 10.0 ** age_med_log
        age_err_minus = age_med_yr - 10.0 ** (age_med_log - age_err_minus_log)
        age_err_plus = 10.0 ** (age_med_log + age_err_plus_log) - age_med_yr
    else:
        mass_err_minus = np.nan
        mass_err_plus = np.nan
        feh_err_minus = np.nan
        feh_err_plus = np.nan
        age_err_minus = np.nan
        age_err_plus = np.nan

    mags = get_model_mag(mass=mass, age=age_yr, feh=feh, av=av)
    fit_result = {
        "mass": mass,
        "mass_err_minus": mass_err_minus,
        "mass_err_plus": mass_err_plus,
        "mass_err": np.nanmean([mass_err_minus, mass_err_plus]),
        "age_yr": age_yr,
        "age_yr_err_minus": age_err_minus,
        "age_yr_err_plus": age_err_plus,
        "age_yr_err": np.nanmean([age_err_minus, age_err_plus]),
        "feh": feh,
        "feh_err_minus": feh_err_minus,
        "feh_err_plus": feh_err_plus,
        "feh_err": np.nanmean([feh_err_minus, feh_err_plus]),
        "av": av,
        "chi2_total": res.fun,
        "model_mags": mags,
    }
    return build_row(hostname, mega_df, phot_df, fit_result)


def run_stars_and_save_values(
    star_names: Iterable[str],
    mega_csv: Path,
    phot_csv: Path,
    out_dir: Path,
    sigma_phot: float = 0.5,
    fallback_sigma_param: float = 0.1,
    nwalkers: int = 32,
    nsteps: int = 4000,
    burn_in: int = 300,
    backend: str = "serial",
    max_workers: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    mega_df = pd.read_csv(mega_csv)
    phot_df = pd.read_csv(phot_csv)
    load_catalogs(str(mega_csv), str(phot_csv))

    stars = list(star_names)
    if backend == "parallel":
        workers = max_workers if max_workers is not None else min(4, len(stars))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(
                executor.map(
                    _run_fit_for_hostname,
                    stars,
                    [mega_csv] * len(stars),
                    [phot_csv] * len(stars),
                    [sigma_phot] * len(stars),
                    [fallback_sigma_param] * len(stars),
                    [nwalkers] * len(stars),
                    [nsteps] * len(stars),
                    [burn_in] * len(stars),
                )
            )
    elif backend == "process":
        workers = max_workers if max_workers is not None else min(4, len(stars))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(
                executor.map(
                    _run_fit_for_hostname,
                    stars,
                    [mega_csv] * len(stars),
                    [phot_csv] * len(stars),
                    [sigma_phot] * len(stars),
                    [fallback_sigma_param] * len(stars),
                    [nwalkers] * len(stars),
                    [nsteps] * len(stars),
                    [burn_in] * len(stars),
                )
            )
    else:
        rows = [
            _run_fit_for_hostname(
                hostname=hostname,
                mega_csv=mega_csv,
                phot_csv=phot_csv,
                sigma_phot=sigma_phot,
                fallback_sigma_param=fallback_sigma_param,
                nwalkers=nwalkers,
                nsteps=nsteps,
                burn_in=burn_in,
            )
            for hostname in stars
        ]

    results_df = pd.DataFrame(rows)
    results_csv = out_dir / "interpolator_results.csv"
    if results_csv.exists():
        old_df = pd.read_csv(results_csv)
        results_df = pd.concat([old_df, results_df], ignore_index=True)

        # optional: remove duplicates by hostname
        results_df = results_df.drop_duplicates(subset="hostname", keep="last")
    results_df.to_csv(results_csv, index=False)

    print(f"Saved: {results_csv}")
    return results_csv
