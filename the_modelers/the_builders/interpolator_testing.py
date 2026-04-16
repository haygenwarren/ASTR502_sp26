from pathlib import Path

from interpolator_plotting import make_age_comparison_plot_from_csv
from interpolator_values import run_stars_and_save_values


def run_stars(
    star_names,
    mega_csv: Path,
    phot_csv: Path,
    out_dir: Path,
    sigma_phot: float = 0.5,
    fallback_sigma_param: float = 0.1,
    nwalkers: int = 32,
    nsteps: int = 4000,
    burn_in: int = 300,
):
    """Run interpolation for stars, save/update CSV values, and generate one age plot."""
    results_csv = run_stars_and_save_values(
        star_names=star_names,
        mega_csv=mega_csv,
        phot_csv=phot_csv,
        out_dir=out_dir,
        sigma_phot=sigma_phot,
        fallback_sigma_param=fallback_sigma_param,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn_in=burn_in,
    )

    age_plot = out_dir / "age_comparison.png"
    make_age_comparison_plot_from_csv(results_csv=results_csv, out_path=age_plot)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    mega_csv = repo_root / "ASTR502_Mega_Target_List.csv"
    phot_csv = repo_root / "ASTR502_Master_Photometry_List.csv"
    out_dir = Path(__file__).resolve().parent / "interpolator_outputs"

    stars_to_run = ["K2-33", "KELT-9", "TOI-2497"]

    run_stars(
        star_names=stars_to_run,
        mega_csv=mega_csv,
        phot_csv=phot_csv,
        out_dir=out_dir,
        sigma_phot=0.5,
        fallback_sigma_param=0.1,
        nwalkers=32,
        nsteps=5000,
        burn_in=300,
    )
