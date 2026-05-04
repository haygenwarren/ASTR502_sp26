from pathlib import Path
import argparse

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
    backend: str = "serial",
    max_workers: int | None = None,
):
    """Run interpolation for stars and save/update CSV values."""
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
        backend=backend,
        max_workers=max_workers,
    )

    return results_csv


def make_age_plot(results_csv: Path, out_dir: Path):
    """Generate an age comparison plot from an existing results CSV."""
    age_plot = out_dir / "age_comparison.png"
    make_age_comparison_plot_from_csv(results_csv=results_csv, out_path=age_plot)
    return age_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interpolator CSV values and/or an age-comparison plot."
    )
    parser.add_argument(
        "--generate-csv",
        action="store_true",
        help="Run run_stars_and_save_values to generate/update the results CSV.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate the age plot from an existing results CSV.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Path to an existing results CSV to use when --plot is set.",
    )
    parser.add_argument(
        "--backend",
        choices=["serial", "parallel", "process"],
        default="serial",
        help="Execution backend for star fitting when generating CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count for --backend=parallel|process (default picks up to 4 workers).",
    )
    args = parser.parse_args()

    run_generate_csv = args.generate_csv
    run_plot = args.plot

    # If neither flag is provided, run both for convenience.
    if not run_generate_csv and not run_plot:
        run_generate_csv = True
        run_plot = True

    repo_root = Path(__file__).resolve().parents[3]
    mega_csv = repo_root / "ASTR502_Mega_Target_List.csv"
    phot_csv = repo_root / "ASTR502_Master_Photometry_List.csv"
    out_dir = Path(__file__).resolve().parents[1] / "interpolator_outputs"

    stars_to_run = ["WASP-59", "HATS-67", "Kepler-198", "Kepler-198", "HD 207496",
                    "Qatar-5", "KOI-351", "Kepler-349", "TOI-1135", "TOI-444"]

    results_csv = args.results_csv

    if run_generate_csv:
        results_csv = run_stars(
            star_names=stars_to_run,
            mega_csv=mega_csv,
            phot_csv=phot_csv,
            out_dir=out_dir,
            sigma_phot=0.5,
            fallback_sigma_param=0.1,
            nwalkers=32,
            nsteps=5000,
            burn_in=300,
            backend=args.backend,
            max_workers=args.workers,
        )

    if run_plot:
        if results_csv is None:
            raise ValueError(
                "--plot requires a results CSV. Provide --results-csv or include --generate-csv."
            )
        make_age_plot(results_csv=results_csv, out_dir=out_dir)
