import argparse
import csv
from pathlib import Path

from the_interpolator import brute_force_likelihood, get_model_mag

BAND_CONFIG = {
    "J": {"mags": ["Jmag"], "errs": ["e_Jmag"]},
    "H": {"mags": ["Hmag"], "errs": ["e_Hmag"]},
    "K": {"mags": ["Kmag"], "errs": ["e_Kmag"]},
    "G": {"mags": ["gaia_Gmag", "gaiamag"], "errs": ["e_gaiamag"]},
    "BP": {"mags": ["gaia_BPmag"], "errs": []},
    "RP": {"mags": ["gaia_RPmag"], "errs": []},
    "W1": {"mags": ["w1mag"], "errs": ["e_w1mag"]},
    "W2": {"mags": ["w2mag"], "errs": ["e_w2mag"]},
    "W3": {"mags": ["w3mag"], "errs": ["e_w3mag"]},
    "W4": {"mags": ["w4mag"], "errs": ["e_w4mag"]},
    "g": {"mags": ["gmag", "gpmag"], "errs": ["e_gmag", "e_gpmag"]},
    "r": {"mags": ["rmag", "rpmag"], "errs": ["e_rmag", "e_rpmag"]},
    "i": {"mags": ["imag", "ipmag"], "errs": ["e_imag", "e_ipmag"]},
    "z": {"mags": ["zmag"], "errs": ["e_zmag"]},
}

DEFAULT_ERR = 0.05


def _safe_float(value):
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _build_index_map(header):
    index_map = {}
    for idx, name in enumerate(header):
        index_map.setdefault(name, []).append(idx)
    return index_map


def _first_available_value(row, index_map, names):
    for name in names:
        for idx in index_map.get(name, []):
            value = _safe_float(row[idx])
            if value is not None:
                return value
    return None


def _collect_observations(row, index_map):
    observed_mags = {}
    observed_errs = {}
    for band, config in BAND_CONFIG.items():
        mag = _first_available_value(row, index_map, config["mags"])
        if mag is None:
            continue
        err = _first_available_value(row, index_map, config["errs"])
        if err is None or err <= 0:
            err = DEFAULT_ERR
        observed_mags[band] = mag
        observed_errs[band] = err
    return observed_mags, observed_errs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare interpolator outputs against the master photometry list.",
    )
    parser.add_argument(
        "--csv",
        default="ASTR502_Master_Photometry_List.csv",
        help="Path to the master photometry CSV file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of rows to process.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    csv_path = Path(args.csv)

    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index_map = _build_index_map(header)

        for idx, row in enumerate(reader, start=1):
            if args.max_rows is not None and idx > args.max_rows:
                break

            observed_mags, observed_errs = _collect_observations(row, index_map)
            if not observed_mags:
                continue

            planet = row[index_map["pl_name"][0]]
            host = row[index_map["hostname"][0]]

            try:
                mass, age, feh = brute_force_likelihood(observed_mags, observed_errs)
            except ValueError as exc:
                print(f"Skipping {planet} ({host}): {exc}")
                continue

            model_mags = get_model_mag(mass, age, feh)
            print(f"{planet} ({host})")
            print(f"  Observed mags: {observed_mags}")
            print(f"  Best fit: mass={mass}, age={age}, feh={feh}")
            print(f"  Model mags: {model_mags}")


if __name__ == "__main__":
    main()
