from pathlib import Path
import csv
import math

from the_interpolator import get_model_mag, brute_force_likelihood

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_LIST_PATH = REPO_ROOT / "ASTR502_Mega_Target_List.csv"
PHOTOMETRY_LIST_PATH = REPO_ROOT / "ASTR502_Master_Photometry_List.csv"


def _parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _find_row_by_hostname(csv_path, hostname):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("hostname", "").strip().lower() == hostname.strip().lower():
                return row
    return None


def _to_absolute_mag(apparent_mag, distance_pc):
    if apparent_mag is None or distance_pc is None or distance_pc <= 0:
        return None
    distance_modulus = 5 * math.log10(distance_pc) - 5
    return apparent_mag - distance_modulus


def load_hostname_test_inputs(hostname):
    target_row = _find_row_by_hostname(TARGET_LIST_PATH, hostname)
    if target_row is None:
        raise ValueError(f"Hostname '{hostname}' not found in {TARGET_LIST_PATH.name}")

    distance_pc = _parse_float(target_row.get("bj_dist_pc"))
    age_gyr = _parse_float(target_row.get("st_age"))
    mass = _parse_float(target_row.get("st_mass"))
    feh = _parse_float(target_row.get("st_met"))

    phot_row = _find_row_by_hostname(PHOTOMETRY_LIST_PATH, hostname)
    if phot_row is None:
        raise ValueError(f"Hostname '{hostname}' not found in {PHOTOMETRY_LIST_PATH.name}")

    apparent_mag_columns = {
        "G": "gaia_Gmag",
        "BP": "gaia_BPmag",
        "RP": "gaia_RPmag",
        "J": "Jmag",
        "H": "Hmag",
        "K": "Kmag",
        "g": "gmag",
        "r": "rmag",
        "i": "imag",
        "z": "zmag",
        "W1": "w1mag",
        "W2": "w2mag",
        "W3": "w3mag",
        "W4": "w4mag",
    }

    apparent_mags = {
        band: _parse_float(phot_row.get(column_name))
        for band, column_name in apparent_mag_columns.items()
    }
    absolute_mags = {
        band: _to_absolute_mag(apparent_mag, distance_pc)
        for band, apparent_mag in apparent_mags.items()
    }

    return {
        "distance_pc": distance_pc,
        "age_gyr": age_gyr,
        "mass": mass,
        "feh": feh,
        "apparent_mags": apparent_mags,
        "absolute_mags": absolute_mags,
    }


# Optional: set this to a hostname in ASTR502_Mega_Target_List.csv to
# auto-populate distance/age/mass/metallicity and absolute magnitudes.
HOSTNAME_FOR_TESTING = None

# observed_mags = {"g": 10.74, "r": 9.13, "i": 7.96, "z": 6.90}
# observed_errs = {"g": 0.005, "r": 0.005, "i": 0.005, "z": 0.005}
#
# mass, age, feh, dm = brute_force_likelihood(observed_mags, observed_errs)
# print(f"Brute-force likelihood results: mass={mass}, age={age}, feh={feh}")

mass = 0.56
age = 9300000
feh = 0.0

distance_pc = None
hostname_absolute_mags = None

if HOSTNAME_FOR_TESTING:
    hostname_data = load_hostname_test_inputs(HOSTNAME_FOR_TESTING)
    distance_pc = hostname_data["distance_pc"]
    mass = hostname_data["mass"] if hostname_data["mass"] is not None else mass
    age_gyr = hostname_data["age_gyr"]
    if age_gyr is not None:
        age = age_gyr * 1_000_000_000
    feh = hostname_data["feh"] if hostname_data["feh"] is not None else feh
    hostname_absolute_mags = hostname_data["absolute_mags"]

    print(f"Loaded hostname '{HOSTNAME_FOR_TESTING}' from catalog files")
    print(
        f"distance_pc={distance_pc}, age_gyr={age_gyr}, mass={mass}, feh={feh}"
    )
    print("Absolute magnitudes:", hostname_absolute_mags)

model_mags = get_model_mag(mass, age, feh)
print("Model magnitudes at best-fit parameters:", model_mags)

# master_list = {"G": 8.80, "BP": 9.75, "RP": 7.92}
# comparison = {
#     band: model_mags[band] - master_list[band]
#     for band in master_list
#     if band in model_mags
# }
# print("Model - master list differences:", comparison)
