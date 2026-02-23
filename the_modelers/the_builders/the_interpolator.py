import numpy as np
from scipy.interpolate import RegularGridInterpolator
from isochrones import get_ichrone
from isochrones.mist import MIST_Isochrone

_REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")

_BAND_COLUMNS = {
    "G": "G_mag",
    "BP": "BP_mag",
    "RP": "RP_mag",
    "J": "J_mag",
    "H": "H_mag",
    "K": "K_mag",
    "W1": "W1_mag",
    "W2": "W2_mag",
    "W3": "W3_mag",
    "W4": "W4_mag",
    "g": "g_mag",
    "r": "r_mag",
    "i": "i_mag",
    "z": "z_mag",
}

_INTERPOLATORS = None
_GRIDS = None
_ACTIVE_BANDS = None


def _build_interpolators(age_grid=None, feh_grid=None, mass_points=200):
    global _ACTIVE_BANDS

    mist = get_ichrone("mist", bands=list(_REQUESTED_BANDS))
    mist.initialize()

    if age_grid is None:
        age_grid = np.logspace(np.log10(1e8), np.log10(13e9), 32)  # 0.1–13 Gyr
    if feh_grid is None:
        feh_grid = np.linspace(-0.5, 0.5, 21)

    isochrones = {}
    mass_mins = []
    mass_maxs = []

    iso0 = mist.isochrone(age=np.log10(age_grid[0]), feh=feh_grid[0])
    available_cols = set(iso0.columns)

    _ACTIVE_BANDS = [b for b in _REQUESTED_BANDS if _BAND_COLUMNS.get(b) in available_cols]

    # missing = [b for b in _REQUESTED_BANDS if b not in _ACTIVE_BANDS]
    # print("Active bands:", _ACTIVE_BANDS)
    # if missing:
    #     print("Skipping missing bands (no column found):", missing)

    for age in age_grid:
        for feh in feh_grid:
            iso = mist.isochrone(age=np.log10(age), feh=feh)
            masses = iso["mass"].to_numpy()
            mass_mins.append(np.nanmin(masses))
            mass_maxs.append(np.nanmax(masses))
            isochrones[(age, feh)] = iso

    mass_min = 0.05
    mass_max = 2.0
    mass_grid = np.linspace(mass_min, mass_max, mass_points)

    magnitude_grids = {
        band: np.empty((mass_grid.size, age_grid.size, feh_grid.size))
        for band in _ACTIVE_BANDS
    }

    for age_index, age in enumerate(age_grid):
        for feh_index, feh in enumerate(feh_grid):
            iso = isochrones[(age, feh)]
            masses = iso["mass"].to_numpy()
            sort_idx = np.argsort(masses)
            masses_sorted = masses[sort_idx]

            for band in _ACTIVE_BANDS:
                col = _BAND_COLUMNS[band]
                values = iso[col].to_numpy()[sort_idx]

                # avoid extrapolation beyond that slice’s mass coverage
                vals = np.interp(
                    mass_grid,
                    masses_sorted,
                    values,
                    left=np.nan,
                    right=np.nan,
                )
                magnitude_grids[band][:, age_index, feh_index] = vals

    interpolators = {
        band: RegularGridInterpolator(
            (mass_grid, age_grid, feh_grid),
            magnitude_grids[band],
            bounds_error=False,
            fill_value=np.nan,
        )
        for band in _ACTIVE_BANDS
    }

    # print("mass range:", mass_grid[0], mass_grid[-1])
    # print("age range:", age_grid[0], age_grid[-1])
    # print("feh range:", feh_grid[0], feh_grid[-1])

    return interpolators, (mass_grid, age_grid, feh_grid)


def _get_interpolators():
    global _INTERPOLATORS, _GRIDS
    if _INTERPOLATORS is None:
        _INTERPOLATORS, _GRIDS = _build_interpolators()
    return _INTERPOLATORS, _GRIDS


def get_model_mag(mass, age, feh):
    interpolators, _ = _get_interpolators()

    mass_arr = np.asarray(mass)
    age_arr = np.asarray(age)
    feh_arr = np.asarray(feh)
    mass_arr, age_arr, feh_arr = np.broadcast_arrays(mass_arr, age_arr, feh_arr)

    points = np.column_stack([mass_arr.ravel(), age_arr.ravel(), feh_arr.ravel()])

    outputs = {}
    for band, interp in interpolators.items():
        outputs[band] = interp(points).reshape(mass_arr.shape)


    if mass_arr.shape == ():
        return {b: outputs[b].item() for b in outputs}

    return outputs

# # Example usage
# print("Results from interpolator and get_model_mag:")
# mags = get_model_mag(0.5, 1e9, 0.0)
# for k in sorted(mags.keys()):
#     print(f"{k}: {mags[k]}")
#
# print("--------------------------------")
# print("Results from brute-force likelihood calculator:")

def _validate_observations(observed_mags, observed_errs):
    if observed_mags is None or observed_errs is None:
        raise ValueError("Both observed_mags and observed_errs are required.")

    bands = sorted(set(observed_mags) & set(observed_errs))
    if not bands:
        raise ValueError("No overlapping bands between observed_mags and observed_errs.")

    ERR_FLOOR = 0.02  # 0.02 mag systematic/model floor
    for band in bands:
        err = observed_errs[band]
        if err is None or err <= 0:
            raise ValueError(f"Non-positive error for band '{band}'.")
        # Apply floor in quadrature
        observed_errs[band] = np.hypot(err, ERR_FLOOR)
    return bands


def brute_force_likelihood(observed_mags, observed_errs):
    observed_errs = dict(observed_errs)  # copy
    bands = _validate_observations(observed_mags, observed_errs)
    interpolators, grids = _get_interpolators()
    mass_grid, age_grid, feh_grid = grids

    bands = _validate_observations(observed_mags, observed_errs)
    bands = [b for b in bands if b in interpolators]
    if not bands:
        raise ValueError("None of the observed bands are available in the interpolator grid.")

    m_grid, a_grid, f_grid = np.meshgrid(mass_grid, age_grid, feh_grid, indexing="ij")
    points = np.column_stack([m_grid.ravel(), a_grid.ravel(), f_grid.ravel()])

    # after you build points, and inside the loop you currently do chi2 += ...
    # instead, compute model mags for all bands first

    model_stack = []
    obs_stack = []
    sig_stack = []

    for band in bands:
        model_mag = interpolators[band](points)
        model_stack.append(model_mag)
        obs_stack.append(np.full_like(model_mag, observed_mags[band], dtype=float))
        sig_stack.append(np.full_like(model_mag, observed_errs[band], dtype=float))

    model_stack = np.vstack(model_stack)  # (nband, npts)
    obs_stack = np.vstack(obs_stack)
    sig_stack = np.vstack(sig_stack)

    valid = np.all(np.isfinite(model_stack), axis=0) & np.all(sig_stack > 0, axis=0)

    w = 1.0 / (sig_stack ** 2)

    return best_mass, best_age, best_feh

# # Example likelihood usage
# observed_mags = {"G": 8.83, "BP": 9.79, "RP": 7.89}
# observed_errs = {"G": 0.02, "BP": 0.03, "RP": 0.02}
# mass, age, feh= brute_force_likelihood(observed_mags, observed_errs)
# print("mass:", mass)
# print("age:", age)
# print("feh:", feh)
