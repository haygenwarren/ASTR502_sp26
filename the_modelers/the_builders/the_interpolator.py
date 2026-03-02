import numpy as np
from scipy.interpolate import RegularGridInterpolator
from isochrones import get_ichrone

_REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")

_BAND_COLUMNS = {
    "G": "G_mag", "BP": "BP_mag", "RP": "RP_mag",
    "J": "J_mag", "H": "H_mag", "K": "K_mag",
    "W1": "W1_mag", "W2": "W2_mag", "W3": "W3_mag", "W4": "W4_mag",
    "g": "g_mag", "r": "r_mag", "i": "i_mag", "z": "z_mag",
}

_MIST = None
_INTERPOLATORS = None
_GRIDS = None
_ACTIVE_BANDS = None


def _get_mist():
    global _MIST
    if _MIST is None:
        _MIST = get_ichrone("mist", bands=list(_REQUESTED_BANDS))
        _MIST.initialize()
    return _MIST


def _select_rows(iso):
    """
    Keep pre-MS (phase=-1), MS (phase=0), and early subgiant (phase=2)
    rows, truncated at the first point where mass turns over.
    """
    selected = iso[iso["phase"].isin([-1, 0, 2])].copy()
    selected = selected.sort_values("eep").reset_index(drop=True)
    masses = selected["mass"].to_numpy()
    cutoff = len(masses)
    for i in range(1, len(masses)):
        if masses[i] < masses[i - 1]:
            cutoff = i
            break
    return selected.iloc[:cutoff]


def _build_interpolators(age_grid=None, feh_grid=None, mass_points=300):
    global _ACTIVE_BANDS

    mist = _get_mist()

    if age_grid is None:
        age_grid = np.logspace(np.log10(1e6), np.log10(13.8e9), 60)
    if feh_grid is None:
        feh_grid = np.linspace(-1.0, 0.5, 31)

    mass_grid = np.linspace(0.1, 3.0, mass_points)

    # Discover which bands are actually available
    iso0 = mist.isochrone(age=np.log10(age_grid[len(age_grid) // 2]), feh=0.0)
    available_cols = set(iso0.columns)
    _ACTIVE_BANDS = [b for b in _REQUESTED_BANDS if _BAND_COLUMNS.get(b) in available_cols]

    magnitude_grids = {
        band: np.full((mass_grid.size, age_grid.size, feh_grid.size), np.nan)
        for band in _ACTIVE_BANDS
    }

    for age_index, age in enumerate(age_grid):
        for feh_index, feh in enumerate(feh_grid):
            try:
                iso = mist.isochrone(age=np.log10(age), feh=feh)
            except Exception:
                continue

            selected = _select_rows(iso)
            if len(selected) < 2:
                continue

            masses = selected["mass"].to_numpy()
            sort_idx = np.argsort(masses)
            masses_sorted = masses[sort_idx]
            _, unique_idx = np.unique(masses_sorted, return_index=True)
            masses_sorted = masses_sorted[unique_idx]

            for band in _ACTIVE_BANDS:
                col = _BAND_COLUMNS[band]
                if col not in selected.columns:
                    continue

                values = selected[col].to_numpy()[sort_idx][unique_idx]

                # Interpolate all mass grid points at once
                in_range = (
                    (mass_grid >= masses_sorted[0]) &
                    (mass_grid <= masses_sorted[-1])
                )
                extrap_mask = (
                    (mass_grid > masses_sorted[-1]) &
                    (mass_grid <= masses_sorted[-1] + 0.1)
                )

                if np.any(in_range):
                    magnitude_grids[band][in_range, age_index, feh_index] = np.interp(
                        mass_grid[in_range], masses_sorted, values
                    )

                if np.any(extrap_mask):
                    dm = masses_sorted[-1] - masses_sorted[-2]
                    dv = values[-1] - values[-2]
                    slope = dv / dm if dm != 0 else 0.0
                    magnitude_grids[band][extrap_mask, age_index, feh_index] = (
                        values[-1] + slope * (mass_grid[extrap_mask] - masses_sorted[-1])
                    )

    # Fill interior NaN gaps along the mass axis only
    for band in _ACTIVE_BANDS:
        grid = magnitude_grids[band]
        for age_index in range(len(age_grid)):
            for feh_index in range(len(feh_grid)):
                col_vals = grid[:, age_index, feh_index]
                valid = np.isfinite(col_vals)
                if valid.sum() < 2:
                    continue
                if not np.all(valid):
                    valid_idx = np.where(valid)[0]
                    lo, hi = valid_idx[0], valid_idx[-1]
                    nan_idx = np.where(~valid)[0]
                    interior_nans = nan_idx[(nan_idx > lo) & (nan_idx < hi)]
                    if len(interior_nans) > 0:
                        nearest = valid_idx[
                            np.argmin(
                                np.abs(interior_nans[:, None] - valid_idx[None, :]), axis=1
                            )
                        ]
                        col_vals[interior_nans] = col_vals[nearest]
                    grid[:, age_index, feh_index] = col_vals

    interpolators = {
        band: RegularGridInterpolator(
            (mass_grid, age_grid, feh_grid),
            magnitude_grids[band],
            bounds_error=False,
            fill_value=np.nan,
        )
        for band in _ACTIVE_BANDS
    }

    return interpolators, (mass_grid, age_grid, feh_grid)


def _get_interpolators():
    global _INTERPOLATORS, _GRIDS
    if _INTERPOLATORS is None:
        _INTERPOLATORS, _GRIDS = _build_interpolators()
    return _INTERPOLATORS, _GRIDS


def get_model_mag(mass, age, feh):
    """
    Parameters
    ----------
    mass : float or array
        Stellar mass in solar masses.
    age : float or array
        Stellar age in years (e.g. 10e6 for 10 Myr, 3e9 for 3 Gyr).
    feh : float or array
        Metallicity [Fe/H] in dex.

    Returns
    -------
    dict mapping band name -> absolute magnitude (float or array)
    """

    interpolators, (mass_grid, age_grid, feh_grid) = _get_interpolators()

    mass_arr = np.asarray(mass, dtype=float)
    age_arr  = np.asarray(age,  dtype=float)
    feh_arr  = np.asarray(feh,  dtype=float)
    mass_arr, age_arr, feh_arr = np.broadcast_arrays(mass_arr, age_arr, feh_arr)

    if np.any(mass_arr < mass_grid[0]) or np.any(mass_arr > mass_grid[-1]):
        print(f"Warning: mass {mass} outside grid [{mass_grid[0]:.2f}, {mass_grid[-1]:.2f}] Msun")
    if np.any(age_arr < age_grid[0]) or np.any(age_arr > age_grid[-1]):
        print(f"Warning: age {age:.3e} outside grid [{age_grid[0]:.3e}, {age_grid[-1]:.3e}] yr")
    if np.any(feh_arr < feh_grid[0]) or np.any(feh_arr > feh_grid[-1]):
        print(f"Warning: feh {feh} outside grid [{feh_grid[0]:.2f}, {feh_grid[-1]:.2f}]")

    points = np.column_stack([mass_arr.ravel(), age_arr.ravel(), feh_arr.ravel()])

    outputs = {}
    for band, interp in interpolators.items():
        outputs[band] = interp(points).reshape(mass_arr.shape)

    if mass_arr.shape == ():
        return {b: float(outputs[b]) for b in outputs}

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





#-------------------------
#Code for grabbing a single isochrone rather than building a grid
#-------------------------

# import numpy as np
# from isochrones import get_ichrone
#
# _REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")
#
# _BAND_COLUMNS = {
#     "G": "G_mag", "BP": "BP_mag", "RP": "RP_mag",
#     "J": "J_mag", "H": "H_mag", "K": "K_mag",
#     "W1": "W1_mag", "W2": "W2_mag", "W3": "W3_mag", "W4": "W4_mag",
#     "g": "g_mag", "r": "r_mag", "i": "i_mag", "z": "z_mag",
# }
#
# _MIST = None
#
#
# def _get_mist():
#     global _MIST
#     if _MIST is None:
#         _MIST = get_ichrone("mist", bands=list(_REQUESTED_BANDS))
#         _MIST.initialize()
#     return _MIST
#
#
# def _select_rows(iso):
#     """
#     Keep pre-MS (phase=-1), MS (phase=0), and early subgiant (phase=2)
#     rows, truncated at the first point where mass turns over.
#     """
#     selected = iso[iso["phase"].isin([-1, 0, 2])].copy()
#     selected = selected.sort_values("eep").reset_index(drop=True)
#     masses = selected["mass"].to_numpy()
#     cutoff = len(masses)
#     for i in range(1, len(masses)):
#         if masses[i] < masses[i - 1]:
#             cutoff = i
#             break
#     return selected.iloc[:cutoff]
#
#
# def get_model_mag(mass, age, feh):
#     """
#     Parameters
#     ----------
#     mass : float
#         Stellar mass in solar masses.
#     age : float
#         Stellar age in years (e.g. 10e6 for 10 Myr, 3e9 for 3 Gyr).
#     feh : float
#         Metallicity [Fe/H] in dex.
#
#     Returns
#     -------
#     dict mapping band name -> absolute magnitude (float)
#     """
#     mist = _get_mist()
#
#     # Fetch the single isochrone at this age and metallicity
#     iso = mist.isochrone(age=np.log10(age), feh=feh)
#     selected = _select_rows(iso)
#
#     if len(selected) < 2:
#         return {b: np.nan for b in _BAND_COLUMNS}
#
#     masses = selected["mass"].to_numpy()
#     sort_idx = np.argsort(masses)
#     masses_sorted = masses[sort_idx]
#
#     # Deduplicate
#     _, unique_idx = np.unique(masses_sorted, return_index=True)
#     masses_sorted = masses_sorted[unique_idx]
#
#     if mass < masses_sorted[0] or mass > masses_sorted[-1] + 0.1:
#         print(f"Warning: mass {mass} outside isochrone range "
#               f"[{masses_sorted[0]:.3f}, {masses_sorted[-1]:.3f}] Msun for "
#               f"age={age:.3e}, feh={feh:.2f}")
#
#     results = {}
#     for band, col in _BAND_COLUMNS.items():
#         if col not in selected.columns:
#             results[band] = np.nan
#             continue
#
#         values = selected[col].to_numpy()[sort_idx][unique_idx]
#
#         if mass <= masses_sorted[-1]:
#             # Standard interpolation within the isochrone mass range
#             results[band] = float(np.interp(mass, masses_sorted, values,
#                                             left=np.nan, right=np.nan))
#         elif mass <= masses_sorted[-1] + 0.1:
#             # Linear extrapolation up to 0.1 Msun beyond the upper boundary
#             dm = masses_sorted[-1] - masses_sorted[-2]
#             dv = values[-1] - values[-2]
#             slope = dv / dm if dm != 0 else 0.0
#             results[band] = float(values[-1] + slope * (mass - masses_sorted[-1]))
#         else:
#             results[band] = np.nan
#
#     return results
