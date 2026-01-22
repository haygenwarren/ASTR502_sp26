import numpy as np
from scipy.interpolate import RegularGridInterpolator
from isochrones import get_ichrone
from isochrones.mist import MIST_Isochrone

# What you WANT to support
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
        age_grid = np.logspace(np.log10(1e8), np.log10(13e9), 16)
    if feh_grid is None:
        feh_grid = np.linspace(-2.0, 0.5, 11)

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

    mass_min = min(mass_mins)
    mass_max = max(mass_maxs)
    if mass_min >= mass_max:
        raise ValueError("No mass range available across the requested age/feh grid.")

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

# Example usage
mags = get_model_mag(1.0, 1e9, 0.0)
for k in sorted(mags.keys()):
    print(f"{k}: {mags[k]}")

print("\n")


