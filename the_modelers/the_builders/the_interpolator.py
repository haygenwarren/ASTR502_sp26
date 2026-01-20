import numpy as np
from scipy.interpolate import RegularGridInterpolator
from isochrones.mist import MIST_Isochrone

_BANDS = ("G", "BP", "RP", "J", "H", "K")
_INTERPOLATORS = None
_GRIDS = None


def _build_interpolators(
    age_grid=None,
    feh_grid=None,
    mass_points=200,
):
    mist = MIST_Isochrone()
    mist.initialize()

    if age_grid is None:
        age_grid = np.logspace(np.log10(1e8), np.log10(13e9), 16)
    if feh_grid is None:
        feh_grid = np.linspace(-2.0, 0.5, 11)

    isochrones = {}
    mass_mins = []
    mass_maxs = []

    for age in age_grid:
        for feh in feh_grid:
            iso = mist.isochrone(age=np.log10(age), feh=feh)
            masses = iso["mass"].to_numpy()
            mass_mins.append(np.nanmin(masses))
            mass_maxs.append(np.nanmax(masses))
            isochrones[(age, feh)] = iso

    mass_min = max(mass_mins)
    mass_max = min(mass_maxs)
    if mass_min >= mass_max:
        raise ValueError("No overlapping mass range across the requested age/feh grid.")

    mass_grid = np.linspace(mass_min, mass_max, mass_points)

    magnitude_grids = {
        band: np.empty((mass_grid.size, age_grid.size, feh_grid.size))
        for band in _BANDS
    }

    for age_index, age in enumerate(age_grid):
        for feh_index, feh in enumerate(feh_grid):
            iso = isochrones[(age, feh)]
            masses = iso["mass"].to_numpy()
            sort_idx = np.argsort(masses)
            masses_sorted = masses[sort_idx]
            for band in _BANDS:
                values = iso[band].to_numpy()[sort_idx]
                magnitude_grids[band][:, age_index, feh_index] = np.interp(
                    mass_grid, masses_sorted, values
                )

    interpolators = {
        band: RegularGridInterpolator(
            (mass_grid, age_grid, feh_grid),
            magnitude_grids[band],
            bounds_error=False,
            fill_value=np.nan,
        )
        for band in _BANDS
    }

    return interpolators, (mass_grid, age_grid, feh_grid)


def _get_interpolators():
    global _INTERPOLATORS, _GRIDS
    if _INTERPOLATORS is None:
        _INTERPOLATORS, _GRIDS = _build_interpolators()
    return _INTERPOLATORS, _GRIDS


def get_model_mag(mass, age, feh):
    """Return interpolated Gaia and NIR magnitudes for given mass, age, [Fe/H].

    Args:
        mass: Stellar mass in solar masses.
        age: Stellar age in years.
        feh: Metallicity [Fe/H].

    Returns:
        Tuple of (G, BP, RP, J, H, K) magnitudes. Scalars for scalar inputs, arrays otherwise.
    """
    interpolators, _ = _get_interpolators()

    mass_arr = np.asarray(mass)
    age_arr = np.asarray(age)
    feh_arr = np.asarray(feh)
    mass_arr, age_arr, feh_arr = np.broadcast_arrays(mass_arr, age_arr, feh_arr)

    points = np.column_stack(
        [mass_arr.ravel(), age_arr.ravel(), feh_arr.ravel()]
    )

    outputs = {}
    for band in _BANDS:
        outputs[band] = interpolators[band](points).reshape(mass_arr.shape)

    if mass_arr.shape == ():
        return (
            outputs["G"].item(),
            outputs["BP"].item(),
            outputs["RP"].item(),
            outputs["J"].item(),
            outputs["H"].item(),
            outputs["K"].item(),
        )

    return (
        outputs["G"],
        outputs["BP"],
        outputs["RP"],
        outputs["J"],
        outputs["H"],
        outputs["K"],
    )
