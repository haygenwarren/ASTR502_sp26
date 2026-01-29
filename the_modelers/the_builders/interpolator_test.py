from the_interpolator import get_model_mag, brute_force_likelihood

observed_mags = {"g": 16.4520000, "r": 14.8454000, "i": 13.668100, "z": 12.6126000}
observed_errs = {"f": 0.0040676, "r": 0.0037485, "i": 0.000602, "z": 0.0039562}

mass, age, feh = brute_force_likelihood(observed_mags, observed_errs)
print(f"Brute-force likelihood results: mass={mass}, age={age}, feh={feh}")

model_mags = get_model_mag(mass, age, feh)
print("Model magnitudes at best-fit parameters:", model_mags)

# master_list = {"G": 8.80, "BP": 9.75, "RP": 7.92}
# comparison = {
#     band: model_mags[band] - master_list[band]
#     for band in master_list
#     if band in model_mags
# }
# print("Model - master list differences:", comparison)
