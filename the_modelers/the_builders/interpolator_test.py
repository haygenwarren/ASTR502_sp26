from the_interpolator import get_model_mag, brute_force_likelihood

# observed_mags = {"g": 10.74, "r": 9.13, "i": 7.96, "z": 6.90}
# observed_errs = {"g": 0.005, "r": 0.005, "i": 0.005, "z": 0.005}
#
# mass, age, feh, dm = brute_force_likelihood(observed_mags, observed_errs)
# print(f"Brute-force likelihood results: mass={mass}, age={age}, feh={feh}")

mass = 0.56
age = 9300000
feh = 0.0

model_mags = get_model_mag(mass, age, feh)
print("Model magnitudes at best-fit parameters:", model_mags)

# master_list = {"G": 8.80, "BP": 9.75, "RP": 7.92}
# comparison = {
#     band: model_mags[band] - master_list[band]
#     for band in master_list
#     if band in model_mags
# }
# print("Model - master list differences:", comparison)
