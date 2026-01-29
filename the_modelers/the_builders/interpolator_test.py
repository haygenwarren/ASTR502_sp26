from the_interpolator import get_model_mag, brute_force_likelihood

observed_mags = {"G": 8.83, "BP": 9.79, "RP": 7.89}
observed_errs = {"G": 0.02, "BP": 0.03, "RP": 0.02}

mass, age, feh = brute_force_likelihood(observed_mags, observed_errs)
print(f"Brute-force likelihood results: mass={mass}, age={age}, feh={feh}")

model_mags = get_model_mag(mass, age, feh)
print("Model magnitudes at best-fit parameters:", model_mags)

master_list = {"G": 8.80, "BP": 9.75, "RP": 7.92}
comparison = {
    band: model_mags[band] - master_list[band]
    for band in master_list
    if band in model_mags
}
print("Model - master list differences:", comparison)
