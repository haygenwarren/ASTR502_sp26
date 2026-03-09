from the_interpolator import load_catalogs, get_bestfit_model_mag_for_star, brute_force_likelihood

load_catalogs(
    "/Users/haygenwarren/Classes/ASTR502/ASTR502_Mega_Target_List.csv",
    "/Users/haygenwarren/Classes/ASTR502/ASTR502_Master_Photometry_List.csv"
)

# observed_mags = {'G': 7.409338833308387, 'BP': 8.411417012861971, 'RP': 6.442523729007716, 'J': 5.06967124945907, 'H': 4.298705321707759, 'K': 4.129954160927307, 'W1': 4.080113523803485, 'W2': 4.049163771576428, 'W3': 3.9371307524328376, 'W4': 3.872219952729174, 'g': 8.988348945552788, 'r': 7.626684040241528, 'i': 6.862445040926976, 'z': 6.3955482575074765}
# observed_errs = {'G': 0.05, 'BP': 0.05, 'RP': 0.05, 'J': 0.05, 'H': 0.05, 'K': 0.05, 'W1': 0.05, 'W2': 0.05, 'W3': 0.05, 'W4': 0.05, 'g': 0.05, 'r': 0.05, 'i': 0.05, 'z': 0.05}
#
# mass, age, feh = brute_force_likelihood(observed_mags, observed_errs)
# print(f"Brute-force likelihood results: mass={mass}, age={age}, feh={feh}")

(best_params, mags) = get_bestfit_model_mag_for_star(
    "KELT-17",
    sigma_phot=0.5,
    fallback_sigma_param=0.1,
    verbose=True
)

print("Best params:", best_params)
print("Model mags:", mags)