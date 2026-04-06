from the_interpolator import load_catalogs, get_bestfit_model_mag_for_star, brute_force_likelihood
import matplotlib.pyplot as plt

load_catalogs(
    "/Users/haygenwarren/Classes/ASTR502/ASTR502_Mega_Target_List.csv",
    "/Users/haygenwarren/Classes/ASTR502/ASTR502_Master_Photometry_List.csv"
)

# mass, age, feh = brute_force_likelihood(observed_mags, observed_errs)
# print(f"Brute-force likelihood results: mass={mass}, age={age}, feh={feh}")

(best_params, mags, walker_plots) = get_bestfit_model_mag_for_star(
    "TOI-2497",
    sigma_phot=0.5,
    fallback_sigma_param=0.1,
    verbose=True
)
for fig in walker_plots.values():
    fig.show()
plt.show()
print("Best params:", best_params)
print("Model mags:", mags)