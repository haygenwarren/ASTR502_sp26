from the_interpolator import get_model_mag, brute_force_likelihood
import numpy as np

mass, age, feh = brute_force_likelihood(8.59, 0.005)
print(f"Brute-force likelihood results: mass={mass}, age={age}, feh={feh}")