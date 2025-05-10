import numpy as np
import math
from scipy.optimize import minimize

# 1. Define the Scling Law Fundtion
# This function predicts the test perplexity given the input variables and a set of coefficients.
# Coefficients order: [A, E_loss, alpha_v, alpha_l, beta]
def predict_ppl_scaling_law1(coeffs, r_vision, r_llm, D_f):
    """
    Predicts test perplexity based on Scaling Law 1.
    coeffs: A list or array [A, E_loss, alpha_v, alpha_l, beta]
    """
    A, E_loss, alpha_v, alpha_l, beta = coeffs

    # Basic validation for inputs and coefficients to prevent math errors
    if r_vision <= 0 or r_llm <= 0 or D_f <= 0:
        return float('inf')
    if A < 0 or alpha_v < 0 or alpha_l < 0 or beta < 0: # Assuming these should be non-negative
        return float('inf')

    try:
        denominator_term = (r_vision**alpha_v) * (r_llm**alpha_l) * (D_f**beta)
        if denominator_term == 0 or math.isinf(denominator_term) or math.isnan(denominator_term):
            return float('inf')

        term_A_over_denom = A / denominator_term
        if math.isinf(term_A_over_denom):
            return float('inf')

        predicted_ppl = term_A_over_denom + E_loss
        
        if math.isnan(predicted_ppl) or math.isinf(predicted_ppl):
            return float('inf')
        return predicted_ppl
    except (OverflowError, ZeroDivisionError, ValueError):
        return float('inf') 

# 2. Define the Huber Loss Function
def huber_loss(error, delta):
    """
    Calculates Huber loss.
    error: The difference (predicted - observed)
    delta: The threshold for switching between quadratic and linear loss.
    """
    abs_error = np.abs(error)
    if abs_error <= delta:
        return 0.5 * error**2
    else:
        return delta * (abs_error - 0.5 * delta)

# 3. Define the Objective Function (to be minimized by L-BFGS)
def objective_function(coeffs, experimental_data, delta_huber):
    """
    Calculates the total Huber loss for the given coefficients and data.
    coeffs: Current guess for [A, E_loss, alpha_v, alpha_l, beta]
    experimental_data: A list of tuples, where each tuple is 
                       ((r_vision, r_llm, D_f), observed_ppl)
    delta_huber: The delta parameter for Huber loss.
    """
    total_loss = 0.0
    num_points = len(experimental_data)

    if num_points == 0:
        return float('inf') # Or handle as an error

    for data_point in experimental_data:
        inputs, observed_ppl = data_point
        r_vision, r_llm, D_f = inputs

        predicted_ppl = predict_ppl_scaling_law1(coeffs, r_vision, r_llm, D_f)

        if math.isinf(predicted_ppl): # Penalize if prediction is invalid
            total_loss += 1e12 # Add a large penalty
            continue

        error = predicted_ppl - observed_ppl
        total_loss += huber_loss(error, delta_huber)
    
    return total_loss / num_points # Return mean Huber loss

# 4. Prepare Experimental Data (Example)
# YOU MUST REPLACE THIS WITH YOUR ACTUAL EXPERIMENTAL DATA
# Each item: ((r_vision, r_llm, D_f), observed_ppl)
# Example: ( (ve_rank, llm_rank, dataset_size), actual_ppl_achieved )
# More data points across a wider range of inputs will lead to better fitting.

example_experimental_data = [
    ((4, 8, 50000), 0.35),
    ((8, 8, 50000), 0.30),
    ((8, 16, 50000), 0.28),
    ((16, 16, 50000), 0.25),
    ((16, 32, 50000), 0.23),
    ((32, 32, 50000), 0.20),
    ((8, 8, 100000), 0.28),
    ((16, 16, 100000), 0.22),
    ((32, 32, 100000), 0.18),
    ((32, 64, 100000), 0.17),
    ((64, 64, 100000), 0.15),
    ((64, 128, 100000), 0.14),
    ((16, 16, 200000), 0.20),
    ((32, 32, 200000), 0.16),
    ((64, 64, 200000), 0.13),
    ((128, 128, 200000), 0.11),
]

# 5. Grid Search for Initialization Ranges & L-BFGS Optimization
# Define ranges for initial guesses for each coefficient.
# These are just examples; you'll need to adjust based on your expectations.
initial_guess_ranges = {
    'A': [1e4, 1e5, 1e6],          # Scaling factor
    'E_loss': [0.01, 0.05, 0.1],   # Irreducible loss
    'alpha_v': [0.05, 0.1, 0.2],   # Exponent for r_vision
    'alpha_l': [0.05, 0.1, 0.2],   # Exponent for r_llm
    'beta': [0.1, 0.2, 0.3]        # Exponent for D_f
}

# Huber loss delta parameter (as per the snapshot)
DELTA_HUBER = 1e-3

# Bounds for coefficients (L-BFGS-B can use these)
# (A, E_loss, alpha_v, alpha_l, beta)
# Example: A > 0, E_loss > 0 (often), exponents > 0
coeff_bounds = [
    (1e-9, None),     # A > 0
    (1e-9, None),     # E_loss > 0 (can be adjusted if E_loss can be ~0 or negative)
    (1e-9, None),     # alpha_v > 0
    (1e-9, None),     # alpha_l > 0
    (1e-9, None)      # beta > 0
]

best_result = None
min_overall_loss = float('inf')

# Iterate through the grid of initial guesses
# This is a simplified grid search; for many parameters, this can be large.
# You might do a more structured search or random sampling for initial points.
count = 0
total_combinations = np.prod([len(v) for v in initial_guess_ranges.values()])
print(f"Starting grid search for initializations ({total_combinations} combinations)...")

for a_init in initial_guess_ranges['A']:
    for e_init in initial_guess_ranges['E_loss']:
        for av_init in initial_guess_ranges['alpha_v']:
            for al_init in initial_guess_ranges['alpha_l']:
                for b_init in initial_guess_ranges['beta']:
                    count += 1
                    initial_coeffs = np.array([a_init, e_init, av_init, al_init, b_init])
                    if count % 10 == 0 or count == total_combinations:
                        print(f"  Running L-BFGS for initialization {count}/{total_combinations}...")
                        print(f"    Initial guess: {initial_coeffs}")

                    # Run the L-BFGS-B optimization
                    result = minimize(
                        objective_function,
                        initial_coeffs,
                        args=(example_experimental_data, DELTA_HUBER),
                        method='L-BFGS-B',
                        bounds=coeff_bounds,
                        options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-7} # Adjust options as needed
                    )

                    if result.success and result.fun < min_overall_loss:
                        min_overall_loss = result.fun
                        best_result = result
                        print(f"    [Success] New best coefficients found by L-BFGS from this init:")
                        print(f"      Coeffs: {result.x}")
                        print(f"      Minimized Mean Huber Loss: {result.fun:.6e}")
                    elif not result.success:
                        print(f"    [Fail] Optimization failed or did not improve for initial guess: {initial_coeffs}. Message: {result.message}")
                        pass


print("\n--- Fitting Complete ---")
if best_result:
    print("Best coefficients found across all initializations:")
    print(f"  A      : {best_result.x[0]:.4e}")
    print(f"  E_loss : {best_result.x[1]:.4f}")
    print(f"  alpha_v: {best_result.x[2]:.4f}")
    print(f"  alpha_l: {best_result.x[3]:.4f}")
    print(f"  beta   : {best_result.x[4]:.4f}")
    print(f"Final minimized mean Huber loss: {min_overall_loss:.6e}")

    # You would now use these best_result.x coefficients in your
    # lora_rank_search_script_v2.py for A_PARAM, E_LOSS_PARAM, etc.

    print("\nVerification: Predicted vs Observed for a few points using fitted coeffs:")
    fitted_coeffs = best_result.x
    for i in range(min(5, len(example_experimental_data))): # Show first 5
        inputs, observed = example_experimental_data[i]
        r_v, r_l, Df = inputs
        predicted = predict_ppl_scaling_law1(fitted_coeffs, r_v, r_l, Df)
        print(f"  Point {i+1}: Inputs=({r_v},{r_l},{Df}), Observed={observed:.4f}, Predicted={predicted:.4f}, Error={predicted-observed:.4f}")

else:
    print("Optimization was not successful for any initial guess, or no initial guesses were provided.")
    print("Consider adjusting initialization ranges, bounds, or optimization options.")
    print("Also, ensure your experimental_data is populated and diverse.")

