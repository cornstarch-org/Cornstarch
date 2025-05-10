import numpy as np
from scipy.optimize import minimize
import math

# --- 1. Define the Scaling Law Function for LLM Steps ---
# This function predicts t_llm given input variables and coefficients.
# Coefficients order: [k_v, gamma_v, delta_v]
def predict_steps_llm_scaling_law2(coeffs, r_llm, D_f):
    """
    Predicts convergence steps for LLM (t_llm) based on Scaling Law 2.
    coeffs: A list or array [k_v, gamma_v, delta_v]
    """
    k_v, gamma_v, delta_v = coeffs

    # Basic validation for inputs and coefficients
    if r_llm <= 0 or D_f <= 0:
        return float('inf')
    # k_v is a scaling constant, typically positive.
    # Exponents gamma_v, delta_v can be positive or negative depending on the relationship.
    # For example, gamma_v > 0 if more rank means more steps.
    if k_v < 0: # k_v should generally be positive as steps are positive
        return float('inf')

    try:
        # Ensure r_llm is positive before raising to a potentially fractional power
        if r_llm <= 0 and gamma_v != 0 and gamma_v % 1 != 0 : # More robust check might be needed for complex numbers
             return float('inf')
        if D_f <= 0 and delta_v != 0 and delta_v % 1 != 0:
             return float('inf')

        predicted_t_llm = k_v * (r_llm**gamma_v) * (D_f**delta_v)
        
        if math.isnan(predicted_t_llm) or math.isinf(predicted_t_llm) or predicted_t_llm < 0:
            # Steps should be non-negative
            return float('inf')
        return predicted_t_llm
    except (OverflowError, ZeroDivisionError, ValueError):
        return float('inf')

# --- 2. Define the Huber Loss Function ---
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

# --- 3. Define the Objective Function for LLM Steps ---
def objective_function_llm_steps(coeffs, experimental_data, delta_huber):
    """
    Calculates the total mean Huber loss for t_llm.
    coeffs: Current guess for [k_v, gamma_v, delta_v]
    experimental_data: List of tuples ((r_llm, D_f), observed_t_llm)
    delta_huber: Delta for Huber loss.
    """
    total_h_loss = 0.0
    num_points = len(experimental_data)

    if num_points == 0:
        print("Warning: Experimental data for t_llm is empty.")
        return float('inf')

    for data_point in experimental_data:
        inputs, observed_t_llm = data_point
        r_llm, D_f = inputs

        predicted_t_llm = predict_steps_llm_scaling_law2(coeffs, r_llm, D_f)

        if math.isinf(predicted_t_llm):
            total_h_loss += 1e12 # Large penalty for invalid prediction
            continue

        error = predicted_t_llm - observed_t_llm
        total_h_loss += huber_loss(error, delta_huber)
    
    return total_h_loss / num_points # Mean Huber loss

# --- 4. Prepare Experimental Data for LLM (Example) ---
# YOU MUST REPLACE THIS WITH YOUR ACTUAL EXPERIMENTAL DATA FOR t_llm
# Each item: ((r_llm, D_f), observed_t_llm)
example_experimental_data_llm = [
    # Format: ((r_llm, D_f), observed_t_llm_steps)
    ((8, 50000), 12000),
    ((16, 50000), 15000),
    ((32, 50000), 18000),
    ((64, 50000), 22000),

    ((16, 100000), 20000),
    ((32, 100000), 25000),
    ((64, 100000), 30000),

    ((32, 200000), 35000),
    ((64, 200000), 45000),
    ((128, 200000), 55000),
]

# --- 5. Grid Search & L-BFGS Optimization for LLM ---
initial_guess_ranges_llm = {
    'k_v': [100, 1000, 5000],         # Scaling factor for steps
    'gamma_v': [0.1, 0.5, 1.0, 1.5],  # Exponent for r_llm (can be >1 or <1)
    'delta_v': [0.1, 0.5, 1.0]        # Exponent for D_f
}

DELTA_HUBER_VISION = 100 # Delta for Huber loss, might need tuning based on typical step counts/errors

# Bounds for coefficients [k_v, gamma_v, delta_v]
coeff_bounds_llm = [
    (1e-9, None),     # k_v > 0
    (None, None),     # gamma_v can be positive or negative (though often positive)
    (None, None)      # delta_v can be positive or negative (though often positive)
]
# If you expect gamma_v and delta_v to be positive, you can set lower bound to 1e-9.

best_result_llm = None
min_loss_llm = float('inf')

count_llm = 0
total_combinations_llm = np.prod([len(v) for v in initial_guess_ranges_llm.values()])
print(f"Starting grid search for t_llm ({total_combinations_llm} initializations)...")
print(f"Huber delta for t_llm: {DELTA_HUBER_VISION}")

if not example_experimental_data_llm:
    print("Error: Experimental data for t_llm is empty. Cannot fit.")
else:
    for kv_init in initial_guess_ranges_llm['k_v']:
        for gv_init in initial_guess_ranges_llm['gamma_v']:
            for dv_init in initial_guess_ranges_llm['delta_v']:
                count_llm += 1
                initial_coeffs_llm = np.array([kv_init, gv_init, dv_init])
                
                if count_llm % 10 == 0 or count_llm == 1 or count_llm == total_combinations_llm:
                    print(f"  Running L-BFGS for t_llm init {count_llm}/{total_combinations_llm}...")
                    print(f"    Initial guess: k_v={initial_coeffs_llm[0]:.1e}, g_v={initial_coeffs_llm[1]:.2f}, d_v={initial_coeffs_llm[2]:.2f}")

                result_llm = minimize(
                    objective_function_llm_steps,
                    initial_coeffs_llm,
                    args=(example_experimental_data_llm, DELTA_HUBER_VISION),
                    method='L-BFGS-B',
                    bounds=coeff_bounds_llm,
                    options={'maxiter': 5000, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False}
                )

                if result_llm.success and result_llm.fun < min_loss_llm:
                    min_loss_llm = result_llm.fun
                    best_result_llm = result_llm
                    print(f"    ---> New best t_llm coeffs: k_v={result_llm.x[0]:.3e}, g_v={result_llm.x[1]:.4f}, d_v={result_llm.x[2]:.4f}")
                    print(f"         Minimized Mean Huber Loss for t_llm: {result_llm.fun:.3e}")

    print("\n--- Fitting Complete for LLM Steps (t_llm) ---")
    if best_result_llm:
        print("Best coefficients found for t_llm:")
        print(f"  k_v     : {best_result_llm.x[0]:.4e}")
        print(f"  gamma_v : {best_result_llm.x[1]:.4f}")
        print(f"  delta_v : {best_result_llm.x[2]:.4f}")
        print(f"Final minimized mean Huber loss for t_llm: {min_loss_llm:.6e}")

        print("\nVerification (t_llm): Predicted vs Observed:")
        fitted_coeffs_llm = best_result_llm.x
        for i in range(min(3, len(example_experimental_data_llm))):
            inputs, observed = example_experimental_data_llm[i]
            r_v, Df_val = inputs
            predicted = predict_steps_llm_scaling_law2(fitted_coeffs_llm, r_v, Df_val)
            print(f"  Point {i+1}: Inputs=({r_v},{Df_val}), Obs={observed:.0f}, Pred={predicted:.0f}, Err={predicted-observed:.0f}")
    else:
        print("t_llm optimization failed for all initial guesses.")
