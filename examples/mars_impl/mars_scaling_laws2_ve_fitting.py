import numpy as np
from scipy.optimize import minimize
import math

# --- 1. Define the Scaling Law Function for Vision Encoder Steps ---
# This function predicts t_vision given input variables and coefficients.
# Coefficients order: [k_v, gamma_v, delta_v]
def predict_steps_vision_scaling_law2(coeffs, r_vision, D_f):
    """
    Predicts convergence steps for Vision Encoder (t_vision) based on Scaling Law 2.
    coeffs: A list or array [k_v, gamma_v, delta_v]
    """
    k_v, gamma_v, delta_v = coeffs

    # Basic validation for inputs and coefficients
    if r_vision <= 0 or D_f <= 0:
        return float('inf')
    # k_v is a scaling constant, typically positive.
    # Exponents gamma_v, delta_v can be positive or negative depending on the relationship.
    # For example, gamma_v > 0 if more rank means more steps.
    if k_v < 0: # k_v should generally be positive as steps are positive
        return float('inf')

    try:
        # Ensure r_vision is positive before raising to a potentially fractional power
        if r_vision <= 0 and gamma_v != 0 and gamma_v % 1 != 0 : # More robust check might be needed for complex numbers
             return float('inf')
        if D_f <= 0 and delta_v != 0 and delta_v % 1 != 0:
             return float('inf')

        predicted_t_vision = k_v * (r_vision**gamma_v) * (D_f**delta_v)
        
        if math.isnan(predicted_t_vision) or math.isinf(predicted_t_vision) or predicted_t_vision < 0:
            # Steps should be non-negative
            return float('inf')
        return predicted_t_vision
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

# --- 3. Define the Objective Function for Vision Encoder Steps ---
def objective_function_vision_steps(coeffs, experimental_data, delta_huber):
    """
    Calculates the total mean Huber loss for t_vision.
    coeffs: Current guess for [k_v, gamma_v, delta_v]
    experimental_data: List of tuples ((r_vision, D_f), observed_t_vision)
    delta_huber: Delta for Huber loss.
    """
    total_h_loss = 0.0
    num_points = len(experimental_data)

    if num_points == 0:
        print("Warning: Experimental data for t_vision is empty.")
        return float('inf')

    for data_point in experimental_data:
        inputs, observed_t_vision = data_point
        r_vision, D_f = inputs

        predicted_t_vision = predict_steps_vision_scaling_law2(coeffs, r_vision, D_f)

        if math.isinf(predicted_t_vision):
            total_h_loss += 1e12 # Large penalty for invalid prediction
            continue

        error = predicted_t_vision - observed_t_vision
        total_h_loss += huber_loss(error, delta_huber)
    
    return total_h_loss / num_points # Mean Huber loss

# --- 4. Prepare Experimental Data for Vision Encoder (Example) ---
# YOU MUST REPLACE THIS WITH YOUR ACTUAL EXPERIMENTAL DATA FOR t_vision
# Each item: ((r_vision, D_f), observed_t_vision)
example_experimental_data_vision = [
    # Format: ((r_vision, D_f), observed_t_vision_steps)
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

# --- 5. Grid Search & L-BFGS Optimization for Vision Encoder ---
initial_guess_ranges_vision = {
    'k_v': [100, 1000, 5000],         # Scaling factor for steps
    'gamma_v': [0.1, 0.5, 1.0, 1.5],  # Exponent for r_vision (can be >1 or <1)
    'delta_v': [0.1, 0.5, 1.0]        # Exponent for D_f
}

DELTA_HUBER_VISION = 100 # Delta for Huber loss, might need tuning based on typical step counts/errors

# Bounds for coefficients [k_v, gamma_v, delta_v]
coeff_bounds_vision = [
    (1e-9, None),     # k_v > 0
    (None, None),     # gamma_v can be positive or negative (though often positive)
    (None, None)      # delta_v can be positive or negative (though often positive)
]
# If you expect gamma_v and delta_v to be positive, you can set lower bound to 1e-9.

best_result_vision = None
min_loss_vision = float('inf')

count_vision = 0
total_combinations_vision = np.prod([len(v) for v in initial_guess_ranges_vision.values()])
print(f"Starting grid search for t_vision ({total_combinations_vision} initializations)...")
print(f"Huber delta for t_vision: {DELTA_HUBER_VISION}")

if not example_experimental_data_vision:
    print("Error: Experimental data for t_vision is empty. Cannot fit.")
else:
    for kv_init in initial_guess_ranges_vision['k_v']:
        for gv_init in initial_guess_ranges_vision['gamma_v']:
            for dv_init in initial_guess_ranges_vision['delta_v']:
                count_vision += 1
                initial_coeffs_vision = np.array([kv_init, gv_init, dv_init])
                
                if count_vision % 10 == 0 or count_vision == 1 or count_vision == total_combinations_vision:
                    print(f"  Running L-BFGS for t_vision init {count_vision}/{total_combinations_vision}...")
                    print(f"    Initial guess: k_v={initial_coeffs_vision[0]:.1e}, g_v={initial_coeffs_vision[1]:.2f}, d_v={initial_coeffs_vision[2]:.2f}")

                result_vision = minimize(
                    objective_function_vision_steps,
                    initial_coeffs_vision,
                    args=(example_experimental_data_vision, DELTA_HUBER_VISION),
                    method='L-BFGS-B',
                    bounds=coeff_bounds_vision,
                    options={'maxiter': 5000, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False}
                )

                if result_vision.success and result_vision.fun < min_loss_vision:
                    min_loss_vision = result_vision.fun
                    best_result_vision = result_vision
                    print(f"    ---> New best t_vision coeffs: k_v={result_vision.x[0]:.3e}, g_v={result_vision.x[1]:.4f}, d_v={result_vision.x[2]:.4f}")
                    print(f"         Minimized Mean Huber Loss for t_vision: {result_vision.fun:.3e}")

    print("\n--- Fitting Complete for Vision Encoder Steps (t_vision) ---")
    if best_result_vision:
        print("Best coefficients found for t_vision:")
        print(f"  k_v     : {best_result_vision.x[0]:.4e}")
        print(f"  gamma_v : {best_result_vision.x[1]:.4f}")
        print(f"  delta_v : {best_result_vision.x[2]:.4f}")
        print(f"Final minimized mean Huber loss for t_vision: {min_loss_vision:.6e}")

        print("\nVerification (t_vision): Predicted vs Observed:")
        fitted_coeffs_vision = best_result_vision.x
        for i in range(min(3, len(example_experimental_data_vision))):
            inputs, observed = example_experimental_data_vision[i]
            r_v, Df_val = inputs
            predicted = predict_steps_vision_scaling_law2(fitted_coeffs_vision, r_v, Df_val)
            print(f"  Point {i+1}: Inputs=({r_v},{Df_val}), Obs={observed:.0f}, Pred={predicted:.0f}, Err={predicted-observed:.0f}")
    else:
        print("t_vision optimization failed for all initial guesses.")
