import math

# --- User-Defined Scaling Law Parameters ---
# These MUST be fitted from your experimental data.

# For Scaling Law 1 (PPL (=L_hat) Prediction)
# L_hat = A * ( (r_vision^alpha_v * r_llm^alpha_l * D_f^beta)^(-1) ) + E_loss
# L(r_vision, r_llm, Df) = A / ((r_vision^alpha_v) * (r_llm^alpha_l) * (Df^beta)) + E
A_PARAM = 1.0e5      # Example: Scaling constant for the loss term
E_LOSS_PARAM = 0.05  # Example: Irreducible error component of the loss
ALPHA_V_PARAM = 0.1  # Example: Scaling exponent for VE rank in loss
ALPHA_L_PARAM = 0.15 # Example: Scaling exponent for LLM rank in loss
BETA_PARAM = 0.2     # Example: Scaling exponent for dataset size in loss

# For Scaling Law 2 (Convergence Steps)
# t_vision = K_V * (r_vision^GAMMA_V) * (D_f^DELTA_V)
# t_llm = K_L * (r_llm^GAMMA_L) * (D_f^DELTA_L)
# GAMMA_V and GAMMA_L are crucial for the pruning rule.
# K_V, DELTA_V, K_L, DELTA_L are used for reporting/verification of t_vision, t_llm.
K_V_PARAM = 1000.0     # Example: Constant for t_vision
GAMMA_V_PARAM = 0.5    # Example: Exponent for r_vision in t_vision (MUST NOT BE ZERO for pruning)
DELTA_V_PARAM = 0.3    # Example: Exponent for D_f in t_vision

K_L_PARAM = 1200.0     # Example: Constant for t_llm
GAMMA_L_PARAM = 0.6    # Example: Exponent for r_llm in t_llm
DELTA_L_PARAM = 0.3    # Example: Exponent for D_f in t_llm

# --- Search Configuration ---
# Define the discrete sets of possible LoRA ranks to consider
POSSIBLE_RANKS_VE = [4, 8, 16, 32, 64, 128, 256]  # Example Vision Encoder ranks
POSSIBLE_RANKS_LLM = [4, 8, 16, 32, 64, 128, 256, 512] # Example LLM ranks
FIXED_DATASET_SIZE = 100000  # Example: Number of image-text pairs in fine-tuning dataset (D_f)

# --- Scaling Law Function Implementations ---

def calc_scaling_laws1(r_vision, r_llm, D_f, A, E_loss, alpha_v, alpha_l, beta):
    """
    Calculates predicted fine-tuning loss (L_hat) based on Scaling Law 1.
    Formula: L_hat = A / ((r_vision^alpha_v) * (r_llm^alpha_l) * (D_f^beta)) + E_loss
    Lower loss corresponds to higher accuracy.
    """
    if r_vision <= 0 or r_llm <= 0 or D_f <= 0:
        # Ranks and dataset size must be positive
        return float('inf')
    try:
        # Calculate the denominator term: (r_vision^alpha_v) * (r_llm^alpha_l) * (D_f^beta)
        denominator_term = (r_vision**alpha_v) * (r_llm**alpha_l) * (D_f**beta)
        if denominator_term == 0:
            # Avoid division by zero if any component makes the term zero
            return float('inf')
        # Calculate the loss
        predicted_L = A / denominator_term + E_loss
        return predicted_L
    except (OverflowError, ZeroDivisionError):
        # Handle potential math errors (e.g., very large numbers)
        return float('inf')

def calc_scaling_laws2_ve(r_vision, D_f, k_v, gamma_v, delta_v):
    """
    Calculates predicted convergence steps for the Vision Encoder (t_vision)
    using Scaling Law 2.
    Formula: t_vision = k_v * (r_vision^gamma_v) * (D_f^delta_v)
    """
    if r_vision <= 0 or D_f <= 0:
        return float('inf')
    try:
        return k_v * (r_vision**gamma_v) * (D_f**delta_v)
    except (OverflowError, ZeroDivisionError):
        return float('inf')

def calc_scaling_laws2_llm(r_llm, D_f, k_l, gamma_l, delta_l):
    """
    Calculates predicted convergence steps for the LLM (t_llm)
    using Scaling Law 2.
    Formula: t_llm = k_l * (r_llm^gamma_l) * (D_f^delta_l)
    """
    if r_llm <= 0 or D_f <= 0:
        return float('inf')
    try:
        return k_l * (r_llm**gamma_l) * (D_f**delta_l)
    except (OverflowError, ZeroDivisionError):
        return float('inf')

# --- Procedure Step 1 & 2: Generate Balanced Rank Pairs based on Convergence Speed Relationship ---

def find_matching_ve_rank(r_llm, gamma_l, gamma_v):
    """
    Calculates the ideal r_vision that aims to balance convergence speeds with r_llm.
    This implements the relationship: r_vision ~ (r_llm)^(gamma_l / gamma_v)
    (This is your Equation 7, derived from t_vision ~ t_llm under simplifying assumptions).
    """
    if gamma_v == 0:
        print("Error: GAMMA_V_PARAM (gamma_v) cannot be zero for the pruning rule r_vision ~ (r_llm)^(gamma_l/gamma_v).")
        return float('nan') # Not a number, indicates an issue
    if r_llm <= 0: # Rank must be positive
        return float('nan')
    try:
        # Calculate the exponent for r_llm
        power_ratio = gamma_l / gamma_v
        # Calculate ideal r_vision
        r_vision_ideal = r_llm ** power_ratio
        return r_vision_ideal
    except (ValueError, OverflowError, ZeroDivisionError):
        # Handle potential math errors
        return float('nan')

def generate_balanced_candidate_pairs(
    possible_r_llm_values, possible_r_ve_values,
    gamma_l_param, gamma_v_param
):
    """
    Generates a pruned set of (r_ve, r_llm) pairs.
    It iterates through possible r_llm values, calculates an ideal r_ve
    using the simplified convergence balance relationship, and then picks the
    closest available discrete r_ve.
    """
    candidate_pairs = []
    if not possible_r_ve_values: # Check if the list of VE ranks is empty
        print("Warning: POSSIBLE_RANKS_VE is empty. No VE ranks to select from for pruning.")
        return []

    for r_llm_current in possible_r_llm_values:
        # Calculate the ideal r_vision for the current r_llm to balance convergence speeds
        r_vision_ideal = find_matching_ve_rank(
            r_llm_current, gamma_l_param, gamma_v_param
        )

        if not math.isnan(r_vision_ideal) and r_vision_ideal > 0:
            # If a valid ideal r_vision is found, find the closest discrete rank
            # from the predefined list of possible VE ranks.
            closest_r_vision_actual = min(
                possible_r_ve_values,
                key=lambda r_ve_actual: abs(r_ve_actual - r_vision_ideal)
            )
            candidate_pairs.append((closest_r_vision_actual, r_llm_current))

    # Convert to a set to remove duplicate pairs (if any), then back to a sorted list.
    unique_candidate_pairs = sorted(list(set(candidate_pairs)))
    return unique_candidate_pairs

# --- Main Execution Block ---

if __name__ == "__main__":
    print("Starting Adaptive LoRA Rank Search...")
    print(f"Target Dataset Size (D_f): {FIXED_DATASET_SIZE}")
    print(f"Available Vision Encoder Ranks (r_ve): {POSSIBLE_RANKS_VE}")
    print(f"Available LLM Ranks (r_llm): {POSSIBLE_RANKS_LLM}")
    
    # Display the pruning rule being used
    pruning_rule_str = f"r_vision ~ r_llm ^ (GAMMA_L / GAMMA_V)"
    if GAMMA_V_PARAM != 0:
        ratio_val = GAMMA_L_PARAM / GAMMA_V_PARAM
        pruning_rule_str += f" where GAMMA_L/GAMMA_V = {GAMMA_L_PARAM}/{GAMMA_V_PARAM} = {ratio_val:.4f}"
    else:
        pruning_rule_str += f" (Error: GAMMA_V is 0, rule is ill-defined)"
    print(f"Pruning Rule for balanced convergence (from Scaling Law 2): {pruning_rule_str}")
    print("-" * 40)

    # Procedure Step 1 & 2: Generate the pruned set of candidate rank pairs
    # This set contains (r_ve, r_llm) pairs expected to have balanced convergence.
    balanced_candidate_set = generate_balanced_candidate_pairs(
        POSSIBLE_RANKS_LLM,
        POSSIBLE_RANKS_VE,
        GAMMA_L_PARAM,
        GAMMA_V_PARAM
    )

    if not balanced_candidate_set:
        print("No candidate rank pairs were generated after pruning. Check parameters or rank lists.")
    else:
        print(f"Generated {len(balanced_candidate_set)} unique candidate (r_ve, r_llm) pairs satisfying the balance condition:")
        # To avoid printing a very long list, show only a few if many candidates
        if len(balanced_candidate_set) <= 20:
            print(f"  Candidates: {balanced_candidate_set}")
        else:
            print(f"  First 10 candidates: {balanced_candidate_set[:10]}")
            print(f"  Last 10 candidates: {balanced_candidate_set[-10:]}")

        print("-" * 40)
        print("Procedure Step 3: Finding optimal pair from pruned set using Scaling Law 1 (minimizing loss)...")

        min_predicted_ppl = float('inf')
        optimal_rank_pair_info = None # Will store dict with details

        for r_ve, r_llm in balanced_candidate_set:
            # Calculate predicted loss for the current pair using Scaling Law 1
            current_ppl = calc_scaling_laws1(
                r_ve, r_llm, FIXED_DATASET_SIZE,
                A_PARAM, E_LOSS_PARAM, ALPHA_V_PARAM, ALPHA_L_PARAM, BETA_PARAM
            )

            # If this pair has lower loss, it's potentially the new optimum
            if current_ppl < min_predicted_ppl:
                min_predicted_ppl = current_ppl
                # Store details of this optimal pair
                optimal_rank_pair_info = {
                    "r_ve": r_ve,
                    "r_llm": r_llm,
                    "ppl": current_ppl
                }
        
        print("-" * 40)
        if optimal_rank_pair_info:
            print("Optimal LoRA Rank Pair Found:")
            print(f"  Vision Encoder Rank (r_ve): {optimal_rank_pair_info['r_ve']}")
            print(f"  LLM Rank (r_llm): {optimal_rank_pair_info['r_llm']}")
            print(f"  Predicted Minimum PPL (L_hat): {optimal_rank_pair_info['ppl']:.6f} (lower is better)")

            # For verification, calculate and display the predicted convergence steps for this optimal pair
            # using the full Scaling Law 2 equations.
            opt_r_ve = optimal_rank_pair_info['r_ve']
            opt_r_llm = optimal_rank_pair_info['r_llm']
            
            t_ve_optimal = calc_scaling_laws2_ve(
                opt_r_ve, FIXED_DATASET_SIZE, K_V_PARAM, GAMMA_V_PARAM, DELTA_V_PARAM
            )
            t_llm_optimal = calc_scaling_laws2_llm(
                opt_r_llm, FIXED_DATASET_SIZE, K_L_PARAM, GAMMA_L_PARAM, DELTA_L_PARAM
            )
            
            print("\n  Predicted convergence characteristics for this optimal pair (from Scaling Law 2):")
            print(f"    t_vision (VE steps): {t_ve_optimal:.0f} steps")
            print(f"    t_llm (LLM steps): {t_llm_optimal:.0f} steps")
            if not (math.isinf(t_ve_optimal) or math.isinf(t_llm_optimal)):
                disparity = abs(t_ve_optimal - t_llm_optimal)
                print(f"    Convergence Disparity |t_vision - t_llm|: {disparity:.0f} steps")
            else:
                print(f"    Convergence Disparity |t_vision - t_llm|: N/A (one or both step counts are infinite)")
        else:
            print("No optimal rank pair found from the pruned set.")
            if balanced_candidate_set:
                 print("This might occur if all candidate pairs resulted in infinite predicted loss,")
                 print("or if GAMMA_V_PARAM is zero, preventing pruning.")
