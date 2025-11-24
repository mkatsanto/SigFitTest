
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize
import scipy.stats as stats
import warnings
from sklearn.preprocessing import normalize
import multiprocessing
import os
from sklearn.decomposition import PCA
        
        

def create_enhanced_unknown_signatures(W, method='orthogonal', n_unknown=None):
    """
    Create biologically meaningful unknown signatures instead of identity matrix
    """
    n_mut_types = W.shape[0]
    
    if n_unknown is None:
        n_unknown = min(n_mut_types, 20)  # Reasonable limit
    
    if method == 'orthogonal':
      
        # Get orthogonal space to W
        U, s, Vt = np.linalg.svd(W, full_matrices=True)
        
        # Take null space vectors (orthogonal to W)
        null_space = U[:, len(s):]  # Columns orthogonal to W's column space
        
        # Limit to requested number
        n_available = null_space.shape[1]
        n_to_use = min(n_unknown, n_available)
        
        if n_to_use > 0:
            # Use available null space vectors
            unknown_sigs = np.abs(null_space[:, :n_to_use])
            
            # Normalize each signature to sum to 1
            valid_signatures = []
            for i in range(unknown_sigs.shape[1]):
                col_sum = np.sum(unknown_sigs[:, i])
                if col_sum > 0:
                    normalized_sig = unknown_sigs[:, i] / col_sum
                    valid_signatures.append(normalized_sig)

            if len(valid_signatures) > 0:
                unknown_sigs = np.column_stack(valid_signatures)
            else:
                # If all signatures are invalid, create one minimal valid signature
                print("Warning: All orthogonal signatures had zero sums, creating minimal signature")
                unknown_sigs = np.ones((n_mut_types, 1)) / n_mut_types
        else:
            # No null space available, create random orthogonal signatures
            print("no null space available, creating random orthogonal signatures")
            unknown_sigs = np.random.random((n_mut_types, n_unknown))
            # Normalize
            for i in range(n_unknown):
                unknown_sigs[:, i] = unknown_sigs[:, i] / np.sum(unknown_sigs[:, i])
    else:  # 'identity'
        unknown_sigs = np.eye(n_mut_types)[:, :n_unknown]
    
    return unknown_sigs



def advanced_signature_filtering(W_extended, original_n_sigs, 
                               correlation_threshold=0.90, 
                               redundancy_threshold=0.99):
    """
    Advanced filtering with multiple criteria:
    1. Remove signatures highly correlated with originals
    2. Remove dummy signatures that are redundant with each other
    3. Keep most informative signatures
    """
    
    correlation_matrix = np.corrcoef(W_extended.T)
    n_sigs = W_extended.shape[1]
    
    to_remove = set()
    
    # Step 1: Remove dummy signatures correlated with COSMIC signatures
    for cosmic_idx in range(original_n_sigs):
        for dummy_idx in range(original_n_sigs, n_sigs):
            if dummy_idx not in to_remove:
                corr = abs(correlation_matrix[cosmic_idx, dummy_idx])
                if corr > correlation_threshold:
                    print(f"Removing dummy {dummy_idx} (corr={corr:.3f} with COSMIC {cosmic_idx})")
                    to_remove.add(dummy_idx)
    
    # Step 2: Remove redundant dummy signatures
    dummy_indices = [i for i in range(original_n_sigs, n_sigs) if i not in to_remove]
    for i, idx1 in enumerate(dummy_indices):
        for idx2 in dummy_indices[i+1:]:
            if idx2 not in to_remove:
                corr = abs(correlation_matrix[idx1, idx2])
                if corr > redundancy_threshold:
                    print(f"Removing redundant dummy {idx2} (corr={corr:.3f} with dummy {idx1})")
                    to_remove.add(idx2)
    
    # Step 3: Check for problematic COSMIC-COSMIC correlations
    for i in range(original_n_sigs):
        for j in range(i+1, original_n_sigs):
            corr = abs(correlation_matrix[i, j])
            if corr > 0.98:  # Very high threshold for COSMIC signatures
                print(f"Warning: High correlation ({corr:.3f}) between COSMIC signatures {i} and {j}")
    
    # Create filtered matrix
    kept_indices = [i for i in range(n_sigs) if i not in to_remove]
    W_filtered = W_extended[:, kept_indices]
    return W_filtered #, kept_indices, list(to_remove)


def custom_attempt1(x, W, thresh_backward=0.001, thresh_forward=None, max_iter=1000, per_trial=True, indices_associated_sigs=None, allow_unknown=False, unknown_penalty=2):
    """Likelihood NNLS with both backward and forward stepwise rountines.

    Notes:
    1. thresh_forward should be greater than thresh_backward. Otherwise the
    loop may run into a dead loop where the same signature is being removed
    and added back within one iteration, although this is caught gracefully in
    the code.
    2. Both thresh_backward and thresh_forward should be nonnegative.
    """
    if thresh_forward is None:
        thresh_forward = thresh_backward
    if thresh_backward > thresh_forward:
        warnings.warn('thresh_backward is greater thresh_forward. This might lead to indefinite loops.', UserWarning)
    
    # Create extended signature matrix W' = [W | I]
    original_n_sigs = W.shape[1]
    n_mut_types = W.shape[0]
    
    if allow_unknown:
        print(f"Using unknown penalty: {unknown_penalty}")
        
        unknown_sigs = create_enhanced_unknown_signatures(W, method="identity", n_unknown=30)
        W_extended = np.column_stack([W, unknown_sigs])
        dummy_indices = set(range(original_n_sigs, W_extended.shape[1]))
        
        

        # Check completeness and linear combinations following proper mathematical principles
        rank_W = np.linalg.matrix_rank(W)
        rank_W_extended = np.linalg.matrix_rank(W_extended)
        n_linear_combos = W_extended.shape[1] - rank_W_extended

        # CORRECTED: Mutational space completeness (does W_extended span R^m?)
        n_mut_types = W_extended.shape[0]  # Dimension of the vector space (m)
        mutational_space_completeness = rank_W_extended / n_mut_types * 100

        # Check if columns are linearly independent
        columns_independent = (n_linear_combos == 0)

        # Check if matrix spans the entire mutation space
        spans_full_space = (rank_W_extended == n_mut_types)

        # Effective new orthogonal directions added
        effective_new_signatures = rank_W_extended - rank_W

        print(f"Original W rank: {rank_W}/{W.shape[1]} signatures")
        print(f"Extended W rank: {rank_W_extended}/{W_extended.shape[1]} signatures")
        print(f"Linear combinations found: {n_linear_combos}")
        print(f"Effective new signatures: {effective_new_signatures}")
        print(f"Mutational space completeness: {mutational_space_completeness:.1f}% ({rank_W_extended}/{n_mut_types})")
        print(f"Columns linearly independent: {columns_independent}")
        print(f"Spans full mutation space: {spans_full_space}")

        # Additional diagnostic: Check if we have a basis
        if spans_full_space and columns_independent:
            print("✓ W_extended forms a complete orthogonal basis for mutation space")
        elif spans_full_space:
            print("✓ W_extended spans mutation space but has redundant signatures")
        elif columns_independent:
            print("→ W_extended is linearly independent but doesn't span full space")
        else:
            print("→ W_extended has redundancies and doesn't span full space")        # identity_matrix = np.eye(n_mut_types)
        
        
        
        # W_extended = np.column_stack([W, identity_matrix])
        # W_extended = advanced_signature_filtering(W_extended, W.shape[1])    
        # dummy_indices = set(range(original_n_sigs, W_extended.shape[1]))
    else:
        W_extended = W.copy()
        dummy_indices = set()
    
    n_sigs_extended = W_extended.shape[1]
    indices_all = np.arange(0, n_sigs_extended)
    
    def objective_function(weights):
        """
        NNLS-equivalent objective: minimize ||W_extended @ weights - x||^2 + L1 penalty
        This replicates sp.optimize.nnls(W, x) behavior with added L1 regularization
        """
        # Ensure non-negativity
        weights = np.maximum(weights, 0)
        
        # Calculate predicted counts: W_extended @ weights
        predicted_counts = W_extended @ weights
        
        # NNLS objective: squared residuals (least squares)
        residuals = predicted_counts - x
        nnls_objective = 0.5 * np.sum(residuals ** 2)  # 0.5 for standard least squares form
        
        # L1 penalty ONLY on dummy signature weights
        l1_penalty = 0.0
        if allow_unknown and len(dummy_indices) > 0:
            dummy_weights = weights[list(dummy_indices)]
            l1_penalty = unknown_penalty * np.sum(dummy_weights)
            
        return nnls_objective + l1_penalty
    
    # Convergence tracking
    convergence_metrics = []
    
    
    
    def callback_function(xk):
        """Enhanced callback to track detailed convergence metrics and convergence issues"""
        obj_val = objective_function(xk)
        residuals = W_extended @ xk - x
        predicted_counts = W_extended @ xk
        
        # Basic metrics
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # Weight distribution metrics
        total_weight = np.sum(xk)
        l1_norm = np.sum(np.abs(xk))
        l2_norm = np.linalg.norm(xk)
        max_weight = np.max(xk)
        min_nonzero_weight = np.min(xk[xk > 1e-10]) if np.any(xk > 1e-10) else 0.0
        
        # Sparsity metrics
        sparsity_1e6 = np.sum(xk > 1e-6)
        sparsity_1e3 = np.sum(xk > 1e-3)
        sparsity_01 = np.sum(xk > 0.01)
        
        # Weight concentration (Gini coefficient-like metric)
        sorted_weights = np.sort(xk)[::-1]
        weight_concentration = np.sum(sorted_weights[:5]) / total_weight if total_weight > 0 else 0
        
        # Penalty components
        l1_penalty = 0.0
        known_weight_sum = 0.0
        unknown_weight_sum = 0.0
        
        if allow_unknown and len(dummy_indices) > 0:
            dummy_weights = xk[list(dummy_indices)]
            l1_penalty = unknown_penalty * np.sum(dummy_weights)
            unknown_weight_sum = np.sum(dummy_weights)
            known_weight_sum = np.sum(xk[:original_n_sigs])
        else:
            known_weight_sum = total_weight
        
        # Gradient information (approximate)
        gradient_norm = 0.0
        condition_number = 0.0
        try:
            # Approximate gradient using finite differences
            eps = 1e-8
            grad_approx = np.zeros_like(xk)
            for i in range(min(10, len(xk))):  # Sample first 10 components
                xk_plus = xk.copy()
                xk_plus[i] += eps
                grad_approx[i] = (objective_function(xk_plus) - obj_val) / eps
            gradient_norm = np.linalg.norm(grad_approx[:10])
            
            # Condition number of active signatures
            active_indices = np.where(xk > 1e-6)[0]
            if len(active_indices) > 1:
                W_active = W_extended[:, active_indices]
                condition_number = np.linalg.cond(W_active.T @ W_active)
        except:
            pass
        
        # Convergence stability metrics
        iteration = len(convergence_metrics) + 1
        objective_change = 0.0
        objective_change_rate = 0.0
        is_oscillating = False
        stagnation_count = 0
        
        if len(convergence_metrics) > 0:
            prev_obj = convergence_metrics[-1]['objective']
            objective_change = obj_val - prev_obj
            objective_change_rate = objective_change / prev_obj if prev_obj != 0 else 0
            
            # Check for oscillation pattern
            if len(convergence_metrics) >= 5:
                recent_changes = [convergence_metrics[i]['objective'] - convergence_metrics[i-1]['objective'] 
                                for i in range(max(1, len(convergence_metrics)-4), len(convergence_metrics))]
                sign_changes = np.sum(np.diff(np.sign(recent_changes)) != 0)
                is_oscillating = sign_changes >= 2
                
            # Check for stagnation
            if len(convergence_metrics) >= 10:
                recent_objectives = [m['objective'] for m in convergence_metrics[-10:]]
                obj_std = np.std(recent_objectives)
                if obj_std < 1e-10 * np.mean(recent_objectives):
                    stagnation_count = 10
        
        # Fit quality metrics
        explained_variance = 1 - np.var(residuals) / np.var(x) if np.var(x) > 0 else 0
        relative_error = np.linalg.norm(residuals) / np.linalg.norm(x) if np.linalg.norm(x) > 0 else 0
        
        # Known vs Unknown signature usage
        known_signatures_active = np.sum(xk[:original_n_sigs] > 1e-6) if allow_unknown else sparsity_1e6
        unknown_signatures_active = np.sum(xk[original_n_sigs:] > 1e-6) if allow_unknown else 0
        
        convergence_metrics.append({
            # Iteration info
            'iteration': iteration,
            'objective': obj_val,
            'objective_change': objective_change,
            'objective_change_rate': objective_change_rate,
            
            # Error metrics
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'explained_variance': explained_variance,
            'relative_error': relative_error,
            
            # Weight distribution
            'total_weight': total_weight,
            'max_weight': max_weight,
            'min_nonzero_weight': min_nonzero_weight,
            'weight_concentration_top5': weight_concentration,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            
            # Sparsity at different thresholds
            'sparsity_1e6': sparsity_1e6,
            'sparsity_1e3': sparsity_1e3,
            'sparsity_01': sparsity_01,
            
            # Known vs Unknown signatures
            'known_weight_sum': known_weight_sum,
            'unknown_weight_sum': unknown_weight_sum,
            'known_signatures_active': known_signatures_active,
            'unknown_signatures_active': unknown_signatures_active,
            'unknown_ratio': unknown_weight_sum / total_weight if total_weight > 0 else 0,
            
            # Penalty components
            'l1_penalty': l1_penalty,
            'data_fit_component': obj_val - l1_penalty,
            'penalty_ratio': l1_penalty / obj_val if obj_val > 0 else 0,
            
            # Optimization health
            'gradient_norm': gradient_norm,
            'condition_number': condition_number,
            'is_oscillating': is_oscillating,
            'stagnation_count': stagnation_count,
        })
        
        # Enhanced warnings with specific diagnostics
        if iteration > 5:
            # Oscillation detection with specific warning
            if is_oscillating:
                print(f"Warning: Objective oscillating at iteration {iteration}")
                
            # Stagnation detection
            if len(convergence_metrics) >= 10:
                recent_changes = [abs(m['objective_change_rate']) for m in convergence_metrics[-5:]]
                if all(change < 1e-8 for change in recent_changes):
                    print(f"Warning: Optimization stagnating at iteration {iteration}")
            
            # Condition number warning - matches Jupyter parsing format
            if condition_number > 1e12:
                print(f"Warning: Poor conditioning (cond={condition_number:.2e}) at iteration {iteration}")
            
            # Weight distribution warnings
            if weight_concentration > 0.99:
                print(f"Warning: Extreme weight concentration at iteration {iteration}")
                
            # Unknown signature dominance warning
            if allow_unknown and unknown_weight_sum / total_weight > 0.5:
                print(f"Warning: Unknown signatures dominating ({unknown_weight_sum/total_weight:.1%}) at iteration {iteration}")
        
        # Progress reporting every 20 iterations - matches Jupyter parsing format
        if iteration % 20 == 0:
            print(f"Iteration {iteration}: obj={obj_val:.3e}, rmse={rmse:.6f}, "
                f"active_sigs={sparsity_1e6}, unknown_ratio={unknown_weight_sum/total_weight:.2%}")
        
    
    
   
# #UNKNOWN IMPLEMENTATION
# def adaptive_unknown_generation(x, W, max_unknowns=10):
#     """
#     Generate unknown signatures based on the actual residual patterns in the data
#     """
#     # Start with NNLS fit
#     h_nnls, _ = sp.optimize.nnls(W, x)
#     residuals = x - W @ h_nnls
    
#     unknown_sigs = []
    
#     # Method 1: Use residual directly as unknown signature
#     if np.sum(np.abs(residuals)) > 0.01 * np.sum(x):
#         residual_sig = np.maximum(residuals, 0)  # Only positive residuals
#         if np.sum(residual_sig) > 0:
#             residual_sig = residual_sig / np.sum(residual_sig)  # Normalize
#             unknown_sigs.append(residual_sig)
    
#     # Method 2: Generate signatures for dominant residual patterns
#     residual_magnitude = np.abs(residuals)
#     top_residual_indices = np.argsort(residual_magnitude)[-5:]  # Top 5 residual positions
    
#     for i, idx in enumerate(top_residual_indices):
#         if len(unknown_sigs) < max_unknowns:
#             # Create focused signature for this mutation type
#             focused_sig = np.zeros_like(residuals)
#             focused_sig[idx] = 1.0  # Pure signature for this mutation type
#             unknown_sigs.append(focused_sig)
    
#     # Method 3: Add some general unknown signatures
#     while len(unknown_sigs) < min(max_unknowns, 5):
#         # Random sparse signature
#         sparse_sig = np.zeros(len(x))
#         active_positions = np.random.choice(len(x), size=np.random.randint(2, 4), replace=False)
#         sparse_sig[active_positions] = np.random.dirichlet(np.ones(len(active_positions)))
#         unknown_sigs.append(sparse_sig)
    
#     if len(unknown_sigs) == 0:
#         # Fallback to identity
#         return np.eye(len(x))[:, :max_unknowns]
    
#     return np.column_stack(unknown_sigs)

# # Usage in your function:
# if allow_unknown:
#     print(f"Using unknown penalty: {unknown_penalty}")
#     unknown_sigs = adaptive_unknown_generation(x, W, max_unknowns=15)
#     W_extended = np.column_stack([W, unknown_sigs])
#     # ... rest of the code
    
    
    
    ## Channel based initializationL
    # sig_activity = np.sum(W_extended > 1e-6, axis=0)  # Count non-zero rows for each signature
    # max_activity = np.max(sig_activity)
    # h_init = np.zeros(n_sigs_extended)
    # # Get NNLS solution as base
    # h_known, _ = sp.optimize.nnls(W, x)  # Original matrix only

    ### 
    # if max_activity > 0:
    #     # Normalize activity scores to [1, 2] range for weighting
    #     h_init[:original_n_sigs] = h_known * (1.0 + (sig_activity[:original_n_sigs] / max_activity))
    #     h_init[original_n_sigs:] = 0.0 
    #     total_weight = np.sum(h_init)
    # if total_weight > 0:
    #     h_init = h_init / total_weight
    # else:
    #     # Fallback if all weights are zero
    #     h_init[0] = 1.0
        
    
    # Initialize with random weights for known signatures, zero for unknown
    # h_init = np.zeros(n_sigs_extended)

    # if allow_unknown:
    #     # Random weights for known signatures
    #     np.random.seed(42)  # For reproducibility
    #     h_init[:original_n_sigs] = np.random.uniform(0.01, 0.1, original_n_sigs)
    #     # Zero weights for unknown signatures
    #     h_init[original_n_sigs:] = 0.0
    # else:
    #     # Random weights for all signatures when not allowing unknown
    #     np.random.seed(42)
    #     h_init = np.random.uniform(0.01, 0.1, n_sigs_extended)

    # # Normalize to ensure weights sum to 1
    # h_init = h_init / np.sum(h_init)
    
    
    ##### NNLS based initialization - DEPRECATED
    h_init = np.zeros(n_sigs_extended)
    if allow_unknown:
        # Initialize known signatures with stable NNLS
        try:
            h_known, _ = sp.optimize.nnls(W, x)  # Original matrix only
            h_init[:original_n_sigs] = h_known
            # Start with ZERO weights for unknown signatures (identity matrix columns)
            h_init[original_n_sigs:] = 0.0  # Explicit zero initialization
        except:
            # Deterministic fallback: small weights for known, zero for unknown
            h_init[:original_n_sigs] = 1e-6
            h_init[original_n_sigs:] = 0.0  # Keep unknowns at zero
    else:
        # For original matrix, use more stable initialization
        try:
            # Use least squares solution projected to positive
            h_ls = np.linalg.lstsq(W_extended, x, rcond=None)[0]
            h_init = np.maximum(h_ls, 1e-6)  # Deterministic minimum
        except:
            h_init = np.ones(n_sigs_extended) * 1e-6  # Deterministic uniform
            
            

            
    # Set up bounds: all weights must be non-negative
    bounds = [(0, None) for _ in range(n_sigs_extended)]
    
    # Solve directly using minimize with the NNLS-equivalent objective function
    try:
        # Record initial state
        callback_function(h_init)
        
        #the one that behaves like nnls when l1=0
        result = minimize(
            fun=objective_function,
            x0=h_init,
            constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1},
            method='L-BFGS-B',
            bounds=bounds,
            callback=callback_function,
            options={
                'maxiter': 1000,        
                'maxfun': 150000,        # Higher function evaluation limit
                'ftol': 1e-12,          # Tighter function tolerance to match NNLS precision
                'gtol': 1e-8,           # Tighter gradient tolerance
                'eps': 1e-8,            
                'disp': False}
        )
        # result = minimize(
        #     fun=objective_function,
        #     x0=h_init,
        #     method='TNC',  # Often more stable than L-BFGS-B for constrained problems
        #     bounds=bounds,
        #     callback=callback_function,
        #     options={
        #         'maxiter': 1000,
        #         'ftol': 1e-8,
        #         'disp': True
        #     }
        # )
    
        if result.success:
            h_final = result.x
            print(f"Optimization converged successfully in {result.nit} iterations")
        else:
            print(f"Optimization failed: {result.message}")
            # Try with different initialization
            h_init_alt = np.random.uniform(0, 0.01, n_sigs_extended)
            convergence_metrics.clear()  # Reset metrics for second attempt
            callback_function(h_init_alt)
            
            result2 = minimize(
                fun=objective_function,
                x0=h_init_alt,
                method='L-BFGS-B',
                bounds=bounds,
                callback=callback_function,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-8}
            )
            h_final = result2.x if result2.success else h_init_alt
            print(f"Second attempt: {'Success' if result2.success else 'Failed'}")
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        h_final = h_init
    
    # Print convergence metrics as DataFrame
    if convergence_metrics:
        convergence_df = pd.DataFrame(convergence_metrics)
        print("\n=== Convergence Metrics ===")
        print(convergence_df.round(6))
        print(f"\nFinal objective value: {convergence_df.iloc[-1]['objective']:.6f}")
        print(f"Final RMSE: {convergence_df.iloc[-1]['rmse']:.6f}")
        print(f"Final sparsity: {convergence_df.iloc[-1]['sparsity_1e6']}/{n_sigs_extended} signatures")
    
    return h_final




# def custom_likelihood_bidirectional(x, W, thresh_backward=0.001, thresh_forward=None, max_iter=1000, per_trial=True, indices_associated_sigs=None, allow_unknown=False, unknown_penalty=2):
#     """Likelihood NNLS with both backward and forward stepwise routines using objective function optimization."""    
    
#     if thresh_forward is None:
#         thresh_forward = thresh_backward
#     if thresh_backward > thresh_forward:
#         warnings.warn('thresh_backward is greater thresh_forward. This might lead to indefinite loops.', UserWarning)
    
#     # Create extended signature matrix W' = [W | I]
#     original_n_sigs = W.shape[1]
#     n_mut_types = W.shape[0]
    
#     if allow_unknown:
#         print(f"Using unknown penalty: {unknown_penalty}")
#         identity_matrix = np.eye(n_mut_types)
#         W_extended = np.column_stack([W, identity_matrix])
#         dummy_indices = set(range(original_n_sigs, W_extended.shape[1]))
#     else:
#         W_extended = W.copy()
#         dummy_indices = set()
    
#     n_sigs_extended = W_extended.shape[1]
#     indices_all = np.arange(0, n_sigs_extended)

#     # SAFETY CHECK: Ensure x has non-zero counts
#     if np.sum(x) == 0:
#         print("Warning: Input x has zero total counts, returning minimal solution")
#         h_fallback = np.zeros(n_sigs_extended)
#         h_fallback[0] = 1.0
#         return h_fallback

#     def objective_function(weights):
#         """
#         Standard signature fitting: predicted_counts = W_extended @ weights
#         """
#         # Ensure non-negativity
#         weights = np.maximum(weights, 0)
        
#         # Standard predicted counts: W_extended @ weights
#         predicted_counts = W_extended @ weights
#         predicted_counts = np.maximum(predicted_counts, 1e-16)  # Avoid log(0)
        
#         # Check for valid predictions
#         if np.sum(predicted_counts) <= 0:
#             return 1e10
        
#         # Likelihood calculation
#         try:
#             if per_trial and np.sum(x) > 0:
#                 # Multinomial formulation (normalized)
#                 p_pred = predicted_counts / np.sum(predicted_counts)
#                 p_pred = np.clip(p_pred, 1e-16, 1.0)
#                 neg_log_likelihood = -np.sum(x * np.log(p_pred)) / np.sum(x)
#             else:
#                 # Poisson formulation
#                 neg_log_likelihood = -np.sum(x * np.log(predicted_counts) - predicted_counts) 
#         except:
#             return 1e10
        
#         # L1 penalty ONLY on dummy signature weights
#         l1_penalty = 0.0
#         if allow_unknown and len(dummy_indices) > 0:
#             dummy_weights = weights[list(dummy_indices)]
#             l1_penalty = unknown_penalty * np.sum(dummy_weights) #* np.sum(x)        
#         return neg_log_likelihood + l1_penalty

#     def optimize_weights(selected_indices):
#         """Helper function to optimize weights for given signature indices"""
        
#         if len(selected_indices) == 0:
#             return np.zeros(n_sigs_extended), np.inf
        
#         # Set up bounds: only selected signatures can be positive
#         bounds = [(0, None) if i in selected_indices else (0, 0) for i in range(n_sigs_extended)]
        
#         # Initialize with NNLS solution
#         h_init = np.zeros(n_sigs_extended)
#         try:
#             h_nnls, _ = sp.optimize.nnls(W_extended[:, selected_indices], x)
#             h_init[selected_indices] = h_nnls
#         except:
#             h_init[selected_indices[0]] = 1.0  # Minimal fallback
        
#         # Optimize with objective function
#         try:
#             result = minimize(
#                 fun=objective_function,
#                 x0=h_init,
#                 method='L-BFGS-B',
#                 bounds=bounds,
#                 options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-5, 'disp': False},
#             )
#             if result.success:
#                 return result.x, result.fun
#             else:
#                 # Fallback to NNLS
#                 h_fallback = np.zeros(n_sigs_extended)
#                 h_nnls, _ = sp.optimize.nnls(W_extended[:, selected_indices], x)
#                 h_fallback[selected_indices] = h_nnls
#                 obj_val = objective_function(h_fallback)
#                 return h_fallback, obj_val
#         except:
#             # Fallback to NNLS
#             h_fallback = np.zeros(n_sigs_extended)
#             h_nnls, _ = sp.optimize.nnls(W_extended[:, selected_indices], x)
#             h_fallback[selected_indices] = h_nnls
#             obj_val = objective_function(h_fallback)
#             return h_fallback, obj_val
#     try:
#         h_all, _ = optimize_weights(indices_all)
#         indices_retained = indices_all[h_all > 1e-8]
#         print("Initial optimization successful")

#     except Exception as e:
#         print(f"Initialization error: {e}")
#         indices_retained = np.array([0])
    
#     # SAFETY CHECK
#     if len(indices_retained) == 0:
#         indices_retained = np.array([0])
#         print("Emergency fallback to first signature")
    
#     print(f"Initial selection: {len(indices_retained)} signatures")
    
#     ### Bidirectional loop
#     i_iter = 0
#     while i_iter < max_iter:
#         i_iter += 1
        
#         if len(indices_retained) == 0:
#             indices_retained = np.array([0])
        
#         ########################## Backward Selection ##########################
#         if len(indices_retained) <= 1:
#             backward_stop = True
#         else:
#             try:
#                 # Get current model objective value
#                 h_current, obj_current = optimize_weights(indices_retained)
                                
#                 # Test removing each signature
#                 best_removal_obj = np.inf
#                 best_remove_idx = None
                
#                 for idx_to_remove in indices_retained:
#                     _indices = np.array([i for i in indices_retained if i != idx_to_remove])
                    
#                     if len(_indices) > 0:
#                         h_test, obj_test = optimize_weights(_indices)
                        
#                         if obj_test < best_removal_obj:
#                             best_removal_obj = obj_test
#                             best_remove_idx = idx_to_remove
                
#                 # Check if removal improves or worsens the objective beyond threshold
#                 obj_diff = best_removal_obj - obj_current
#                 if obj_diff >= thresh_backward:  # Worse by more than threshold
#                     backward_stop = True

#                 else:
#                     backward_stop = False
#                     indices_retained = np.array([i for i in indices_retained if i != best_remove_idx])
                    
#                     if len(indices_retained) == 0:
#                         indices_retained = np.array([0])
#                         backward_stop = True
                        
#             except Exception as e:
#                 print(f"Backward selection error: {e}")
#                 backward_stop = True
        
#         ########################## Forward Selection ##########################
#         indices_others = np.array(sorted(list(set(indices_all) - set(indices_retained))))
#         if len(indices_others) == 0:
#             forward_stop = True
#         else:
#             try:
#                 # Get current model objective value
#                 h_current, obj_current = optimize_weights(indices_retained)
                
#                 # BIAS: Test known signatures first
#                 if allow_unknown:
#                     known_candidates = [idx for idx in indices_others if idx < original_n_sigs]
#                     dummy_candidates = [idx for idx in indices_others if idx >= original_n_sigs]
#                     test_order = known_candidates + dummy_candidates[:]  # Limit dummy testing
#                 else:
#                     test_order = indices_others[:20]  # Limit total testing
                
#                 best_addition_obj = obj_current
#                 best_add_idx = None
                
#                 for idx_to_add in test_order:
#                     _indices = np.sort(np.append(indices_retained, idx_to_add))
                    
#                     h_test, obj_test = optimize_weights(_indices)
                    

#                     if obj_test < best_addition_obj:
#                         best_addition_obj = obj_test
#                         best_add_idx = idx_to_add
                
#                 # Check if addition improves objective beyond threshold
#                 obj_diff = obj_current - best_addition_obj
#                 if obj_diff <= thresh_forward:  # Improvement less than threshold
#                     forward_stop = True
#                 else:
#                     forward_stop = False
#                     indices_retained = np.sort(np.append(indices_retained, best_add_idx))
                    
#             except Exception as e:
#                 print(f"Forward selection error: {e}")
#                 forward_stop = True
        
#         ######################## Stopping Criterion ########################
#         if backward_stop and forward_stop:
#             break
        
#         # Prevent infinite loops
#         if i_iter > 10 and len(indices_retained) > 50:
#             print("Too many signatures selected, stopping early")
#             break
    
#     print(f"Bidirectional selection completed: {len(indices_retained)} signatures retained")
#     print(f"Retained signatures: {indices_retained}")   
#     ### Handle associated signatures (only for known signatures)
#     if indices_associated_sigs is not None:
#         for ind_pair in indices_associated_sigs:
#             if any(item in ind_pair for item in indices_retained if item < original_n_sigs):
#                 valid_pairs = [i for i in ind_pair if i < original_n_sigs]
#                 indices_retained = np.unique(np.append(indices_retained, valid_pairs))
    
#     ### Final optimization with selected signatures
#     try:
#         h_final, _ = optimize_weights(indices_retained)
#     except Exception as e:
#         print(f"Final optimization failed: {e}, using NNLS fallback")
#         h_final, _ = sp.optimize.nnls(W_extended[:, indices_retained], x)
#         h_final = _fill_vector(h_final, indices_retained, n_sigs_extended)
    
#     # FINAL CHECK: Ensure we return valid results
#     if np.sum(h_final) == 0:
#         print("Warning: Final result has zero weights, using minimal fallback")
#         h_final = np.zeros(n_sigs_extended)
#         if len(indices_retained) > 0:
#             h_final[indices_retained[0]] = 1.0
#         else:
#             h_final[0] = 1.0

 
#     return h_final

def _fill_vector(x, indices, L):
    if len(x) != len(indices):
        raise ValueError('x and indices are not of the same length.')
    x_filled = np.zeros(L)
    x_filled[indices] = x
    return x_filled

def _multinomial_loglikelihood(x, p, epsilon=1e-16, per_trial=True):
    """Log likelihood of multinomial distribution: logP(x|p).

    Parameters
    ----------
    x : 1-d numpy array
        Observations (counts)
    p : 1-d numpy array
        Event probabilities. p should be summed to 1.

    Notes
    ----------
    1. Result is sum_i x_i log(p_i), ignoring the constant factor independent
        of p.
    2. Scipy.stats.multinomial.logpmf does not work, because it will give -inf
        for cases where there are zeros in p.
    """
    p = p.astype(float)  # In case p is of type int
    p = p.clip(epsilon)
    p = p/np.sum(p)
    if per_trial:
        return np.sum(x*np.log(p))/np.sum(x)
    else:
        return np.sum(x*np.log(p))

def musical_nnls_likelihood_bidirectional(x, W, thresh_backward=0.001, thresh_forward=None, max_iter=1000, per_trial=True, indices_associated_sigs=None):
    """Likelihood NNLS with both backward and forward stepwise rountines.

    Notes:
    1. thresh_forward should be greater than thresh_backward. Otherwise the
    loop may run into a dead loop where the same signature is being removed
    and added back within one iteration, although this is caught gracefully in
    the code.
    2. Both thresh_backward and thresh_forward should be nonnegative.
    """
    if thresh_forward is None:
        thresh_forward = thresh_backward
    if thresh_backward > thresh_forward:
        warnings.warn('thresh_backward is greater thresh_forward. This might lead to indefinite loops.', UserWarning)
    n_sigs = W.shape[1]
    indices_all = np.arange(0, n_sigs)
    ### Initial NNLS
    h, _ = sp.optimize.nnls(W, x)
    # indices_retained = indices_all[h > 0]
    # indices_others = np.array(sorted(list(set(indices_all) - set(indices_retained))))
    # ### Didirectional loop
    # i_iter = 0
    # while i_iter < max_iter:
    #     i_iter += 1
    #     ########################## Backward ##########################
    #     if len(indices_retained) == 1:
    #         backward_stop = True
    #     else:
    #         # Log likelihood of current full model
    #         h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    #         p_current = W[:, indices_retained] @ h
    #         p_current = p_current/np.sum(p_current)
    #         loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
    #         loglikelihoods = []
    #         # Log likelihoods of model that removes 1 signature
    #         for index in indices_retained:
    #             _indices = np.array([i for i in indices_retained if i != index])
    #             p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
    #             p = p/np.sum(p)
    #             loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
    #         loglikelihoods = np.array(loglikelihoods)
    #         # Log likelihood ratios
    #         loglikelihoods = loglikelihood_current - loglikelihoods
    #         # Test
    #         if np.min(loglikelihoods) >= thresh_backward:
    #             backward_stop = True
    #             #print(i_iter, 'Remove', backward_stop, None, indices_retained)
    #             #print(np.min(loglikelihoods))
    #         else:
    #             backward_stop = False
    #             index_remove = indices_retained[np.argmin(loglikelihoods)]
    #             indices_retained = np.array([i for i in indices_retained if i != index_remove])
    #             #print(i_iter, 'Remove', backward_stop, index_remove, indices_retained)
    #             #print(np.min(loglikelihoods))
    #     ########################## Forward ##########################
    #     indices_others = np.array(sorted(list(set(indices_all) - set(indices_retained))))
    #     if len(indices_others) == 0:
    #         forward_stop = True
    #     else:
    #         # Log likelihood of current full model
    #         h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    #         p_current = W[:, indices_retained] @ h
    #         p_current = p_current/np.sum(p_current)
    #         loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
    #         loglikelihoods = []
    #         # Log likelihoods of model that adds 1 signature
    #         for index in indices_others:
    #             _indices = np.sort(np.append(indices_retained, index))
    #             p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
    #             p = p/np.sum(p)
    #             loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
    #         loglikelihoods = np.array(loglikelihoods)
    #         # Log likelihood ratios
    #         loglikelihoods = loglikelihoods - loglikelihood_current
    #         # Test
    #         if np.max(loglikelihoods) <= thresh_forward:
    #             forward_stop = True
    #             #print(i_iter, 'Add', forward_stop, None, indices_retained)
    #             #print(np.max(loglikelihoods))
    #         else:
    #             forward_stop = False
    #             index_add = indices_others[np.argmax(loglikelihoods)]
    #             indices_retained = np.sort(np.append(indices_retained, index_add))
    #             #print(i_iter, 'Add', forward_stop, index_add, indices_retained)
    #             #print(np.max(loglikelihoods))
    #     ######################## Stopping criterion ########################
    #     if backward_stop and forward_stop:
    #         break
    #     if not backward_stop and not forward_stop and index_remove == index_add:
    #         warnings.warn('The same signature is being removed and added back within one iteration, suggesting ill convergence.',
    #                       UserWarning)
    #         break
    # if i_iter >= max_iter:
    #     warnings.warn('Max_iter reached, suggesting that the problem may not converge. Or try increasing max_iter.',
    #                   UserWarning)
    # ### Final NNLS
    # if indices_associated_sigs is not None:
    #     for ind_pair in indices_associated_sigs:
    #         if any(item in ind_pair for item in indices_retained):
    #             indices_retained = np.unique(np.append(indices_retained, ind_pair))
    # print("0k")
    # h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    # h = _fill_vector(h, indices_retained, n_sigs)
    return h