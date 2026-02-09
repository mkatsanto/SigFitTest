
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
        

def recalculate_TAE(df_pred, df_true):
    """
    Recalculate the Total Absolute Error (TAE) between predicted and true values.
    """
    muts_true = df_true.sum()  # total mutations per sample
    df_pred1 = df_pred.reindex(index=df_true.index, fill_value=0)
    TAE = df_true.sub(df_pred1, axis=0).abs().sum() / (2 * (muts_true))   # total absolute error (aka "fitting error") for each sample     
    TAE = pd.DataFrame(TAE)
    return TAE


def create_enhanced_unknown_signatures(W, method='identity', n_unknown=None):
    """
    Create biologically meaningful unknown signatures instead of identity matrix
    """
    n_mut_types = W.shape[0]
    if n_unknown is None:
        n_unknown = min(n_mut_types, 20)
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
    
    elif method == 'identity':
        # Find the maximum value for each mutation type across all signatures
        max_values_per_channel = W.max(axis=1)
        
        # Get indices of channels with lowest maximum values
        # Sort by max values and take the first n_unknown indices
        lowest_channels = np.argsort(max_values_per_channel)[:n_unknown]
        
        # Create identity matrix for only the selected channels
        unknown_sigs = np.zeros((n_mut_types, n_unknown))
        for i, channel_idx in enumerate(lowest_channels):
            unknown_sigs[channel_idx, i] = 1.0
        #print(f"Selected {n_unknown} channels with lowest max values:")
        #for i, channel_idx in enumerate(lowest_channels):
            #print(f"  Channel {channel_idx}: max_value = {max_values_per_channel[channel_idx]:.6f}")
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


def create_diverse_initialization(W_extended, x, start_idx, original_n_sigs):
    """Create diverse initialization strategies with improved approaches"""
    n_sigs = W_extended.shape[1]
    h_init = np.zeros(n_sigs)
    total_muts = np.sum(x)
    
    if start_idx == 0:
        # NNLS initialization with 0 for unknown

        h_known, residual = sp.optimize.nnls(W_extended[:, :original_n_sigs], x)
        h_init[:original_n_sigs] = h_known
        h_init[original_n_sigs:] = 0

    elif start_idx == 1:
        # Residual-focused initialization
        try:
            h_nnls, _ = sp.optimize.nnls(W_extended[:, :original_n_sigs], x)
            residuals = x - W_extended[:, :original_n_sigs] @ h_nnls
            
            # Assign known signatures with some noise
            h_init[:original_n_sigs] = h_nnls * np.random.uniform(0.8, 1.2, original_n_sigs)
            
            # Initialize unknown signatures proportional to unexplained variance
            if original_n_sigs < n_sigs:
                residual_strength = np.sum(np.abs(residuals))
                n_unknown = n_sigs - original_n_sigs
                # Distribute residual strength among unknown signatures
                h_init[original_n_sigs:] = residual_strength * 0.1 / n_unknown
        except:
            h_init[:original_n_sigs] = total_muts * 0.95 / original_n_sigs
            h_init[original_n_sigs:] = total_muts * 0.05 / (n_sigs - original_n_sigs)
            
    elif start_idx == 2:
        # Sparse random initialization with emphasis on high-variance channels
        np.random.seed(42 + start_idx)
        
        # Select signatures with highest contribution potential
        n_active = min(10, original_n_sigs)
        
        # Compute channel variance to select most informative signatures
        channel_variance = np.var(W_extended[:, :original_n_sigs], axis=0)
        active_indices = np.argsort(channel_variance)[-n_active:]
        
        h_init[active_indices] = np.random.uniform(0.01, 0.1, n_active) * total_muts / n_active
        
        # Small values for unknown
        if original_n_sigs < n_sigs:
            h_init[original_n_sigs:] = 1e-3
    else: # this is not actually used
        # Random perturbation of NNLS with adaptive noise
        try:
            h_nnls, _ = sp.optimize.nnls(W_extended[:, :original_n_sigs], x)
            
            # Adaptive noise based on weight magnitude
            noise_scale = 0.1 * h_nnls
            noise = np.random.normal(0, noise_scale, original_n_sigs)
            h_init[:original_n_sigs] = np.maximum(h_nnls + noise, 1e-6)
            
            # Unknown signatures get small random values
            if original_n_sigs < n_sigs:
                h_init[original_n_sigs:] = np.random.uniform(1e-4, 1e-3, n_sigs - original_n_sigs) * total_muts
        except:
            h_init = np.random.uniform(0.001, 0.05, n_sigs) * total_muts / n_sigs
    
    # Normalize to match total mutation count
    current_sum = np.sum(h_init)
    if current_sum > 0:
        h_init = h_init * (total_muts / current_sum)
    else:
        # Fallback: uniform distribution
        h_init[:] = total_muts / n_sigs
    
    # Ensure non-negativity and minimum threshold
    h_init = np.maximum(h_init, 1e-10)
    
    return h_init


# this cannot help much as the unknown weights stll can not be correctly assigned
# to the extension signatures
def solution_based_initialization(true_weights, W_extended, x, original_n_sigs):
    """Create initialization based on known weights
        Used only in simulations to study the effect 
        of initialization on the objective convergence"""
    n_sigs = W_extended.shape[1]
    h_init = np.zeros(n_sigs)
    
    try:
        h_known = true_weights
        h_init[:original_n_sigs] = h_known
        h_init[original_n_sigs:] = 0.0
    except:
        h_init[:original_n_sigs] = 1e-3 
        
    # Normalize
    total = np.sum(h_init)
    if total > 0:
        return h_init / total
    else:
        h_init[0] = 1.0
        return h_init


# def enrich_signature_references(x, W, sample_name, outdir_prefix, approach, channel_number, thresh_backward=0.001, thresh_forward=None, max_iter=1000, per_trial=True, indices_associated_sigs=None, allow_unknown=False, unknown_penalty=2, true_weights=None):
#     """Likelihood NNLS with extended signatures and penalty for unknown signatures
#     and multi-start optimization for robustness"""    
    
#     if thresh_forward is None:
#         thresh_forward = thresh_backward
#     if thresh_backward > thresh_forward:
#         warnings.warn('thresh_backward is greater thresh_forward. This might lead to indefinite loops.', UserWarning)
    
#     # Create extended signature matrix W' = [W | I]
#     original_n_sigs = W.shape[1]
#     n_mut_types = W.shape[0]
    
#     if allow_unknown:
#         print(f"Using unknown penalty: {unknown_penalty}", "method:", approach, "channel_number:", channel_number)
#         unknown_sigs = create_enhanced_unknown_signatures(W, method=approach, n_unknown=channel_number)
#         W_extended = np.column_stack([W, unknown_sigs])
#         dummy_indices = set(range(original_n_sigs, W_extended.shape[1]))
#     else:
#         W_extended = W.copy()
#         dummy_indices = set()
    
#     n_sigs_extended = W_extended.shape[1]
    
#     def objective_function(weights):
#         """Simple least squares implementation but with
#         L1 penalty on unknown signatures"""
#         weights = np.maximum(weights, 0)
#         predicted_counts = W_extended @ weights
#         residuals = predicted_counts - x
#         nnls_objective = 0.5 * np.sum(residuals ** 2)
        
#         l1_penalty = 0.0
#         if allow_unknown and len(dummy_indices) > 0:
#             dummy_weights = weights[list(dummy_indices)]
#             l1_penalty = unknown_penalty * np.sum(dummy_weights)
#         return nnls_objective + l1_penalty 
    
#     # Convergence tracking
#     convergence_metrics = []
#     def callback_function(xk): #, state=None):
#         """Callback function for recording convergence metrics with mutation assignment tracking"""
        
#         weights = np.maximum(xk, 0)
#         obj_val = objective_function(weights)
#         predicted_counts = W_extended @ weights
#         residuals = predicted_counts - x
        
#         # MUTATION ASSIGNMENT TRACKING 
#         total_mutations = np.sum(x)
        
#         # Unassigned: mutations not explained by model
#         unassigned = np.maximum(x - predicted_counts, 0)
#         unassigned_total = np.sum(unassigned)
        
#         # Assigned: mutations successfully explained
#         assigned = np.minimum(predicted_counts, x)
#         assigned_total = np.sum(assigned)
        
#         # Overfitted: model predicts more than observed
#         overfitted = np.maximum(predicted_counts - x, 0)
#         overfitted_total = np.sum(overfitted)
        
#         # === SIGNATURE-SPECIFIC CONTRIBUTIONS (NEW) ===
#         known_weights = weights[:original_n_sigs]
#         unknown_weights = weights[original_n_sigs:] if allow_unknown else np.array([])
        
#         unknown_weight_sum = np.sum(unknown_weights) if allow_unknown else 0.0
        
#         # Mutations assigned to known vs unknown signatures
#         known_contrib = np.sum(W_extended[:, :original_n_sigs] @ known_weights)
#         unknown_contrib = np.sum(W_extended[:, original_n_sigs:] @ unknown_weights) if allow_unknown else 0.0
        
#         # Count active signatures (weight > threshold)
#         active_threshold = 5  # mutations
#         known_active = np.sum(known_weights > active_threshold)
#         unknown_active = np.sum(unknown_weights > active_threshold) if allow_unknown else 0
        
        
#         # === TAE CALCULATION (NEW) ===
#         if true_weights is not None:
#             TAE = recalculate_TAE(
#                 pd.DataFrame(known_weights.reshape(-1, 1)),
#                 pd.DataFrame(true_weights)
#             )
#             tae_value = TAE.values[0][0]
#         else:
#             tae_value = np.nan
        
#         # objective metrics
#         mse = np.mean(residuals ** 2)
#         rmse = np.sqrt(mse)
    
#         if allow_unknown and len(dummy_indices) > 0:
#             l1_penalty = unknown_penalty * unknown_weight_sum
#         iteration = len(convergence_metrics) + 1
#         convergence_metrics.append({
#             'iteration': iteration,
#             'objective': obj_val,
#             'data_fit_component': obj_val - l1_penalty,
#             'l1_penalty': l1_penalty,
#             'total_mutations': total_mutations,
#             'assigned_mutations': assigned_total,
#             'unassigned_mutations': unassigned_total,
#             'overfitted_mutations': overfitted_total,
#             'known_mutations': known_contrib,
#             'unknown_mutations': unknown_contrib,
#             'TAE': tae_value,
#             'rmse': rmse,
#             'known_signatures_active': known_active,
#             'unknown_signatures_active': unknown_active,
#         })


#     # MULTI-START OPTIMIZATION
#     n_starts = 3 if allow_unknown else 2
#     best_result = None
#     best_objective = np.inf
#     all_results = []
#     all_convergence_metrics = {}  # Store convergence metrics for each attempt
#     bounds = [(0, None) for _ in range(n_sigs_extended)]    
#     print(f"Starting multi-start optimization with {n_starts} attempts...")

#     for start_idx in range(4):
#         print(f"\n=== Optimization attempt {start_idx + 1}/{n_starts} ===")
        
#         # Reset convergence metrics for each start
#         convergence_metrics.clear()
        
#         if allow_unknown and start_idx == n_starts:
#             solution_based_initialization(true_weights, W_extended, x, original_n_sigs)
#         else:
#             # Create diverse initialization
#             h_init = create_diverse_initialization(W_extended, x, start_idx, original_n_sigs)
        
#         try:
#             # Record initial state
#             callback_function(h_init)
            
#             result = minimize(
#                 fun=objective_function,
#                 x0=h_init,
#                 bounds=bounds,
#                 constraints = {'type': 'eq', 'fun': lambda h: h.sum() - np.sum(x)},
#                 method="SLSQP", #'L-BFGS-B',
#                 callback=callback_function,
#                 options={
#                     'maxiter': 10000,        
#                     'maxfun': 150000,
#                     'ftol': 1e-9, 
#                     'gtol': 1e-6, 
#                     'eps': 1e-8,            
#                     'disp': True}
#             )
             
#             final_obj = objective_function(result.x)
#             all_results.append({
#                 'start': start_idx,
#                 'success': result.success,
#                 'objective': final_obj,
#                 'nit': result.nit,
#                 'result': result,
#                 'final_weights': result.x.copy()
#             })
            
#             # Store convergence metrics for this attempt
#             all_convergence_metrics[start_idx] = convergence_metrics.copy()
            
#             print(f"Attempt {start_idx + 1}: {'Success' if result.success else 'Failed'}, "
#                   f"Obj={final_obj:.6e}, Iterations={result.nit}")
            
#             if result.success and final_obj < best_objective:
#                 best_objective = final_obj
#                 best_result = result
#                 best_start_idx = start_idx  # Track which attempt was best
#                 print(f"New best result found!")
                
#         except Exception as e:
#             print(f"Attempt {start_idx + 1} failed with exception: {e}")
#             all_results.append({
#                 'start': start_idx,
#                 'success': False,
#                 'objective': np.inf,
#                 'nit': 0,
#                 'result': None,
#                 'error': str(e)
#             })
    
#     # Analyze results
#     successful_results = [r for r in all_results if r['success']]
#     if len(successful_results) > 1:
#         objectives = [r['objective'] for r in successful_results]
#         obj_std = np.std(objectives)
#         obj_mean = np.mean(objectives)
        
#         print(f"\n=== Multi-Start Analysis ===")
#         print(f"Successful attempts: {len(successful_results)}/{n_starts}")
#         print(f"Objective values: {[f'{obj:.6e}' for obj in objectives]}")
#         print(f"Best objective: {min(objectives):.6e}")
#         print(f"Variability (std/mean): {obj_std/obj_mean:.3f}")
        
#         if obj_std / obj_mean > 0.05:  # More than 5% relative variability
#             print(f"⚠️  High variability detected - optimization landscape may be complex")
#         else:
#             print(f"✓ Good consistency across starts")
    
#     # Use best result or fallback
#     if best_result and best_result.success:
#         h_final = best_result.x
#         best_convergence = all_convergence_metrics.get(best_start_idx, [])
#         print(f"\nOptimization converged successfully with best result from {n_starts} attempts")
#     elif len(successful_results) > 0:
#         # Use any successful result
#         first_successful = successful_results[0]
#         h_final = first_successful['result'].x
#         best_start_idx = first_successful['start']
#         best_convergence = all_convergence_metrics.get(best_start_idx, [])
#         print(f"\nUsing successful result (not necessarily optimal)")
#     else:
#         h_final = np.zeros(n_sigs_extended)
#         best_convergence = []  # No valid convergence metrics
        
#     # Save convergence metrics (using the best run's metrics only)
#     if best_convergence:
#         convergence_df = pd.DataFrame(best_convergence)
#         convergence_df.to_csv(f"{outdir_prefix}/{unknown_penalty}_{sample_name}_convergence_metrics.csv", index=False)
#         print(f"Saved convergence metrics from best optimization attempt (start_idx={best_start_idx})")
#     else:
#         print("No convergence metrics to save (all attempts failed)")
        
#     return h_final

# def enrich_signature_references(x, W, sample_name, outdir_prefix, approach, channel_number, 
#                                 thresh_backward=0.001, thresh_forward=None, max_iter=1000, 
#                                 per_trial=True, indices_associated_sigs=None, allow_unknown=False, 
#                                 unknown_penalty=2, true_weights=None):
#     """
#     Iterative signature refinement with L1 penalty on unknown signatures.
#     Similar to SigProfiler's backward selection approach.
#     """
    
#     original_n_sigs = W.shape[1]
#     n_mut_types = W.shape[0]
#     convergence_metrics = []
    
#     # --- STEP 1: Initialize with known signatures ---
#     h_known_init, _ = sp.optimize.nnls(W, x)
#     residuals = x - W @ h_known_init
#     positive_residuals = np.maximum(residuals, 0)
#     total_unexplained = np.sum(positive_residuals)
    
#     print(f"Total unexplained: {total_unexplained:.1f} ({100*total_unexplained/np.sum(x):.2f}%)")
    
#     # --- STEP 2: Add unknown signatures if needed ---
#     if allow_unknown and total_unexplained > 0:
#         # Compute z-scores for high-residual channels
#         channel_number = 20
#         selected_channels = np.argsort(positive_residuals)[-channel_number:]
#         selected_channels = selected_channels[positive_residuals[selected_channels] > 0]  # Keep only positive
        
#         if len(selected_channels) > 0:
#             print(f"Selected {len(selected_channels)} high-residual channels")
            
#             # Create identity-based unknown signatures
#             unknown_sigs = np.zeros((n_mut_types, len(selected_channels)))
#             unknown_sigs[selected_channels, np.arange(len(selected_channels))] = 1.0
            
#             W_extended = np.column_stack([W, unknown_sigs])
#             dummy_indices = set(range(original_n_sigs, W_extended.shape[1]))
#         else:
#             W_extended = W.copy()
#             dummy_indices = set()
#     else:
#         W_extended = W.copy()
#         dummy_indices = set()
    
#     active_indices = list(range(W_extended.shape[1]))
    
#     # --- OBJECTIVE FUNCTION ---
#     def objective(weights, W_curr, dummy_curr):
#         predicted = W_curr @ np.maximum(weights, 0)
#         sse = 0.5 * np.sum((predicted - x) ** 2)
#         l1_penalty = unknown_penalty * np.sum(weights[list(dummy_curr)]) if dummy_curr else 0
#         return sse + l1_penalty
    
#     # --- TRACKING FUNCTION ---
#     def track_metrics(weights, iter_num, W_curr, dummy_curr):
#         weights = np.maximum(weights, 0)
#         predicted = W_curr @ weights
        
#         # Map to full signature space
#         full_weights = np.zeros(W_extended.shape[1])
#         full_weights[active_indices] = weights
        
#         known_w = full_weights[:original_n_sigs]
#         unknown_w = full_weights[original_n_sigs:]
        
#         convergence_metrics.append({
#             'iteration': iter_num,
#             'n_sigs_active': len(active_indices),
#             'objective': objective(weights, W_curr, dummy_curr),
#             'l1_penalty': unknown_penalty * np.sum(unknown_w) if allow_unknown else 0,
#             'total_mutations': np.sum(x),
#             'assigned_mutations': np.sum(np.minimum(predicted, x)),
#             'unassigned_mutations': np.sum(np.maximum(x - predicted, 0)),
#             'overfitted_mutations': np.sum(np.maximum(predicted - x, 0)),
#             'known_mutations': np.sum(W_extended[:, :original_n_sigs] @ known_w),
#             'unknown_mutations': np.sum(W_extended[:, original_n_sigs:] @ unknown_w) if allow_unknown else 0,
#             'TAE': recalculate_TAE(pd.DataFrame(known_w.reshape(-1, 1)), pd.DataFrame(true_weights)).values[0][0] if true_weights is not None else np.nan,
#             'rmse': np.sqrt(np.mean((predicted - x) ** 2)),
#             'known_signatures_active': np.sum(known_w > 5),
#             'unknown_signatures_active': np.sum(unknown_w > 5) if allow_unknown else 0,
#         })
    
#     # --- ITERATIVE REFINEMENT ---
#     print(f"\n{'='*80}\n=== Iterative Refinement ===")
#     print(f"Initial signatures: {len(active_indices)} (Known: {original_n_sigs}, Unknown: {len(dummy_indices)})\n{'='*80}\n")
    
#     for outer_iter in range(1, max_iter + 1):
#         print(f"\n--- Iteration {outer_iter}/{max_iter} (Active: {len(active_indices)}) ---")
        
#         # Build current signature matrix
#         W_curr = W_extended[:, active_indices]
#         dummy_curr = {i for i, idx in enumerate(active_indices) if idx in dummy_indices}
        
#         # Optimize
#         h_init, _ = sp.optimize.nnls(W_curr, x)
#         track_metrics(h_init, 0, W_curr, dummy_curr)
        
#         result = minimize(
#             fun=lambda w: objective(w, W_curr, dummy_curr),
#             x0=h_init,
#             bounds=[(0, None)] * len(active_indices),
#             constraints={'type': 'eq', 'fun': lambda h: h.sum() - np.sum(x)},
#             method="SLSQP",
#             callback=lambda xk: track_metrics(xk, len(convergence_metrics), W_curr, dummy_curr),
#             options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
#         )
        
#         weights_curr = np.maximum(result.x, 0)
        
#         # --- SIGPROFILER BACKWARD ELIMINATION ---
#         if len(active_indices) == 1:
#             print(f"  ✓ Only 1 signature remaining - stopping")
#             break

#         # Compute current model objective (lower is better)
#         obj_current = objective(weights_curr, W_curr, dummy_curr)

#         # Test removal of each signature
#         obj_increases = []
#         for i in range(len(active_indices)):
#             # Create model without signature i
#             indices_without_i = [j for j in range(len(active_indices)) if j != i]
#             W_reduced = W_curr[:, indices_without_i]
#             dummy_reduced = {j for j, orig_j in enumerate(indices_without_i) if orig_j in dummy_curr}
            
#             # Refit without signature i
#             h_reduced, _ = sp.optimize.nnls(W_reduced, x)
#             obj_reduced = objective(h_reduced, W_reduced, dummy_reduced)
            
#             # Objective increase (how much worse does model get?)
#             increase = obj_reduced - obj_current  # Positive = worse fit
#             obj_increases.append(increase)

#         obj_increases = np.array(obj_increases)
#         min_increase = np.min(obj_increases)
#         sig_to_remove_idx = np.argmin(obj_increases)

#         print(f"  Objective increases: min={min_increase:.6f}, max={np.max(obj_increases):.6f}")
#         print(f"  Threshold: {thresh_backward:.6f}")

#         # Remove signature if objective increase is below threshold
#         if min_increase < thresh_backward:
#             removed_global_idx = active_indices[sig_to_remove_idx]
#             is_unknown = removed_global_idx in dummy_indices
#             sig_type = "Unknown" if is_unknown else "Known"
            
#             print(f"  ✗ Removing signature {removed_global_idx} ({sig_type}) - objective increase: {min_increase:.6f}")
            
#             active_indices = [idx for j, idx in enumerate(active_indices) if j != sig_to_remove_idx]
            
#             if len(active_indices) == 0:
#                 print(f"  ⚠️  All removed - keeping best signature")
#                 active_indices = [active_indices[np.argmax(weights_curr)]]
#                 break
#         else:
#             print(f"  ✓ Converged - all signatures significant (min increase: {min_increase:.6f} >= {thresh_backward:.6f})")
#             break
    
#     # --- FINAL FIT ---
#     print(f"\n{'='*80}\n=== Final Optimization ===")
    
#     # Add associated signatures if specified
#     if indices_associated_sigs:
#         for pair in indices_associated_sigs:
#             if any(idx in active_indices for idx in pair):
#                 active_indices = sorted(set(active_indices) | set(pair))
    
#     W_final = W_extended[:, active_indices]
#     dummy_final = {i for i, idx in enumerate(active_indices) if idx in dummy_indices}
    
#     # Final optimization with L1 penalty
#     h_init_final, _ = sp.optimize.nnls(W_final, x)
    
#     result_final = minimize(
#         fun=lambda w: objective(w, W_final, dummy_final),
#         x0=h_init_final,
#         bounds=[(0, None)] * len(active_indices),
#         constraints={'type': 'eq', 'fun': lambda h: h.sum() - np.sum(x)},
#         method="SLSQP",
#         options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
#     )
    
#     h_final_active = np.maximum(result_final.x, 0)
    
#     # Map to full space
#     h_final = np.zeros(W_extended.shape[1])
#     h_final[active_indices] = h_final_active
    
#     # Print summary
#     n_known = sum(1 for idx in active_indices if idx < original_n_sigs)
#     n_unknown = len(active_indices) - n_known
#     print(f"Active: {len(active_indices)} (Known: {n_known}, Unknown: {n_unknown})")
#     print(f"Iterations: {outer_iter}")
    
#     if true_weights is not None:
#         tae = recalculate_TAE(pd.DataFrame(h_final[:original_n_sigs].reshape(-1, 1)), pd.DataFrame(true_weights))
#         print(f"TAE: {tae.values[0][0]:.4f}")
    
#     unassigned = np.sum(np.maximum(x - W_extended @ h_final, 0))
#     print(f"Unassigned: {unassigned:.0f} ({100*unassigned/np.sum(x):.2f}%)\n{'='*80}\n")
    
#     # Save metrics
#     if convergence_metrics:
#         pd.DataFrame(convergence_metrics).to_csv(
#             f"{outdir_prefix}/{unknown_penalty}_{sample_name}_convergence_metrics.csv", index=False
#         )
    
#     # Create output DataFrame
#     known_names = W.columns.tolist() if isinstance(W, pd.DataFrame) else [f"Signature_{i}" for i in range(original_n_sigs)]
#     unknown_names = [f"Unknown_{i}" for i in range(W_extended.shape[1] - original_n_sigs)]
#     all_names = known_names + unknown_names
    
#     return pd.DataFrame(h_final, index=all_names, columns=[sample_name])




def enrich_signature_references(x, W, sample_name, outdir_prefix, approach, channel_number, 
                                thresh_backward=0.01, thresh_forward=0.05, max_iter=1000, 
                                per_trial=True, indices_associated_sigs=None, allow_unknown=False, 
                                unknown_penalty=2, true_weights=None):
    """
    Signature refinement with L1 penalty on unknown signatures.
    Uses SigProfilerAssignment's leave-one-out backward elimination strategy.
    """
    W_names = W.columns.tolist() if isinstance(W, pd.DataFrame) else [f"Signature_{i}" for i in range(W.shape[1])]
    W = W.values if isinstance(W, pd.DataFrame) else W.copy()
    original_n_sigs = W.shape[1]
    n_mut_types = W.shape[0]
    convergence_metrics = []
    
    # Use a mutable container to hold active_indices so callback can see updates
    state = {
        'active_indices': list(range(W.shape[1])),
        'dummy_indices': set()
    }
    
    # --- STEP 1: Initialize with known signatures ---
    # h_known_init, _ = sp.optimize.nnls(W, x)
    # residuals = x - W @ h_known_init
    # positive_residuals = np.maximum(residuals, 0)
    # total_unexplained = np.sum(positive_residuals)
    
    #print(f"Total unexplained: {total_unexplained:.1f} ({100*total_unexplained/np.sum(x):.2f}%)")
    
    
    W_extended = W.copy()
    # --- OBJECTIVE FUNCTION (SSE + L1 penalty) ---
    def objective(weights, W_curr, dummy_curr):
        predicted = W_curr @ np.maximum(weights, 0)
        sse = 0.5 * np.sum((predicted - x) ** 2)
        l1_penalty = unknown_penalty * np.sum(np.abs(weights[list(dummy_curr)])) if dummy_curr else 0
        return sse + l1_penalty
    
    # --- TRACKING FUNCTION (now uses state dict) ---
    def track_metrics(weights, iter_num, W_curr, dummy_curr):
        weights = np.maximum(weights, 0)
        predicted = W_curr @ weights
        
        # Get current active indices from state
        active_indices = state['active_indices']
        
        # Map to full signature space
        full_weights = np.zeros(W_extended.shape[1])
        full_weights[active_indices] = weights
        
        known_w = full_weights[:original_n_sigs]
        unknown_w = full_weights[original_n_sigs:]
        
        convergence_metrics.append({
            'iteration': iter_num,
            'n_sigs_active': len(active_indices),  # ✓ Now reflects current active set
            'objective': objective(weights, W_curr, dummy_curr),
            'l1_penalty': unknown_penalty * np.sum(unknown_w) if allow_unknown else 0,
            'total_mutations': np.sum(x),
            'assigned_mutations': np.sum(np.minimum(predicted, x)),
            'unassigned_mutations': np.sum(np.maximum(x - predicted, 0)),
            'overfitted_mutations': np.sum(np.maximum(predicted - x, 0)),
            'known_mutations': np.sum(W_extended[:, :original_n_sigs] @ known_w),
            'unknown_mutations': np.sum(W_extended[:, original_n_sigs:] @ unknown_w) if allow_unknown else 0,
            'TAE': recalculate_TAE(pd.DataFrame(known_w.reshape(-1, 1)), pd.DataFrame(true_weights)).values[0][0] if true_weights is not None else np.nan,
            'rmse': np.sqrt(np.mean((predicted - x) ** 2)),
            'known_signatures_active': np.sum(known_w > 5),  # ✓ Now based on full_weights
            'unknown_signatures_active': np.sum(unknown_w > 5) if allow_unknown else 0,
        })
    
    # --- INITIAL OPTIMIZATION WITH L1 PENALTY ---
    print(f"\n{'='*80}\n=== Initial Optimization with L1 Penalty ===")
    print(f"Signatures: {W_extended.shape[1]} (Known: {original_n_sigs}, Unknown: {len(state['dummy_indices'])})")
    print(f"L1 Penalty: {unknown_penalty}\n{'='*80}\n")
    
    active_indices = state['active_indices']
    W_curr = W_extended[:, active_indices]
    dummy_curr = {i for i, idx in enumerate(active_indices) if idx in state['dummy_indices']}
    
    # Initialize with NNLS
    h_init, _ = sp.optimize.nnls(W_curr, x)
    track_metrics(h_init, 0, W_curr, dummy_curr)
    
    # --- SIGPROFILER-STYLE BACKWARD ELIMINATION ---
    if thresh_backward is not None and thresh_backward > 0 and len(active_indices) > 1:
        print(f"\n{'='*80}\n=== SigProfiler Backward Elimination ===")
        print(f"Threshold: {thresh_backward}\n{'='*80}\n")
        
        elimination_round = 0
        
        while len(active_indices) > 1 and elimination_round < max_iter:
            elimination_round += 1
            print(f"\n--- Round {elimination_round} (Active: {len(active_indices)}) ---")
            
            # NNLS reconstruction for cosine similarity baseline
            h_current_nnls, _ = sp.optimize.nnls(W_curr, x)
            reconstruction_current_nnls = W_curr @ h_current_nnls
            
            cos_sim_current = np.dot(x, reconstruction_current_nnls) / (
                np.linalg.norm(x) * np.linalg.norm(reconstruction_current_nnls)
            )
            
            print(f"  Current cosine similarity: {cos_sim_current:.6f}")
            
            # === LEAVE-ONE-OUT with NNLS ===
            cos_sim_drops = []
            
            for i in range(len(active_indices)):
                indices_leave_one_out = [j for j in range(len(active_indices)) if j != i]
                
                if len(indices_leave_one_out) == 0:
                    cos_sim_drops.append(np.inf)
                    continue
                
                W_reduced = W_curr[:, indices_leave_one_out]
                h_reduced, _ = sp.optimize.nnls(W_reduced, x)
                reconstruction_reduced = W_reduced @ h_reduced
                
                norm_x = np.linalg.norm(x)
                norm_recon = np.linalg.norm(reconstruction_reduced)
                
                if norm_recon > 0:
                    cos_sim_reduced = np.dot(x, reconstruction_reduced) / (norm_x * norm_recon)
                else:
                    cos_sim_reduced = 0.0
                
                drop = cos_sim_current - cos_sim_reduced
                cos_sim_drops.append(drop)
            
            cos_sim_drops = np.array(cos_sim_drops)
            min_drop = np.min(cos_sim_drops)
            sig_to_remove_local_idx = np.argmin(cos_sim_drops)
            sig_to_remove_global_idx = active_indices[sig_to_remove_local_idx]
            
            is_unknown = sig_to_remove_global_idx in state['dummy_indices']
            sig_type = "Unknown" if is_unknown else "Known"
            
            print(f"  Cosine similarity drops: min={min_drop:.6f}, max={np.max(cos_sim_drops):.6f}, mean={np.mean(cos_sim_drops):.6f}")
            print(f"  Candidate to remove: Sig {sig_to_remove_global_idx} ({sig_type})")
            
            # === REMOVAL DECISION ===
            if min_drop < thresh_backward:
                print(f"  ✗ REMOVING Sig {sig_to_remove_global_idx} ({sig_type})")
                print(f"     Reason: Cosine similarity drop ({min_drop:.6f}) < threshold ({thresh_backward:.6f})")
                
                # ✓ UPDATE STATE
                active_indices = [idx for j, idx in enumerate(active_indices) if j != sig_to_remove_local_idx]
                state['active_indices'] = active_indices  # ✓ Update state dict
                
                # Re-build signature matrix
                W_curr = W_extended[:, active_indices]
                dummy_curr = {i for i, idx in enumerate(active_indices) if idx in state['dummy_indices']}
                
                # Re-optimize with L1 penalty
                h_init_new, _ = sp.optimize.nnls(W_curr, x)
                
                # ✓ Track metrics BEFORE optimization (with updated active_indices)
                track_metrics(h_init_new, len(convergence_metrics), W_curr, dummy_curr)
                
                # result_new = minimize(
                #     fun=lambda w: objective(w, W_curr, dummy_curr),
                #     x0=h_init_new,
                #     bounds=[(0, None)] * len(active_indices),
                #     constraints={'type': 'eq', 'fun': lambda h: h.sum() - np.sum(x)},
                #     method="SLSQP",
                #     callback=lambda xk: track_metrics(xk, len(convergence_metrics), W_curr, dummy_curr),
                #     options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
                # )
                
                # weights_optimized = np.maximum(result_new.x, 0)
                
            else:
                print(f"  ✓ CONVERGED - All signatures significant")
                print(f"     Reason: Min cosine similarity drop ({min_drop:.6f}) >= threshold ({thresh_backward:.6f})")
                break
        
        print(f"\n{'='*80}")
        print(f"Elimination complete after {elimination_round} rounds")
        print(f"Final active signatures: {len(active_indices)}")
        print(f"{'='*80}\n")
        
    #     # --- SIGPROFILER-STYLE BACKWARD ELIMINATION (Round 1) ---
    # if thresh_backward is not None and thresh_backward > 0 and len(active_indices) > 1:
    #     print(f"\n{'='*80}\n=== SigProfiler Backward Elimination (Round 1) ===")
    #     print(f"Threshold: {thresh_backward}\n{'='*80}\n")
        
    #     elimination_round = 0
        
    #     while len(active_indices) > 1 and elimination_round < max_iter:
    #         elimination_round += 1
    #         print(f"\n--- Round {elimination_round} (Active: {len(active_indices)}) ---")
            
    #         # Current model fit
    #         h_current, _ = sp.optimize.nnls(W_curr, x)
    #         reconstruction_current = W_curr @ h_current
            
    #         # Current cosine similarity
    #         cos_sim_current = np.dot(x, reconstruction_current) / (
    #             np.linalg.norm(x) * np.linalg.norm(reconstruction_current)
    #         )
            
    #         print(f"  Current cosine similarity: {cos_sim_current:.6f}")
            
    #         # === TEST REMOVAL OF EACH SIGNATURE ===
    #         signature_removed_this_round = False
            
    #         for i in range(len(active_indices)):
    #             # Create model WITHOUT signature i
    #             indices_without_i = [j for j in range(len(active_indices)) if j != i]
                
    #             if len(indices_without_i) == 0:
    #                 continue
                
    #             W_reduced = W_curr[:, indices_without_i]
                
    #             # Refit WITHOUT signature i
    #             h_reduced, _ = sp.optimize.nnls(W_reduced, x)
    #             reconstruction_reduced = W_reduced @ h_reduced
                
    #             # Compute cosine similarity
    #             norm_x = np.linalg.norm(x)
    #             norm_recon = np.linalg.norm(reconstruction_reduced)
                
    #             if norm_recon > 0:
    #                 cos_sim_reduced = np.dot(x, reconstruction_reduced) / (norm_x * norm_recon)
    #             else:
    #                 cos_sim_reduced = 0.0
                
    #             # Cosine similarity drop
    #             drop = cos_sim_current - cos_sim_reduced
                
    #             sig_global_idx = active_indices[i]
    #             is_unknown = sig_global_idx in state['dummy_indices']
    #             sig_type = "Unknown" if is_unknown else "Known"
                
    #             # === IMMEDIATE REMOVAL DECISION (like SigProfilerAssignment) ===
    #             if drop < thresh_backward:
    #                 print(f"  ✗ REMOVING Sig {sig_global_idx} ({sig_type})")
    #                 print(f"     Cosine drop: {drop:.6f} < threshold: {thresh_backward:.6f}")
                    
    #                 # Remove signature
    #                 active_indices = [idx for j, idx in enumerate(active_indices) if j != i]
    #                 state['active_indices'] = active_indices
                    
    #                 # Rebuild signature matrix
    #                 W_curr = W_extended[:, active_indices]
    #                 dummy_curr = {i for i, idx in enumerate(active_indices) if idx in state['dummy_indices']}
                    
    #                 # Track metrics after removal
    #                 h_new, _ = sp.optimize.nnls(W_curr, x)
    #                 track_metrics(h_new, len(convergence_metrics), W_curr, dummy_curr)
                    
    #                 signature_removed_this_round = True
    #                 break  # ← RESTART: Exit inner loop and start over
            
    #         # Check if we should continue
    #         if not signature_removed_this_round:
    #             print(f"  ✓ CONVERGED - All signatures significant")
    #             print(f"     No signature had drop < {thresh_backward:.6f}")
    #             break  # Exit while loop - done!
        
    #     if elimination_round >= max_iter:
    #         print(f"  ⚠️  Max iterations reached ({max_iter})")
        
    #     print(f"\n{'='*80}")
    #     print(f"Elimination complete after {elimination_round} rounds")
    #     print(f"Final active signatures: {len(active_indices)}")
    #     print(f"{'='*80}\n")
    
    # Final optimization with L1 penalty
    h_init_new, _ = sp.optimize.nnls(W_curr, x)
    result_new = minimize(
        fun=lambda w: objective(w, W_curr, dummy_curr),
        x0=h_init_new,
        bounds=[(0, None)] * len(active_indices),
        #constraints={'type': 'eq', 'fun': lambda h: h.sum() - np.sum(x)},
        method="SLSQP",
        callback=lambda xk: track_metrics(xk, len(convergence_metrics), W_curr, dummy_curr),
        options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
    )
    weights_optimized = np.maximum(result_new.x, 0)
    if allow_unknown:
        h_known_init, _ = sp.optimize.nnls(W_curr, x)
        residuals = x - W_curr @ h_known_init
        positive_residuals = np.maximum(residuals, 0)
        
        predicted_counts = W_curr @ weights_optimized
        unassigned = np.maximum(x - predicted_counts, 0)
        
        if np.sum(unassigned) > 0:
            mean = np.mean(unassigned)
            std = np.std(unassigned)
            # threshold = mean + 1 * std
            # selected_channels = np.where(unassigned > threshold)[0]
            # selected_channels = np.where(unassigned > 0)[0]

        if np.sum(positive_residuals) > 0:
            # Select high-residual channels
            selected_channels = np.argsort(positive_residuals)[-channel_number:]
            #selected_channels = selected_channels[positive_residuals[selected_channels] > 0]
            if len(selected_channels) > 0:
                print(f"Adding {len(selected_channels)} unknown signatures for high-residual channels")
                
                # Create identity matrix for unknown signatures
                unknown_sigs = np.zeros((n_mut_types, len(selected_channels)))
                unknown_sigs[selected_channels, np.arange(len(selected_channels))] = 1.0
                
                # Update W_extended and indices
                W_extended = np.column_stack([W_extended, unknown_sigs])
                new_unknown_indices = list(range(W_extended.shape[1] - len(selected_channels), W_extended.shape[1]))
                
                # Update state
                active_indices = active_indices + new_unknown_indices
                state['active_indices'] = active_indices
                state['dummy_indices'] = set(new_unknown_indices)
                
                # Rebuild W_curr and dummy_curr with new dimensions
                W_curr = W_extended[:, active_indices]
                dummy_curr = {i for i, idx in enumerate(active_indices) if idx in state['dummy_indices']}
                h_init_new, _ = sp.optimize.nnls(W_curr, x)
                
                print(f"  W_curr: {W_curr.shape}, h_init_new: {h_init_new.shape}, active: {len(active_indices)}")


    # === COMBINE UNKNOWN SIGNATURES INTO ONE ===
    if allow_unknown and len(dummy_curr) > 0:
        unknown_active_local = list(dummy_curr)
        
        if len(unknown_active_local) > 1:
            print(f"\n{'='*80}\n=== Combining Unknown Signatures ===")
            print(f"Combining {len(unknown_active_local)} unknown signatures into one\n{'='*80}\n")
            
            # Fit current model to get weights (using L1-penalized optimization)
            h_temp, _ = sp.optimize.nnls(W_curr, x)
            result_temp = minimize(
                fun=lambda w: objective(w, W_curr, dummy_curr),
                x0=h_temp,
                bounds=[(0, None)] * len(active_indices),
                method="SLSQP",
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            unknown_weights = result_temp.x[unknown_active_local]
            total_unknown_weight = np.sum(unknown_weights)
            
            # Create weighted combination of unknown signature profiles
            unknown_sigs = W_curr[:, unknown_active_local]
            if total_unknown_weight > 0:
                combined_unknown = unknown_sigs @ (unknown_weights / total_unknown_weight)
            else:
                combined_unknown = np.mean(unknown_sigs, axis=1)
            combined_unknown = combined_unknown.reshape(-1, 1)
            
            # ✓ CREATE NEW W_CURR = ORIGINAL W + COMBINED UNKNOWN
            W_curr = np.column_stack([W, combined_unknown])
            
            # ✓ UPDATE INDICES: Only known signatures + 1 combined unknown
            combined_unknown_idx = W.shape[1]  # Index right after original signatures
            active_indices = list(range(original_n_sigs)) + [combined_unknown_idx]
            
            # Update state
            state['active_indices'] = active_indices
            state['dummy_indices'] = {combined_unknown_idx}
            
            # Update dummy_curr (combined unknown is last column)
            dummy_curr = {len(active_indices) - 1}
            
            print(f"  Created new W_curr: Original W ({W.shape[1]} sigs) + Combined Unknown (1 sig)")
            print(f"  New W_curr shape: {W_curr.shape}")
            print(f"  Active indices: {len(active_indices)} (Known: {original_n_sigs}, Unknown: 1)")
            


            # ✓ SECOND ROUND OF PRUNING ON W_curr (ORIGINAL + COMBINED UNKNOWN)
            print(f"\n{'='*80}\n=== Second Round: Pruning Original + Combined Unknown ===")
            print(f"Threshold: {thresh_backward}\n{'='*80}\n")
            
            elimination_round = 0
            
            while len(active_indices) > 1 and elimination_round < max_iter:
                elimination_round += 1
                print(f"\n--- Round {elimination_round} (Active: {len(active_indices)}) ---")
                
                # Current model fit
                h_current_nnls, _ = sp.optimize.nnls(W_curr, x)
                reconstruction_current_nnls = W_curr @ h_current_nnls
                
                cos_sim_current = np.dot(x, reconstruction_current_nnls) / (
                    np.linalg.norm(x) * np.linalg.norm(reconstruction_current_nnls)
                )
                
                print(f"  Current cosine similarity: {cos_sim_current:.6f}")
                
                # === LEAVE-ONE-OUT TEST ===
                cos_sim_drops = []
                
                for i in range(len(active_indices)):
                    indices_leave_one_out = [j for j in range(len(active_indices)) if j != i]
                    
                    if len(indices_leave_one_out) == 0:
                        cos_sim_drops.append(np.inf)
                        continue
                    
                    W_reduced = W_curr[:, indices_leave_one_out]
                    h_reduced, _ = sp.optimize.nnls(W_reduced, x)
                    reconstruction_reduced = W_reduced @ h_reduced
                    
                    norm_recon = np.linalg.norm(reconstruction_reduced)
                    
                    if norm_recon > 0:
                        cos_sim_reduced = np.dot(x, reconstruction_reduced) / (np.linalg.norm(x) * norm_recon)
                    else:
                        cos_sim_reduced = 0.0
                    
                    drop = cos_sim_current - cos_sim_reduced
                    cos_sim_drops.append(drop)
                
                cos_sim_drops = np.array(cos_sim_drops)
                min_drop = np.min(cos_sim_drops)
                sig_to_remove_local_idx = np.argmin(cos_sim_drops)
                sig_to_remove_global_idx = active_indices[sig_to_remove_local_idx]
                
                is_unknown = sig_to_remove_global_idx in state['dummy_indices']
                sig_type = "Combined Unknown" if is_unknown else "Known"
                
                print(f"  Cosine drops: min={min_drop:.6f}, max={np.max(cos_sim_drops):.6f}")
                print(f"  Candidate: Sig {sig_to_remove_global_idx} ({sig_type})")
                
                # === REMOVAL DECISION ===
                if min_drop < thresh_backward:
                    print(f"  ✗ REMOVING Sig {sig_to_remove_global_idx} ({sig_type})")
                    print(f"     Drop ({min_drop:.6f}) < threshold ({thresh_backward:.6f})")
                    
                    # Remove from active_indices
                    kept_local_indices = [j for j in range(len(active_indices)) if j != sig_to_remove_local_idx]
                    active_indices = [active_indices[j] for j in kept_local_indices]
                    state['active_indices'] = active_indices
                    
                    # Rebuild W_curr by keeping only columns at kept_local_indices
                    W_curr = W_curr[:, kept_local_indices]
                    
                    # Update dummy_curr
                    dummy_curr = {i for i, idx in enumerate(active_indices) if idx in state['dummy_indices']}
                    
                    # Track metrics
                    h_new, _ = sp.optimize.nnls(W_curr, x)
                    track_metrics(h_new, len(convergence_metrics), W_curr, dummy_curr)
                    
                else:
                    print(f"  ✓ CONVERGED - All signatures significant")
                    print(f"     Min drop ({min_drop:.6f}) >= threshold ({thresh_backward:.6f})")
                    break
            
            print(f"\n{'='*80}")
            print(f"Second pruning complete after {elimination_round} rounds")
            print(f"Final active signatures: {len(active_indices)}")
            print(f"{'='*80}\n")

    # === FINAL OPTIMIZATION ===
    h_init_new, _ = sp.optimize.nnls(W_curr, x)

    def track_metrics_wrapper(xk):
        return track_metrics(xk, len(convergence_metrics), W_curr, dummy_curr)

    result_new = minimize(
        fun=lambda w: objective(w, W_curr, dummy_curr),
        x0=h_init_new,
        bounds=[(0, None)] * len(active_indices),
        method="SLSQP",
        callback=track_metrics_wrapper,
        options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
    )

    weights_optimized = np.maximum(result_new.x, 0)

    # ✓ BUILD FULL WEIGHT VECTOR FOR W_EXTENDED
    h_final = np.zeros(W_extended.shape[1])
    h_final[active_indices] = weights_optimized
    
    # Print summary
    n_known = sum(1 for idx in active_indices if idx < original_n_sigs)
    n_unknown = len(active_indices) - n_known
    print(f"Active signatures: {len(active_indices)} (Known: {n_known}, Unknown: {n_unknown})")
    
    if true_weights is not None:
        tae = recalculate_TAE(pd.DataFrame(h_final[:original_n_sigs].reshape(-1, 1)), pd.DataFrame(true_weights))
        print(f"TAE: {tae.values[0][0]:.4f}")
    
    final_obj = objective(weights_optimized, W_curr, dummy_curr)
    print(f"Final objective: {final_obj:.6f}")
    
    reconstruction_final = W_extended @ h_final
    cos_sim_final = np.dot(x, reconstruction_final) / (np.linalg.norm(x) * np.linalg.norm(reconstruction_final))
    print(f"Final cosine similarity: {cos_sim_final:.6f}")
    
    unassigned = np.sum(np.maximum(x - reconstruction_final, 0))
    print(f"Unassigned: {unassigned:.0f} ({100*unassigned/np.sum(x):.2f}%)\n{'='*80}\n")
    
    # Save metrics
    if convergence_metrics:
        pd.DataFrame(convergence_metrics).to_csv(
            f"{outdir_prefix}/{unknown_penalty}_{sample_name}_convergence_metrics.csv", index=False
        )
    
    # Extract known signature names from original W
    known_names = W_names
    # Create unknown signature names
    n_unknown_total = W_extended.shape[1] - original_n_sigs
    unknown_names = [f"Unknown_{i}" for i in range(n_unknown_total)]
    
    # Combine all names
    all_names = known_names + unknown_names
    
    # Return DataFrame with proper names
    return pd.DataFrame(h_final, index=all_names, columns=[sample_name])






# MUSICAL RELATED FUNCTIONS

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