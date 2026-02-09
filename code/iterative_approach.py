
import numpy as np
import pandas as pd
import scipy as sp
import warnings
from pathlib import Path


def recalculate_TAE(df_pred, df_true):
    """
    Recalculate the Total Absolute Error (TAE) between predicted and true values.
    """
    muts_true = df_true.sum()  # total mutations per sample
    df_pred1 = df_pred.reindex(index=df_true.index, fill_value=0)
    TAE = df_true.sub(df_pred1, axis=0).abs().sum() / (2 * (muts_true))   # total absolute error (aka "fitting error") for each sample     
    TAE = pd.DataFrame(TAE)
    return TAE


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def sigprofiler_backward_elimination(W, x, thresh_backward=0.01, max_iter=100):
    """
    SigProfilerAssignment-style backward elimination using cosine similarity.
    
    Parameters:
    -----------
    W : np.ndarray
        Signature matrix (channels × signatures)
    x : np.ndarray
        Mutation counts (channels)
    thresh_backward : float
        Minimum cosine similarity drop to keep a signature (default: 0.01)
    max_iter : int
        Maximum elimination rounds
    
    Returns:
    --------
    np.ndarray: Final weights for all signatures (zeros for removed ones)
    """
    n_sigs = W.shape[1]
    active_indices = list(range(n_sigs))
    
    print(f"\n{'='*80}")
    print(f"=== SigProfilerAssignment-Style Backward Elimination ===")
    print(f"Cosine Similarity Drop Threshold: {thresh_backward}")
    print(f"{'='*80}\n")
    
    for elimination_round in range(1, max_iter + 1):
        if len(active_indices) <= 1:
            print(f"Only 1 signature remaining - stopping")
            break
        
        print(f"\n--- Round {elimination_round} (Active: {len(active_indices)}) ---")
        
        # Current model
        W_curr = W[:, active_indices]
        h_current, _ = sp.optimize.nnls(W_curr, x)
        reconstruction_current = W_curr @ h_current
        
        # Current cosine similarity
        cos_sim_current = cosine_similarity(x, reconstruction_current)
        print(f"  Current cosine similarity: {cos_sim_current:.6f}")
        
        # Leave-one-out testing
        cos_sim_drops = []
        
        for i in range(len(active_indices)):
            # Remove signature i
            indices_without_i = [j for j in range(len(active_indices)) if j != i]
            
            if len(indices_without_i) == 0:
                cos_sim_drops.append(np.inf)
                continue
            
            # Refit without signature i
            W_reduced = W_curr[:, indices_without_i]
            h_reduced, _ = sp.optimize.nnls(W_reduced, x)
            reconstruction_reduced = W_reduced @ h_reduced
            
            # Cosine similarity drop
            cos_sim_reduced = cosine_similarity(x, reconstruction_reduced)
            drop = cos_sim_current - cos_sim_reduced
            cos_sim_drops.append(drop)
        
        cos_sim_drops = np.array(cos_sim_drops)
        min_drop = np.min(cos_sim_drops)
        sig_to_remove_local_idx = np.argmin(cos_sim_drops)
        sig_to_remove_global_idx = active_indices[sig_to_remove_local_idx]
        
        print(f"  Cosine drops: min={min_drop:.6f}, max={np.max(cos_sim_drops):.6f}")
        print(f"  Candidate to remove: Signature {sig_to_remove_global_idx}")
        
        # Removal decision
        if min_drop < thresh_backward:
            print(f"  ✗ REMOVING Signature {sig_to_remove_global_idx}")
            print(f"     Reason: Cosine drop ({min_drop:.6f}) < threshold ({thresh_backward:.6f})")
            
            # Remove signature
            active_indices = [idx for j, idx in enumerate(active_indices) if j != sig_to_remove_local_idx]
        else:
            print(f"  ✓ CONVERGED - All signatures significant")
            print(f"     Reason: Min cosine drop ({min_drop:.6f}) >= threshold ({thresh_backward:.6f})")
            break
    
    # Final fit with active signatures
    W_final = W[:, active_indices]
    h_final_active, _ = sp.optimize.nnls(W_final, x)
    
    # Map to full signature space
    h_final = np.zeros(n_sigs)
    h_final[active_indices] = h_final_active
    
    print(f"\n{'='*80}")
    print(f"Elimination complete after {elimination_round} rounds")
    print(f"Final active signatures: {len(active_indices)}")
    print(f"{'='*80}\n")
    
    return h_final


def leave_out_unassigned(x, W, true_weights, sample_name=None, zscore_threshold=3, 
                         thresh_backward=0.01, max_iter=3000, 
                         prune_signatures_flag=True, output_dir=None):
    """
    Iteratively remove significant unassigned mutations, then apply SigProfiler-style pruning.
    
    Parameters:
    -----------
    x : np.ndarray
        Observed mutation counts (channels)
    W : np.ndarray
        Signature matrix (channels × signatures)
    true_weights : np.ndarray
        True weights for TAE calculation
    sample_name : str
        Name of the sample (for logging)
    zscore_threshold : float
        Z-score threshold for unassigned mutation removal
    thresh_backward : float
        Cosine similarity drop threshold for SigProfiler pruning (default: 0.01)
    max_iter : int
        Maximum iterations for unassigned removal
    prune_signatures_flag : bool
        Whether to apply SigProfiler-style signature pruning
    output_dir : str or Path
        Directory to save convergence file
    
    Returns:
    --------
    np.ndarray: Final weights
    """
    # Replace None values in x with 0
    x = np.array([0 if v is None else v for v in x], dtype=np.float64)

    # Initialize convergence tracking
    convergence_data = []
    
    unassigned_list = []
    x_fit = x.copy()
    
    # PHASE 1: Iterative unassigned mutation removal (no pruning)
    print("\n=== PHASE 1: Iterative Unassigned Removal ===")
    for i in range(max_iter):
        # Fit WITHOUT pruning
        weights, _ = sp.optimize.nnls(W, x_fit)
        
        predicted_counts = W @ weights
        unassigned = np.maximum(x_fit - predicted_counts, 0)
        assigned = np.minimum(predicted_counts, x_fit)
        overfitted = np.maximum(predicted_counts - x_fit, 0)
        
        # Calculate metrics
        overfitted_sum = np.sum(overfitted)
        unassigned_sum = np.sum(unassigned)
        assigned_sum = np.sum(assigned)
        x_fit_sum = np.sum(x_fit)
        predicted_sum = np.sum(predicted_counts)
        total_unassigned = np.sum(unassigned_list)
        
        # Calculate TAE
        TAE = recalculate_TAE(pd.DataFrame(weights), pd.DataFrame(true_weights))
        tae_value = TAE.values[0][0]
        
        # Count active signatures
        n_active = np.sum(weights > 5)  # Using 5 mutations as threshold
        
        # Calculate cosine similarity
        cos_sim = cosine_similarity(x_fit, predicted_counts)
        
        # Store convergence data
        convergence_data.append({
            'iteration': i + 1,
            'phase': 'removal',
            'x_fit_sum': x_fit_sum,
            'predicted_sum': predicted_sum,
            'assigned': assigned_sum,
            'unassigned_current': unassigned_sum,
            'unassigned_total': total_unassigned,
            'overfitted': overfitted_sum,
            'TAE': tae_value,
            'cosine_similarity': cos_sim,
            'n_active_signatures': n_active,
            'zscore_threshold': zscore_threshold,
            'thresh_backward': thresh_backward
        })

        # Remove significant unassigned mutations (z-score approach)
        if np.sum(unassigned) > 0:
            mean = np.mean(unassigned)
            std = np.std(unassigned)
            
            if std == 0:
                print("Standard deviation is zero. No significant unassigned mutations to remove.")
                break
            
            threshold = mean + zscore_threshold * std
            removed = np.zeros_like(unassigned)
            above = unassigned > threshold
            removed[above] = unassigned[above] - threshold
            
            if removed[above].sum() == 0:
                print("No significant unassigned mutations left to remove.")
                break
            
            x_fit -= removed
            x_fit = np.maximum(x_fit, 0)
            unassigned_list.append(np.sum(removed))
        else:
            unassigned_list.append(0)
            print("No unassigned mutations left.")
            break

        print(f"Iter {i+1}: x_fit_sum={x_fit_sum:.2f}, predicted_sum={predicted_sum:.2f}, "
              f"assigned={assigned_sum:.2f}, unassigned_total={total_unassigned:.2f}, "
              f"overfitted={overfitted_sum:.2f}, TAE={tae_value:.4f}, cos_sim={cos_sim:.4f}, active_sigs={n_active}")
        
        # Early stopping if overfitting is minimal
        if overfitted_sum < 100:
            print("Overfitting below threshold. Stopping removal phase.")
            break
    
    # PHASE 2: SigProfilerAssignment-style signature pruning
    print("\n=== PHASE 2: SigProfilerAssignment-Style Pruning ===")
    if prune_signatures_flag:
        weights = sigprofiler_backward_elimination(W, x_fit, thresh_backward, max_iter=100)
        
        # Calculate final metrics after pruning
        predicted_counts = W @ weights
        unassigned = np.maximum(x_fit - predicted_counts, 0)
        assigned = np.minimum(predicted_counts, x_fit)
        overfitted = np.maximum(predicted_counts - x_fit, 0)
        
        overfitted_sum = np.sum(overfitted)
        unassigned_sum = np.sum(unassigned)
        assigned_sum = np.sum(assigned)
        predicted_sum = np.sum(predicted_counts)
        total_unassigned = np.sum(unassigned_list)
        
        TAE = recalculate_TAE(pd.DataFrame(weights), pd.DataFrame(true_weights))
        tae_value = TAE.values[0][0]
        n_active = np.sum(weights > 5)
        cos_sim = cosine_similarity(x_fit, predicted_counts)
        
        # Add final pruned state to convergence data
        convergence_data.append({
            'iteration': len(convergence_data) + 1,
            'phase': 'pruning',
            'x_fit_sum': x_fit_sum,
            'predicted_sum': predicted_sum,
            'assigned': assigned_sum,
            'unassigned_current': unassigned_sum,
            'unassigned_total': total_unassigned,
            'overfitted': overfitted_sum,
            'TAE': tae_value,
            'cosine_similarity': cos_sim,
            'n_active_signatures': n_active,
            'zscore_threshold': zscore_threshold,
            'thresh_backward': thresh_backward
        })
        
        print(f"After SigProfiler pruning: predicted_sum={predicted_sum:.2f}, assigned={assigned_sum:.2f}, "
              f"unassigned_total={total_unassigned:.2f}, overfitted={overfitted_sum:.2f}, "
              f"TAE={tae_value:.4f}, cos_sim={cos_sim:.4f}, active_sigs={n_active}")
    else:
        # If no pruning, just do final fit
        print("\n=== PHASE 2: Final Fit (No Pruning) ===")
        weights, _ = sp.optimize.nnls(W, x_fit)
        
        TAE = recalculate_TAE(pd.DataFrame(weights), pd.DataFrame(true_weights))
        tae_value = TAE.values[0][0]
        n_active = np.sum(weights > 5)
        
        print(f"Final fit: TAE={tae_value:.4f}, active_sigs={n_active}")

    print(f"\nFinal: {n_active} active signatures, TAE={tae_value:.4f}")
    
    # Create convergence DataFrame
    convergence_df = pd.DataFrame(convergence_data)
    
    # Save convergence file if output directory is provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sample_str = f"_{sample_name}" if sample_name else ""
        filename = f"convergence{sample_str}_sigprofiler_style.csv"
        filepath = output_path / filename
        
        convergence_df.to_csv(filepath, index=False)
        print(f"Convergence data saved to: {filepath}")
    
    return weights

# def leave_out_unassigned_percentile(x, W, true_weights, sample_name=None, remove_percentile=90, 
#                                     thresh_backward=0.001, thresh_forward=0.001, 
#                                     max_iter=3000, prune_signatures_flag=True, output_dir=None):
#     """
#     Iteratively remove top percentile of unassigned mutations and prune signatures.
    
#     Parameters similar to leave_out_unassigned, but uses percentile instead of z-score.
    
#     Returns:
#     --------
#     tuple: (weights, convergence_df)
#     """

#     x = np.array([0 if v is None else v for v in x], dtype=np.float64)
    
#     # Initialize convergence tracking
#     convergence_data = []
    
#     unassigned_list = []
#     x_fit = x.copy()
    
#     for i in range(max_iter):
#         # Fit with signature pruning
#         if prune_signatures_flag:
#             weights = iterative_pruning_nnls(W, x_fit, thresh_backward, thresh_forward, max_iter=100)
#         else:
#             weights, _ = sp.optimize.nnls(W, x_fit)
        
#         predicted_counts = W @ weights
#         unassigned = np.maximum(x_fit - predicted_counts, 0)
#         assigned = np.minimum(predicted_counts, x_fit)
#         overfitted = np.maximum(predicted_counts - x_fit, 0)
        
#         # Calculate metrics
#         overfitted_sum = np.sum(overfitted)
#         unassigned_sum = np.sum(unassigned)
#         assigned_sum = np.sum(assigned)
#         x_fit_sum = np.sum(x_fit)
#         predicted_sum = np.sum(predicted_counts)
#         total_unassigned = np.sum(unassigned_list)
        
#         # Calculate TAE
#         TAE = recalculate_TAE(pd.DataFrame(weights), pd.DataFrame(true_weights))
#         tae_value = TAE.values[0][0]
        
#         # Count active signatures
#         n_active = np.sum(weights > thresh_backward)
        
#         # Store convergence data
#         convergence_data.append({
#             'iteration': i + 1,
#             'x_fit_sum': x_fit_sum,
#             'predicted_sum': predicted_sum,
#             'assigned': assigned_sum,
#             'unassigned_current': unassigned_sum,
#             'unassigned_total': total_unassigned,
#             'overfitted': overfitted_sum,
#             'TAE': tae_value,
#             'n_active_signatures': n_active,
#             'percentile_threshold': remove_percentile,
#             'thresh_backward': thresh_backward,
#             'thresh_forward': thresh_forward
#         })

#         # Remove top percentile of unassigned mutations
#         if np.sum(unassigned) > 0:
#             unassigned_nonzero = unassigned[unassigned > 0]
            
#             if len(unassigned_nonzero) == 0:
#                 print("No non-zero unassigned mutations.")
#                 break
            
#             percentile_threshold = np.percentile(unassigned_nonzero, remove_percentile)
            
#             removed = np.zeros_like(unassigned)
#             above_threshold = unassigned > percentile_threshold
#             removed[above_threshold] = unassigned[above_threshold] - percentile_threshold
            
#             if removed[above_threshold].sum() == 0:
#                 print("No mutations above percentile threshold.")
#                 break
            
#             x_fit -= removed
#             x_fit = np.maximum(x_fit, 0)
#             unassigned_list.append(np.sum(removed))
#         else:
#             unassigned_list.append(0)
#             break

#         print(f"Iter {i+1}: x_fit_sum={x_fit_sum:.2f}, predicted_sum={predicted_sum:.2f}, "
#               f"assigned={assigned_sum:.2f}, unassigned_total={total_unassigned:.2f}, "
#               f"overfitted={overfitted_sum:.2f}, TAE={tae_value:.4f}, active_sigs={n_active}")
        
#         if overfitted_sum < 100:
#             print("Overfitting below threshold. Stopping.")
#             break

#     print(f"Final: {n_active} active signatures, TAE={tae_value:.4f}")
    
#     # Create convergence DataFrame
#     convergence_df = pd.DataFrame(convergence_data)
    
#     # Save convergence file if output directory is provided
#     if output_dir is not None:
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         sample_str = f"_{sample_name}" if sample_name else ""
#         filename = f"convergence{sample_str}_percentile.csv"
#         filepath = output_path / filename
        
#         convergence_df.to_csv(filepath, index=False)
#         print(f"Convergence data saved to: {filepath}")
    
#     return weights
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# def recalculate_TAE(df_pred, df_true):
#     """
#     Recalculate the Total Absolute Error (TAE) between predicted and true values.
#     """
#     muts_true = df_true.sum()  # total mutations per sample
    
#     df_pred1 = df_pred.reindex(index=df_true.index, fill_value=0)
#     TAE = df_true.sub(df_pred1, axis=0).abs().sum() / (2 * (muts_true))   # total absolute error (aka "fitting error") for each sample     
#     TAE = pd.DataFrame(TAE)
#     return TAE


# # def leave_out_unassigned(x, W, true_weights, remove_fraction=0.8, threshold=10, thresh_backward=0.001, thresh_forward=None, max_iter=3000):
# #     """Leave out the unassigned mutations"""
    
# #     if thresh_forward is None:
# #         thresh_forward = thresh_backward
# #     if thresh_backward > thresh_forward:
# #         warnings.warn('thresh_backward is greater thresh_forward. This might lead to indefinite loops.', UserWarning)
    
# #     unassigned_list = []
# #     max_iter = max_iter
# #     x_fit = x.copy().astype(np.float64)
# #     for i in range(max_iter):
# #         weights, residual_norm = sp.optimize.nnls(W, x_fit)
# #         predicted_counts = W @ weights

# #         unassigned = np.maximum(x_fit - predicted_counts, 0)
# #         assigned = np.minimum(predicted_counts, x_fit)
        
# #         overfitted = np.maximum(predicted_counts - x_fit, 0)
# #         overfitted_sum = np.sum(overfitted)
# #         # Calculate percentile threshold on unassigned mutations
# #         if np.sum(unassigned) > 0:
# #             # Only consider non-zero unassigned values for percentile
# #             unassigned_nonzero = unassigned[unassigned > 0]
# #             if len(unassigned_nonzero) > 0:
# #                 percentile_threshold = np.percentile(unassigned_nonzero, remove_fraction * 100)
                
# #                 # Remove only mutations above the percentile threshold
# #                 removed = np.zeros_like(unassigned)
# #                 above_threshold = unassigned > percentile_threshold
# #                 removed[above_threshold] = unassigned[above_threshold] - percentile_threshold
                
# #                 x_fit -= removed
# #                 x_fit = np.maximum(x_fit, 0)
# #                 unassigned_list.append(np.sum(removed))
# #             else:
# #                 unassigned_list.append(0)
# #                 break
# #         else:
# #             unassigned_list.append(0)
# #             break
# #         print(weights, true_weights)
# #         TAE = recalculate_TAE(pd.DataFrame(weights), pd.DataFrame(true_weights))
        
# #         print(f"Sum of x_fit: {(x_fit.sum().sum()):.2f}, Sum of predicted_counts: {(predicted_counts.sum().sum()):.2f},Assigned mutations: {np.sum(assigned):.2f}, Unassigned mutations: {np.sum(unassigned_list):.2f}, Overfitted mutations: {overfitted_sum:.2f}, TAE: {TAE.values[0][0]:.4f}")
# #     print("okay done")
# #     return weights


# # Z- SCORE IMPLEMENTATION 
# def leave_out_unassigned(x, W, true_weights, zscore_threshold=3, thresh_backward=0.001, thresh_forward=None, max_iter=3000):
#     """Iteratively remove only the significant part of unassigned mutations (above z-score threshold) for each channel."""

#     if thresh_forward is None:
#         thresh_forward = thresh_backward
#     if thresh_backward > thresh_forward:
#         warnings.warn('thresh_backward is greater thresh_forward. This might lead to indefinite loops.', UserWarning)

#     # Replace None values in x with 0
#     x = np.array([0 if v is None else v for v in x])

#     # def objective_function(weights, x_fit):
#     #     weights = np.maximum(weights, 0)
#     #     predicted_counts = W @ weights
#     #     return 0.5 * np.sum((x_fit - predicted_counts) ** 2)

#     # bounds = [(0, None) for _ in range(len(W.T))]
#     # h_init, _ = sp.optimize.nnls(W, x)

#     unassigned_list = []
#     max_iter = max_iter
#     x_fit = x.copy().astype(np.float64)
#     for i in range(max_iter):
#         weights, residual_norm = sp.optimize.nnls(W, x_fit)
        
#         predicted_counts = W @ weights
#         unassigned = np.maximum(x_fit - predicted_counts, 0)
#         assigned = np.minimum(predicted_counts, x_fit)
#         overfitted = np.maximum(predicted_counts - x_fit, 0)
#         overfitted_sum = np.sum(overfitted)

#         if np.sum(unassigned) > 0:
#             mean = np.mean(unassigned)
#             std = np.std(unassigned)
#             threshold = mean + zscore_threshold * std
#             removed = np.zeros_like(unassigned)
#             above = unassigned > threshold
#             removed[above] = unassigned[above] - threshold
            
#             if removed[above].sum() == 0:
#                 print("No significant unassigned mutations left to remove.")
#                 break
            
#             x_fit -= removed
#             x_fit = np.maximum(x_fit, 0)
#             unassigned_list.append(np.sum(removed))
#         else:
#             unassigned_list.append(0)
#             print("No significant unassigned mutations left to remove.")
#             break

#         TAE = recalculate_TAE(pd.DataFrame(weights), pd.DataFrame(true_weights))
        
    
#         print(f"Sum of x_fit: {(x_fit.sum().sum()):.2f}, Sum of predicted_counts: {(predicted_counts.sum().sum()):.2f}, Assigned mutations: {np.sum(assigned):.2f}, Unassigned mutations: {np.sum(unassigned_list):.2f}, Overfitted mutations: {overfitted_sum:.2f}, TAE: {TAE.values[0][0]:.4f}")
#         if overfitted_sum < 100:
#             break

    # return weights