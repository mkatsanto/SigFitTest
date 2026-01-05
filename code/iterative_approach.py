
import numpy as np
import pandas as pd
import scipy as sp
import warnings
        

def recalculate_TAE(df_pred, df_true):
    """
    Recalculate the Total Absolute Error (TAE) between predicted and true values.
    """
    muts_true = df_true.sum()  # total mutations per sample
    
    df_pred1 = df_pred.reindex(index=df_true.index, fill_value=0)
    TAE = df_true.sub(df_pred1, axis=0).abs().sum() / (2 * (muts_true))   # total absolute error (aka "fitting error") for each sample     
    TAE = pd.DataFrame(TAE)
    return TAE


def leave_out_unassigned(x, W, true_weights, remove_fraction=0.8, threshold=10, thresh_backward=0.001, thresh_forward=None, max_iter=3000):
    """Leave out the unassigned mutations"""
    
    if thresh_forward is None:
        thresh_forward = thresh_backward
    if thresh_backward > thresh_forward:
        warnings.warn('thresh_backward is greater thresh_forward. This might lead to indefinite loops.', UserWarning)
    
    unassigned_list = []
    max_iter = max_iter
    x_fit = x.copy().astype(np.float64)
    for i in range(max_iter):
        weights, residual_norm = sp.optimize.nnls(W, x_fit)
        predicted_counts = W @ weights

        unassigned = np.maximum(x_fit - predicted_counts, 0)
        assigned = np.minimum(predicted_counts, x_fit)
        
        overfitted = np.maximum(predicted_counts - x_fit, 0)
        overfitted_sum = np.sum(overfitted)
        # Calculate percentile threshold on unassigned mutations
        if np.sum(unassigned) > 0:
            # Only consider non-zero unassigned values for percentile
            unassigned_nonzero = unassigned[unassigned > 0]
            if len(unassigned_nonzero) > 0:
                percentile_threshold = np.percentile(unassigned_nonzero, remove_fraction * 100)
                
                # Remove only mutations above the percentile threshold
                removed = np.zeros_like(unassigned)
                above_threshold = unassigned > percentile_threshold
                removed[above_threshold] = unassigned[above_threshold] - percentile_threshold
                
                x_fit -= removed
                x_fit = np.maximum(x_fit, 0)
                unassigned_list.append(np.sum(removed))
            else:
                unassigned_list.append(0)
                break
        else:
            unassigned_list.append(0)
            break
        print(weights, true_weights)
        TAE = recalculate_TAE(pd.DataFrame(weights), pd.DataFrame(true_weights))
        
        print(f"Sum of x_fit: {(x_fit.sum().sum()):.2f}, Sum of predicted_counts: {(predicted_counts.sum().sum()):.2f},Assigned mutations: {np.sum(assigned):.2f}, Unassigned mutations: {np.sum(unassigned_list):.2f}, Overfitted mutations: {overfitted_sum:.2f}, TAE: {TAE.values[0][0]:.4f}")
    print("okay done")
    return weights


# Z- SCORE IMPLEMENTATION 
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

#     return weights