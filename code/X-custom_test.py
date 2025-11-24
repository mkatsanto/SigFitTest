#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd

sys.path.append('/Users/mariakatsantoni/sigfittest')
from custom_method import custom_attempt1

X = pd.read_csv('data/data_for_MutationalPatterns.dat', index_col = 0, sep = '\t')
W = pd.read_csv('../input/COSMIC_v3_SBS_GRCh38.txt', sep = '\t', index_col = 0)
MP_index = pd.read_csv('../input/mut_matrix_order_MutationalPatterns.dat', header = None).squeeze()
W = W.reindex(MP_index)

# to do run for multiple lambda and fix the results
for unknown_penalty in [5, 10,15,20,30,50,100]:
    H_results = []
    for sample_name in X.columns:
        print(f"Processing sample: {sample_name}")
        x = X[sample_name].values  # Get 1D array for this sample
        
        # Call custom_likelihood_bidirectional directly
        h = custom_attempt1(
            x=x,
            W=W.values,
            thresh_backward=0.001,
            thresh_forward=0.001,
            max_iter=1000,
            per_trial=True,
            indices_associated_sigs=None,
            allow_unknown=True,
            unknown_penalty=unknown_penalty,
        )
        
        H_results.append(h)
        print(f"  Result: {np.sum(h > 1e-8)} non-zero signatures")

    # Convert to DataFrame
    H_results = np.array(H_results).T
    print(f"Result shape: {H_results.shape}")

    # Handle signature names (account for potential unknown signatures)
    if W.shape[1] < H_results.shape[0]:
        known_sigs = W.columns.tolist()
        n_unknown = H_results.shape[0] - W.shape[1]
        unknown_sigs = [f'Unknown_{i+1}' for i in range(n_unknown)]
        all_sig_names = known_sigs + unknown_sigs
    else:
        all_sig_names = W.columns.tolist()

    H = pd.DataFrame(H_results, index=all_sig_names, columns=X.columns)

    # Save results
    print(f"Current working directory: {os.getcwd()}")

    os.makedirs('../signature_results', exist_ok=True)
    H.to_csv(f'../signature_results/custom_test-contribution_orthgonal_{unknown_penalty}.dat')
    print(f"Results saved! Final shape: {H.shape}")



