#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
#sys.path.append('/Users/mariakatsantoni/sigfittest')
from extended_method import enrich_signature_references, musical_nnls_likelihood_bidirectional

approach='identity'
outdir = f'../signature_results_identity/'


penalties = [5,25,50,200]


X = pd.read_csv('data/data_for_MutationalPatterns.dat', index_col = 0, sep = '	')
W = pd.read_csv('../input/COSMIC_v3_SBS_GRCh38.txt', sep = '	', index_col = 0)
MP_index = pd.read_csv('../input/mut_matrix_order_MutationalPatterns.dat', header = None).squeeze()
W = W.reindex(MP_index)
true_weights = pd.read_csv('/Users/mariakatsantoni/sigfittest/SigFitTest/code/data/true_weights_Head-SCC_SET6_w0_1000000.dat', sep='\t', index_col=0).T * 1000000
# Keep extra true_weights indexes and put them at the end
extra_idxs = [idx for idx in true_weights.index if idx not in W.columns]
ordered_idxs = list(W.columns) + extra_idxs
true_weights = true_weights.reindex(index=ordered_idxs)
true_weights.fillna(0, inplace=True)
os.makedirs(outdir, exist_ok=True)


# Iterate over unknown penalties
for unknown_penalty in penalties:
    H_results = []
    for sample_name in X.columns:
        print(f"Processing sample: {sample_name}")
        x = X[sample_name].values  # Get 1D array for this sample
        # Call custom_likelihood_bidirectional directly
        h = enrich_signature_references(
            x=x,
            W=W.values,
            sample_name=sample_name,
            outdir_prefix=outdir,
            approach=approach,  
            channel_number=W.shape[0],
            thresh_backward=0.001,
            thresh_forward=0.001,
            max_iter=1000,
            per_trial=True,
            indices_associated_sigs=None,
            allow_unknown=True,
            unknown_penalty=unknown_penalty,
            true_weights=true_weights[sample_name].values,
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
    H.to_csv(f'{outdir}/custom_test-contribution_{approach}_{unknown_penalty}.dat')
    print(f"Results saved! Final shape: {H.shape}")


