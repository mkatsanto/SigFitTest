#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
#sys.path.append('/Users/mariakatsantoni/sigfittest')
from extended_method import enrich_signature_references, musical_nnls_likelihood_bidirectional

approach='extended'
outdir = f'../signature_results_extended'


penalties = [0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000, 50000]


X = pd.read_csv('data/data_for_MutationalPatterns.dat', index_col = 0, sep = '	')
W = pd.read_csv('../input/COSMIC_v3_SBS_GRCh38.txt', sep = '	', index_col = 0)
MP_index = pd.read_csv('../input/mut_matrix_order_MutationalPatterns.dat', header = None).squeeze()
W = W.reindex(MP_index)
true_weights = pd.read_csv('/Users/mariakatsantoni/sigfittest/SigFitTest/code/data/true_weights_ColoRect-AdenoCA_SET6_w0_50000.dat', sep='\t', index_col=0).T * 50000
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
            W=W,
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
    # # Convert to DataFrame
    # H_results = np.array(H_results).T
    # print(f"Result shape: {H_results.shape}")
    # H = pd.DataFrame(H_results, index=all_sig_names, columns=X.columns)
    # H.to_csv(f'{outdir}/custom_test-contribution_{approach}_{unknown_penalty}.dat')
    # print(f"Results saved! Final shape: {H.shape}")

    # Concatenate DataFrames column-wise
    H = pd.concat(H_results, axis=1)

    # Save
    output_file = f'{outdir}/custom_test-contribution_{approach}_{unknown_penalty}.dat'
    H.to_csv(output_file, sep='	', index=True, header=True)
    print(f"Results saved to {output_file}")

