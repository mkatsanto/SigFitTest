#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from iterative_approach import leave_out_unassigned

approach='iterative'
outdir = f'../signature_results_iterative/'


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

H_results = []
for sample_name in X.columns:
    print(f"Processing sample: {sample_name}")
    #Call custom_likelihood_bidirectional directly
    # h = leave_out_unassigned(
    #     x=X[sample_name].values,
    #     W=W.values,
    #     true_weights=true_weights[sample_name].values,
    #     remove_fraction=0.30,
    #     threshold=0.2,
    #     thresh_backward=0.001,
    #     thresh_forward=0.001,
    #     max_iter=10
    # )


    h = leave_out_unassigned(
        x=X[sample_name].values,
        W=W.values,
        true_weights=true_weights[sample_name].values,
        sample_name=sample_name,
        zscore_threshold=1,
        thresh_backward=0.001,
        max_iter=5000,
        prune_signatures_flag=True,
        output_dir=outdir
    )
    H_results.append(h)

# Convert to DataFrame
H_results = np.array(H_results).T
print(f"Result shape: {H_results.shape}")
all_sig_names = W.columns.tolist()
H = pd.DataFrame(H_results, index=all_sig_names, columns=X.columns)
H.to_csv(f'{outdir}/custom_test-contribution_{approach}.dat')
print(f"Results saved! Final shape: {H.shape}")    

