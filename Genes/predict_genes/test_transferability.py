import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dadapy.metric_comparisons import MetricComparisons
from scipy.stats import rankdata
import pickle
from scipy.spatial.distance import pdist, squareform
import argparse

def nns_index_array(data, metric="euclidean", maxk=1):
    """
    Computes the indices of the k nearest neighbors to each point.

    Args:
        data (np.ndarray or list(list(float))): dataset of shape (N,D), with N points and D features
        metric (str, default="euclidean"): name of the distance employed
        maxk (int, default=1): number of nearest neighbors considered in the Information Imbalance calculation

    Returns:
        NNs (np.ndarray): array of shape (N,k). The ij-th element is the index of the j-th nearest 
                          neighbor to point i.
    """
    N = data.shape[0]

    pairwise_dist = squareform(pdist(data, metric=metric))
    pairwise_dist = pairwise_dist.astype('float')

    rank_matrix = rankdata(pairwise_dist, method='average', axis=1)
    
    NNs = np.zeros((N, maxk+1), dtype=int)
    for i in np.arange(N):
        NNs[i, :] = np.argpartition(rank_matrix[i], np.arange(maxk+1))[:maxk+1]
    return np.array(NNs)

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", dest="k",
                    default=1, type=int,
                    help="Number of neighbors to compute Information Imbalance")
parser.add_argument("-n_coords", "--n_coords", dest="n_coords",
                    default=30, type=int,
                    help="Maximum number of optimal features to select")
parser.add_argument("-n_best", "--n_best", dest="n_best",
                    default=1, type=int,
                    help="Parameter of beam search (number of optimal tuples tested with additional variables at each step)")
parser.add_argument("-i_fold", "--i_fold", dest="i_fold",
                    default=0, type=int,
                    help="Index of the fold in the 3-fold cross-validation")
args = parser.parse_args()

genes_df = pd.read_csv("../data/gene.yyy")

# extract numpy arrays from pandas dataframes
genes = (genes_df.T).iloc[1:,:].to_numpy(dtype=float)

# add noise to break neighbor degeneracies (both in gene and pseudotime spaces)
np.random.seed(1998)
genes += np.random.normal(loc=0., scale=1e-6, size=genes.shape)

# shuffle indices of genes datasets 
new_indices = np.random.choice(np.arange(genes.shape[0]), replace=False, size=genes.shape[0])
genes = genes[new_indices]

# cross-validation
N_fold = genes.shape[0] // 3
fold_indices = [[np.arange(N_fold), np.arange(N_fold,2*N_fold), np.arange(2*N_fold,3*N_fold)],
                [np.arange(N_fold,2*N_fold), np.arange(N_fold), np.arange(2*N_fold,3*N_fold)],
                [np.arange(2*N_fold,3*N_fold), np.arange(N_fold), np.arange(N_fold,2*N_fold)]]

#for i_fold in range(3): 
njobs = 16

# split train and validation indices
train_indices = fold_indices[args.i_fold][0]
val_indices1 = fold_indices[args.i_fold][1]
val_indices2 = fold_indices[args.i_fold][2]

# compute greedy imbalance on train set
target_ranks_genes = nns_index_array(genes[train_indices], maxk=len(train_indices)-1)
d = MetricComparisons(genes[train_indices], njobs=njobs)
variables_train, imbalances_train, _ = d.greedy_feature_selection_target(target_ranks=target_ranks_genes, 
                                                            n_best=args.n_best,
                                                            n_coords=args.n_coords,
                                                            k=args.k,
                                                            symm=False)

# first validation
target_ranks_genes = nns_index_array(genes[val_indices1], maxk=len(val_indices1)-1)
d = MetricComparisons(genes[val_indices1], njobs=njobs)
imbalances_val1 = d.return_inf_imb_target_selected_coords(target_ranks=target_ranks_genes, 
                                                              coord_list=variables_train, 
                                                              k=args.k)

# second validation
target_ranks_genes = nns_index_array(genes[val_indices2], maxk=len(val_indices2)-1)
d = MetricComparisons(genes[val_indices2], njobs=njobs)
imbalances_val2 = d.return_inf_imb_target_selected_coords(target_ranks=target_ranks_genes, 
                                                              coord_list=variables_train, 
                                                              k=args.k)

pickle.dump([variables_train, imbalances_train, imbalances_val1, imbalances_val2],
            open(f"./pickles_transferability/nbest{args.n_best}_ncoords{args.n_coords}_k{args.k}_fold{args.i_fold}.p","wb"))