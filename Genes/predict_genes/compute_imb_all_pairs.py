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
                    default=30, type=int,
                    help="Number of neighbors to compute Information Imbalance")
parser.add_argument("-n_coords", "--n_coords", dest="n_coords",
                    default=25, type=int,
                    help="Maximum number of optimal features to select")
parser.add_argument("-n_best", "--n_best", dest="n_best",
                    default=5, type=int,
                    help="Parameter of beam search (number of optimal tuples tested with additional variables at each step)")
args = parser.parse_args()

features, _ = pickle.load(open(f"./pickles/nbest{args.n_best}_ncoords{args.n_coords}_k{args.k}.p","rb"))
genes_df = pd.read_csv("../data/gene.yyy")

# extract numpy arrays from pandas dataframes
genes = (genes_df.T).iloc[1:,:].to_numpy(dtype=float)

# select only columns corresponding to optimal 25 features
genes = genes[:,features[24]]

# add noise to break neighbor degeneracies
genes += np.random.normal(loc=0., scale=1e-6, size=genes.shape)

# compute imbalance for all the pairs
njobs = 8
d = MetricComparisons(genes, njobs=njobs, maxk=genes.shape[0]-1)
imbalances = d.return_inf_imb_matrix_of_coords(k=args.k)

pickle.dump([features[24], imbalances], open(f"./pickles/matrix_imbalances_ncoords{args.n_coords}_k{args.k}.p","wb"))