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
parser.add_argument("-feature_ref", "--feature_ref", dest="feature_ref",
                    default=0, type=int,
                    help="Optimal feature in space A")
args = parser.parse_args()

features, _ = pickle.load(open(f"./pickles/nbest5_ncoords30_k{args.k}.p","rb"))
features_selected = np.array(features[24])
feature_ref = np.array(features[24])[args.feature_ref]
genes_df = pd.read_csv("../data/gene.yyy")

# extract numpy arrays from pandas dataframes
genes = (genes_df.T).iloc[1:,:].to_numpy(dtype=float)
features_nonselected = np.delete(np.arange(genes.shape[1]), features_selected)
print(genes.shape, features_selected.shape, features_nonselected.shape)

# add noise to break neighbor degeneracies
genes += np.random.normal(loc=0., scale=1e-6, size=genes.shape)

# compute imbalance of selected features vs nonselected ones
target_ranks_genes = nns_index_array(genes, maxk=genes.shape[0]-1)
njobs = 8
d = MetricComparisons(genes, njobs=njobs)

imbalances = np.zeros((len(features_nonselected),2))
for i_feat, feature_B in enumerate(features_nonselected):
    imbalances[i_feat] = d.return_inf_imb_two_selected_coords(coords1=[feature_ref], coords2=[feature_B], k=args.k)

pickle.dump([features_selected, imbalances], open(f"./pickles/imbalances_selected{args.feature_ref}_vs_nonselected_k{args.k}.p","wb"))