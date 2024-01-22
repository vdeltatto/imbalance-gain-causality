import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dadapy.metric_comparisons import MetricComparisons
from scipy.stats import rankdata
import pickle
from scipy.spatial.distance import pdist, squareform
import argparse
from itertools import combinations
from tqdm import tqdm

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
parser.add_argument("-i_parallel", "--i_parallel", dest="i_parallel",
                    default=0, type=int,
                    help="Number of parallel job (from 0 to 14)")
args = parser.parse_args()

features, _ = pickle.load(open(f"./pickles/nbest5_ncoords30_k{args.k}.p","rb"))
all_2plets = list(combinations(features[24], 2))

pairs_2plets = []
for i in range(len(all_2plets)):
    for j in range(i+1,len(all_2plets)):
        intersection = set(all_2plets[i]).intersection(all_2plets[j])
        if len(intersection) == 0:
            pairs_2plets.append([all_2plets[i], all_2plets[j]])
pairs_2plets = np.array(pairs_2plets)
print(pairs_2plets.shape)

genes_df = pd.read_csv("../data/gene.yyy")

# extract numpy arrays from pandas dataframes
genes = (genes_df.T).iloc[1:,:].to_numpy(dtype=float)

# add noise to break neighbor degeneracies
genes += np.random.normal(loc=0., scale=1e-6, size=genes.shape)

# compute imbalance for all the pairs
njobs = 8
d = MetricComparisons(genes, njobs=njobs, maxk=genes.shape[0]-1)

imbalances = np.zeros((pairs_2plets.shape[0] // 15,2))

start_pair_index = pairs_2plets.shape[0] // 15 * args.i_parallel
last_pair_index = pairs_2plets.shape[0] // 15 * (args.i_parallel + 1)
for i_imb, i_pair  in tqdm(enumerate(range(start_pair_index,last_pair_index))):
    imbalances[i_imb] = d.return_inf_imb_two_selected_coords(coords1=pairs_2plets[i_pair][0], coords2=pairs_2plets[i_pair][1], k=args.k)

pickle.dump([pairs_2plets, imbalances], open(f"./pickles/imbalances_pairs_2plets_ncoords25_k{args.k}_iparallel{args.i_parallel}.p","wb"))