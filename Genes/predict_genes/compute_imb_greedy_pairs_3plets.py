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
                    help="Number of parallel job (from 0 to 9)")
args = parser.parse_args()

# read best 25 features and construct all pairs (25*24/2)
features_single, _ = pickle.load(open("./pickles/matrix_imbalances_ncoords30_k30.p","rb"))
all_2plets = np.array(list(combinations(features_single, 2)))

# read data for pairs of 2plets
imbs = np.zeros((15,2530,2))
for i_parallel in range(15):
    features_2plet, imbs[i_parallel] = pickle.load(open(f"./pickles/imbalances_pairs_2plets_ncoords25_k30_iparallel{i_parallel}.p","rb"))
imbs = imbs.reshape((37950,2))

# select indices of the 10 2plets pairs that have the most asymmetric information imbalance
indices_largest_diff = np.argsort(np.abs(imbs[:,0] - imbs[:,1]))[-10:] 

# for each of the 10 pairs of 2plets selected, add one feature to space A and one to space B (and viceversa)
# (distinct to the ones already present) and compute info imbalance between 3plets: (25-4)*(25-5) for each pair
# of 2plets (tot: 10*21*20)
pairs_3plets = np.zeros((10*21*20, 2, 3), dtype=int)

index_3plet = 0
for index_of_pair_2plet in indices_largest_diff:
    for new_feat_1, new_feat_2 in all_2plets:
        if new_feat_1 not in features_2plet[index_of_pair_2plet] and new_feat_2 not in features_2plet[index_of_pair_2plet]:
            pairs_3plets[index_3plet,0] = np.append(features_2plet[index_of_pair_2plet,0], [new_feat_1])
            pairs_3plets[index_3plet,1] = np.append(features_2plet[index_of_pair_2plet,1], [new_feat_2])
            pairs_3plets[index_3plet+1,0] = np.append(features_2plet[index_of_pair_2plet,0], [new_feat_2])
            pairs_3plets[index_3plet+1,1] = np.append(features_2plet[index_of_pair_2plet,1], [new_feat_1])
            index_3plet += 2

genes_df = pd.read_csv("../data/gene.yyy")

# extract numpy arrays from pandas dataframes
genes = (genes_df.T).iloc[1:,:].to_numpy(dtype=float)

# add noise to break neighbor degeneracies
genes += np.random.normal(loc=0., scale=1e-6, size=genes.shape)

# compute imbalance for all the pairs
njobs = 8
d = MetricComparisons(genes, njobs=njobs, maxk=genes.shape[0]-1)

imbalances = np.zeros((pairs_3plets.shape[0] // 10,2))

start_pair_index = pairs_3plets.shape[0] // 10 * args.i_parallel
last_pair_index = pairs_3plets.shape[0] // 10 * (args.i_parallel + 1)
for i_imb, i_pair  in tqdm(enumerate(range(start_pair_index,last_pair_index))):
    imbalances[i_imb] = d.return_inf_imb_two_selected_coords(coords1=pairs_3plets[i_pair][0], coords2=pairs_3plets[i_pair][1], k=args.k)

pickle.dump([pairs_3plets, imbalances], open(f"./pickles/imbalances_greedy_pairs_3plets_ncoords25_k{args.k}_iparallel{args.i_parallel}.p","wb"))

