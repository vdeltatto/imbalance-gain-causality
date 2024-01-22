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
                    help="Number of parallel job (from 0 to 39)")
args = parser.parse_args()

# read best 25 features
features_25, _ = pickle.load(open("./pickles/matrix_imbalances_ncoords30_k30.p","rb"))

# read all combinations of singlet pairs such that feature A is among the 25 most informative, and fetaure B is
# among the 203 least informative (features_singlets has shape (25*203, 2))
features_singlets = np.zeros((len(features_25),203,2))
imbalances = np.zeros((len(features_25),203,2))
for ifeat, feature_ref in enumerate(features_25):
    features_singlets[ifeat], imbalances[ifeat] = pickle.load(open(f"./pickles/imbalances_selected{ifeat}_vs_nonselected_k30.p","rb"))
features_singlets = features_singlets.reshape((len(features_25)*203,2))
imbalances = imbalances.reshape((len(features_25)*203,2))

# select indices of the 20 singlets pairs that have the most asymmetric information imbalance
indices_largest_diff = np.argsort(np.abs(imbalances[:,0] - imbalances[:,1]))[-20:] 
print(indices_largest_diff)


# for each of the 20 pairs of 2plets selected, add one informative feature to space A and one noninformative to space B 
# (distinct to the ones already present) and compute info imbalance between 3plets (25-1)*(203-1) 3plets
pairs_2plets = np.zeros((20*(25-1)*(203-1), 2, 2), dtype=int)

index_2plet = 0
for index_singlet in indices_largest_diff:
    for new_feat_1, new_feat_2 in features_singlets:
        if new_feat_1 != features_singlets[index_singlet,0] and new_feat_2 != features_singlets[index_singlet,1]:
            pairs_2plets[index_2plet,0] = np.append(features_singlets[index_singlet,0], [new_feat_1])
            pairs_2plets[index_2plet,1] = np.append(features_singlets[index_singlet,1], [new_feat_2])
            index_2plet += 1

genes_df = pd.read_csv("../data/gene.yyy")

# extract numpy arrays from pandas dataframes
genes = (genes_df.T).iloc[1:,:].to_numpy(dtype=float)

# add noise to break neighbor degeneracies
genes += np.random.normal(loc=0., scale=1e-6, size=genes.shape)

# compute imbalance for all the pairs
njobs = 8
d = MetricComparisons(genes, njobs=njobs, maxk=genes.shape[0]-1)

imbalances = np.zeros((pairs_2plets.shape[0] // 40,2))

start_pair_index = pairs_2plets.shape[0] // 40 * args.i_parallel
last_pair_index = pairs_2plets.shape[0] // 40 * (args.i_parallel + 1)
for i_imb, i_pair  in tqdm(enumerate(range(start_pair_index,last_pair_index))):
    imbalances[i_imb] = d.return_inf_imb_two_selected_coords(coords1=pairs_2plets[i_pair][0], coords2=pairs_2plets[i_pair][1], k=args.k)

pickle.dump([pairs_2plets, imbalances], open(f"./pickles/imbalances_greedy_pairs_2plets_infovsnoninfo_ncoords25_k{args.k}_iparallel{args.i_parallel}.p","wb"))

