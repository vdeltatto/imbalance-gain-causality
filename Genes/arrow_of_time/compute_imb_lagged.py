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

def moving_average_allgenes(a, window_length):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[window_length:,:] = ret[window_length:,:] - ret[:-window_length,:]
    return ret[window_length - 1:,:] / window_length

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", dest="k",
                    default=5, type=int,
                    help="Number of neighbors to compute Information Imbalance")
parser.add_argument("-window_length", "--window_length", dest="window_length",
                    default=50, type=int,
                    help="Length of window used for moving average")
parser.add_argument("-tau_max", "--tau_max", dest="tau_max",
                    default=100, type=int,
                    help="Lag for arrow of time test")
args = parser.parse_args()

# read data
genes_df = pd.read_csv("../data/gene.yyy")
times_df = pd.read_csv("../data/time.yyy")

# sort pseudotimes
times_df = times_df.sort_values(by=['Calcium Signaling Pseudotime'])

# Reorder points in gene dataset according to labels in time dataset (re-ordered according to pseudotime)
labels_points = times_df['Sample'].values.astype(str)
genes_df = genes_df[np.append(['Unnamed: 0'],labels_points)]

# extract numpy arrays from pandas dataframes
times = times_df['Calcium Signaling Pseudotime'].to_numpy()
genes = (genes_df.T).iloc[1:,:].to_numpy(dtype=float)

# add noise to break neighbor degeneracies (both in gene and pseudotime spaces)
times += np.random.normal(loc=0., scale=1e-6, size=times.shape)
genes += np.random.normal(loc=0., scale=1e-6, size=genes.shape)

# compute moving average
genes_moving_avg = moving_average_allgenes(genes, window_length=args.window_length)

# read best 25 features (to predict full gene space) from previous analysis
features_25, _ = pickle.load(open("../predict_genes/pickles/matrix_imbalances_ncoords30_k30.p","rb"))
genes_moving_avg = +genes_moving_avg[:,features_25]

# select times
times = np.arange(args.tau_max+1,genes_moving_avg.shape[0]-args.tau_max-1,dtype=int) # completely neglect correlations
X0 = genes_moving_avg[times]

# compute lagged info imbalance
njobs = 8
taus = np.arange(-args.tau_max,args.tau_max)
imbalances = np.zeros((len(taus),2))
for i_tau, tau in enumerate(taus):

    Xtau = genes_moving_avg[times+tau]
    d = MetricComparisons(np.column_stack((X0,Xtau)), njobs=njobs, maxk=times.shape[0]-1)
    imbalances[i_tau] = d.return_inf_imb_two_selected_coords(coords1=np.arange(len(features_25)), 
                                                             coords2=np.arange(len(features_25),2*len(features_25)), 
                                                             k=args.k)

with open(f"./pickles/best25feats_k{args.k}_window{args.window_length}_taumax{args.tau_max}.p","wb") as load:
    pickle.dump([features_25, taus, imbalances], load)
