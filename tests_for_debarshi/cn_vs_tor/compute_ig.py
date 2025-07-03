import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1, type=int, help="Label of traj ensemble")
parser.add_argument("-coord", "--coord", dest="coord",
                    default=0, help="Coordinate of CN feature")
parser.add_argument("-k", "--k", dest="k",
                    default=5, type=int, help="Number of nearest neighbors")
args = parser.parse_args()

if args.coord == "all":
    coord = np.array([0,1,2,3,4])
else:
    coord = [int(args.coord)]

trajs = np.load(
    f"/scratch/vdeltatt/imbalance-gain-causality/tests_for_debarshi/data/1_cn_tor.npy"
)

# remove thermalization part
trajs = trajs[:,5:]

# set parameters, initialize variables
n_jobs = 8
t0 = (args.seed-1) * 30
taus = np.linspace(0,100,21,dtype=int)
alphas = np.linspace(0,20,500)
info_imbalances_X_to_Y = np.zeros((len(taus), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(taus), len(alphas)))

# extract X(0) and Y(0) and standardize them
std_X = 2*np.pi
std_Y = (trajs[:,t0,coord].reshape((-1,len(coord)))).std(axis=0)[0]
X0 = trajs[:,t0,5:7] / std_X # take only tor1, tor2 as space A
Y0 = trajs[:,t0,coord] / std_Y
for i_tau, tau in tqdm(enumerate(taus)):
    # extract X(tau) and Y(tau) and standardize them
    Xtau = trajs[:,t0 + tau,5:7] / std_X
    Ytau = trajs[:,t0 + tau,coord] / std_Y

    d = MetricComparisons(maxk=X0.shape[0]-1, n_jobs=n_jobs)
    # X->Y test
    info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, 
        period_cause=1, period_effect=1000, # large period just to avoid PBCs for CNs
        weights=alphas, k=args.k
    )
    # Y->X test
    info_imbalances_Y_to_X[i_tau] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, 
        period_cause=1000, period_effect=1, # large period just to avoid PBCs for CNs
        weights=alphas, k=args.k
    )

# save data
pickle.dump([taus, alphas, info_imbalances_X_to_Y, info_imbalances_Y_to_X], 
            open(f"./pickles_ig/seed{args.seed}_coord{args.coord}_k{args.k}.p","wb"))