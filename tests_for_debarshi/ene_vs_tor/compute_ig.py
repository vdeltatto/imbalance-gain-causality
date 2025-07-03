import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1, type=int, help="Label of traj ensemble")
parser.add_argument("-case", "--case", dest="case",
                    type=int, help="Defines target space")
parser.add_argument("-k", "--k", dest="k",
                    default=5, type=int, help="Number of nearest neighbors")
args = parser.parse_args()

if args.case == 0:
    coords = [1,4]
elif args.case == 1:
    coords = [2,5]
elif args.case == 2:
    coords = [0,3]

trajs = np.load(
    f"/scratch/vdeltatt/imbalance-gain-causality/tests_for_debarshi/ene_vs_tor/data/ene_tor.npy"
)

# remove thermalization part (first 5 ps)
trajs = trajs[:,5:]

# set parameters, initialize variables
n_jobs = 8
t0 = (args.seed-1) * 30
taus = np.linspace(0,100,21,dtype=int)
alphas = np.linspace(0,1,300) #np.linspace(0,20,500)
info_imbalances_X_to_Y = np.zeros((len(taus), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(taus), len(alphas)))

# extract X(0) and Y(0) and standardize them
X0 = trajs[:,t0,-3:] # take only tor1, tor2 as space A
Y0 = trajs[:,t0,coords].sum(axis=-1)[:,np.newaxis]
std_X = 2*np.pi
std_Y = Y0.std(ddof=1)
X0 = X0 / (std_X * np.sqrt(3))
Y0 = Y0 / std_Y
Y0 = Y0 - np.min(Y0) # translate to positive values (only for PBC option)
for i_tau, tau in tqdm(enumerate(taus)):
    # extract X(tau) and Y(tau) and standardize them
    Xtau = trajs[:,t0 + tau,-3:] / (std_X * np.sqrt(3))
    Ytau = trajs[:,t0 + tau,coords].sum(axis=-1)[:,np.newaxis] / std_Y
    Ytau = Ytau - np.min(Ytau) # translate to positive values (only for PBC option)

    d = MetricComparisons(maxk=X0.shape[0]-1, n_jobs=n_jobs)
    # X->Y test
    info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, 
        period_cause=1/np.sqrt(3), period_effect=10000, # large period just to avoid PBCs for II / IP /IW feats
        weights=alphas, k=args.k
    )
    # Y->X test
    info_imbalances_Y_to_X[i_tau] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, 
        period_cause=10000, period_effect=1/np.sqrt(3),  # large period just to avoid PBCs for II / IP /IW feats
        weights=alphas, k=args.k
    )

# save data
pickle.dump([taus, alphas, info_imbalances_X_to_Y, info_imbalances_Y_to_X], 
            open(f"./pickles_ig/seed{args.seed}_case{args.case}_k{args.k}.p","wb"))