import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
import knncmi
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed",
                        default=0, type=int,
                        help="random seed of trajectory")
parser.add_argument("-minzero", "--minzero", dest="minzero",
                        default=1, type=int,
                        help="Whether TE min of TE is set to zero or not")
parser.add_argument("-k", "--k", dest="k",
                        default=3, type=int,
                        help="number of NNs used for TE computation")
args = parser.parse_args()

epsilons = np.linspace(0.,1.5,31)
sample_traj = pickle.load(open(f"../../fig3_errors/trajs/lorenz96/seed0_ieps0.p","rb"))
traj_length = sample_traj.shape[0]
N = 5000
D = 40 # dimensionality of each system
tau = 30
sample_times = np.linspace(100000,352000,N,dtype=int)

transfer_entropy_X_to_Y = np.zeros(len(epsilons))
transfer_entropy_Y_to_X = np.zeros(len(epsilons))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../fig3_errors/trajs/lorenz96/seed{args.seed}_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X0 = trajectory[sample_times,1:D+1]
    Y0 = trajectory[sample_times,D+1:]
    Xtau = trajectory[sample_times+tau,1:D+1]
    Ytau = trajectory[sample_times+tau,D+1:]

    del trajectory

    dataset = pd.DataFrame(np.column_stack((X0,Y0,Xtau,Ytau))) #, columns=variable_names)

    # compute transfer entropy in both directions
    transfer_entropy_X_to_Y[ieps] = (
        knncmi.cmi(list(np.arange(D)), list(np.arange(3*D,4*D)), list(np.arange(D,2*D)), k=args.k, data=dataset, minzero=args.minzero)
    )
    transfer_entropy_Y_to_X[ieps] = (
        knncmi.cmi(list(np.arange(D,2*D)), list(np.arange(2*D,3*D)), list(np.arange(D)), k=args.k, data=dataset, minzero=args.minzero)
    )

pickle.dump(
    [transfer_entropy_X_to_Y, transfer_entropy_Y_to_X], 
    open(f"./pickles/lorenz96/seed{args.seed}_k{args.k}_minzero{bool(args.minzero)}.p","wb")
)