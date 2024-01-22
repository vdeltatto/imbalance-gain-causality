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

epsilons = np.linspace(0.,0.3,31)
sample_traj = pickle.load(open(f"../../fig3_errors/trajs/lorenz/seed0_ieps0.p","rb"))
traj_length = sample_traj.shape[0]
N = 5000
tau = 5
sample_times = np.linspace(100000,300000,N,dtype=int)

variable_names = ['x1_0', 'x2_0', 'x3_0', 'y1_0', 'y2_0', 'y3_0', 
                  'x1_tau', 'x2_tau', 'x3_tau', 'y1_tau', 'y2_tau', 'y3_tau']
transfer_entropy_X_to_Y = np.zeros(len(epsilons))
transfer_entropy_Y_to_X = np.zeros(len(epsilons))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../fig3_errors/trajs/lorenz/seed{args.seed}_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X0 = trajectory[sample_times,1:4]
    Y0 = trajectory[sample_times,4:]
    Xtau = trajectory[sample_times+tau,1:4]
    Ytau = trajectory[sample_times+tau,4:]

    dataset = pd.DataFrame(np.column_stack((X0,Y0,Xtau,Ytau)), columns=variable_names)

    # compute transfer entropy in both directions
    transfer_entropy_X_to_Y[ieps] = (
        knncmi.cmi(['x1_0','x2_0','x3_0'], ['y1_tau', 'y2_tau', 'y3_tau'], ['y1_0', 'y2_0', 'y3_0'], k=args.k, data=dataset, minzero=args.minzero)
    )
    transfer_entropy_Y_to_X[ieps] = (
        knncmi.cmi(['y1_0','y2_0','y3_0'], ['x1_tau', 'x2_tau', 'x3_tau'], ['x1_0', 'x2_0', 'x3_0'], k=args.k, data=dataset, minzero=args.minzero)
    )

pickle.dump(
    [transfer_entropy_X_to_Y, transfer_entropy_Y_to_X], 
    open(f"./pickles/lorenz/seed{args.seed}_k{args.k}_minzero{bool(args.minzero)}.p","wb")
)