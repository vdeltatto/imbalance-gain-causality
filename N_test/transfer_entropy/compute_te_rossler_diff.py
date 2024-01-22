import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
import knncmi
import pandas as pd
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding
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

ieps = 12
epsilons = np.linspace(0.,0.25,30)
print(f"Epsilon X->Y is {epsilons[ieps]:.3f}")
trajectory = pickle.load(open(f"../../fig3_errors/trajs/rossler_diff/seed{args.seed}_ieps{ieps}.p","rb"))

tau = 20
E = 3
tau_e = 1
traj_length = trajectory.shape[0]
Ns = np.append([100, 200], np.arange(500,5500,500))

variable_names = ['x1_0', 'x2_0', 'x3_0', 'y1_0', 'y2_0', 'y3_0', 
                  'x1_tau', 'x2_tau', 'x3_tau', 'y1_tau', 'y2_tau', 'y3_tau']
transfer_entropy_X_to_Y = np.zeros(len(Ns))
transfer_entropy_Y_to_X = np.zeros(len(Ns))

for iN, N in enumerate(Ns):
    
    sample_times = np.linspace(100000,traj_length- tau_e*E - tau -1, N, dtype=int)

    X_time_delay = construct_time_delay_embedding(X=trajectory[:,1], E=E, tau_e=tau_e)
    Y_time_delay = construct_time_delay_embedding(X=trajectory[:,4], E=E, tau_e=tau_e)    
    X0 = X_time_delay[sample_times]
    Y0 = Y_time_delay[sample_times]
    Xtau = X_time_delay[sample_times+tau]
    Ytau = Y_time_delay[sample_times+tau]

    dataset = pd.DataFrame(np.column_stack((X0,Y0,Xtau,Ytau)), columns=variable_names)

    # compute transfer entropy in both directions
    transfer_entropy_X_to_Y[iN] = (
        knncmi.cmi(['x1_0','x2_0','x3_0'], ['y1_tau', 'y2_tau', 'y3_tau'], ['y1_0', 'y2_0', 'y3_0'], k=args.k, data=dataset, minzero=args.minzero)
    )
    transfer_entropy_Y_to_X[iN] = (
        knncmi.cmi(['y1_0','y2_0','y3_0'], ['x1_tau', 'x2_tau', 'x3_tau'], ['x1_0', 'x2_0', 'x3_0'], k=args.k, data=dataset, minzero=args.minzero)
    )
    
pickle.dump(
    [transfer_entropy_X_to_Y, transfer_entropy_Y_to_X], 
    open(f"./pickles/rossler_diff/seed{args.seed}_k{args.k}_minzero{bool(args.minzero)}.p","wb")
)