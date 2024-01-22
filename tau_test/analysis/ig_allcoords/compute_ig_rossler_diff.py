import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tau", "--tau", dest="tau",
                    default=30, type=int,
                    help="Time lag tau")
parser.add_argument("-i_eps", "--i_eps", dest="i_eps",
                    default=14, type=int,
                    help="Epsilon index")
args = parser.parse_args()

epsilons = np.linspace(0.,0.15,30)
sample_traj = pickle.load(open(f"../../../fig3_errors/trajs/rossler_diff/seed0_ieps0.p","rb"))
traj_length = sample_traj.shape[0]
N = 5000
D = 3 # dimensionality of each system
k = 20
sample_times = np.linspace(100000,202500,N,dtype=int)
n_jobs = 8
alphas = np.linspace(0.,0.5,100)

info_imbalances_X_to_Y = np.zeros(len(alphas))
info_imbalances_Y_to_X = np.zeros(len(alphas))

trajectory = pickle.load(open(f"../../../fig3_errors/trajs/rossler_diff/seed0_ieps{args.i_eps}.p","rb"))
assert trajectory.shape == sample_traj.shape

X0 = trajectory[sample_times,1:D+1]
Y0 = trajectory[sample_times,D+1:]
Xtau = trajectory[sample_times+args.tau,1:D+1]
Ytau = trajectory[sample_times+args.tau,D+1:]

d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
info_imbalances_X_to_Y = d.return_inf_imb_causality(
    cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
info_imbalances_Y_to_X = d.return_inf_imb_causality(
    cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/rossler_diff/ieps{args.i_eps}_tau{args.tau}.p","wb"))