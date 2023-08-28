import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys

seed = int(sys.argv[1])
epsilons = np.linspace(0.,0.25,30)
sample_traj = pickle.load(open(f"../../trajs/rossler_diff/seed0_ieps0.p","rb"))
traj_length = sample_traj.shape[0]
N = 5000
tau = 20
k = 1
sample_times = np.linspace(100000,traj_length-tau-1,N,dtype=int)
n_jobs = 8
alphas = np.linspace(0.,0.25,50)

info_imbalances_X_to_Y = np.zeros((len(epsilons), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(epsilons), len(alphas)))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../trajs/rossler_diff/seed{seed}_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X0 = trajectory[sample_times,1:4]
    Y0 = trajectory[sample_times,4:]
    Xtau = trajectory[sample_times+tau,1:4]
    Ytau = trajectory[sample_times+tau,4:]

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_Y[ieps] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    info_imbalances_Y_to_X[ieps] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/rossler_diff/seed{seed}.p","wb"))