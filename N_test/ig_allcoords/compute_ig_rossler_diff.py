import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys

seed = int(sys.argv[1])

ieps = 12
epsilons = np.linspace(0.,0.25,30)
print(f"Epsilon X->Y is {epsilons[ieps]:.3f}")
trajectory = pickle.load(open(f"../../fig3_errors/trajs/rossler_diff/seed{seed}_ieps{ieps}.p","rb"))
traj_length = trajectory.shape[0]
tau = 20
k = 1
n_jobs = 8
alphas = np.linspace(0.,0.25,50)

Ns = np.append([100, 200], np.arange(500,5500,500))

info_imbalances_X_to_Y = np.zeros((len(Ns), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(Ns), len(alphas)))

for iN, N in enumerate(Ns):

    sample_times = np.linspace(100000,traj_length-tau-1,N,dtype=int)

    X0 = trajectory[sample_times,1:4]
    Y0 = trajectory[sample_times,4:]
    Xtau = trajectory[sample_times+tau,1:4]
    Ytau = trajectory[sample_times+tau,4:]

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_Y[iN] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    info_imbalances_Y_to_X[iN] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/rossler_diff/seed{seed}.p","wb"))