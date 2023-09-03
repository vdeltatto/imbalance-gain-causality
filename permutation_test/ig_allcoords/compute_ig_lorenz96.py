import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys

seed = int(sys.argv[1])
np.random.seed(seed=seed)

epsilons = np.linspace(0.,1.5,31)
sample_traj = pickle.load(open(f"../../fig3_errors/trajs/lorenz96/seed0_ieps0.p","rb"))
traj_length = sample_traj.shape[0]
N = 5000
D = 40 # dimensionality of each system
tau = 30
k = 20
sample_times = np.linspace(100000,352000,N,dtype=int)
n_jobs = 8
alphas = np.linspace(0.,1.5,50)

info_imbalances_X_to_Y = np.zeros((len(epsilons), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(epsilons), len(alphas)))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../fig3_errors/trajs/lorenz96/seed0_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X0 = trajectory[sample_times,1:D+1]
    Y0 = trajectory[sample_times,D+1:]
    Xtau = trajectory[sample_times+tau,1:D+1]
    Ytau = trajectory[sample_times+tau,D+1:]

    random_indices = np.random.choice(np.arange(N), size=N, replace=False)

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_Y[ieps] = d.return_inf_imb_causality(
        cause_present=X0[random_indices], effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    info_imbalances_Y_to_X[ieps] = d.return_inf_imb_causality(
        cause_present=Y0[random_indices], effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/lorenz96/seed{seed}.p","wb"))