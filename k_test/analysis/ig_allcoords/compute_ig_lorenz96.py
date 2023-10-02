import numpy as np
from dadapy.metric_comparisons import MetricComparisons
from imbalance_gain import scan_alphas
from utilities import compute_rank_matrix
import pickle
import sys

k = int(sys.argv[1])
epsilons = np.linspace(0.,1.5,31)
sample_traj = pickle.load(open(f"../../../fig3_errors/trajs/lorenz96/seed0_ieps0.p","rb"))
traj_length = sample_traj.shape[0]
N = 5000
D = 40 # dimensionality of each system
tau = 30
sample_times = np.linspace(100000,352000,N,dtype=int)
n_jobs = 8
alphas = np.linspace(0.,1.5,50)

info_imbalances_X_to_Y = np.zeros((len(epsilons), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(epsilons), len(alphas)))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../../fig3_errors/trajs/lorenz96/seed0_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X0 = trajectory[sample_times,1:D+1]
    Y0 = trajectory[sample_times,D+1:]
    Xtau = trajectory[sample_times+tau,1:D+1]
    Ytau = trajectory[sample_times+tau,D+1:]

    #d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    #info_imbalances_X_to_Y[ieps] = d.return_inf_imb_causality(
    #    cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    #info_imbalances_Y_to_X[ieps] = d.return_inf_imb_causality(
    #    cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

    rank_matrix_Ytau = compute_rank_matrix(Ytau)
    info_imbalances_X_to_Y[ieps] = scan_alphas(
        cause_present=X0, effect_present=Y0, rank_matrix_effect_future=rank_matrix_Ytau, 
        alphas=alphas, k=k, n_jobs=n_jobs
    )

    rank_matrix_Xtau = compute_rank_matrix(Xtau)
    info_imbalances_Y_to_X[ieps] = scan_alphas(
        cause_present=Y0, effect_present=X0, rank_matrix_effect_future=rank_matrix_Xtau, 
        alphas=alphas, k=k, n_jobs=n_jobs
    )

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/lorenz96/k{k}.p","wb"))