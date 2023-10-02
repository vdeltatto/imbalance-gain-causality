import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding

seed = int(sys.argv[1])
np.random.seed(seed=seed)

seed = int(sys.argv[1])
sample_traj = pickle.load(open(f"../../fig3_errors/trajs/lorenz96/seed0_ieps0.p","rb"))
N = 5000
D = 40
tau = 30
k = 20
n_jobs = 8
alphas = np.linspace(0.,1.5,50)

# parameters for time-delay embeddings
Es = np.arange(1,40+1)
tau_e = 1
sample_times = np.linspace(100000,352000,N,dtype=int)

info_imbalances_X_to_Y = np.zeros((len(Es), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(Es), len(alphas)))

for iE, E in enumerate(Es):
    trajectory = pickle.load(open(f"../../fig3_errors/trajs/lorenz96/seed0_ieps15.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X_time_delay, Y_time_delay = construct_time_delay_embedding(X=trajectory[:,1], Y=trajectory[:,D+1], E=E, tau_e=tau_e)
    X0 = X_time_delay[sample_times]
    Y0 = Y_time_delay[sample_times]
    Xtau = X_time_delay[sample_times+tau]
    Ytau = Y_time_delay[sample_times+tau]

    random_indices = np.random.choice(np.arange(N), size=N, replace=False)

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_Y[iE] = d.return_inf_imb_causality(
        cause_present=X0[random_indices], effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    info_imbalances_Y_to_X[iE] = d.return_inf_imb_causality(
        cause_present=Y0[random_indices], effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/lorenz96/seed{seed}.p","wb"))