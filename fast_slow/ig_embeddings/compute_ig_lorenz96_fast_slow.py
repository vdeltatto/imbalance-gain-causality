import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding

seed = int(sys.argv[1])
E = 3
tau_e = 1
sample_traj = pickle.load(open(f"../trajs/lorenz96/seed{seed}_itau0.p","rb"))
traj_length = sample_traj.shape[0] - tau_e*E
N = 5000
D = 40
tau = 30
taus_eq = np.logspace(-1,1,21)
k = 20
n_jobs = 8
alphas = np.linspace(0.,1.5,50)

# parameters for time-delay embeddings
E = 30
tau_e = 1
sample_times = np.linspace(100000,352000,N,dtype=int)

info_imbalances_X_to_Y = np.zeros((len(taus_eq), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(taus_eq), len(alphas)))

for itau_eq, tau_eq in enumerate(taus_eq):
    trajectory = pickle.load(open(f"../trajs/lorenz96/seed{seed}_itau{itau_eq}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    # construct time-delay embeddings
    X_time_delay = construct_time_delay_embedding(X=trajectory[:,1], E=E, tau_e=tau_e)
    Y_time_delay = construct_time_delay_embedding(X=trajectory[:,D+1], E=E, tau_e=tau_e)

    X0 = X_time_delay[sample_times]
    Y0 = Y_time_delay[sample_times]
    Xtau = X_time_delay[sample_times+tau]
    Ytau = Y_time_delay[sample_times+tau]

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_Y[itau_eq] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    info_imbalances_Y_to_X[itau_eq] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/lorenz96/seed{seed}.p","wb"))