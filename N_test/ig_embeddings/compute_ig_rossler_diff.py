import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding

seed = int(sys.argv[1])

ieps = 12
epsilons = np.linspace(0.,0.25,30)
print(f"Epsilon X->Y is {epsilons[ieps]:.3f}")
trajectory = pickle.load(open(f"../../fig3_errors/trajs/rossler_diff/seed{seed}_ieps{ieps}.p","rb"))
E = 3
tau_e = 1
traj_length = trajectory.shape[0] - tau_e*E
tau = 20
k = 1
n_jobs = 8
alphas = np.linspace(0.,0.25,50)


Ns = np.append([100, 200], np.arange(500,5500,500))

info_imbalances_X_to_Y = np.zeros((len(Ns), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(Ns), len(alphas)))

for iN, N in enumerate(Ns):

    sample_times = np.linspace(100000,traj_length-tau-1,N,dtype=int)

    X_time_delay = construct_time_delay_embedding(X=trajectory[:,1], E=E, tau_e=tau_e)
    Y_time_delay = construct_time_delay_embedding(X=trajectory[:,4], E=E, tau_e=tau_e)
    X0 = X_time_delay[sample_times]
    Y0 = Y_time_delay[sample_times]
    Xtau = X_time_delay[sample_times+tau]
    Ytau = Y_time_delay[sample_times+tau]

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_Y[iN] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    info_imbalances_Y_to_X[iN] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/rossler_diff/seed{seed}.p","wb"))