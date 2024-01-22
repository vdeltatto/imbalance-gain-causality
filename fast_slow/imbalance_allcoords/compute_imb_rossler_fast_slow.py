import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys

seed = int(sys.argv[1])
sample_traj = pickle.load(open(f"../trajs/seed{seed}_itau0.p","rb"))
traj_length = sample_traj.shape[0]
N = 5000
tau = 20
taus_eq = np.logspace(-1,1,21)
k = 1
sample_times = np.linspace(100000,traj_length-200-1,N,dtype=int)
n_jobs = 8

info_imbalances_X_to_X = np.zeros((len(taus_eq)))
info_imbalances_Y_to_Y = np.zeros((len(taus_eq)))

for itau_eq, tau_eq in enumerate(taus_eq):
    trajectory = pickle.load(open(f"../trajs/seed{seed}_itau{itau_eq}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X0 = trajectory[sample_times,1:4]
    Y0 = trajectory[sample_times,4:]
    Xtau = trajectory[sample_times+int(np.round(tau/tau_eq)),1:4]
    Ytau = trajectory[sample_times+tau,4:]

    print(f"Actual rescaled tau itau_eq = {itau_eq}: {tau/tau_eq}")

    # X(0)->X(tau)
    d = MetricComparisons(np.column_stack((X0,Xtau)), maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_X[itau_eq], _ = d.return_inf_imb_two_selected_coords(
        coords1=[0,1,2], coords2=[3,4,5], k=k)
    
    # Y(0)->Y(tau)
    d = MetricComparisons(np.column_stack((Y0,Ytau)), maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_Y_to_Y[itau_eq], _ = d.return_inf_imb_two_selected_coords(
        coords1=[0,1,2], coords2=[3,4,5], k=k)

pickle.dump([info_imbalances_X_to_X, info_imbalances_Y_to_Y], open(f"./pickles/seed{seed}.p","wb"))