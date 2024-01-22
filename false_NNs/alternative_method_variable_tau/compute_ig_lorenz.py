import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding

seed = int(sys.argv[1])
traj = pickle.load(open(f"/scratch/vdeltatt/imbalance-gain-causality/fig3_errors/trajs/lorenz/seed{seed}_ieps0.p","rb"))
N = 5000
tau = 30
k = 1
n_jobs = 8

# parameters for time-delay embeddings
Es = np.arange(1,50)
tau_e = 1
sample_times = np.linspace(100000,300000,N,dtype=int)

info_imbalances = np.zeros((len(Es), 2))

for iE, E in enumerate(Es):

    X0 = construct_time_delay_embedding(X=traj[:,1], E=E, tau_e=tau_e, sample_times=sample_times)
    Xtau = construct_time_delay_embedding(X=traj[:,1], E=E, tau_e=tau_e, sample_times=sample_times+tau+E) # N.B. tau + E instead of tau
    X_all = np.column_stack((X0, Xtau))

    d = MetricComparisons(X_all, maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances[iE] = d.return_inf_imb_two_selected_coords(coords1=np.arange(E), coords2=np.arange(E,2*E), k=k)

pickle.dump(info_imbalances, open(f"./pickles/lorenz/seed{seed}.p","wb"))