import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding

seed = int(sys.argv[1])
traj = pickle.load(open(f"/scratch/vdeltatt/imbalance-gain-causality/fig3_errors/trajs/lorenz96/seed{seed}_ieps0.p","rb"))
N = 5000
D = 40
k = 20
n_jobs = 8
Es = np.arange(1,100)

# parameters for time-delay embeddings
E = 30
tau_e = 1
sample_times = np.linspace(100000,352000,N,dtype=int)

info_imbalances = np.zeros((len(Es),2))
X_time_delay = construct_time_delay_embedding(X=traj[:,1], E=int(Es[-1]+1), tau_e=tau_e, sample_times=sample_times)

for iE, E in enumerate(Es):

    d = MetricComparisons(X_time_delay, maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances[iE] = d.return_inf_imb_two_selected_coords(coords1=np.arange(E), coords2=np.arange(E+1), k=k)

pickle.dump(info_imbalances, open(f"./pickles/lorenz96/seed{seed}.p","wb"))