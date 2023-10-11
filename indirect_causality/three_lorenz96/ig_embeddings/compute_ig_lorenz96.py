import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding

ieps = int(sys.argv[1])
trajectory = pickle.load(open(f"../trajs/lorenz96/ieps{ieps}.p","rb"))
N = 5000
D = 40
tau = 30
k = 20
n_jobs = 8
alphas = np.linspace(0.,1.5,200)

# parameters for time-delay embeddings
E = 30
tau_e = 1
sample_times = np.linspace(5000,299000,N,dtype=int)


# X,Y
X_time_delay, Y_time_delay = construct_time_delay_embedding(X=trajectory[:,1], Y=trajectory[:,D+1], E=E, tau_e=tau_e)
X0 = X_time_delay[sample_times]
Y0 = Y_time_delay[sample_times]
Xtau = X_time_delay[sample_times+tau]
Ytau = Y_time_delay[sample_times+tau]

d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
info_imbalances_X_to_Y = d.return_inf_imb_causality(
    cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
info_imbalances_Y_to_X = d.return_inf_imb_causality(
    cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/lorenz96/XY_emb_ieps{ieps}.p","wb"))

# Z,Y
Z_time_delay, Y_time_delay = construct_time_delay_embedding(X=trajectory[:,2*D+1], Y=trajectory[:,D+1], E=E, tau_e=tau_e)
Z0 = Z_time_delay[sample_times]
Y0 = Y_time_delay[sample_times]
Ztau = Z_time_delay[sample_times+tau]
Ytau = Y_time_delay[sample_times+tau]

d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
info_imbalances_Z_to_Y = d.return_inf_imb_causality(
    cause_present=Z0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
info_imbalances_Y_to_Z = d.return_inf_imb_causality(
    cause_present=Y0, effect_present=Z0, effect_future=Ztau, weights=alphas, k=k)

pickle.dump([info_imbalances_Z_to_Y, info_imbalances_Y_to_Z], open(f"./pickles/lorenz96/ZY_emb_ieps{ieps}.p","wb"))
