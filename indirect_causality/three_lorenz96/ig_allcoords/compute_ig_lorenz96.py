import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys

ieps = int(sys.argv[1])
trajectory = pickle.load(open(f"../trajs/lorenz96/ieps{ieps}.p","rb"))
traj_length = trajectory.shape[0]
N = 5000
D = 40 # dimensionality of each system
tau = 30
k = 20
sample_times = np.linspace(5000,299000,N,dtype=int)
n_jobs = 8
alphas = np.linspace(0.,1.5,200)

# X and Y
#X0 = trajectory[sample_times,1:D+1]
#Y0 = trajectory[sample_times,D+1:2*D+1]
#Xtau = trajectory[sample_times+tau,1:D+1]
#Ytau = trajectory[sample_times+tau,D+1:2*D+1]
#
#d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
#info_imbalances_X_to_Y = d.return_inf_imb_causality(
#    cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
#info_imbalances_Y_to_X = d.return_inf_imb_causality(
#    cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)
#
#pickle.dump([info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles/lorenz96/XY_ieps{ieps}.p","wb"))

# Z and Y
Z0 = trajectory[sample_times,2*D+1:]
Y0 = trajectory[sample_times,D+1:2*D+1]
Ztau = trajectory[sample_times+tau,2*D+1:]
Ytau = trajectory[sample_times+tau,D+1:2*D+1]

d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
info_imbalances_Z_to_Y = d.return_inf_imb_causality(
    cause_present=Z0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
info_imbalances_Y_to_Z = d.return_inf_imb_causality(
    cause_present=Y0, effect_present=Z0, effect_future=Ztau, weights=alphas, k=k)

pickle.dump([info_imbalances_Z_to_Y, info_imbalances_Y_to_Z], open(f"./pickles/lorenz96/ZY_ieps{ieps}.p","wb"))
