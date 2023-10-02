import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
#sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding, compute_rank_matrix
from imbalance_gain import scan_alphas


k = int(sys.argv[1])
epsilons = np.linspace(0.,0.3,31)
sample_traj = pickle.load(open(f"../../../fig3_errors/trajs/lorenz96/seed0_ieps0.p","rb"))
N = 5000
D = 40
tau = 30
n_jobs = 8
alphas = np.linspace(0.,1.5,50)

# parameters for time-delay embeddings
E = 30
tau_e = 1
sample_times = np.linspace(100000,352000,N,dtype=int)

info_imbalances_X_to_Y = np.zeros((len(epsilons), len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(epsilons), len(alphas)))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../../fig3_errors/trajs/lorenz96/seed0_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X_time_delay, Y_time_delay = construct_time_delay_embedding(X=trajectory[:,1], Y=trajectory[:,D+1], E=E, tau_e=tau_e)
    X0 = X_time_delay[sample_times]
    Y0 = Y_time_delay[sample_times]
    Xtau = X_time_delay[sample_times+tau]
    Ytau = Y_time_delay[sample_times+tau]

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