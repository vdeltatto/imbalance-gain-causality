import numpy as np
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding
from comparison_methods import compute_cross_mapping

seed = int(sys.argv[1])
epsilons = np.linspace(0.,0.25,30)
sample_traj = pickle.load(open(f"../../trajs/rossler_id/seed0_ieps0.p","rb"))
N = 5000

# parameters for time-delay embeddings
E = 3
tau_e = 5
traj_length = sample_traj.shape[0] - tau_e*E
Ls = np.linspace(100,30000,50,dtype=int)

ccm_X_to_Y = np.zeros((len(epsilons),len(Ls)))
ccm_Y_to_X = np.zeros((len(epsilons),len(Ls)))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../trajs/rossler_id/seed{seed}_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X_time_delay, Y_time_delay = construct_time_delay_embedding(X=trajectory[:,1], Y=trajectory[:,4], E=E, tau_e=tau_e)

    for iL, L in enumerate(Ls):
        sample_times = np.linspace(100000,204000,L,dtype=int)
        ccm_X_to_Y[ieps,iL], ccm_Y_to_X[ieps,iL] = compute_cross_mapping(X_time_delay[sample_times], 
                                                                         Y_time_delay[sample_times])

pickle.dump([ccm_X_to_Y, ccm_Y_to_X], open(f"./pickles/rossler_id/seed{seed}.p","wb"))