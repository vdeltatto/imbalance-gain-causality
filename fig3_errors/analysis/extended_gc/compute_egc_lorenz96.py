import numpy as np
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding
from comparison_methods import compute_extended_granger_index

seed = int(sys.argv[1])
epsilons = np.linspace(0.,0.3,31)
sample_traj = pickle.load(open(f"../../trajs/lorenz96/seed0_ieps0.p","rb"))
N = 5000
D = 40

# parameters for time-delay embeddings
E = int(30+1)
tau_e = 7
traj_length = sample_traj.shape[0] - tau_e*E
sample_times = np.linspace(100000,352000,N,dtype=int)

egc_index_X_to_Y = np.zeros(len(epsilons))
egc_index_Y_to_X = np.zeros(len(epsilons))

for ieps, eps in enumerate(epsilons):
    trajectory = pickle.load(open(f"../../trajs/lorenz96/seed{seed}_ieps{ieps}.p","rb"))
    assert trajectory.shape == sample_traj.shape

    X_time_delay, Y_time_delay = construct_time_delay_embedding(X=trajectory[:,1], Y=trajectory[:,D+1], E=E, tau_e=tau_e)
    egc_index_X_to_Y[ieps], egc_index_Y_to_X[ieps] = compute_extended_granger_index(X_time_delay[sample_times], 
                                                                                    Y_time_delay[sample_times], 
                                                                                    n_neighborhoods=200, 
                                                                                    ks=[500], 
                                                                                    seed=1998)

pickle.dump([egc_index_X_to_Y, egc_index_Y_to_X], open(f"./pickles/lorenz96/seed{seed}.p","wb"))