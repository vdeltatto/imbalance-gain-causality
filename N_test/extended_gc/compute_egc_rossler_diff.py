import numpy as np
import pickle
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding
from comparison_methods import compute_extended_granger_index

seed = int(sys.argv[1])

ieps = 12
epsilons = np.linspace(0.,0.25,30)
print(f"Epsilon X->Y is {epsilons[ieps]:.3f}")
trajectory = pickle.load(open(f"../../fig3_errors/trajs/rossler_diff/seed{seed}_ieps{ieps}.p","rb"))

# parameters for time-delay embeddings
E = int(3+1)
tau_e = 1
traj_length = trajectory.shape[0] - tau_e*E
Ns = np.append([100, 200], np.arange(500,5500,500))

egc_index_X_to_Y = np.zeros(len(Ns))
egc_index_Y_to_X = np.zeros(len(Ns))

for iN, N in enumerate(Ns):

    sample_times = np.linspace(100000,traj_length-tau_e-1,N,dtype=int)

    X_time_delay = construct_time_delay_embedding(X=trajectory[:,1], E=E, tau_e=tau_e)
    Y_time_delay = construct_time_delay_embedding(X=trajectory[:,4], E=E, tau_e=tau_e)
    
    egc_index_X_to_Y[iN], egc_index_Y_to_X[iN] = compute_extended_granger_index(X_time_delay[sample_times], 
                                                                                    Y_time_delay[sample_times], 
                                                                                    n_neighborhoods=int(200/5000*N), 
                                                                                    ks=[int(200/5000*N)], 
                                                                                    seed=1998)

pickle.dump([egc_index_X_to_Y, egc_index_Y_to_X], open(f"./pickles/rossler_diff/seed{seed}.p","wb"))