import numpy as np
import pandas as pd
from dadapy.metric_comparisons import MetricComparisons
import pickle
from scipy.io import loadmat
import argparse
from tqdm import tqdm
import knncmi
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i_subject", "--i_subject", dest="i_subject",
                    default=0, type=int,
                    help="Index of the subject, from 0 to 20")
parser.add_argument("-channel_X", "--channel_X", dest="channel_X",
                    default="P1", type=str,
                    help="Channel X string")
parser.add_argument("-channel_Y", "--channel_Y", dest="channel_Y",
                    default="FC1", type=str,
                    help="Channel Y string")
parser.add_argument("-E", "--E", dest="E",
                    default=20, type=int,
                    help="Embedding length")
parser.add_argument("-tau_e", "--tau_e", dest="tau_e",
                    default=1, type=int,
                    help="Embedding time")
parser.add_argument("-k", "--k", dest="k",
                    default=3, type=int,
                    help="Number of neighbors")
parser.add_argument("-t0", "--t0", dest="t0",
                    default=100, type=int,
                    help="Initial time")
parser.add_argument("-minzero", "--minzero", dest="minzero",
                        default=1, type=int,
                        help="Whether TE min of TE is set to zero or not")
args = parser.parse_args()

# subjects names
subjects = (['03CM','04ML','05NK','06AT','07YC','08RD','09NT','10EM','11AD','12AD',
             '13AB','14LS','15AC','16AG','17SM','18NB','19LS','20FC','21AB','22RT','23ST'])
subject = subjects[args.i_subject]

# read data here
electrodes = pd.read_csv("./TiBET_dataset/electrodes.csv", header=None).to_numpy(dtype=str).flatten()
channel_X_index = np.where(electrodes==args.channel_X)[0][0]
channel_Y_index = np.where(electrodes==args.channel_Y)[0][0]

X = loadmat(f"./TiBET_dataset/{subject}_filled_offset.mat")['mat'][channel_X_index].T
Y = loadmat(f"./TiBET_dataset/{subject}_filled_offset.mat")['mat'][channel_Y_index].T
assert X.shape == Y.shape, f"Error: shapes of X ({X.shape}) and Y ({Y.shape}) do not match!"

n_jobs = 4
alphas = np.linspace(0.,1.5,300)
taus = np.arange(0,X.shape[1]-args.t0-args.E,1)

transfer_entropy_X_to_Y = np.zeros(len(taus))
transfer_entropy_Y_to_X = np.zeros(len(taus))

# extract time-delay embeddings at time 0
window_t0 = np.arange(args.t0, args.t0 + args.tau_e*args.E, args.tau_e) #  E time-points between t0 and t0 + (E-1)*tau_e
X0 = X[:,window_t0]
Y0 = Y[:,window_t0]
for i_tau, tau in tqdm(enumerate(taus)):
    # extract time-delay embeddings at time tau
    Xtau = X[:,window_t0+tau]
    Ytau = Y[:,window_t0+tau]

    dataset = pd.DataFrame(np.column_stack((X0,Y0,Xtau,Ytau))) #, columns=variable_names)

    # compute transfer entropy in both directions
    transfer_entropy_X_to_Y[i_tau] = (
        knncmi.cmi(list(np.arange(args.E)), list(np.arange(3*args.E,4*args.E)), list(np.arange(args.E,2*args.E)), k=args.k, data=dataset, minzero=args.minzero)
    )
    transfer_entropy_Y_to_X[i_tau] = (
        knncmi.cmi(list(np.arange(args.E,2*args.E)), list(np.arange(2*args.E,3*args.E)), list(np.arange(args.E)), k=args.k, data=dataset, minzero=args.minzero)
    )

# save data
pickle.dump([taus, transfer_entropy_X_to_Y, transfer_entropy_Y_to_X], 
            open(f"./pickles/TE_sub{subject}_X{args.channel_X}_Y{args.channel_Y}_t0{args.t0}_E{args.E}_tauE{args.tau_e}_k{args.k}.p","wb"))