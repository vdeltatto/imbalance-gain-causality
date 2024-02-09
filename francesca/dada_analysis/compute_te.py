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
                    help="Index of the subject, from 0 to 18")
parser.add_argument("-channel_X", "--channel_X", dest="channel_X",
                    default="Fz", type=str,
                    help="Channel Y string")
parser.add_argument("-channel_Y", "--channel_Y", dest="channel_Y",
                    default="POz", type=str,
                    help="Channel Y string")
parser.add_argument("-E", "--E", dest="E",
                    default=12, type=int,
                    help="Embedding length")
parser.add_argument("-tau_e", "--tau_e", dest="tau_e",
                    default=1, type=int,
                    help="Embedding time")
parser.add_argument("-k", "--k", dest="k",
                    default=3, type=int,
                    help="Number of neighbors")
parser.add_argument("-t0", "--t0", dest="t0",
                    default=0, type=int,
                    help="Initial time")
parser.add_argument("-minzero", "--minzero", dest="minzero",
                    default=1, type=int,
                    help="Whether TE min of TE is set to zero or not")
args = parser.parse_args()

# subjects names
subjects = (["001","002","003","004","005","006","007","008","009","010",
             "011","012","013","014","015","016","017","018","019"])
subject = subjects[args.i_subject]
durations = [300, 400, 500, 600, 700, 800, 900]

# read data here
electrodes = pd.read_csv("/scratch/vdeltatt/imbalance-gain-causality/EEG_analysis/dada_dataset/sub-001_task-baseline_duration-300.csv")['labels']
channel_X_index = np.where(electrodes==args.channel_X)[0][0]
channel_Y_index = np.where(electrodes==args.channel_Y)[0][0]

with open("/scratch/vdeltatt/imbalance-gain-causality/EEG_analysis/pickles_osf/dataset_offset.p", "rb") as f:
    data = pickle.load(f)

# construct X and Y datasets
X = data[subject,300][:,:,channel_X_index] # initialization with first duration
Y = data[subject,300][:,:,channel_Y_index]

for duration in durations[1:]: # skip first duration (300)
    X = np.concatenate((X, data[subject,duration][:,:,channel_X_index]), axis=0)
    Y = np.concatenate((Y, data[subject,duration][:,:,channel_Y_index]), axis=0)

n_jobs = 4
alphas = np.linspace(0.,1.5,300)
taus = np.arange(0,X.shape[1]-args.t0-args.E)

transfer_entropy_X_to_Y = np.zeros((len(taus),len(alphas)))
transfer_entropy_Y_to_X = np.zeros((len(taus),len(alphas)))

# extract time-delay embeddings at time 0
X0 = X[:,:args.E]
Y0 = Y[:,:args.E]
for i_tau, tau in tqdm(enumerate(taus)):
    # extract time-delay embeddings at time tau
    Xtau = X[:,tau:tau + args.E]
    Ytau = Y[:,tau:tau + args.E]

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