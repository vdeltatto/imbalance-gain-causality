import numpy as np
import pandas as pd
from dadapy.metric_comparisons import MetricComparisons
import pickle
from scipy.io import loadmat
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i_subject", "--i_subject", dest="i_subject",
                    default=0, type=int,
                    help="Index of the subject, from 0 to 18")
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
                    default=20, type=int,
                    help="Number of neighbors")
parser.add_argument("-t0", "--t0", dest="t0",
                    default=0, type=int,
                    help="Initial time")
args = parser.parse_args()

# subjects names
subjects = (["001","002","003","004","005","006","007","008","009","010",
             "011","012","013","014","015","016","017","018","019"])
subject = subjects[args.i_subject]

# read data here
electrodes = pd.read_csv("../dada_dataset/sub-001_task-baseline_duration-300.csv")['labels']
channel_Y_index = np.where(electrodes==args.channel_Y)[0][0]

with open("../pickles_osf/fourth_dataset_onset_dichotomous_test.p", "rb") as f: #before: dataset_onset_dichotomous_test
    data = pickle.load(f)
#assert data[subject,0].shape == data[subject,1].shape, "Something wrong in the data!" #not with second dataset!!
Y = np.row_stack((data[subject,0][:,:,channel_Y_index], data[subject,1][:,:,channel_Y_index]))
#Y /= np.std(Y)
X0 = np.concatenate((np.zeros(data[subject,0].shape[0]), np.ones(data[subject,1].shape[0]))).reshape(-1,1)

n_jobs = 4
alphas = np.linspace(0.,50,100)
taus = np.arange(0,Y.shape[1]-args.t0-args.E,1)

info_imbalances_Y_to_X = np.zeros((len(taus),len(alphas)))
info_imbalances_X_to_Y = np.zeros((len(taus),len(alphas)))

# extract time-delay embeddings at time 0
window_t0 = np.arange(args.t0, args.t0 + args.tau_e*args.E, args.tau_e) #  E time-points between t0 and t0 + (E-1)*tau_e
Y0 = Y[:,window_t0]
for i_tau, tau in tqdm(enumerate(taus)):
    # extract time-delay embeddings at time tau
    Ytau = Y[:,window_t0+tau]

    # scan Information Imbalance as a function of alpha
    d = MetricComparisons(maxk=X0.shape[0]-1, njobs=n_jobs)
    info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=args.k)

# save data
pickle.dump([taus, info_imbalances_X_to_Y], 
            open(f"./pickles/pickles_fourth/II_sub{subject}_Xdichotomous_Y{args.channel_Y}_t0{args.t0}_E{args.E}_tauE{args.tau_e}_k{args.k}.p","wb"))