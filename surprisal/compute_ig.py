import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm
import scienceplots
import argparse
import sys
sys.path.append('/scratch/vdeltatt/imbalance-gain-causality')
from utilities import construct_time_delay_embedding
from dadapy.metric_comparisons import MetricComparisons



parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", dest="k",
                    default=25, type=int,
                    help="Number of neighbors to compute Information Imbalance")
parser.add_argument("-E_X", "--E_X", dest="E_X",
                    default=3, type=int,
                    help="Embedding dimension for X")
parser.add_argument("-E_Y", "--E_Y", dest="E_Y",
                    default=3, type=int,
                    help="Embedding dimension for Y")
parser.add_argument("-tau_e", "--tau_e", dest="tau_e",
                    default=1, type=int,
                    help="Embedding time")
parser.add_argument("-part", "--part", dest="part",
                    default='F', type=str,
                    help="Participant label")
parser.add_argument("-area", "--area", dest="area",
                    default='all', type=str,
                    help="Area of fMRI signal")
args = parser.parse_args()


brain_data_subjects = pickle.load(open("./data/brain_data_part.pkl","rb"))
sfl_gpt2 = pickle.load(open("./data/sfl_gpt2.pkl","rb"))

taus = np.arange(0,20)
alphas = np.linspace(0,1.5,100)
n_jobs = 4

X = brain_data_subjects[args.area][args.part] / np.std(brain_data_subjects[args.area][args.part]) 
#Y = sfl_gpt2[:,0] / np.std(sfl_gpt2[:,0]) # surprisal
#Y = sfl_gpt2[:,1] / np.std(sfl_gpt2[:,1]) # log(freq)
Y = sfl_gpt2[:,2] / np.std(sfl_gpt2[:,2]) # length

X_time_delay = construct_time_delay_embedding(X=X, E=args.E_X, tau_e=args.tau_e)
Y_time_delay = construct_time_delay_embedding(X=Y, E=args.E_Y, tau_e=args.tau_e)

sample_times = np.linspace(0,X_time_delay.shape[0]-np.max(taus)-1,500,dtype=int)
X0 = X_time_delay[sample_times]
Y0 = Y_time_delay[sample_times]

info_imbalances_X_to_Y = np.zeros((len(taus),len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(taus),len(alphas)))
for i_tau, tau in tqdm(enumerate(taus)):
    Xtau = X_time_delay[sample_times+tau]
    Ytau = Y_time_delay[sample_times+tau]

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)
    info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=args.k)
    info_imbalances_Y_to_X[i_tau] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=args.k)
    
pickle.dump([alphas, taus, info_imbalances_X_to_Y, info_imbalances_Y_to_X], 
            open(f"./pickles_l/part{args.part}_area{args.area}_EX{args.E_X}_EY{args.E_Y}_taue{args.tau_e}_k{args.k}.p","wb"))