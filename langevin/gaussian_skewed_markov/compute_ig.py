import numpy as np
import pandas as pd
from dadapy.metric_comparisons import MetricComparisons
import pickle
from scipy.io import loadmat
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                    default=8005000, type=int, help="Number of steps")
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1998, type=int, help="Random seed")
parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                    default=1, type=int, help="Sampling time")
args = parser.parse_args()

with open(f"./pickles_traj/seed{args.seed}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}.p","rb") as f:
    traj = pickle.load(f)

# remove thermalization part
initial_steps = 2000
traj = traj[initial_steps:]
traj += np.random.normal(size=traj.shape)
#traj = np.clip(traj, 0, 99)

# set parameters, initialize variables
n_jobs = 8
k = 50
taus = np.arange(5,505,5,dtype=int)
alphas = np.linspace(0,1,100)
samples = np.linspace(0,8005000-initial_steps-taus[-1]-1,2000,dtype=int)

info_imbalances_X_to_Y = np.zeros((len(taus),len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(taus),len(alphas)))

# extract X(0) and Y(0)
X0 = traj[samples,0].reshape((-1,1))
Y0 = traj[samples,1].reshape((-1,1))
for i_tau, tau in tqdm(enumerate(taus)):
    # extract X(tau) and Y(tau)
    Xtau = traj[samples+tau,0].reshape((-1,1))
    Ytau = traj[samples+tau,1].reshape((-1,1))

    # X->Y test
    d = MetricComparisons(maxk=X0.shape[0]-1, n_jobs=n_jobs)
    info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    
    # Y->X test
    info_imbalances_Y_to_X[i_tau] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

# save data
pickle.dump([taus, info_imbalances_X_to_Y, info_imbalances_Y_to_X], 
            open(f"./pickles_imb/seed{args.seed}_k{k}.p","wb"))