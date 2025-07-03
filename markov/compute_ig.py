import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from dadapy import MetricComparisons
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed",
                    default=0, type=int, help="Random seed")
args = parser.parse_args()


def markov_chain(nsteps=1000, seed=0):
    np.random.seed(seed)

    states = [0,1,2]
    transition_matrix = np.array([[0.0, 0.5, 0.5],
                                  [0.5, 0.4, 0.1],
                                  [0.5, 0.1, 0.4]])
    traj = np.zeros(nsteps+1, dtype=int)
    traj[0] = np.random.choice(states)

    for istep in range(1,nsteps+1):
        traj[istep] = np.random.choice(states, p=transition_matrix[traj[istep-1]])

    return traj

def convert_traj(traj, to="y"):
    if to == "x":
        return traj
    if to == "y":
        traj[traj==1] = 0
        traj[traj==2] = 1
        return traj

taus = np.arange(1,11)
nsteps = 100000
initial_steps = 100
ntrajs = 20
seeds = np.arange(ntrajs)
N = 2500
sample_times = np.linspace(initial_steps, nsteps-taus[-1]-1, N, dtype=int)
n_jobs = 8
alphas = np.linspace(0,20,500)
k = 25

seeds = np.arange(ntrajs)
imbs_X_to_Y = np.zeros((len(taus), len(alphas)))
imbs_Y_to_X = np.zeros((len(taus), len(alphas)))

traj = markov_chain(nsteps=nsteps+initial_steps, seed=args.seed)[initial_steps+1:]

# extract X(0) and Y(0)
X0 = traj[sample_times].reshape((-1,1)) + 0.01*np.random.normal(size=(len(sample_times),1))
Y0 = convert_traj(traj[sample_times]).reshape((-1,1)) + 0.01*np.random.normal(size=(len(sample_times),1))

for i_tau, tau in enumerate(taus):
    # extract X(tau) and Y(tau)
    Xtau = traj[sample_times+tau].reshape((-1,1))  + 0.01*np.random.normal(size=(len(sample_times),1))
    Ytau = convert_traj(traj[sample_times+tau]).reshape((-1,1))  + 0.01*np.random.normal(size=(len(sample_times),1))

    d = MetricComparisons(maxk=X0.shape[0]-1, n_jobs=n_jobs)
    imbs_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)

    # Y->X test
    imbs_Y_to_X[i_tau] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

# save data
pickle.dump([taus, alphas, imbs_X_to_Y, imbs_Y_to_X], open(f"./pickles_ig/seed{args.seed}_k{k}.p","wb"))