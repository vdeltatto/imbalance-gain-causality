import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import knncmi
from pyitlib import discrete_random_variable as drv


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
variable_names = ['x_0','y_0','x_tau','y_tau']

seeds = np.arange(ntrajs)
TEs_X_to_Y = np.zeros((ntrajs, len(taus)))
TEs_Y_to_X = np.zeros((ntrajs, len(taus)))

for seed in tqdm(seeds):
    traj = markov_chain(nsteps=nsteps+initial_steps, seed=seed)[initial_steps+1:]

    # extract X(0) and Y(0)
    X0 = traj[sample_times]
    Y0 = convert_traj(traj[sample_times])
    
    for i_tau, tau in enumerate(taus):
        # extract X(tau) and Y(tau)
        Xtau = traj[sample_times+tau]
        Ytau = convert_traj(traj[sample_times+tau])

        # compute (discrete) transfer entropy in both directions
        TEs_X_to_Y[seed, i_tau] = (
            drv.information_mutual_conditional(X=X0, Y=Ytau, Z=Y0, base=np.exp(1))
        )
        TEs_Y_to_X[seed, i_tau] = (
        drv.information_mutual_conditional(X=Y0, Y=Xtau, Z=X0, base=np.exp(1))
        )

# save data
pickle.dump([taus, TEs_X_to_Y, TEs_Y_to_X], open(f"./pickles_te/allseeds.p","wb"))