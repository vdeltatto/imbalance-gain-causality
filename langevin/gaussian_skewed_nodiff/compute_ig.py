import numpy as np
import pandas as pd
from dadapy.metric_comparisons import MetricComparisons
import pickle
from scipy.io import loadmat
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-dt", "--dt", dest="dt",
                    default=0.01, type=float, help="Timestep of integrator")
parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                    default=2000, type=int, help="Number of steps")
parser.add_argument("-temperature", "--temperature", dest="temperature",
                    default=1, type=float, help="Temperature")
parser.add_argument("-friction_x", "--friction_x", dest="friction_x",
                    default=1.0, type=float, help="Friction of variable X")
parser.add_argument("-friction_y", "--friction_y", dest="friction_y",
                    default=1.0, type=float, help="Friction of variable Y")
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1998, type=int, help="Random seed")
parser.add_argument("-N", "--N", dest="N",
                    default=2500, type=int, help="Number of trajectories")
parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                    default=1, type=int, help="Sampling time")
args = parser.parse_args()

with open(f"./pickles_traj/seed{args.seed}_N{args.N}_dt{args.dt}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}_frictx{args.friction_x}_fricty{args.friction_y}_temp{args.temperature}.p","rb") as f:
    traj = pickle.load(f)

# remove thermalization part
traj = traj[1000:]

# set parameters, initialize variables
n_jobs = 8
k = 25
taus = np.arange(5,215,10,dtype=int)
alphas = np.linspace(0,1,300)

info_imbalances_X_to_Y = np.zeros((len(taus),len(alphas)))
info_imbalances_Y_to_X = np.zeros((len(taus),len(alphas)))

# extract X(0) and Y(0)
X0 = traj[0,:,0].reshape((-1,1))
Y0 = traj[0,:,1].reshape((-1,1))
for i_tau, tau in tqdm(enumerate(taus)):
    # extract X(tau) and Y(tau)
    Xtau = traj[tau,:,0].reshape((-1,1))
    Ytau = traj[tau,:,1].reshape((-1,1))

    # X->Y test
    d = MetricComparisons(maxk=X0.shape[0]-1, n_jobs=n_jobs)
    info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
    
    # Y->X test
    info_imbalances_Y_to_X[i_tau] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

# save data
pickle.dump([taus, info_imbalances_X_to_Y, info_imbalances_Y_to_X], 
            open(f"./pickles_imb/seed{args.seed}_N{args.N}_k{k}.p","wb"))