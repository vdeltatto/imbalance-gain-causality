import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import knncmi
from pyitlib import discrete_random_variable as drv


parser = argparse.ArgumentParser()
parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                    default=1000000, type=int, help="Number of steps")
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1998, type=int, help="Random seed")
parser.add_argument("-k", "--k", dest="k",
                    default=7, type=int, help="Number of neighbors")
parser.add_argument("-minzero", "--minzero", dest="minzero",
                    default=True, type=bool, help="minzero argument")
parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                    default=1, type=int, help="Samling stride")
args = parser.parse_args()

with open(f"./pickles_traj/seed{args.seed}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}.p","rb") as f:
    traj = pickle.load(f)

# remove thermalization part
initial_steps = 20000
traj = traj[initial_steps:]

# set parameters, initialize variables
taus = np.arange(5,505,5,dtype=int)
variable_names = ['x_0', 'y_0', 'x_tau', 'y_tau']
samples = np.linspace(0,8005000-initial_steps-taus[-1]-1,2000,dtype=int)

TEs_X_to_Y = np.zeros((len(taus)))
TEs_Y_to_X = np.zeros((len(taus)))
TEs_X_to_Y_discrete = np.zeros((len(taus)))
TEs_Y_to_X_discrete = np.zeros((len(taus)))

# extract X(0) and Y(0)
X0 = traj[samples,0]
Y0 = traj[samples,1]
for i_tau, tau in tqdm(enumerate(taus)):
    # extract X(tau) and Y(tau)
    Xtau = traj[samples+tau,0]
    Ytau = traj[samples+tau,1]

    ## compute (continuous) transfer entropy in both directions
    dataset = pd.DataFrame(np.column_stack((X0,Y0,Xtau,Ytau)), columns=variable_names)
    TEs_X_to_Y[i_tau] = (
        knncmi.cmi(['x_0'], ['y_tau'], ['y_0'], k=args.k, data=dataset, minzero=args.minzero)
    )
    TEs_Y_to_X[i_tau] = (
        knncmi.cmi(['y_0'], ['x_tau'], ['x_0'], k=args.k, data=dataset, minzero=args.minzero)
    )

    # compute (discrete) transfer entropy in both directions
    #TEs_X_to_Y_discrete[i_tau] = (
    #    drv.information_mutual_conditional(X=X0,Y=Ytau,Z=Y0, base=np.exp(1))
    #)
    #TEs_Y_to_X_discrete[i_tau] = (
    #   drv.information_mutual_conditional(X=Y0,Y=Xtau,Z=X0, base=np.exp(1))
    #)

# save data
pickle.dump([taus, TEs_X_to_Y, TEs_Y_to_X], 
            open(f"./pickles_te/seed{args.seed}_k{args.k}.p","wb"))