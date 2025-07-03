import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import knncmi
from pyitlib import discrete_random_variable as drv


parser = argparse.ArgumentParser()
parser.add_argument("-dt", "--dt", dest="dt",
                    default=0.01, type=float, help="Timestep of integrator")
parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                    default=2000, type=int, help="Number of steps")
parser.add_argument("-temperature", "--temperature", dest="temperature",
                    default=1, type=float, help="Temperature")
parser.add_argument("-friction_x", "--friction_x", dest="friction_x",
                    default=1, type=float, help="Friction of variable X")
parser.add_argument("-friction_y", "--friction_y", dest="friction_y",
                    default=0.1, type=float, help="Friction of variable Y")
parser.add_argument("-friction_z", "--friction_z", dest="friction_z",
                    default=10, type=float, help="Friction of variable Z")
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1998, type=int, help="Random seed")
parser.add_argument("-N", "--N", dest="N",
                    default=2500, type=int, help="Number of trajectories")
parser.add_argument("-k", "--k", dest="k",
                    default=7, type=int, help="Number of neighbors")
parser.add_argument("-minzero", "--minzero", dest="minzero",
                    default=True, type=bool, help="minzero argument")
parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                    default=1, type=int, help="Samling stride")
args = parser.parse_args()

with open(f"./pickles_traj/seed{args.seed}_N{args.N}_dt{args.dt}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}_frictx{args.friction_x}_fricty{args.friction_y}_frictz{args.friction_z}_temp{args.temperature}.p","rb") as f:
    traj = pickle.load(f)

# remove thermalization part 
traj = traj[1000:]

# set parameters, initialize variables
taus = np.concatenate(([1],np.arange(5,215,10,dtype=int)))
variable_names = ['x_0', 'y_0', 'z_0', 'x_tau', 'y_tau', 'z_tau']

TEs_X_to_Y = np.zeros((len(taus)))
TEs_Y_to_X = np.zeros((len(taus)))

TEs_X_to_Y_givenZ = np.zeros((len(taus)))
TEs_Y_to_X_givenZ = np.zeros((len(taus)))

# extract X(0) and Y(0)
X0 = traj[0,:,0].reshape((-1,1))
Y0 = traj[0,:,1].reshape((-1,1))
Z0 = traj[0,:,2].reshape((-1,1))
for i_tau, tau in tqdm(enumerate(taus)):
    # extract X(tau) and Y(tau)
    Xtau = traj[tau,:,0].reshape((-1,1))
    Ytau = traj[tau,:,1].reshape((-1,1))
    Ztau = traj[tau,:,2].reshape((-1,1))

    dataset = pd.DataFrame(np.column_stack((X0,Y0,Z0,Xtau,Ytau,Ztau)), columns=variable_names)

    # compute (continuous) transfer entropy in both directions
    TEs_X_to_Y[i_tau] = (
        knncmi.cmi(['x_0'], ['y_tau'], ['y_0'], k=args.k, data=dataset, minzero=args.minzero)
    )
    TEs_Y_to_X[i_tau] = (
        knncmi.cmi(['y_0'], ['x_tau'], ['x_0'], k=args.k, data=dataset, minzero=args.minzero)
    )

    # compute (continuous) conditional transfer entropy in both directions
    TEs_X_to_Y_givenZ[i_tau] = (
        knncmi.cmi(['x_0'], ['y_tau'], ['y_0','z_0'], k=args.k, data=dataset, minzero=args.minzero)
    )
    TEs_Y_to_X_givenZ[i_tau] = (
        knncmi.cmi(['y_0'], ['x_tau'], ['x_0','z_0'], k=args.k, data=dataset, minzero=args.minzero)
    )

# save data
pickle.dump([taus, TEs_X_to_Y, TEs_Y_to_X, TEs_X_to_Y_givenZ, TEs_Y_to_X_givenZ], 
            open(f"./pickles_te/seed{args.seed}_N{args.N}_k{args.k}.p","wb"))