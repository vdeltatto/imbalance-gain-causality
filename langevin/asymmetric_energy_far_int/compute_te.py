import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import knncmi
from pyitlib import discrete_random_variable as drv


def from_continuous_to_discrete_traj(traj):
    centers_x = np.array([0.0,5.0,12.0])
    centers_y = np.array([0.0,5.0])

    traj_discrete_x = np.argmin(np.abs(traj[:,:,0,np.newaxis] - centers_x[np.newaxis,np.newaxis,:]), axis=-1)
    traj_discrete_y = np.argmin(np.abs(traj[:,:,1,np.newaxis] - centers_y[np.newaxis,np.newaxis,:]), axis=-1)

    return np.concatenate((traj_discrete_x[:,:,np.newaxis],traj_discrete_y[:,:,np.newaxis]), axis=-1)


parser = argparse.ArgumentParser()
parser.add_argument("-dt", "--dt", dest="dt",
                    default=0.01, type=float, help="Timestep of integrator")
parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                    default=1000000, type=int, help="Number of steps")
parser.add_argument("-temperature", "--temperature", dest="temperature",
                    default=1, type=float, help="Temperature")
parser.add_argument("-friction", "--friction", dest="friction",
                    default=1, type=float, help="Friction")
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1998, type=int, help="Random seed")
parser.add_argument("-N", "--N", dest="N",
                    default=2500, type=int, help="Number of trajectories")
parser.add_argument("-k", "--k", dest="k",
                    default=7, type=int, help="Number of neighbors")
parser.add_argument("-minzero", "--minzero", dest="minzero",
                    default=True, type=bool, help="minzero argument")
parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                    default=1000, type=int, help="Samling stride")
args = parser.parse_args()

with open(f"./pickles_traj/seed{args.seed}_N{args.N}_dt{args.dt}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}_frict{args.friction}_temp{args.temperature}.p","rb") as f:
    traj = pickle.load(f)

# remove thermalization part
traj = traj[100:]

# convert traj to discrete numbers
traj_discrete = from_continuous_to_discrete_traj(traj)

# set parameters, initialize variables
taus = np.arange(10,770,40,dtype=int)
variable_names = ['x_0', 'y_0', 'x_tau', 'y_tau']

TEs_X_to_Y = np.zeros((len(taus)))
TEs_Y_to_X = np.zeros((len(taus)))
TEs_X_to_Y_discrete = np.zeros((len(taus)))
TEs_Y_to_X_discrete = np.zeros((len(taus)))

# extract X(0) and Y(0)
X0 = traj[0,:,0].reshape((-1,1))
Y0 = traj[0,:,1].reshape((-1,1))
X0_discrete = traj_discrete[0,:,0]
Y0_discrete = traj_discrete[0,:,1]
for i_tau, tau in tqdm(enumerate(taus)):
    # extract X(tau) and Y(tau)
    Xtau = traj[tau,:,0].reshape((-1,1))
    Ytau = traj[tau,:,1].reshape((-1,1))
    Xtau_discrete = traj_discrete[tau,:,0]
    Ytau_discrete = traj_discrete[tau,:,1]

    dataset = pd.DataFrame(np.column_stack((X0,Y0,Xtau,Ytau)), columns=variable_names)

    # compute (continuous) transfer entropy in both directions
    TEs_X_to_Y[i_tau] = (
        knncmi.cmi(['x_0'], ['y_tau'], ['y_0'], k=args.k, data=dataset, minzero=args.minzero)
    )
    TEs_Y_to_X[i_tau] = (
        knncmi.cmi(['y_0'], ['x_tau'], ['x_0'], k=args.k, data=dataset, minzero=args.minzero)
    )

    # compute (discrete) transfer entropy in both directions
    TEs_X_to_Y_discrete[i_tau] = (
        drv.information_mutual_conditional(X=X0_discrete,Y=Ytau_discrete,Z=Y0_discrete, base=np.exp(1))
    )
    TEs_Y_to_X_discrete[i_tau] = (
       drv.information_mutual_conditional(X=Y0_discrete,Y=Xtau_discrete,Z=X0_discrete, base=np.exp(1))
    )

# save data
pickle.dump([taus, TEs_X_to_Y, TEs_Y_to_X, TEs_X_to_Y_discrete, TEs_Y_to_X_discrete], 
            open(f"./pickles_te/seed{args.seed}_N{args.N}_k{args.k}.p","wb"))