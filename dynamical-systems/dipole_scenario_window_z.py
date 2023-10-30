import pickle
import argparse
import numpy as np
from tqdm import tqdm

from scipy.integrate import odeint
from numpy.random import random

###################################################################

def lorenz_96_with_perturbation(xz, t, N, Fx, Fz, epsilon_zx, t0, tf):
    """
    Coupled Lorenz 96 systems, Z[0] -> X[0]
    """
    x = xz[:N]
    z = xz[N:]
    # Setting up vectors
    dx = np.zeros(N)
    dz = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        dx[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + Fx
    dx[0] +=  epsilon_zx * z[0] * (np.heaviside(t-t0, 1) + np.heaviside(tf-t, 1)-1) # apply perturbation only to one variable
    for i in range(N):
        dz[i] = (z[(i + 1) % N] - z[i - 2]) * z[i - 1] - z[i] + Fz
    return np.concatenate((dx, dz))


###################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", "--N", dest="N",
                        default=40, type=int,
                        help="Number of variables of systems X and Y")
    parser.add_argument("-Fx", "--Fx", dest="Fx",
                        default=6, type=float,
                        help="Value of the forcing constant for system Z")
    parser.add_argument("-Fz", "--Fz", dest="Fz",
                        default=5, type=float,
                        help="Value of the forcing constant for system Z")
    parser.add_argument("-eps_zx", "--epsilon_zx", dest="epsilon_zx",
                        default=1.0, type=float,
                        help="Value of the coupling parameter Z->X")
    parser.add_argument("-nsamples", "--nsamples", dest="nsamples",
                        default=2000, type=int,
                        help="Number of samples of generated trajectory")
    parser.add_argument("-nsamples_decorr", "--nsamples_decorr", dest="nsamples_decorr",
                        default=1000, type=int,
                        help="Number of initial samples to discard")
    parser.add_argument("-dt", "--dt", dest="dt",
                        default=0.03, type=float,
                        help="Integration time step")
    parser.add_argument("-t0", "--t0", dest="t0",
                        default=1500, type=float,
                        help="Starting time of perturbation")
    parser.add_argument("-tf", "--tf", dest="tf",
                        default=1530, type=float,
                        help="Final time of perturbation")
    parser.add_argument("-uf", "--undersample_factor",
                        dest="undersample_factor",
                        default=2, type=int,
                        help="Undersample factor")
    parser.add_argument("-rs", "--seed", dest="seed",
                        default=1998, type=int,
                        help="Random seed")
    parser.add_argument("-out", "--output_folder", dest="output_folder",
                        default="trajectory_lorenz96_systems", type=str,
                        help="Output folder")
    args = parser.parse_args()

    # set initial conditions
    np.random.seed(seed=args.seed)
    # set integration parameters
    nsteps = args.nsamples * args.undersample_factor
    nsteps_decorr = args.nsamples_decorr * args.undersample_factor
    times = np.arange(0.0, nsteps*args.dt, args.dt)
    start_perturbation = args.dt * args.undersample_factor * args.t0
    end_perturbation = args.dt * args.undersample_factor * args.tf

    for itraj in tqdm(range(5000)):

        # new initialization (for tests in dipole_scenario_stat_window)
        x0 = args.Fx * np.ones(args.N) + 0.1 * random(size=args.N)
        z0 = args.Fz * np.ones(args.N) + 0.1 * random(size=args.N)
        xz0 = np.concatenate((x0, z0))

        # integrate equations
        trajectory = odeint(lorenz_96_with_perturbation, xz0, times,
                            args=(args.N, args.Fx, args.Fz, args.epsilon_zx, start_perturbation, end_perturbation))
        trajectory = np.append(times[:, np.newaxis], trajectory, axis=-1)

        # delete first part of trajectory
        trajectory = trajectory[nsteps_decorr:]

        # undersample trajectory
        undersample_times = np.arange(0, trajectory.shape[0], args.undersample_factor)
        trajectory = trajectory[undersample_times]

        # save trajectory in pickle format
        pickle.dump(trajectory, open(f"{args.output_folder}/itraj{itraj}.p", "wb"))