import pickle
import argparse
import numpy as np

from scipy.integrate import odeint
from scipy import signal
from numpy.random import random

###################################################################

def lorenz_96_with_perturbation(xz, t, N, Fx, Fz, epsilon_zx, L, duty):
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
    dx[0] +=  epsilon_zx * z[0] * 0.5 * (1 + signal.square(t*2*np.pi/L, duty=duty)) # apply window perturbation only to one variable
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
                        default=3120000, type=int,
                        help="Number of samples of generated trajectory")
    parser.add_argument("-dt", "--dt", dest="dt",
                        default=0.03, type=float,
                        help="Integration time step")
    parser.add_argument("-L", "--L", dest="L",
                        default=600, type=int,
                        help="Length of single trajectory in ensemble")
    parser.add_argument("-E", "--E", dest="E",
                        default=30, type=int,
                        help="Length of perturbation window")
    parser.add_argument("-uf", "--undersample_factor",
                        dest="undersample_factor",
                        default=2, type=int,
                        help="Undersample factor")
    parser.add_argument("-rs", "--seed", dest="seed",
                        default=1998, type=int,
                        help="Random seed")
    parser.add_argument("-out", "--output", dest="output_filename",
                        default="trajectory_lorenz96_systems.p", type=str,
                        help="Output file name")
    args = parser.parse_args()

    # set initial conditions
    np.random.seed(seed=args.seed)

    # new initialization (for tests in dipole_scenario_stat_window)
    x0 = args.Fx * np.ones(args.N) + 0.1 * random(size=args.N)
    z0 = args.Fz * np.ones(args.N) + 0.1 * random(size=args.N)
    xz0 = np.concatenate((x0, z0))

    # set integration parameters
    nsteps = args.nsamples*args.undersample_factor
    times = np.arange(0.0, nsteps*args.dt, args.dt)
    single_traj_length = args.dt * args.undersample_factor * args.L
    duty = 30 / args.L

    # integrate equations
    trajectory = odeint(lorenz_96_with_perturbation, xz0, times,
                        args=(args.N, args.Fx, args.Fz, args.epsilon_zx, single_traj_length, duty))
    trajectory = np.append(times[:, np.newaxis], trajectory, axis=-1)

    # undersample trajectory
    undersample_times = np.arange(0, trajectory.shape[0], args.undersample_factor)
    trajectory = trajectory[undersample_times]

    # save trajectory in pickle format
    pickle.dump(trajectory, open(args.output_filename, "wb"))