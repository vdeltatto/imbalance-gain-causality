import pickle
import argparse
import numpy as np

from scipy.integrate import odeint
from numpy.random import random

###################################################################


def coupled_lorenz_96(xy, t, N, F1, F2, epsilon):
    """
    Unidirectionally coupled Lorenz96 systems X->Y with constant forcing
    """
    x = xy[:N]
    y = xy[N:]
    # Setting up vectors
    dx = np.zeros(N)
    dy = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        dx[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F1
    for i in range(N):
        dy[i] = (y[(i + 1) % N] - y[i - 2]) * y[i - 1] - y[i] + F2 + \
                epsilon*x[i]
    return np.append(dx, dy)

###################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", "--N", dest="N",
                        default=40, type=int,
                        help="Number of variables of systems X and Y")
    parser.add_argument("-F1", "--F1", dest="F1",
                        default=5, type=int,
                        help="Value of the forcing constant for system X")
    parser.add_argument("-F2", "--F2", dest="F2",
                        default=6, type=int,
                        help="Value of the forcing constant for system Y")
    parser.add_argument("-eps", "--epsilon", dest="epsilon",
                        default=1.0, type=float,
                        help="Value of the coupling parameter X->Y")
    parser.add_argument("-nsamples", "--nsamples", dest="nsamples",
                        default=252500, type=int,
                        help="Number of samples of generated trajectory")
    parser.add_argument("-dt", "--dt", dest="dt",
                        default=0.03, type=float,
                        help="Integration time step")
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
    x0 = args.F1 * np.ones(args.N)
    x0[0] += 0.01 * random() # Add small perturbation to the first variable
    y0 = args.F2 * np.ones(args.N)
    y0[0] += 0.01 * random()
    xy0 = np.append(x0, y0)

    # set integration parameters
    nsteps = args.nsamples*args.undersample_factor
    times = np.arange(0.0, nsteps*args.dt, args.dt)

    # integrate equations
    trajectory = odeint(coupled_lorenz_96, xy0, times,
                        args=(args.N, args.F1, args.F2, args.epsilon))
    trajectory = np.append(times[:, np.newaxis], trajectory, axis=-1)

    # undersample trajectory
    undersample_times = np.arange(0, trajectory.shape[0],
                                  args.undersample_factor)
    trajectory = trajectory[undersample_times]

    # save trajectory in pickle format
    pickle.dump(trajectory, open(args.output_filename, "wb"))
