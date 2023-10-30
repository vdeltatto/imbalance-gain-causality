import pickle
import argparse
import numpy as np

from scipy.integrate import odeint
from numpy.random import random

###################################################################


def three_lorenz_96(xyz, t, N, Fx, Fy, Fz, epsilon_xy, epsilon_zx):
    """
    Unidirectionally coupled Lorenz96 systems X->Y with constant forcing
    """
    x = xyz[:N]
    y = xyz[N:2*N]
    z = xyz[2*N:]
    print(x.shape, y.shape, z.shape)
    # Setting up vectors
    dx = np.zeros(N)
    dy = np.zeros(N)
    dz = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        dx[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + Fx + \
                epsilon_zx*z[i]
    for i in range(N):
        dy[i] = (y[(i + 1) % N] - y[i - 2]) * y[i - 1] - y[i] + Fy + \
                epsilon_xy*x[i]
    for i in range(N):
        dz[i] = (z[(i + 1) % N] - z[i - 2]) * z[i - 1] - z[i] + Fz
    return np.concatenate((dx, dy, dz))

###################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", "--N", dest="N",
                        default=40, type=int,
                        help="Number of variables of systems X and Y")
    parser.add_argument("-Fx", "--Fx", dest="Fx",
                        default=5, type=float,
                        help="Value of the forcing constant for system X")
    parser.add_argument("-Fy", "--Fy", dest="Fy",
                        default=6, type=float,
                        help="Value of the forcing constant for system Y")
    parser.add_argument("-Fz", "--Fz", dest="Fz",
                        default=5.5, type=float,
                        help="Value of the forcing constant for system Z")
    parser.add_argument("-eps_xy", "--epsilon_xy", dest="epsilon_xy",
                        default=1.0, type=float,
                        help="Value of the coupling parameter X->Y")
    parser.add_argument("-eps_zx", "--epsilon_zx", dest="epsilon_zx",
                        default=1.0, type=float,
                        help="Value of the coupling parameter Z->X")
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
    x0 = args.Fx * np.ones(args.N)
    x0[0] += 0.01 * random() # Add small perturbation to the first variable
    y0 = args.Fy * np.ones(args.N)
    y0[0] += 0.01 * random()
    z0 = args.Fz * np.ones(args.N)
    z0[0] += 0.01 * random()
    xyz0 = np.concatenate((x0, y0, z0))
    print(xyz0.shape)

    # set integration parameters
    nsteps = args.nsamples*args.undersample_factor
    times = np.arange(0.0, nsteps*args.dt, args.dt)

    # integrate equations
    trajectory = odeint(three_lorenz_96, xyz0, times,
                        args=(args.N, args.Fx, args.Fy, args.Fx, args.epsilon_xy, args.epsilon_zx))
    trajectory = np.append(times[:, np.newaxis], trajectory, axis=-1)

    # undersample trajectory
    undersample_times = np.arange(0, trajectory.shape[0],
                                  args.undersample_factor)
    trajectory = trajectory[undersample_times]

    # save trajectory in pickle format
    pickle.dump(trajectory, open(args.output_filename, "wb"))
