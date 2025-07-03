import pickle
import argparse
import numpy as np

from scipy.integrate import odeint
from numpy.random import random

###################################################################

def coupled_systems(xyz, t):

    x = +xyz[:40]    # Lorenz 96 with D=40 variables
    y = +xyz[40:43]  # Lorenz with D=3 variables
    z = +xyz[43:46]  # Rossler with D=3 variables
    w = +xyz[46:66]  # Lorenz 96 with D=20 variables
    l = +xyz[66:96]  # Lorenz 96 with D=30 variables
    m = +xyz[96:99]  # Rossler with D=3

    dx = np.zeros(40)
    dy = np.zeros(3)
    dz = np.zeros(3)
    dw = np.zeros(20)
    dl = np.zeros(30)
    dm = np.zeros(3)

    # Parameters of Lorenz 96 (x,w,l)
    Fx = 8
    Fw = 7
    Fl = 6

    # Parameters of Lorenz (y)
    sigma = 10.
    beta = 8. / 3
    rho = 28.

    # Parameters of Rosslers (z,m)
    omega_z = 1.015
    omega_m = 0.985
    a = 0.15
    b = 0.2
    c = 10.

    # Coupling parameters
    eps_xz = 0.1     # L96 -> Rossler
    eps_yz = 0.1     # Lorenz -> Rossler
    eps_zw = 0.7     # Rossler -> L96
    eps_xl = 0.5     # L96 -> L96
    eps_wl = 1.0     # L96 -> L96
    eps_wm = 0.1     # L96 -> Rossler

    ################################ x ################################
    for i in range(40):
        dx[i] = (x[(i + 1) % 40] - x[i - 2]) * x[i - 1] - x[i] + Fx

    ################################ y ################################
    dy[0] = sigma * (y[1] - y[0])
    dy[1] = (y[0] * (rho - y[2]) - y[1])
    dy[2] = (y[0] * y[1] - beta * y[2])

    ################################ z ################################
    dz[0] = (-omega_z * z[1] - z[2]) + eps_xz * z[0] + eps_yz * y[0]
    dz[1] = (omega_z * z[0] + a * z[1])
    dz[2] = (b + z[2] * (z[0] - c))

    ################################ w ################################
    for i in range(20):
        dw[i] = (w[(i + 1) % 20] - w[i - 2]) * w[i - 1] - w[i] + Fw
    dw[0] += eps_zw * z[0]
    dw[1] += eps_zw * z[1]
    dw[2] += eps_zw * z[2]

    ################################ l ################################
    for i in range(30):
        dl[i] = (l[(i + 1) % 30] - l[i - 2]) * l[i - 1] - l[i] + Fl + eps_xl * x[i]
    for i in range(20):
        dl[i] += eps_wl * w[i]**2

    ################################ m ################################
    dm[0] = (-omega_m * m[1] - m[2]) + eps_wm * m[0]
    dm[1] = (omega_m * m[0] + a * m[1])
    dm[2] = (b + m[2] * (m[0] - c))

    return np.concatenate((dx, dy, dz, dw, dl, dm))

###################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-nsamples", "--nsamples", dest="nsamples",
                        default=252500, type=int,
                        help="Number of samples of generated trajectory")
    parser.add_argument("-dt", "--dt", dest="dt",
                        default=0.01, type=float,
                        help="Integration time step")
    parser.add_argument("-uf", "--undersample_factor",
                        dest="undersample_factor",
                        default=5, type=int,
                        help="Undersample factor")
    parser.add_argument("-rs", "--seed", dest="seed",
                        default=1998, type=int,
                        help="Random seed")
    parser.add_argument("-out", "--output", dest="output_filename",
                        default="traj_complicated.p", type=str,
                        help="Output file name")
    args = parser.parse_args()

    # set integration parameters
    nsteps = args.nsamples * args.undersample_factor
    times = np.arange(0.0, nsteps * args.dt, args.dt)

    # initialize systems
    np.random.seed(seed=args.seed)
    x0 = 8 * np.ones(40)
    x0[0] += 0.01 * np.random.random()
    y0 = np.array([1, 1, 1]) * (0.5 + np.random.random(3))
    z0 = np.array([1, 1, 1]) * (0.5 + np.random.random(3))
    w0 = 7 * np.ones(20)
    w0[0] += 0.01 * np.random.random()
    l0 = 6 * np.ones(30)
    l0[0] += 0.01 * np.random.random()
    m0 = np.array([1, 1, 1]) * (0.5 + np.random.random(3))
    xyz0 = np.concatenate((x0, y0, z0, w0, l0, m0))

    # integrate equations
    trajectory = odeint(coupled_systems, xyz0, times)
    trajectory = np.append(times[:, np.newaxis], trajectory, axis=-1) # add time component, [:,0]
    
    # undersample trajectory
    undersample_times = np.arange(0, trajectory.shape[0],
                                  args.undersample_factor)
    trajectory = trajectory[undersample_times]

    # save trajectory in pickle format
    pickle.dump(trajectory, open(args.output_filename, "wb"))