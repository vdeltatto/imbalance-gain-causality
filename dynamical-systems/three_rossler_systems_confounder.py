import pickle
import argparse
import numpy as np

from scipy.integrate import ode
from numpy.random import random
###################################################################

# definition of the system equations


def coupled_rossler_systems(t, xyz, omega_x, omega_y, omega_z, epsilon_zx, epsilon_zy):
    """
    Three coupled Rossler systems, Z->X and Z->Y (Z is a confounder)
    """
    x_1, x_2, x_3, y_1, y_2, y_3, z_1, z_2, z_3 = xyz

    x_dot_1 = -omega_x*x_2 - x_3 + epsilon_zx*(z_1 - x_1)
    x_dot_2 = omega_x*x_1 + 0.15*x_2
    x_dot_3 = 0.2 + x_3*(x_1-10.)

    y_dot_1 = -omega_y*y_2 - y_3 + epsilon_zy*(z_1 - y_1)
    y_dot_2 = omega_y*y_1 + 0.15*y_2
    y_dot_3 = 0.2 + y_3*(y_1-10.)

    z_dot_1 = -omega_z*z_2 - z_3
    z_dot_2 = omega_z*z_1 + 0.15*z_2
    z_dot_3 = 0.2 + z_3*(z_1-10.)

    return [x_dot_1, x_dot_2, x_dot_3, y_dot_1, y_dot_2, y_dot_3, z_dot_1, z_dot_2, z_dot_3]


def jacobian(t, xyz, omega_x, omega_y, omega_z, epsilon_zx, epsilon_zy):
    x_1, x_2, x_3, y_1, y_2, y_3, z_1, z_2, z_3 = xyz

    return [[-epsilon_zx, -omega_x, -1., 0., 0., 0., epsilon_zx, 0., 0.],
            [omega_x, 0.15, 0., 0., 0., 0., 0., 0., 0.],
            [x_3, 0., x_1-10., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., -epsilon_zy, -omega_y, -1., epsilon_zy, 0., 0.],
            [0., 0., 0., omega_y, 0.15, 0., 0., 0., 0.],
            [0., 0., 0., y_3, 0., y_1-10., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., -omega_z, -1.],
            [0., 0., 0., 0., 0., 0., omega_z, 0.15, 0.],
            [0., 0., 0., 0., 0., 0., z_3, 0., z_1-10.]]


###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ox", "--omega_x", dest="omega_x",
                        default=1.015, type=float,
                        help="Value of parameter omega for system X")
    parser.add_argument("-oy", "--omega_y", dest="omega_y",
                        default=0.985, type=float,
                        help="Value of parameter omega for system Y")
    parser.add_argument("-oz", "--omega_z", dest="omega_z",
                        default=1.005, type=float,
                        help="Value of parameter omega for system Z")
    parser.add_argument("-epszx", "--epsilon_zx", dest="epsilon_zx",
                        default=0.08, type=float,
                        help="Value of the coupling parameter Z->X")
    parser.add_argument("-epszy", "--epsilon_zy", dest="epsilon_zy",
                        default=0.05, type=float,
                        help="Value of the coupling parameter Z->Y")
    parser.add_argument("-nsamples", "--nsamples", dest="nsamples",
                        default=305000, type=int,
                        help="Number of samples of generated trajectory")
    parser.add_argument("-dt", "--dt", dest="dt",
                        default=0.0785, type=float,
                        help="Integration time step")
    parser.add_argument("-uf", "--undersample_factor",
                        dest="undersample_factor",
                        default=4, type=int,
                        help="Undersample factor")
    parser.add_argument("-rs", "--seed", dest="seed",
                        default=1998, type=int,
                        help="Random seed")
    parser.add_argument("-ii", "--integrator", dest="integrator",
                        default="dop853", type=str,
                        help="ODE integrator")
    parser.add_argument("-out", "--output", dest="output_filename",
                        default="trajectory_rossler_systems.p", type=str,
                        help="Output file name")
    args = parser.parse_args()

    # set parameters of the two systems
    params = [args.omega_x, args.omega_y, args.omega_z, args.epsilon_zx, args.epsilon_zy]

    # set initial conditions
    np.random.seed(seed=args.seed)
   
    x0 = 10*(random(size=3)-0.5)
    y0 = 10*(random(size=3)-0.5)
    z0 = 10*(random(size=3)-0.5)
    xyz0 = np.concatenate((x0, y0, z0))
    t0 = 0.

    # set integration parameters
    nsteps = args.nsamples * args.undersample_factor
    t_end = nsteps*args.dt

    # integrate equations
    r = ode(coupled_rossler_systems, jacobian).set_integrator(args.integrator)
    r.set_initial_value(xyz0, t0).set_f_params(*params).set_jac_params(*params)

    trajectory = np.empty((args.nsamples+1, 10))

    istep = 0
    isample = 0
    while r.successful() and r.t < t_end:
        new_time = r.t + args.dt
        new_point = r.integrate(r.t + args.dt)
        if istep % args.undersample_factor == 0:
            trajectory[isample, 0] = new_time
            trajectory[isample, 1:] = new_point
            isample = isample + 1
        istep = istep + 1

    # save trajectory in pickle format
    pickle.dump(trajectory, open(args.output_filename, "wb"))
