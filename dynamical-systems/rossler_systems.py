import pickle
import argparse
import numpy as np

from scipy.integrate import ode
from numpy.random import random
###################################################################

# definition of the system equations


def coupled_rossler_systems(t, xy, omega_1, omega_2, epsilon):
    """
    Unidirectionally coupled Rossler systems X->Y
    """
    x_1, x_2, x_3, y_1, y_2, y_3 = xy

    x_dot_1 = -omega_1*x_2 - x_3
    x_dot_2 = omega_1*x_1 + 0.15*x_2
    x_dot_3 = 0.2 + x_3*(x_1-10.)

    y_dot_1 = -omega_2*y_2 - y_3 + epsilon*(x_1 - y_1)
    y_dot_2 = omega_2*y_1 + 0.15*y_2
    y_dot_3 = 0.2 + y_3*(y_1-10.)

    return [x_dot_1, x_dot_2, x_dot_3, y_dot_1, y_dot_2, y_dot_3]


def jacobian(t, xy, omega_1, omega_2, epsilon):
    x_1, x_2, x_3, y_1, y_2, y_3 = xy

    return [[0., -omega_1, -1., 0., 0., 0.],
            [omega_1, 0.15, 0., 0., 0., 0.],
            [x_3, 0., x_1-10., 0., 0., 0.],
            [epsilon, 0., 0., -epsilon, -omega_2, -1.],
            [0., 0., 0., omega_2, 0.15, 0.],
            [0., 0., 0., y_3, 0., y_1-10.]]


###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o1", "--omega_1", dest="omega_1",
                        default=1.015, type=float,
                        help="Value of parameter omega for system X")
    parser.add_argument("-o2", "--omega_2", dest="omega_2",
                        default=0.985, type=float,
                        help="Value of parameter omega for system Y")
    parser.add_argument("-eps", "--epsilon", dest="epsilon",
                        default=0.07, type=float,
                        help="Value of the coupling parameter X->Y")
    parser.add_argument("-nsamples", "--nsamples", dest="nsamples",
                        default=105000, type=int,
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
    parser.add_argument("-x0", "--x_0", dest="x0", nargs=3,
                        default=[11.120979, 17.496796, 51.023544], type=float,
                        help="Initial conditions X0")
    parser.add_argument("-ii", "--integrator", dest="integrator",
                        default="dop853", type=str,
                        help="ODE integrator")
    parser.add_argument("-out", "--output", dest="output_filename",
                        default="trajectory_rossler_systems.p", type=str,
                        help="Output file name")
    args = parser.parse_args()

    # set parameters of the two systems
    params = [args.omega_1, args.omega_2, args.epsilon]

    # set initial conditions
    np.random.seed(seed=args.seed)
    x0 = args.x0
    y0 = [
            x0[0]*(0.5 + random()),
            x0[1]*(0.5 + random()),
            x0[2]*(0.5 + random())
         ]
    xy0 = np.append(x0, y0)
    t0 = 0.

    # set integration parameters
    nsteps = args.nsamples * args.undersample_factor
    t_end = nsteps*args.dt

    # integrate equations
    r = ode(coupled_rossler_systems, jacobian).set_integrator(args.integrator)
    r.set_initial_value(xy0, t0).set_f_params(*params).set_jac_params(*params)

    trajectory = np.empty((args.nsamples+1, 7))

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
