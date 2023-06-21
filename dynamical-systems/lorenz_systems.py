import pickle
import argparse
import numpy as np

from scipy.integrate import ode
from numpy.random import random

###################################################################


# definition of the system equations
def coupled_lorenz_systems(t, xy, epsilon_12, epsilon_21):
    """
    Bidirectionally coupled Lorenz systems X <-> Y
    """
    x_1, x_2, x_3, y_1, y_2, y_3 = xy

    x_dot_1 = 10.*(-x_1+x_2)
    x_dot_2 = 28*x_1 - x_2 - x_1*x_3 + epsilon_21*y_2**2
    x_dot_3 = x_1*x_2 - 8./3.*x_3

    y_dot_1 = 10.*(-y_1+y_2)
    y_dot_2 = 28*y_1 - y_2 - y_1*y_3 + epsilon_12*x_2**2
    y_dot_3 = y_1*y_2 - 8./3.*y_3

    return [x_dot_1, x_dot_2, x_dot_3, y_dot_1, y_dot_2, y_dot_3]


def jacobian(t, xy, epsilon_12, epsilon_21):
    x_1, x_2, x_3, y_1, y_2, y_3 = xy

    return [[-10., 10., 0., 0., 0., 0.],
            [28.-x_3, -1., -x_1, 0., 2.*epsilon_21*y_2, 0.],
            [x_2, x_1, -8./3., 0., 0., 0.],
            [0., 0., 0., -10., 10., 0.],
            [0., 2.*epsilon_12*x_2, 0., 28.-y_3, -1., -y_1],
            [0., 0., 0., y_2, y_1, -8./3.]]


###################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-eps12", "--epsilon_12", dest="epsilon_12",
                        default=0.1, type=float,
                        help="Value of the coupling parameter X->Y")
    parser.add_argument("-eps21", "--epsilon_21", dest="epsilon_21",
                        default=0.1, type=float,
                        help="Value of the coupling parameter Y->X")
    parser.add_argument("-nsamples", "--nsamples", dest="nsamples",
                        default=205000, type=int,
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
    parser.add_argument("-x0", "--x_0", dest="x0", nargs=3,
                        default=[1.0, 1.0, 1.0], type=float,
                        help="Initial conditions X0")
    parser.add_argument("-ii", "--integrator", dest="integrator",
                        default="dop853", type=str,
                        help="ODE integrator")
    parser.add_argument("-out", "--output", dest="output_filename",
                        default="trajectory_lorenz_systems.p", type=str,
                        help="Output file name")
    args = parser.parse_args()

    # set parameters of the two systems
    params = [args.epsilon_12, args.epsilon_21]

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
    r = ode(coupled_lorenz_systems, jacobian).set_integrator(args.integrator)
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
