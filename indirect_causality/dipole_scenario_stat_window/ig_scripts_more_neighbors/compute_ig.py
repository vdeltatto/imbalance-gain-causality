import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-coupling", "--coupling", dest="coupling",
                    default="epszx_1", type=str,
                    help="couling case: one of 'noZ', 'epszx_1' and 'epszx_3'")
parser.add_argument("-case", "--case", dest="case",
                    default="XY", type=str,
                    help="systems among which IG is computed")
parser.add_argument("-iY", "--iY", dest="iY",
                    default="XY", type=int,
                    help="Index of Y variable")
parser.add_argument("-seed", "--seed", dest="seed",
                    default=0, type=int,
                    help="seed of trajectory")
args = parser.parse_args()

trajectories = np.zeros((5000, 1000, 81))
for itraj in np.arange(5000):
    trajectories[itraj] = pickle.load(open(f"../trajs_{args.coupling}/seed{args.seed}/itraj{itraj}.p","rb"))

traj_length = trajectories.shape[1]
N = 5000
D = 40 # dimensionality of each system
taus_forward = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400])
taus_backward = -np.flip(taus_forward)
taus = np.append(taus_backward,taus_forward)
k = 20
E = 30
tau_e = 1
t0 = 500
n_jobs = 12
alphas = np.linspace(0.,1.5,300)

if args.case == "XY":

    info_imbalances_Y_to_X = np.zeros((len(taus),len(alphas)))
    info_imbalances_X_to_Y = np.zeros((len(taus),len(alphas)))

    X0 = trajectories[:,t0-E:t0, 1]
    Y0 = trajectories[:,t0-E:t0, args.iY]
    for i_tau, tau in enumerate(taus):

        d = MetricComparisons(maxk=trajectories.shape[0]-1, njobs=n_jobs)
        Xtau = trajectories[:,t0+tau-E:t0+tau, 1]
        Ytau = trajectories[:,t0+tau-E:t0+tau, args.iY]

        info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
        info_imbalances_Y_to_X[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

    pickle.dump([taus, info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles_{args.coupling}/XY{args.iY}_seed{args.seed}.p","wb"))

elif args.case == "ZY":

    info_imbalances_Z_to_Y = np.zeros((len(taus),len(alphas)))
    info_imbalances_Y_to_Z = np.zeros((len(taus),len(alphas)))

    Z0 = trajectories[:,t0-E:t0, D+1]
    Y0 = trajectories[:,t0-E:t0, args.iY]
    for i_tau, tau in enumerate(taus):

        d = MetricComparisons(maxk=trajectories.shape[0]-1, njobs=n_jobs)
        Ztau = trajectories[:,t0+tau-E:t0+tau, D+1]
        Ytau = trajectories[:,t0+tau-E:t0+tau, args.iY]

        info_imbalances_Z_to_Y[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
        info_imbalances_Y_to_Z[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=Z0, effect_future=Ztau, weights=alphas, k=k)

    pickle.dump([taus, info_imbalances_Z_to_Y, info_imbalances_Y_to_Z], open(f"./pickles_{args.coupling}/ZY{args.iY}_seed{args.seed}.p","wb"))