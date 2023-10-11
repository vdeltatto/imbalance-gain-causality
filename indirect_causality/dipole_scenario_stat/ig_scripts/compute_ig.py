import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import argparse
from utilities import construct_time_delay_embedding

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case", dest="case",
                    default="XY", type=str,
                    help="systems among which IG is computed")
parser.add_argument("-seed", "--seed", dest="seed",
                    default=0, type=int,
                    help="seed of trajectory")
args = parser.parse_args()

#trajectory = pickle.load(open(f"../trajs/seed{args.seed}.p","rb"))
trajectory = pickle.load(open(f"../trajs_noZ/seed{args.seed}.p","rb"))
traj_length = trajectory.shape[0]
N = 5000
D = 40 # dimensionality of each system
taus_forward = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200])
taus_backward = -np.flip(taus_forward)
taus = np.append(taus_backward,taus_forward)
k = 20
E = 30
tau_e = 1
sample_times_forward = np.linspace(505000,805000,N,dtype=int)
sample_times_backward = np.linspace(205000,505000,N,dtype=int)
n_jobs = 12
alphas = np.linspace(0.,1.5,300)
alphas_large = np.linspace(0.,20,300)
alphas_small = np.linspace(0.,0.1,300)

if args.case == "XY":

    info_imbalances_Y_to_X_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_X_to_Y_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_Y_to_X_b = np.zeros((len(taus_backward),len(alphas)))
    info_imbalances_X_to_Y_b = np.zeros((len(taus_backward),len(alphas)))

    X_time_delay = construct_time_delay_embedding(X=trajectory[:,1], E=E, tau_e=tau_e)
    Y_time_delay = construct_time_delay_embedding(X=trajectory[:,2], E=E, tau_e=tau_e)

    X0 = X_time_delay[sample_times_forward]
    Y0 = Y_time_delay[sample_times_forward]
    for i_tau, tau in enumerate(taus_forward):

        d = MetricComparisons(maxk=len(sample_times_forward)-1, njobs=n_jobs)
        Xtau = X_time_delay[sample_times_forward+tau]
        Ytau = Y_time_delay[sample_times_forward+tau]

        info_imbalances_X_to_Y_f[i_tau] = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
        info_imbalances_Y_to_X_f[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)

    X0 = X_time_delay[sample_times_backward]
    Y0 = Y_time_delay[sample_times_backward]
    for i_tau, tau in enumerate(taus_backward):

        d = MetricComparisons(maxk=len(sample_times_backward)-1, njobs=n_jobs)
        Xtau = X_time_delay[sample_times_backward+tau]
        Ytau = Y_time_delay[sample_times_backward+tau]

        info_imbalances_X_to_Y_b[i_tau] = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
        info_imbalances_Y_to_X_b[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)
        
    info_imbalances_X_to_Y = np.row_stack((info_imbalances_X_to_Y_b, info_imbalances_X_to_Y_f))
    info_imbalances_Y_to_X = np.row_stack((info_imbalances_Y_to_X_b, info_imbalances_Y_to_X_f))

    pickle.dump([taus, info_imbalances_X_to_Y, info_imbalances_Y_to_X], open(f"./pickles_noZ/XY_seed{args.seed}.p","wb"))
    del info_imbalances_Y_to_X_f, info_imbalances_X_to_Y_f, info_imbalances_Y_to_X_b, info_imbalances_X_to_Y_b

elif args.case == "ZX":

    info_imbalances_Z_to_X_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_X_to_Z_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_Z_to_X_b = np.zeros((len(taus_backward),len(alphas)))
    info_imbalances_X_to_Z_b = np.zeros((len(taus_backward),len(alphas)))

    Z_time_delay = construct_time_delay_embedding(X=trajectory[:,D+1], E=E, tau_e=tau_e)
    X_time_delay = construct_time_delay_embedding(X=trajectory[:,1], E=E, tau_e=tau_e)

    Z0 = Z_time_delay[sample_times_forward]
    X0 = X_time_delay[sample_times_forward]
    for i_tau, tau in enumerate(taus_forward):

        d = MetricComparisons(maxk=len(sample_times_forward)-1, njobs=n_jobs)
        Ztau = Z_time_delay[sample_times_forward+tau]
        Xtau = X_time_delay[sample_times_forward+tau]

        info_imbalances_Z_to_X_f[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)
        info_imbalances_X_to_Z_f[i_tau] = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Z0, effect_future=Ztau, weights=alphas, k=k)

    Z0 = Z_time_delay[sample_times_backward]
    X0 = X_time_delay[sample_times_backward]
    for i_tau, tau in enumerate(taus_backward):

        d = MetricComparisons(maxk=len(sample_times_backward)-1, njobs=n_jobs)
        Ztau = Z_time_delay[sample_times_backward+tau]
        Xtau = X_time_delay[sample_times_backward+tau]

        info_imbalances_Z_to_X_b[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=X0, effect_future=Xtau, weights=alphas, k=k)
        info_imbalances_X_to_Z_b[i_tau] = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Z0, effect_future=Ztau, weights=alphas, k=k)

    info_imbalances_Z_to_X = np.row_stack((info_imbalances_Z_to_X_b, info_imbalances_Z_to_X_f))
    info_imbalances_X_to_Z = np.row_stack((info_imbalances_X_to_Z_b, info_imbalances_X_to_Z_f))

    pickle.dump([taus, info_imbalances_Z_to_X, info_imbalances_X_to_Z], open(f"./pickles_noZ/ZX_seed{args.seed}.p","wb"))
    del info_imbalances_Z_to_X_f, info_imbalances_X_to_Z_f, info_imbalances_Z_to_X_b, info_imbalances_X_to_Z_b

elif args.case == "ZY":

    info_imbalances_Z_to_Y_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_Y_to_Z_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_Z_to_Y_b = np.zeros((len(taus_backward),len(alphas)))
    info_imbalances_Y_to_Z_b = np.zeros((len(taus_backward),len(alphas)))

    Z_time_delay = construct_time_delay_embedding(X=trajectory[:,D+1], E=E, tau_e=tau_e)
    Y_time_delay = construct_time_delay_embedding(X=trajectory[:,2], E=E, tau_e=tau_e)

    Z0 = Z_time_delay[sample_times_forward]
    Y0 = Y_time_delay[sample_times_forward]
    for i_tau, tau in enumerate(taus_forward):

        d = MetricComparisons(maxk=len(sample_times_forward)-1, njobs=n_jobs)
        Ztau = Z_time_delay[sample_times_forward+tau]
        Ytau = Y_time_delay[sample_times_forward+tau]

        info_imbalances_Z_to_Y_f[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
        info_imbalances_Y_to_Z_f[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=Z0, effect_future=Ztau, weights=alphas, k=k)

    Z0 = Z_time_delay[sample_times_backward]
    Y0 = Y_time_delay[sample_times_backward]
    for i_tau, tau in enumerate(taus_backward):

        d = MetricComparisons(maxk=len(sample_times_backward)-1, njobs=n_jobs)
        Ztau = Z_time_delay[sample_times_backward+tau]
        Ytau = Y_time_delay[sample_times_backward+tau]

        info_imbalances_Z_to_Y_b[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=k)
        info_imbalances_Y_to_Z_b[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=Z0, effect_future=Ztau, weights=alphas, k=k)

    info_imbalances_Z_to_X = np.row_stack((info_imbalances_Z_to_Y_b, info_imbalances_Z_to_Y_f))
    info_imbalances_X_to_Z = np.row_stack((info_imbalances_Y_to_Z_b, info_imbalances_Y_to_Z_f))

    pickle.dump([taus, info_imbalances_Z_to_X, info_imbalances_X_to_Z], open(f"./pickles_noZ/ZY_seed{args.seed}.p","wb"))
    del info_imbalances_Z_to_Y_f, info_imbalances_Y_to_Z_f, info_imbalances_Z_to_Y_b, info_imbalances_Y_to_Z_b

elif args.case == "ZX_noemb":

    info_imbalances_Z_to_X_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_X_to_Z_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_Z_to_X_b = np.zeros((len(taus_backward),len(alphas)))
    info_imbalances_X_to_Z_b = np.zeros((len(taus_backward),len(alphas)))

    Z_time_delay = construct_time_delay_embedding(X=trajectory[:,D+1], E=E, tau_e=tau_e)
    X_time_delay = construct_time_delay_embedding(X=trajectory[:,1], E=E, tau_e=tau_e)

    Z0 = trajectory[sample_times_forward,D+1].reshape(-1, 1)
    X0 = X_time_delay[sample_times_forward]
    for i_tau, tau in enumerate(taus_forward):

        d = MetricComparisons(maxk=len(sample_times_forward)-1, njobs=n_jobs)
        Ztau = trajectory[sample_times_forward+tau,D+1].reshape(-1, 1)
        Xtau = X_time_delay[sample_times_forward+tau]

        info_imbalances_Z_to_X_f[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=X0, effect_future=Xtau, weights=alphas_large, k=k)
        info_imbalances_X_to_Z_f[i_tau] = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Z0, effect_future=Ztau, weights=alphas_small, k=k)

    Z0 = trajectory[sample_times_backward,D+1].reshape(-1, 1)
    X0 = X_time_delay[sample_times_backward]
    for i_tau, tau in enumerate(taus_backward):

        d = MetricComparisons(maxk=len(sample_times_backward)-1, njobs=n_jobs)
        Ztau = trajectory[sample_times_backward+tau,D+1].reshape(-1, 1)
        Xtau = X_time_delay[sample_times_backward+tau]

        info_imbalances_Z_to_X_b[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=X0, effect_future=Xtau, weights=alphas_large, k=k)
        info_imbalances_X_to_Z_b[i_tau] = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Z0, effect_future=Ztau, weights=alphas_small, k=k)

    info_imbalances_Z_to_X = np.row_stack((info_imbalances_Z_to_X_b, info_imbalances_Z_to_X_f))
    info_imbalances_X_to_Z = np.row_stack((info_imbalances_X_to_Z_b, info_imbalances_X_to_Z_f))

    pickle.dump([taus, info_imbalances_Z_to_X, info_imbalances_X_to_Z], open(f"./pickles_noZ/ZX_seed{args.seed}_noembforZ.p","wb"))
    del info_imbalances_Z_to_X_f, info_imbalances_X_to_Z_f, info_imbalances_Z_to_X_b, info_imbalances_X_to_Z_b

elif args.case == "ZY_noemb":

    info_imbalances_Z_to_Y_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_Y_to_Z_f = np.zeros((len(taus_forward),len(alphas)))
    info_imbalances_Z_to_Y_b = np.zeros((len(taus_backward),len(alphas)))
    info_imbalances_Y_to_Z_b = np.zeros((len(taus_backward),len(alphas)))

    Z_time_delay = construct_time_delay_embedding(X=trajectory[:,D+1], E=E, tau_e=tau_e)
    Y_time_delay = construct_time_delay_embedding(X=trajectory[:,2], E=E, tau_e=tau_e)

    Z0 = trajectory[sample_times_forward,D+1].reshape(-1, 1)
    Y0 = Y_time_delay[sample_times_forward]
    for i_tau, tau in enumerate(taus_forward):

        d = MetricComparisons(maxk=len(sample_times_forward)-1, njobs=n_jobs)
        Ztau = trajectory[sample_times_forward+tau,D+1].reshape(-1, 1)
        Ytau = Y_time_delay[sample_times_forward+tau]

        info_imbalances_Z_to_Y_f[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=Y0, effect_future=Ytau, weights=alphas_large, k=k)
        info_imbalances_Y_to_Z_f[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=Z0, effect_future=Ztau, weights=alphas_small, k=k)

    Z0 = trajectory[sample_times_backward,D+1].reshape(-1, 1)
    Y0 = Y_time_delay[sample_times_backward]
    for i_tau, tau in enumerate(taus_backward):

        d = MetricComparisons(maxk=len(sample_times_backward)-1, njobs=n_jobs)
        Ztau = trajectory[sample_times_backward+tau,D+1].reshape(-1, 1)
        Ytau = Y_time_delay[sample_times_backward+tau]

        info_imbalances_Z_to_Y_b[i_tau] = d.return_inf_imb_causality(
            cause_present=Z0, effect_present=Y0, effect_future=Ytau, weights=alphas_large, k=k)
        info_imbalances_Y_to_Z_b[i_tau] = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=Z0, effect_future=Ztau, weights=alphas_small, k=k)

    info_imbalances_Z_to_X = np.row_stack((info_imbalances_Z_to_Y_b, info_imbalances_Z_to_Y_f))
    info_imbalances_X_to_Z = np.row_stack((info_imbalances_Y_to_Z_b, info_imbalances_Y_to_Z_f))

    pickle.dump([taus, info_imbalances_Z_to_X, info_imbalances_X_to_Z], open(f"./pickles_noZ/ZY_seed{args.seed}_noembforZ.p","wb"))
    del info_imbalances_Z_to_Y_f, info_imbalances_Y_to_Z_f, info_imbalances_Z_to_Y_b, info_imbalances_Y_to_Z_b
