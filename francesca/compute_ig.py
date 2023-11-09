import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-subject", "--subject", dest="subject",
                    default="001", type=str,
                    help="Subject string")
parser.add_argument("-channel_X", "--channel_X", dest="channel_X",
                    default="POz", type=str,
                    help="Channel X string")
parser.add_argument("-channel_Y", "--channel_Y", dest="channel_Y",
                    default="Fz", type=str,
                    help="Channel Y string")
parser.add_argument("-E", "--E", dest="E",
                    default=30, type=int,
                    help="Embedding length")
parser.add_argument("-k", "--k", dest="k",
                    default=20, type=int,
                    help="Number of neighbors")
parser.add_argument("-t0", "--t0", dest="t0",
                    default=300, type=int,
                    help="Initial time")
args = parser.parse_args()

# read data here
# X =
# Y = 
assert X.shape == Y.shape, f"Error: shapes of X ({X.shape}) and Y ({Y.shape}) do not match!"

E = 30 # embedding length
tau_e = 1 # embedding time
t0 = 500
n_jobs = 12
alphas = np.linspace(0.,1.5,300)
taus = np.arange(0,500,5) # change it!

info_imbalances_Y_to_X = np.zeros((len(taus),len(alphas)))
info_imbalances_X_to_Y = np.zeros((len(taus),len(alphas)))

# extract time-delay embeddings at time 0
window_t0 = np.arange(t0-tau_e*(E-1),t0+tau_e,tau_e)
X0 = X[:,window_t0]
Y0 = Y[:,window_t0]
for i_tau, tau in enumerate(taus):

    # extract time-delay embeddings at time tau
    Xtau = X[:,window_t0+tau]
    Ytau = Y[:,window_t0+tau]

    # scan Information Imbalance as a function of alpha
    d = MetricComparisons(maxk=X0.shape[0]-1, njobs=n_jobs)
    info_imbalances_X_to_Y[i_tau] = d.return_inf_imb_causality(
        cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas, k=args.k)
    info_imbalances_Y_to_X[i_tau] = d.return_inf_imb_causality(
        cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas, k=args.k)

# save data
pickle.dump([taus, info_imbalances_X_to_Y, info_imbalances_Y_to_X], 
            open(f"./pickles/II_sub{args.subject}_X{args.channel_X}_Y{args.channel_Y}_E{args.E}_k{args.k}.p","wb"))