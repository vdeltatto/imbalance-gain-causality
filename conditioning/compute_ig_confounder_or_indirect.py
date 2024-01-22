import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1, type=int,
                    help="Seed of trajectory, between 0 and 25")
parser.add_argument("-case", "--case", dest="case",
                    default="confounder", type=str,
                    help="Case of third system: 'confounder', 'indirect' or 'Zuncoupled' ")
parser.add_argument("-direction", "--direction", dest="direction",
                    default="X->Y", type=str,
                    help="Direction of causality test: 'X->Y' or 'Y->X' ")
args = parser.parse_args()

assert args.case != "Zuncoupled", "This script is only for 'confounder' and 'indirect' cases!"

traj_sample = pickle.load(open(f"../trajs_xyz/three_rosslers_{args.case}/ieps0_seed0.p","rb"))
traj_length = traj_sample.shape[0]
assert traj_length == 205001, f"Unexpected {traj_length} time points!"
del traj_sample

tau = 20
N = 5000
sample_times = np.linspace(100000,traj_length-tau-1,N,dtype=int)
k = 1
alphas_X = np.linspace(0,0.25,50)
alphas_Y = np.linspace(0,0.25,50)
alphas_Z = np.linspace(0,0.25,50)
n_jobs = 8

for ieps in tqdm(range(30)):
    trajectory = pickle.load(
        open(f"../trajs_xyz/three_rosslers_{args.case}/ieps{ieps}_seed{args.seed}.p","rb")
    )
    assert trajectory.shape[0] == traj_length, f"Error: unexpected {trajectory.shape[0]} points!"
    
    X0 = trajectory[sample_times,1:4]
    Y0 = trajectory[sample_times,4:7]
    Z0 = trajectory[sample_times,7:]

    Xtau = trajectory[sample_times+tau,1:4]
    Ytau = trajectory[sample_times+tau,4:7]
    Ztau = trajectory[sample_times+tau,7:]

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)

    if args.direction == "X->Y":
        imbs_X_to_Y_nocond = d.return_inf_imb_causality(
            cause_present=X0, effect_present=Y0, effect_future=Ytau, weights=alphas_X, k=k)
        
        imbs_Y_to_Y, imbs_X_to_Y = d.return_inf_imb_causality_conditioning(cause_present=X0, 
                                                                        effect_present=Y0, 
                                                                        conditioning_present=Z0,
                                                                        effect_future=Ytau, 
                                                                        weights_cause=alphas_X, 
                                                                        weights_conditioning=alphas_Z,
                                                                        k=k)

        pickle.dump([alphas_X, alphas_Y, alphas_Z, imbs_X_to_Y_nocond, imbs_Y_to_Y, imbs_X_to_Y],
                    open(f"./pickles/rossler_{args.case}/XtoY_seed{args.seed}_ieps{ieps}.p","wb"))

    elif args.direction == "Y->X":
        imbs_Y_to_X_nocond = d.return_inf_imb_causality(
            cause_present=Y0, effect_present=X0, effect_future=Xtau, weights=alphas_Y, k=k)

        imbs_X_to_X, imbs_Y_to_X = d.return_inf_imb_causality_conditioning(cause_present=Y0,
                                                                        effect_present=X0,
                                                                        conditioning_present=Z0,
                                                                        effect_future=Xtau,
                                                                        weights_cause=alphas_Y, 
                                                                        weights_conditioning=alphas_Z,
                                                                        k=k)
        pickle.dump([alphas_X, alphas_Y, alphas_Z, imbs_Y_to_X_nocond, imbs_X_to_X, imbs_Y_to_X],
                    open(f"./pickles/rossler_{args.case}/YtoX_seed{args.seed}_ieps{ieps}.p","wb"))

    else:
        print("Error: '--direction' flag must be either 'X->Y' or 'Y->X'.")