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
#parser.add_argument("-ieps", "--ieps", dest="ieps",
#                    default=1, type=int,
#                    help="Index of eps coupling, between 0 and 29")
parser.add_argument("-case", "--case", dest="case",
                    default="confounder", type=str,
                    help="Case of third system: 'confounder', 'indirect' or 'Zuncoupled' ")
parser.add_argument("-direction", "--direction", dest="direction",
                    default="X->Y", type=str,
                    help="Direction of causality test: 'X->Y' or 'Y->X' ")
args = parser.parse_args()

assert args.case == "Zuncoupled", "This script is only for 'Zuncoupled' case!"

trajectory_z = pickle.load(open(f"../trajs_xyz/three_rosslers_Zuncoupled/seed{args.seed}.p","rb"))[:205001]
traj_length = trajectory_z.shape[0]

tau = 20
N = 5000
sample_times = np.linspace(100000,traj_length-tau-1,N,dtype=int)
k = 1
alphas_X = np.linspace(0,0.25,50)
alphas_Y = np.linspace(0,0.25,50)
alphas_Z = np.linspace(0,0.25,50)
n_jobs = 8

Z0 = trajectory_z[sample_times,7:]
Ztau = trajectory_z[sample_times+tau,7:]

for ieps in tqdm([0]):#range(30)):
    trajectory_xy = pickle.load(
        open(f"/scratch/vdeltatt/imbalance-gain-causality/fig3_errors/trajs/rossler_diff/seed{args.seed}_ieps{ieps}.p","rb")
    )
    assert trajectory_z.shape[0] == trajectory_xy.shape[0], "Error: trajs of x,y and of z have different number of points!"
    
    X0 = trajectory_xy[sample_times,1:4]
    Y0 = trajectory_xy[sample_times,4:7]

    Xtau = trajectory_xy[sample_times+tau,1:4]
    Ytau = trajectory_xy[sample_times+tau,4:7]

    d = MetricComparisons(maxk=len(sample_times)-1, njobs=n_jobs)

    if args.direction == "X->Y":
        imbs_Y_to_Y, imbs_X_to_Y = d.return_inf_imb_causality_conditioning(cause_present=X0, 
                                                                        effect_present=Y0, 
                                                                        conditioning_present=Z0,
                                                                        effect_future=Ytau, 
                                                                        weights_cause=alphas_X, 
                                                                        weights_conditioning=alphas_Z,
                                                                        k=k)
        pickle.dump([alphas_X, alphas_Y, alphas_Z, imbs_Y_to_Y, imbs_X_to_Y],
                    open(f"./pickles/rossler_{args.case}/XtoY_seed{args.seed}_ieps{ieps}.p","wb"))
        
    elif args.direction == "Y->X":
        imbs_X_to_X, imbs_Y_to_X = d.return_inf_imb_causality_conditioning(cause_present=Y0,
                                                                        effect_present=X0,
                                                                        conditioning_present=Z0,
                                                                        effect_future=Xtau,
                                                                        weights_cause=alphas_Y, 
                                                                        weights_conditioning=alphas_Z,
                                                                        k=k)
        pickle.dump([alphas_X, alphas_Y, alphas_Z, imbs_X_to_X, imbs_Y_to_X],
                    open(f"./pickles/rossler_{args.case}/YtoX_seed{args.seed}_ieps{ieps}.p","wb"))

    else:
        print("Error: '--direction' flag must be either 'X->Y' or 'Y->X'.")