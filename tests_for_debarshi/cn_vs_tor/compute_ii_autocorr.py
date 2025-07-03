import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import pickle
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed",
                    default=1, type=int, help="Label of traj ensemble")
parser.add_argument("-coord", "--coord", dest="coord",
                    default=0, help="Coordinate of CN feature")
parser.add_argument("-k", "--k", dest="k",
                    default=5, type=int, help="Number of nearest neighbors")
args = parser.parse_args()

if args.coord == "all":
    coord = np.array([0,1,2,3,4])
else:
    coord = int(args.coord)
print(f"Target CN space includes feature n. {coord}")

trajs = np.load(
    f"/scratch/vdeltatt/imbalance-gain-causality/tests_for_debarshi/data/1_cn_tor.npy"
)

# remove thermalization part
trajs = trajs[:,5:]

# set parameters, initialize variables
n_jobs = 8
t0 = (args.seed-1) * 30
taus = np.linspace(0,100,21,dtype=int)
#alphas = np.linspace(0,1,300)
info_imbalances = np.zeros((len(taus),2))

# extract Y(0)
Y0 = trajs[:,t0,coord]
for i_tau, tau in tqdm(enumerate(taus)):
    # extract Y(tau)
    Ytau = trajs[:,t0 + tau,coord]

    data = np.column_stack((Y0, Ytau))
    print(data.shape)

    # compute Delta(Y0->Ytau)
    d = MetricComparisons(data, maxk=data.shape[0]-1, n_jobs=n_jobs)
    coords1 = [0,1,2,3,4] if args.coord == "all" else [0]
    coords2 = [5,6,7,8,9] if args.coord == "all" else [1]
    info_imbalances[i_tau] = d.return_inf_imb_two_selected_coords(
        coords1=coords1, coords2=coords2, k=args.k
    )

# save data
pickle.dump([taus, info_imbalances], 
            open(f"./pickles_ii_autocorr/seed{args.seed}_coord{args.coord}_k{args.k}.p","wb"))