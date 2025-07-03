import os
import sys
import pickle
import argparse
import numpy as np
import warnings
from joblib import Parallel, delayed
from dadapy.metric_comparisons import MetricComparisons

warnings.filterwarnings("ignore", category=UserWarning)


##############################################################################################################
#!#!!!!!!!!!!!!!!!!!!!!!!!!!!!!! main() function where all the magic happens !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#!#
##############################################################################################################
def main():
    # number of trajectories to consider
    Ntrajs = 1000

    # max number of rows of colvar*.txt files to read
    Nrows = 50000

    data_path = "../../5-nve/colvar_data/"
    file_name_list = [data_path + "colvar" + str(i) + ".txt" for i in range(0, Ntrajs, 1)]
    file_name_list_hbonds = [data_path + "hbonds_smooth_" + str(i) + ".txt" for i in range(0, Ntrajs, 1)]

    # read the initial starting time for IG estimation
    # this is to recycle the same trajectory for independent estimates
    parser = argparse.ArgumentParser()
    # initial time for tau estimation
    parser.add_argument("--ic", dest="IC", default=None, type=int)
    args = parser.parse_args()
    if args.IC is None:
        sys.exit("Error: IC (Initial Condition) must be given by user with --ic: it can be any integer number starting from 1")
    IC = args.IC

    two_pi = 2 * np.pi

    # get current working directory
    current_directory = os.getcwd()

    # Create a new directory called 'iib_data'
    save_data_directory = os.path.join(current_directory, "iib_data")
    os.makedirs(save_data_directory, exist_ok=True)
    print("Saving IIB/IG data in directory:", save_data_directory)

    test_traj = np.loadtxt(file_name_list[0])

    # number of frames of interest in each trajectory
    Nframes = test_traj.shape[0]

    # '-1' since 'time' is not a feature
    Nfeatures = test_traj.shape[1] - 1

    print(f"Number of frames/time-steps: {Nframes}")
    print(f"Number of 'features': {Nfeatures}")

    # the embedding dimension
    E = 1

    # 1 frame = 10 fs (in NVE sims)
    tau_e = 1

    # dt between each frame of simulation (in ps)
    dt = 0.01

    # decorrelation time (in ps)
    decorrelation_time = 30

    # decorrelation time (in frames)
    decorrelation_frames = int(decorrelation_time / dt)

    print(f"Decorrelation time (in ps): {decorrelation_time}")
    print(f"Decorrelation time (in frames): {decorrelation_frames}")

    # starting frame for each traj
    # discard first 5 ps (500 frames) if we start at the very beginning, could contain artifacts (nvt -> nve)
    discard_starting_frames = 500

    # initial time for simulation based on the Initial Condition (IC) index
    t_init = ((IC - 1) * decorrelation_frames) + discard_starting_frames

    # how often we check for causality (in frames) == spacing between tau's
    # every 20 frames = 20 frames * 10 fs = 200 fs = 0.2 ps
    tau_gap = 100

    # the 'tau' to start from in the range of values in (taus)
    # useful if a job has to be restarted and we want to skip the already processed 'tau' values
    tau_start = 0

    # max duration of time lag; 10K = 100 ps (in frames)
    tau_end = 10000

    # 'tau' --> how often to check for causality from initial starting frame
    taus = np.arange(tau_start, tau_end, tau_gap)

    # maximum number of neighbours to consider
    k = 5

    # range of alphas, for the driving variable, to find alpha which gives the maximum IG (imblanace gain)
    alphas = np.linspace(0.0, 1.0, 100)

    # number of jobs to parallelize over
    Njobs = 16

    # get std devs
    std_devs = get_std_devs(file_name_list_hbonds[0], Nrows)

    print("A list of standard deviations of all CVs:", std_devs)

    #!# pre-load all the data (all the colvar*.txt files)
    all_data_tor = []
    all_data_hbonds = []
    for itraj in range(0, Ntrajs, 1):
        file_name_tor = data_path + "colvar" + str(itraj) + ".txt"
        file_name_hbonds = data_path + "hbonds_smooth_" + str(itraj) + ".txt"
        tmp_data_tor = np.loadtxt(file_name_tor, max_rows=Nrows, usecols=(13, 14, 15))
        tmp_data_hbonds = np.loadtxt(file_name_hbonds, max_rows=Nrows)
        all_data_tor.append(tmp_data_tor)
        all_data_hbonds.append(tmp_data_hbonds)

    # shape: (Nrows, Nfeatures, Ntrajs)
    data_tor = np.dstack(all_data_tor)
    data_hbonds = np.dstack(all_data_hbonds)
    # shape: (Ntrajs, Nrows, Nfeatures)
    data_tor = np.rollaxis(data_tor, -1)
    data_hbonds = np.rollaxis(data_hbonds, -1)

    # deallocate list due to huge memory usage
    del all_data_tor
    del all_data_hbonds

    #!# all values must be made +ve to work with DADApy distance function
    #!# so translate the values that can be negative
    min_values = np.min(data_hbonds, axis=(0, 1))  # 1 for each feature
    # subtract either 0 (if minimum is > 0), or the most -ve value
    data_hbonds = data_hbonds[:, :, :] - np.minimum(0, min_values)

    # rescale torsions using mod(2 pi)
    data_tor = data_tor % (two_pi)

    # all data
    data_array = np.dstack([data_hbonds, data_tor])

    # find the largest value in the data-set (ignoring time column)
    # ensure that the period for non-periodic variables is > this maximum value
    large_period = 1000 * np.max(data_hbonds)

    print(f"Initial condition: {IC}")

    print(f"Constructing the time-delayed embeddings at the starting time = {t_init}")
    #!# construct X and Y at time=t0 (starting time)
    (
        tor_vec_t0,
        cn_vec_t0,
    ) = construct_Xt_Yt(
        data_array=data_array,
        Ntrajs=Ntrajs,
        t=t_init,
        E=E,
        tau_e=tau_e,
        std_devs=std_devs,
        Njobs=Njobs,
    )

    #!# build the DADApy object necessary
    d = MetricComparisons(maxk=Ntrajs - 1, n_jobs=Njobs)

    print(f"Constructing the 'ranks present' matrices at the starting time = {t_init}")

    #!# 'ranks_present' has shape (len(alphas), N, maxk+1)

    ranks_present__tor_to_cn = d.return_ranks_present_for_all_weights(
        cause_present=tor_vec_t0,
        effect_present=cn_vec_t0,
        weights=alphas,
        period_cause=1.0,
        period_effect=large_period,
    )
    ranks_present__cn_to_tor = d.return_ranks_present_for_all_weights(
        cause_present=cn_vec_t0,
        effect_present=tor_vec_t0,
        weights=alphas,
        period_cause=large_period,
        period_effect=1.0,
    )

    for tau in taus:
        print("Tau = ", tau)

        #!# construct X and Y at time=tau (the time we are checking causality at)
        (
            tor_vec_tau,
            cn_vec_tau,
        ) = construct_Xt_Yt(
            data_array=data_array,
            Ntrajs=Ntrajs,
            t=t_init + tau,
            E=E,
            tau_e=tau_e,
            std_devs=std_devs,
            Njobs=Njobs,
        )

        print(f"Starting IIB for tau={t_init+tau}, k={k}, t_init={t_init}, IC={IC}")

        #!# Torsion Vector <-> CN Vector
        iib_tor_to_cn = d.return_inf_imb_causality_input_rank(
            ranks_present=ranks_present__tor_to_cn,
            effect_future=cn_vec_tau,
            k=k,
            period_effect=large_period,
        )
        iib_cn_to_tor = d.return_inf_imb_causality_input_rank(
            ranks_present=ranks_present__cn_to_tor,
            effect_future=tor_vec_tau,
            k=k,
            period_effect=1.0,
        )

        # create file to save data in
        save_data_file = os.path.join(
            save_data_directory,
            f"iib_IC-{IC}_tau-{tau}_k-{k}.p",
        )

        # save IIB calculation
        pickle.dump(
            [
                iib_tor_to_cn,
                iib_cn_to_tor,
            ],
            open(save_data_file, "wb"),
        )

        print(f"Saved IIB data for tau={t_init+tau}, k={k}, t_init={t_init}, IC={IC}")


##############################################################################################################
##############################################################################################################


##############################################################################################################
#!#!!!!!!!!!!!!!!!!!!!!!!!! Compute standard deviations from a sample trajectory !!!!!!!!!!!!!!!!!!!!!!!!!!#!#
##############################################################################################################
def get_std_devs(file_name, Nrows):
    print("Calculating standard deviations from a FULL sample trajectory:", file_name)
    data = np.loadtxt(file_name, max_rows=Nrows)

    #!# hbonds: NT, CT, O1, O2, NE1
    cn_NT = data[:, 0]
    cn_CT = data[:, 1]
    cn_O1 = data[:, 2]
    cn_O2 = data[:, 3]
    cn_NE1 = data[:, 4]

    std_devs = [
        cn_NT.std(axis=0),
        cn_CT.std(axis=0),
        cn_O1.std(axis=0),
        cn_O2.std(axis=0),
        cn_NE1.std(axis=0),
    ]

    return std_devs


##############################################################################################################
##############################################################################################################


##############################################################################################################
#!#!!!!!!!!!!!!!!!!!!!!!!! Read colvar.txt files and create time-delayed embeddings !!!!!!!!!!!!!!!!!!!!!!!#!#
##############################################################################################################
def return_time_delayed_embeddings(data, t, E, tau_e, std_devs):
    embed_times = np.arange(t, t + (E * tau_e), tau_e)

    #!# NT, CT, O1, O2, NE1, tor1, tor2, tor3
    #!# all 4 CN components must be divided using the same stddev, to ensure consistency
    ## otherwise we end up adding noise
    cn_NT = (data[embed_times, 0]) / std_devs[0]
    cn_CT = (data[embed_times, 1]) / std_devs[0]
    cn_O1 = (data[embed_times, 2]) / std_devs[0]
    cn_O2 = (data[embed_times, 3]) / std_devs[0]
    cn_NE1 = (data[embed_times, 4]) / std_devs[0]
    #!# all torsions must be scaled by 2pi to be between (0,1)
    tor1 = (data[embed_times, 5]) / (2 * np.pi)
    tor2 = (data[embed_times, 6]) / (2 * np.pi)
    tor3 = (data[embed_times, 7]) / (2 * np.pi)

    embedded_array = np.array(
        [
            cn_NT,
            cn_CT,
            cn_O1,
            cn_O2,
            cn_NE1,
            tor1,
            tor2,
            tor3,
        ]
    )

    return embedded_array


##############################################################################################################
##############################################################################################################


##############################################################################################################
#!#!!!!!!!!!!!!!!!!!!!!!!! Construct driving and driven variable matrices at time=t !!!!!!!!!!!!!!!!!!!!!!!#!#
##############################################################################################################
def construct_Xt_Yt(data_array, Ntrajs, t, E, tau_e, std_devs, Njobs):
    #!# construct the time-delayed embedding matrices for each individual trajectory
    #!# then recompile them together to form the full dataset of shape (Ntrajs, E/tau_e) == (Ntrajs, len(embed_times))
    #!# this is done in parallel with Joblib = Njobs
    (
        cn_NT,
        cn_CT,
        cn_O1,
        cn_O2,
        cn_NE1,
        tor1,
        tor2,
        tor3,
    ) = np.swapaxes(
        Parallel(n_jobs=Njobs)(
            delayed(return_time_delayed_embeddings)(
                data=data_array[itraj, :, :],
                t=t,
                E=E,
                tau_e=tau_e,
                std_devs=std_devs,
            )
            for itraj in range(0, Ntrajs, 1)
        ),
        axis1=0,
        axis2=1,
    )

    #!# reshape all the CVs into (Ntrajs,len(embed_times))
    cn_NT = cn_NT.reshape((Ntrajs, -1))
    cn_CT = cn_CT.reshape((Ntrajs, -1))
    cn_O1 = cn_O1.reshape((Ntrajs, -1))
    cn_O2 = cn_O2.reshape((Ntrajs, -1))
    cn_NE1 = cn_NE1.reshape((Ntrajs, -1))
    tor1 = tor1.reshape((Ntrajs, -1))
    tor2 = tor2.reshape((Ntrajs, -1))
    tor3 = tor3.reshape((Ntrajs, -1))

    #!# if you want a new CV to be a combination of diff existing CVs, proceed as below
    #!# compile all the relevant CVs (in this case X,Y,Z dipole components) into a single array
    #!# this array will have shape (num_CVs, Ntrajs, len(embed_times))
    #!# remember each individual CV array has shape (Ntrajs, len(embed_times))
    #!# so, we flatten this 3D embedding array and reshape it as (Ntrajs, num_CVs*len(embed_times))
    #!# this procedure is fine since the flattening doesn't change the distances that are computed
    cn_vec = np.array([cn_NT, cn_CT, cn_O1, cn_O2, cn_NE1])
    cn_vec = cn_vec.reshape(Ntrajs, -1)
    tor_vec = np.array([tor1, tor2, tor3])
    tor_vec = tor_vec.reshape(Ntrajs, -1)
    #!# we further scale the full 3-component or 3-dim dipole vector by sqrt(3)
    #!# this is to ensure the respective 'alpha' ranges are compatible with 1-dim CVs
    cn_vec = cn_vec / np.sqrt(5)
    tor_vec = tor_vec / np.sqrt(3)
    # print("Combined vector shapes:", tor_vec.shape, cn_vec.shape)

    return (
        tor_vec,
        cn_vec,
    )


##############################################################################################################
##############################################################################################################

if __name__ == "__main__":
    main()
