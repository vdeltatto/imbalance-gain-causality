#!/bin/bash
#SBATCH --job-name=emb_test_EEG #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_ig.sh ::: {0..20}
python3 compute_ig.py --i_subject ${1} --channel_X 'P1' --channel_Y 'FC1' --tau_e 1 --k 7 --t0 50