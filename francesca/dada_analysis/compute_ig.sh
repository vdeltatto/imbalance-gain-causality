#!/bin/bash
#SBATCH --job-name=EEG #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_ig.sh ::: {0..18} ::: CP6 C6 ::: FC2 Fz
python3 compute_ig.py --i_subject ${1} --channel_X ${2} --channel_Y ${3} --E 12 --tau_e 1 --k 20 --t0 0