#!/bin/bash
#SBATCH --job-name=TE_EEG #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_te.sh ::: {0..20} ::: CP6 C6 ::: Fz FC2
python3 compute_te.py --i_subject ${1} --channel_X ${2} --channel_Y ${3} --E 45 --tau_e 1 --k 3 --t0 50 --minzero 0