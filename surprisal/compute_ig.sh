#!/bin/bash
#SBATCH --job-name=s_vs_fmri #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_ig.sh ::: F H I J K L M N ::: all PostTemp AntTemp AngularG IFG MFG IFGorb pCingulate

python3 compute_ig.py --k 25 --E_X 3 --E_Y 3 --tau_e 1 --part ${1} --area ${2}