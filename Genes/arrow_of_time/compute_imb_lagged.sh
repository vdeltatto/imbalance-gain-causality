#!/bin/bash
#SBATCH --job-name=imb_lagged #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j           # Standard output
#SBATCH --error=./log_sbatch/%x.o%j            # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_imb_lagged.sh ::: 1 5 10 20 30 50 ::: 10 20 30 40 50 100
python3 compute_imb_lagged.py --k ${1} --window_length ${2} --tau_max 300