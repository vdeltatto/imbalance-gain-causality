#!/bin/bash
#SBATCH --job-name=taux #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_ig_xy_taux.sh ::: {0..30} ::: 10 20 30 40 50 60 70 80

python3 compute_ig.py --seed ${1} --coupling "noZ" --tau_X0 ${2} --iY 2