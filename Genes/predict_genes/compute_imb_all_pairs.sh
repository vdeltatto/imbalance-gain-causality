#!/bin/bash
#SBATCH --job-name=matrix #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j           # Standard output
#SBATCH --error=./log_sbatch/%x.o%j            # Standard error
#SBATCH --mail-type=FAIL,BEGIN                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#SBATCH --dependency=afterok:10732572
#send using sbatch compute_imb_all_pairs.sh
python3 compute_imb_all_pairs.py --n_coords 30 --k 30 --n_best 5