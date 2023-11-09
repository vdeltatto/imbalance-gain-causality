#!/bin/bash
#SBATCH --job-name=greedy_genes #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL,END             # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch test_transferability.sh ::: 0 1 2

python3 test_transferability.py --n_coords 150 --k 10 --n_best 5 --i_fold ${1}
#python3 test_transferability.py --n_coords 30 --k 10 --n_best 20 --i_fold ${1} #--n_best 20 in tests to predict pseudotime
