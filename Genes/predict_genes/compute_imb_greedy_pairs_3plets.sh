#!/bin/bash
#SBATCH --job-name=3plets #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j           # Standard output
#SBATCH --error=./log_sbatch/%x.o%j            # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#SBATCH --dependency=afterok:10838281,10838282,10838283,10838284,10838285,10838286,10838287,10838288,10838289,10838290,10838291,10838292,10838293,10838294,10838295
#send using parallel sbatch compute_imb_greedy_pairs_3plets.sh ::: {0..9}
python3 compute_imb_greedy_pairs_3plets.py --k 30 --i_parallel ${1}