#!/bin/bash
#SBATCH --job-name=xy #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#SBATCH --dependency=afterok:10459015,10459016,10459017,10459018,10459019,10459020,10459021,10459022,10459023,10459024,10459025,10459026,10459027,10459028,10459029,10453673
#send using parallel sbatch compute_ig.sh ::: {0..15}

python3 compute_ig.py --case "XY" --ieps ${1} 