#!/bin/bash
#SBATCH --job-name=indir #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_ig_confounder_or_indirect.sh ::: {0..25} ::: 'X->Y' 'Y->X' ::: 'confounder' 'indirect'
python3 compute_ig_confounder_or_indirect.py --seed ${1} --direction ${2} --case ${3}