#!/bin/bash
#SBATCH --job-name=lor96emb #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch compute_ig_lorenz96.sh ::: 1 2 3 4 5 10 15 20 25 30 35 40 45 50 100 200 250 500 1000 1500 2000 2500 3000 4000
python3 compute_ig_lorenz96.py ${1}