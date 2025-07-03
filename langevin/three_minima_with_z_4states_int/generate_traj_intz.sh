#!/bin/bash
#SBATCH --job-name=int_z #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch generate_traj_intz.sh ::: {0..3} ::: 0 16
#python3 generate_traj_intz.py --n_steps 5 --seed ${1} --sampling_stride 1 --int_value ${2} # for debugging
python3 generate_traj_intz.py --n_steps 5000000 --seed ${1} --sampling_stride 1000 --int_value ${2}
