#!/bin/bash
#SBATCH --job-name=ros_id #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch run_rossler_fast_slow_smallertau_largerundersample.sh ::: {0..25}

# Loop over taus
tau=5
unders=20

python3 ../dynamical-systems/rossler_systems_fast_slow.py --seed ${1} --nsamples 205000 --epsilon 0.1 --omega_1 1.015 --omega_2 1.015 --tau ${tau} --undersample_factor ${unders} --output "./trajs_smallertau_largerundersample/seed${1}_tau${tau}_unders${unders}.p"