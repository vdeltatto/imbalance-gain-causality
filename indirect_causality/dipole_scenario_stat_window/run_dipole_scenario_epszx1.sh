#!/bin/bash
#SBATCH --job-name=lor96stat #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2,gpu2,long1,long2,wide1,wide2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch run_dipole_scenario_epszx1.sh ::: {0..30}

seed=${1}
python3 ../../dynamical-systems/dipole_scenario_window_z.py --nsamples 2000 --epsilon_zx 1.0 --t0 1500 --tf 1530 --Fx 6 --Fz 5 --seed ${seed} --output_folder "./trajs_epszx_1/seed${seed}" 