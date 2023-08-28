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
#send using parallel sbatch run_rossler_identical.sh ::: {0..25}

start_eps=0.0
end_eps=0.25
n_eps=30
stride=$(bc <<< "scale=20; (${end_eps}-${start_eps})/(${n_eps}-1)")

# Loop using floating-point arithmetic
for ((i=0; i<${n_eps}; i++)); do
    eps=$(bc <<< "scale=10; ${start_eps} + ${i} * ${stride}")
    python3 ../dynamical-systems/rossler_systems.py --seed ${1} --nsamples 205000 --epsilon ${eps} --omega_1 1.015 --omega_2 1.015 --output "./trajs/rossler_id/seed${1}_ieps${i}.p"
done

