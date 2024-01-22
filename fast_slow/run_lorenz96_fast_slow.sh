#!/bin/bash
#SBATCH --job-name=lor96 #name of job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 #max time required
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_sbatch/%x.o%j              # Standard output
#SBATCH --error=./log_sbatch/%x.o%j               # Standard error
#SBATCH --mail-type=FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=vdeltatt@sissa.it    # Where to send the mail
#send using parallel sbatch run_lorenz96_fast_slow.sh ::: {0..25}

# 21 taus between 0.1 and 10 with log spacing (base 10)
taus=(0.1         0.12589254  0.15848932  0.19952623  0.25118864
      0.31622777  0.39810717  0.50118723  0.63095734  0.79432823
      1.0         1.25892541  1.58489319  1.99526231  2.51188643
      3.16227766  3.98107171  5.01187234  6.30957344  7.94328235
      10.0)

# Loop over taus
itau=0
for tau in "${taus[@]}"
do
    python3 ../dynamical-systems/lorenz_96_systems_fast_slow.py --seed ${1} --nsamples 352500 --epsilon 1.0 --tau ${tau} --output "./trajs/lorenz96/seed${1}_itau${itau}.p"
    ((itau++))
done