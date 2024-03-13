#!/usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:03:00
#SBATCH --output=task3.txt

cd $SLURM_SUBMIT_DIR

g++ task3.cxx

./a.out 65536