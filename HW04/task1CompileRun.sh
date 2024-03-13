#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output=task1.txt

cd $SLURM_SUBMIT_DIR

g++ task1.cxx

./a.out 1024 1024