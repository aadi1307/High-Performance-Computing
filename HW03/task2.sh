#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:03:00
#SBATCH -J task2
#SBATCH -o task2Slurm.out -e task2Slurm.err
#SBATCH -c 2

nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++12 -o task2

./task2


