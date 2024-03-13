#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:03:00
#SBATCH -J task1
#SBATCH -o task1Slurm.out -e task1Slurm.err
#SBATCH -c 2

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++12 -o task1

./task1


