#!/usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --partition=instruction
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=4
#SBATCH --output=task3.out
#SBATCH --mem=20G

module load nvidia/cuda/11.8.0


g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

./task3

