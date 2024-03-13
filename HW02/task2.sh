#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:03:00
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
#SBATCH -c 2

g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2
./task2 6 8


