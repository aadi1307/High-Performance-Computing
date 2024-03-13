#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:07:00
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
#SBATCH -c 2

g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3
./task3


