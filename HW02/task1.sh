#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:07:00
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
#SBATCH -c 2
#SBATCH --job-name=task1
#SBATCH --output=task1_output.txt
#SBATCH --ntasks=1

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

./task1 34

module load gcc

for exp in {10..30}; do
    n=$((2**$exp))
    echo "Running for n=$n"
    ./task1 $n >> task1_times.txt
done




