#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1.out
#SBATCH --mem=20G

module load nvidia/cuda/11.8.0


g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp


values=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)


for val in "${values[@]}"; do
    ./task1 1024 $val  
done
