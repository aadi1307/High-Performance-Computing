#!/usr/bin/env zsh
#SBATCH --job-name=task1_thrust
#SBATCH --partition=instruction
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1_thrust.out
#SBATCH --mem=20G

module load nvidia/cuda/11.8.0


nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_thrust


values=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)


for val in "${values[@]}"; do
    ./task1_thrust $val  
done
