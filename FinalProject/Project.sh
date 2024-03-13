#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J MyJob
#SBATCH -o project.out -e project.err
#SBATCH --gres=gpu:1 -c 1
./YourCudaProject
