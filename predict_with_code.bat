#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --time 3-00:00:00

echo STARTING
python3 ./utils/predictor_trial.py
