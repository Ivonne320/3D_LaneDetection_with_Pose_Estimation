#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --time 3-00:00:00

echo STARTING

find /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation/ -path '**/*.jpg' | xargs -n 1 -P 8 python3 -m openpifpaf.predict \
--checkpoint ./outputs/24kps/shufflenetv2k16-231003-154606-openlane-slurm1481085.pkl.epoch048 \
--debug-indices cif:0 caf:0 \
--loader-workers=8 --force-complete-pose \
--batch-size=32 --long-edge=452 \
--image-output ./predictions/24kps-48epoch/images \
--json-output ./predictions/24kps-48epoch/jsons

