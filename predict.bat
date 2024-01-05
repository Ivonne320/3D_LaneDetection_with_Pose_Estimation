#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 48G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00

echo STARTING


find /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation/ -path '**/*.jpg' | xargs -n 1 -P 8 python3 -m openpifpaf.predict --checkpoint /home/yiwang/CIVIL-459-Project/outputs/48kps/v16-sample/coco_same/shufflenetv2k16-240102-003036-openlane-slurm1546464.pkl.epoch069 \
--debug-indices cif:0 caf:0 \
--loader-workers=8 --force-complete-pose --instance-threshold=0.1 \
--batch-size=32 \
--long-edge=846 \
--json-output=./predictions/48kps-69epoch-eval/ \





