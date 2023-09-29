#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --partition gpu
#SBATCH --gres gpu:4
#SBATCH --time 3-00:00:00

echo STARTING

python3 -m openpifpaf.train --lr=0.001 --momentum=0.9 --b-scale=5.0 \
--epochs=1000 \
--lr-warm-up-factor=0.3 \
--batch-size=4  --val-batches=20 --val-interval=10 \
--weight-decay=1e-5 \
--dataset=openlane --openlane-upsample=2 \
--checkpoint ./outputs/fulldata_24kps/shufflenetv2k16-230923-035115-openlane-slurm1474685.pkl.epoch031 \
--openlane-train-annotations ./data_openlane_1000_24kps/annotations/openlane_keypoints_training.json \
--openlane-val-annotations ./data_openlane_1000_24kps/annotations/openlane_keypoints_validation.json \
--openlane-train-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training \
--openlane-val-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation  \
--loader-workers 1 --output ./outputs/fulldata_24kps/