#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --time 3-00:00:00

echo STARTING

python3 -m torch.distributed.launch \
-m openpifpaf.train --ddp --lr=0.001 --momentum=0.9 --b-scale=5.0 \
--epochs=1000 \
--lr-warm-up-factor=0.3 \
--batch-size=32  --val-batches=20 --val-interval=10 \
--weight-decay=1e-5 \
--dataset=openlane --openlane-upsample=2 \
--checkpoint=./outputs/uniform_24kps_25/shufflenetv2k16-230930-001818-openlane-slurm1478977.pkl.epoch005 \
--openlane-train-annotations ./data_uniform_24kps_quarter/annotations/openlane_keypoints_sample_10training.json \
--openlane-val-annotations ./data_uniform_24kps_quarter/annotations/openlane_keypoints_sample_10validation.json \
--openlane-train-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training \
--openlane-val-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation  \
--loader-workers 1 --output ./outputs/uniform_24kps_25/