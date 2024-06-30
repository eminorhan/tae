#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=00:15:00
#SBATCH --job-name=train_segmentation
#SBATCH --output=train_segmentation_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

srun python -u train.py \
	--dataset coco \
	--data_path /vast/eo41/data/coco \
	--model deeplabv3_resnet50 \
	--weights_backbone ResNet50_Weights.IMAGENET1K_V1 \
	--batch_size_per_gpu 16 \
	--lr 0.02 \
	--aux_loss

echo "Done"