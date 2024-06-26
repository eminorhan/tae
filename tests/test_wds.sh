#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train_tae
#SBATCH --output=train_tae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# 64
srun python -u test_wds.py \
	--batch_size_per_gpu 2048 \
	--input_size 256 \
	--num_workers 16 \
	--train_data_path "/scratch/projects/lakelab/data_frames/imagenet-21k-wds/imagenet_w21-train-{0000..2047}.tar"

echo "Done"