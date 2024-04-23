#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=00:05:00
#SBATCH --job-name=test_dtype
#SBATCH --output=test_dtype_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# 32 - 1024 - 256
srun python -u ../evaluate.py \
	--model 'tae_patch16_vocab16_px256' \
	--resume ../outputs/tae_patch16_vocab16_px256/tae_patch16_vocab16_px256_checkpoint.pth \
	--batch_size_per_gpu 128 \
	--input_size 256 \
	--num_workers 16 \
	--val_data_path /scratch/eo41/imagenet/val \

echo "Done"