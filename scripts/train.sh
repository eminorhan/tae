#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_tae
#SBATCH --output=train_tae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# # 16 - 32 - 256
# srun python -u ../train.py \
# 	--model 'tae_base_patch16_vocab32_px256' \
# 	--resume '' \
# 	--accum_iter 1 \
# 	--batch_size_per_gpu 256 \
# 	--input_size 256 \
# 	--lr 0.0001 \
# 	--min_lr 0.0001 \
# 	--weight_decay 0.0 \
# 	--num_workers 16 \
# 	--output_dir /scratch/eo41/tae/outputs \
# 	--train_data_path /scratch/work/public/imagenet/train \
# 	--val_data_path /scratch/eo41/imagenet/val \
# 	--save_prefix tae_base_patch16_vocab32_px256

# 32 - 256 - 256
srun python -u ../train.py \
	--model 'tae_base_patch32_vocab256_px256' \
	--resume '' \
	--accum_iter 1 \
	--batch_size_per_gpu 256 \
	--input_size 256 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--weight_decay 0.0 \
	--num_workers 16 \
	--output_dir /scratch/eo41/tae/outputs \
	--train_data_path /scratch/work/public/imagenet/train \
	--val_data_path /scratch/eo41/imagenet/val \
	--save_prefix tae_base_patch32_vocab256_px256

# # 128 - 2048 - 256
# srun python -u ../train.py \
# 	--model 'tae_huge_patch128_vocab2048_px256' \
# 	--resume '' \
# 	--accum_iter 1 \
# 	--batch_size_per_gpu 256 \
# 	--input_size 256 \
# 	--lr 0.0001 \
# 	--min_lr 0.0001 \
# 	--weight_decay 0.0 \
# 	--num_workers 16 \
# 	--output_dir /scratch/eo41/tae/outputs \
# 	--train_data_path /scratch/work/public/imagenet/train \
# 	--val_data_path /scratch/eo41/imagenet/val \
# 	--save_prefix tae_huge_patch128_vocab2048_px256

echo "Done"