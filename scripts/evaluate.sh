#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=00:15:00
#SBATCH --job-name=evaluate_tae
#SBATCH --output=evaluate_tae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# # 16 - 128 - 256
# srun python -u ../train.py \
# 	--model 'tae_base_patch16_vocab128_px256' \
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
# 	--save_prefix tae_base_patch16_vocab128_px256

# 32 - 1024 - 256
srun python -u ../evaluate.py \
	--model 'tae_base_patch32_vocab1024_px256' \
	--resume ../outputs/tae_base_patch32_vocab1024_px256_checkpoint.pth \
	--batch_size_per_gpu 5000 \
	--input_size 256 \
	--num_workers 16 \
	--output_dir ../outputs \
	--val_data_path /scratch/eo41/imagenet/val \
	--save_prefix tae_base_patch32_vocab1024_px256 \
	--display

# # 64 - 8192 - 256
# srun python -u ../train.py \
# 	--model 'tae_base_patch64_vocab8192_px256' \
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
# 	--save_prefix tae_base_patch64_vocab8192_px256

echo "Done"