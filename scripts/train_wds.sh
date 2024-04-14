#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=16:00:00
#SBATCH --job-name=train_tae_wds
#SBATCH --output=train_tae_wds_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

MODELS=(
	tae_base_patch16_vocab16_px256
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# 21k
srun python -u ../train_wds.py \
	--model ${MODEL} \
	--resume '' \
	--accum_iter 1 \
	--batch_size_per_gpu 256 \
	--input_size 256 \
	--lr 0.0001 \
	--weight_decay 0.0 \
	--num_workers 16 \
	--save_freq 10000 \
	--output_dir /scratch/eo41/tae/outputs/${MODEL} \
	--train_data_path "/scratch/projects/lakelab/data_frames/imagenet-21k-wds/imagenet_w21-train-{0000..2047}.tar" \
	--val_data_path /scratch/eo41/imagenet/val \
	--save_prefix ${MODEL} \
	--display \
	--compile

# # 1k
# srun python -u ../train_wds.py \
# 	--model ${MODEL} \
# 	--resume '' \
# 	--accum_iter 1 \
# 	--batch_size_per_gpu 256 \
# 	--input_size 256 \
# 	--lr 0.0001 \
# 	--min_lr 0.0001 \
# 	--weight_decay 0.0 \
# 	--num_workers 16 \
# 	--save_freq 10000 \
# 	--output_dir /scratch/eo41/tae/outputs/${MODEL}_1k \
# 	--train_data_path "/scratch/projects/lakelab/data_frames/imagenet-1k-wds/imagenet1k-train-{0000..1023}.tar" \
# 	--val_data_path /scratch/eo41/imagenet/val \
# 	--save_prefix ${MODEL}_1k \
# 	--display

echo "Done"