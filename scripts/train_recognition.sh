#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=750GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train_recognition
#SBATCH --output=train_recognition_%A_%a.out
#SBATCH --array=1

MODELS=(
	tae_patch16_vocab16_px256
	tae_patch16_vocab64_px256
	tae_patch16_vocab256_px256
	tae_patch32_vocab64_px256
	tae_patch32_vocab256_px256
	tae_patch32_vocab1024_px256
	tae_patch64_vocab256_px256
	tae_patch64_vocab1024_px256
	tae_patch64_vocab4096_px256
	tae_patch128_vocab1024_px256
	tae_patch128_vocab4096_px256
	tae_patch128_vocab16384_px256
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# 21k
srun python -u ../train_recognition.py \
	--model vit_recognition_numpatches256_vocab64_small \
	--model_ckpt '' \
	--num_classes 1000 \
	--accum_iter 1 \
	--batch_size 256 \
	--input_size 256 \
	--lr 0.0001 \
	--weight_decay 0.0 \
	--save_freq 1000 \
	--output_dir /scratch/eo41/tae/outputs_recognition/${MODEL} \
	--train_data_path "/scratch/projects/lakelab/data_frames/imagenet-1k-processed/${MODEL}/imagenet_1k_train_tae_patch16_vocab64_px256.pth" \
	--val_data_path "/scratch/projects/lakelab/data_frames/imagenet-1k-processed/${MODEL}/imagenet_1k_val_tae_patch16_vocab64_px256.pth" \
	--save_prefix imagenet_1k_${MODEL} \
	--compile

echo "Done"