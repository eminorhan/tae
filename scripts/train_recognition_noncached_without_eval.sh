#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_recognition_noncached_without_eval
#SBATCH --output=train_recognition_noncached_without_eval_%A_%a.out
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

srun python -u ../train_recognition_noncached_without_eval.py \
	--encoder ${MODEL} \
	--encoder_ckpt /scratch/eo41/tae/outputs/${MODEL}/${MODEL}_checkpoint.pth \
	--model vit_recognition_numpatches256_vocab64_base \
	--model_ckpt '' \
	--num_classes 19167 \
	--batch_size 896 \
	--input_size 256 \
	--num_workers 16 \
	--save_freq 10000 \
	--output_dir /scratch/eo41/tae/outputs_recognition/${MODEL} \
	--train_data_path "/scratch/projects/lakelab/data_frames/imagenet-21k-wds/imagenet_w21-train-{0000..2047}.tar" \
	--save_prefix imagenet_21k

echo "Done"