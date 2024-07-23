#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --time=168:00:00
#SBATCH --job-name=train_recognition_in19k
#SBATCH --output=train_recognition_in19k_%A_%a.out
#SBATCH --array=0-11

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

srun python -u train_recognition_in19k.py \
	--encoder ${MODEL} \
	--encoder_ckpt /scratch/eo41/tae/outputs/${MODEL}/${MODEL}_checkpoint.pth \
	--model vit_recognition_numpatches256_vocab256_base \
	--model_ckpt '' \
	--num_classes 19167 \
	--batch_size 896 \
	--input_size 256 \
	--max_lr 0.0001 \
	--min_lr 0.00001 \
	--switch_it 400000 \
	--num_its 500001 \
	--num_workers 16 \
	--save_freq 100000 \
	--output_dir /scratch/eo41/tae/outputs_recognition/in19k/${MODEL} \
	--train_data_path "/scratch/projects/lakelab/data_frames/imagenet-21k-wds/imagenet_w21-train-{0000..2047}.tar" \
	--save_prefix in19k

echo "Done"