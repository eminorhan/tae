#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --time=168:00:00
#SBATCH --job-name=train_recognition_in1k
#SBATCH --output=train_recognition_in1k_%A_%a.out
#SBATCH --array=0-11

ENCODERS=(
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
ENCODER=${ENCODERS[$SLURM_ARRAY_TASK_ID]}

MODELS=(
	vit_recognition_numpatches256_vocab16_base
	vit_recognition_numpatches256_vocab64_base
	vit_recognition_numpatches256_vocab256_base
	vit_recognition_numpatches64_vocab64_base
	vit_recognition_numpatches64_vocab256_base
	vit_recognition_numpatches64_vocab1024_base
	vit_recognition_numpatches16_vocab256_base
	vit_recognition_numpatches16_vocab1024_base
	vit_recognition_numpatches16_vocab4096_base
	vit_recognition_numpatches4_vocab1024_base
	vit_recognition_numpatches4_vocab4096_base
	vit_recognition_numpatches4_vocab16384_base
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

srun python -u train_recognition_in1k.py \
	--encoder ${ENCODER} \
	--encoder_ckpt /scratch/eo41/tae/outputs/${ENCODER}/${ENCODER}_checkpoint.pth \
	--model ${MODEL} \
	--model_ckpt /scratch/eo41/tae/outputs_recognition/in19k/${MODEL}/in19k_${MODEL}_checkpoint.pth \
	--num_classes 1000 \
	--epochs 100 \
	--batch_size 896 \
	--input_size 256 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir /scratch/eo41/tae/outputs_recognition/in1k/${MODEL} \
	--train_data_path /scratch/work/public/imagenet/train \
	--val_data_path /scratch/eo41/imagenet/val \
	--save_prefix in1k

echo "Done"