#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=03:00:00
#SBATCH --job-name=encode_tae
#SBATCH --output=encode_tae_%A_%a.out
#SBATCH --array=1,4,7,10

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

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

srun python -u ../encode.py \
	--model ${MODEL} \
	--resume /scratch/eo41/tae/outputs/${MODEL}/${MODEL}_checkpoint.pth \
	--batch_size_per_gpu 256 \
	--input_size 256 \
	--maxcount 4 \
	--data_len 1281167 \
	--num_workers 16 \
	--output_dir /scratch/projects/lakelab/data_frames/imagenet-1k-processed/${MODEL} \
	--save_prefix "imagenet_1k" \
	--data_path "/scratch/projects/lakelab/data_frames/imagenet-1k-wds/imagenet1k-train-{0000..2047}.tar"
	# --data_path "/scratch/projects/lakelab/data_frames/imagenet-21k-wds/imagenet_w21-train-{0000..2047}.tar"

echo "Done"