#!/usr/bin/bash
#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                # Number of gpu
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=./log/%j.log   # Standard output and error log, the program output will be here
​
# you can always have this
eval "$(conda shell.bash hook)"
# you environment
source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate dpr
​
export TQDM_DISABLE=1
# code

# mem=256gb

python dense_retriever.py \
    model_file=/home/nnishika/DPR/dpr/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp \
    qa_dataset=stqa_decomp \
    ctx_datatsets=[stqa_wiki] \
    encoded_ctx_files=[\"/projects/tir3/users/nnishika/StqaIndexDPR/*\"] \
    batch_size=1 \
    out_file=dprout/dpr_on_stqa_decomps.json
