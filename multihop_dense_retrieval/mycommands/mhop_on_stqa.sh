#!/bin/bash
#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                # Number of gpu
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=./log/%j.log   # Standard output and error log, the program output will be here

eval "$(conda shell.bash hook)"
# you environment
source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate MDR
​
export TQDM_DISABLE=1
export OMP_NUM_THREADS=1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \

# you can't run this on tir GPUs bc it runs out of memory unless the index is small

python scripts/eval/eval_mhop_retrieval.py /home/nnishika/stqaout/stqa_to_hotpot.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    models/q_encoder.pt \
    --batch-size 1 \
    --beam-size 10 \
    --topk 10 \
    --shared-encoder \
    --model-name roberta-base \
    --save-path /home/nnishika/mdrout/mdr_stqa_retrieval_top10.json
#   --gpu

