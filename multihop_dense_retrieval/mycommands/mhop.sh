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


# you can't run this on tir GPUs bc it runs out of memory unless the index is small

python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    /home/nnishika/StqaIndexToy/StqaIndexToy.npy \
    /home/nnishika/StqaIndexToy/id2doc.json \
    models/q_encoder.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base
    # --gpu
    # --save-path saving/mdr_hotpot_retrieval.json
'''
python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    # data/hotpot_index/wiki_index.npy \
    # data/hotpot_index/wiki_id2doc.json \
    /home/nnishika/StqaIndexToy/StqaIndexToy.npy \
    /home/nnishika/StqaIndexToy/id2doc.json \
    models/q_encoder.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu
    # --save-path saving/mdr_hotpot_retrieval.json
'''
