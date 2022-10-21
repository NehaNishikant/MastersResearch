#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                # Number of gpu
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=./log/%j.log   # Standard output and error log, the program output will be here

source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate MDR
​
export TQDM_DISABLE=1
export OMP_NUM_THREADS=1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \

# you can't run this on tir GPUs bc it runs out of memory unless the index is small

# --mem=300gb --gres=gpu:1

python scripts/eval/eval_mhop_queries.py /home/nnishika/mdrout/debugging_frank/frank_queries_to_hotpot.json \
    /projects/tir3/users/nnishika/StqaIndex/StqaIndex.npy \
    /projects/tir3/users/nnishika/StqaIndex/id2doc.json \
    models/q_encoder.pt \
    --batch-size 1 \
    --beam-size 2 \
#   --topk 1 \
    --shared-encoder \
    --model-name roberta-base
#   --save-path /home/nnishika/mdrout/mdr_stqa_decomps_retrieval.json
#   --gpu

