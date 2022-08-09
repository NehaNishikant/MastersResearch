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
conda activate base
​
export TQDM_DISABLE=1
# code

git add strategyqa
git add multihop_dense_retrieval/scripts
git add multihop_dense_retrieval/mdr
git add multihop_dense_retrieval/mycommands
git add DPR/dpr
git add DPR/mycommands
git add DPR/conf
git add DPR/dense_retriever.py
git add DPR/download_wiki.py
git add DPR/generate_dense_embeddings.py
git add DPR/setup.py
git add DPR/train_dense_encoder.py
git add DPR/train_extractive_reader.py
git add data_utils.py
git add data_utils.sh
git add git_add.sh
git add requirements_base.txt
git add requirements_mdr.txt
git add requirements_dpr.txt
git add requirements_stqa.txt
git add test/readme.txt
