#!/usr/bin/bash
#SBATCH --mem=256gb #I think?                   # Job memory request
#SBATCH --gres=gpu:1                # Number of gpu
​
# you can always have this
eval "$(conda shell.bash hook)"
# you environment
source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate stqa
​
export TQDM_DISABLE=1
# code
python3 -m src.models.iterative.iterate_dataset \
    --data data/strategyqa/train.json \
    --output-file /home/nnishika/stqaout/mdr_trainfile.json \
    -g 0
