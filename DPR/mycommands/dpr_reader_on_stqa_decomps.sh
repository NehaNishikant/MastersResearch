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

python train_extractive_reader.py \
    prediction_results_file=dprout/dpr_reader_on_stqa_decomps.json \
    eval_top_docs=[10,20,40,50,80,100] \      
    dev_files={path to the retriever results file to evaluate} \
    model_file= /home/nnishika/DPR/dpr/downloads/ ... \
    train.dev_batch_size=80 \
    passages_per_question_predict=100 \
    encoder.sequence_length=350
