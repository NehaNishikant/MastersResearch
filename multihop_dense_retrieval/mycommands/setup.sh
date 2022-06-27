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

'''
pip install -r requirements.txt
conda install faiss-gpu -c pytorch #cudatoolkit=10.2 -c pytorch
conda install pytorch -c pytorch #cudatoolkit=10.2 -c pytorch
'''
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./ #--global-option="--cpp_ext" ./

python setup.py develop
