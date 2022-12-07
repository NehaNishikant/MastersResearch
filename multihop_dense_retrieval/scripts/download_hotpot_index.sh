#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
#!/bin/bash 
# Make data and model folder. 
mkdir data
mkdir models

# Download data 
cd data

mkdir hotpot_index
cd hotpot_index
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_id2doc.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_index.npy

echo "Finished downloading data!"

