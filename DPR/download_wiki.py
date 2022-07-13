"i think this is copied from generate_dense_embeddings.py"

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple
import json

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])

    print("instantiated")

    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict) 

    print("loaded")
    
    f = open("dpr_corpus.json", "a")
    for k, v in all_passages_dict.items():
        print("k: ", k)
        print("v: ", type(v))
        
        f.write(json.dumps({"title": v.title, "text": v.text})+'\n')
        #break

    print("all passages: ", all_passages)

    f.close()
    print("wrote to file")

main()
