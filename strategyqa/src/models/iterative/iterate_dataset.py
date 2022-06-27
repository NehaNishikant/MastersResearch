"""
formats and adds information to stqa's trainfile so 
that mdr can be finetunes in stqa. this includes 
adding positive and negative paragaphs for each
question.
"""

# copied from src/models/iterative/run_model.py

import torch
import json
import logging
import time
from copy import deepcopy
from typing import Optional

from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.training.metrics import BooleanAccuracy

from tqdm import tqdm

import os
print(os.getcwd())

from src.data.dataset_readers.strategy_qa_reader import StrategyQAReader
from src.models.iterative.reference_utils import (
    fill_in_references,
    get_reachability,
)

logger = logging.getLogger(__name__)

def main(
    gpu: int,
    generated_decompositions_paths: Optional[str],
    data: str,
    output_file: str,
    overrides="{}",
):
    import_module_and_submodules("src")

    logger.info("src imported")

    overrides_dict = {}
    overrides_dict.update(json.loads(overrides))

    dataset_reader_bm25 = StrategyQAReader(paragraphs_source="IR-Q")
    dataset_reader_gold = StrategyQAReader(paragraphs_source="ORA-P")

    logger.info("dataset readers initialized")

    logger.info("Reading the dataset:")
    logger.info("Reading file at %s", data)
    dataset = None
    with open(data, mode="r", encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    logger.info("dataset loaded")

    output = []
    bad_bm25_count = 0
    bad_gold_count = 0
    for json_obj in tqdm(dataset):
        item = dataset_reader_bm25.json_to_item(json_obj)
        paragraphs_bm25 = []
        try:
            paragraphs_bm25 = dataset_reader_bm25.get_paragraphs(**item)["unified"]
        except:
            bad_bm25_count +=1

        item = dataset_reader_gold.json_to_item(json_obj)
        parargaphs_pos = []
        try:
            paragraphs_pos = dataset_reader_gold.get_paragraphs(**item)["unified"]
        except:
            bad_gold_count +=1



        #set minus
        pos_titles = []
        for para in paragraphs_pos:
            # print("para: ", para)
            pos_titles.append(para["title"])
        paragraphs_neg = []
        for para in paragraphs_bm25:
            if para["title"] not in pos_titles:
                paragraphs_neg.append(para)

        record = {
                "question": json_obj["question"], 
                "answers": [json_obj["answer"]],
                "_id": json_obj["qid"],
                "type": None, #for now
                "pos_paras": paragraphs_pos,
                "neg_paras": paragraphs_neg,
                }
        output.append(record)

        # break #comment later

    logger.info("dataset read")

    print("bad bm25 count: ", bad_bm25_count)
    print("bad gold count: ", bad_gold_count)

    if output_file != None:
        f = open(output_file, "w")
        json.dump(output, f)
    
    logger.info("wrote to file")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parse = argparse.ArgumentParser(description="Iterative model for StrategyQA")
    parse.add_argument("-g", "--gpu", type=int, default="", help="CUDA device")
    parse.add_argument("--generated-decompositions-paths", type=str, nargs="*")
    parse.add_argument("--data", type=str)
    parse.add_argument("--output-file", type=str)
    parse.add_argument("-o", "--overrides", type=str, default="{}", help="Overrides")
    args = parse.parse_args()

    main(**vars(args))
