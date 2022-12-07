# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
Evaluating trained retrieval model.

Usage:
python eval_mhop_retrieval.py ${EVAL_DATA} ${CORPUS_VECTOR_PATH} ${CORPUS_DICT} ${MODEL_CHECKPOINT} \
     --batch-size 50 \
     --beam-size-1 20 \
     --beam-size-2 5 \
     --topk 20 \
     --shared-encoder \
     --gpu \
     --save-path ${PATH_TO_SAVE_RETRIEVAL}

"""

import os, sys
sys.path.append(os.getcwd()) 

import argparse
import collections
import json
import logging
from os import path
import time
import copy

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from mdr.retrieval.models.mhop_retriever import RobertaRetriever
from mdr.retrieval.utils.basic_tokenizer import SimpleTokenizer
from mdr.retrieval.utils.utils import (load_saved, move_to_cuda, para_has_answer)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=2, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-len', type=int, default=70)
    parser.add_argument('--max-c-len', type=int, default=300)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='roberta-base')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--save-index', action="store_true")
    parser.add_argument('--only-eval-ans', action="store_true")
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
    args = parser.parse_args()
    
    logger.info("Loading data...")
    # dataset items
    # ds_items = [json.loads(_) for _ in open(args.raw_data).readlines()]
    ds_items = json.load(open(args.raw_data))
    print("Dataset size: ", len(ds_items))


    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RobertaRetriever(bert_config, args)
    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()
    

    logger.info("Building index...")
    d = 768 #length of each vector
    xb = np.load(args.indexpath).astype('float32')

    index = faiss.IndexFlatIP(d)     
    index.add(xb)
    if args.gpu:         
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    if args.save_index:
        faiss.write_index(index, "data/hotpot_index/wiki_index_hnsw_roberta")
    
        
    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    if isinstance(id2doc["0"], list):
        for k, v in id2doc.items():
            if len(v) == 4:
                id2doc[k] = {"title": v[0], "text": v[1], "para_id": v[3]}
            else:
                id2doc[k] = {"title":v[0], "text": v[1]}

    logger.info(f"Corpus size {len(id2doc)}")


    logger.info("Encoding questions and searching")
    counter = 0
    retrieval_outputs = []
    for record in ds_items:
        
        print(counter)
        counter +=1

        # get rid of operation hops
        total_annotators = 3
        is_operation = [0 for i in range(len(record["decomposition"]))]
        for anno in record["evidence"]:
            for i in range(len(anno)):
                for ev in anno[i]:
                    if ev == "operation":
                        is_operation[i] += 1

        # subqs = ([subq[:-1] for subq in record["decomposition"]])
        subqs = []
        for i in range(len(record["decomposition"])):
            if is_operation[i] < total_annotators:
                subqs.append(record["decomposition"][i])
        
        # args.beam_size = args.beam_size // len(subqs)
        # args.topk = args.topk // len(subqs)

        # queries = record["subq_queries"]
        total_annotators = len(record["evidence"])
        subq_evs = [[] for i in range(total_annotators)]
        subq_queries = [[] for i in range(total_annotators)]

        Ds = [[] for i in range(total_annotators)]
        Is = [[] for i in range(total_annotators)]

        for subq_idx in range(len(subqs)):
            
            # build query from previous subq evidence
            for anno_idx in range(total_annotators):

                subq = subqs[subq_idx]
                subq_docs = [id2doc[doc_id]["text"] for doc_id in subq_evs[anno_idx]] 
                if isinstance(subq_queries[anno_idx], list):
                    if len(subq_queries[anno_idx]) > 0:
                        subq_queries[anno_idx][0] = subq
                        subq_queries[anno_idx] += subq_docs
                    else:
                        subq_queries[anno_idx] = subq
                else:
                    subq_queries[anno_idx] = [subq] + subq_docs
                

            # update subq evidence
            subq_evs = [[] for i in range(total_annotators)]             
            for anno_idx in range(total_annotators):
                subq_ev = record["evidence"][anno_idx][subq_idx]
                for psg_list in subq_ev:
                    if isinstance(psg_list, list):
                        for (psg_name, psg_id) in psg_list:
                            subq_evs[anno_idx].append(psg_id)

            print("subq queries: ", subq_queries)

            for anno_idx in range(len(subq_queries)):

                query = subq_queries[anno_idx]
                if len(query) == 0:
                    continue

                queries = [query]

                with torch.no_grad():
            
                    #encodes queries
                    queries_encodes = tokenizer.batch_encode_plus(queries, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
                    queries_encodes = move_to_cuda(dict(queries_encodes))
                    q_embeds = model.encode_q(queries_encodes["input_ids"], queries_encodes["attention_mask"], queries_encodes.get("token_type_ids", None))
                    q_embeds_numpy = q_embeds.cpu().contiguous().numpy()


                    # gets beam_size number of docs that have min dist away from question
                    # D, I = distances and IDs of these^ docs    
                    # shape = num queries x beam_size 
                    D, I = index.search(q_embeds_numpy, args.beam_size)

                    for q_idx in range(len(queries)):
                        for _, doc_id in enumerate(I[q_idx]): #loops thru each doc ID from the search for b_idx^th query
                            doc = id2doc[str(doc_id)]["text"]
                            if "roberta" in  args.model_name and doc.strip() == "":
                                doc = id2doc[str(doc_id)]["title"]
                                D[q_idx][_] = float("-inf")

                    Ds[anno_idx].append(D)
                    Is[anno_idx].append(I)


        anno_scores = [0 for i in range(total_annotators)]
        anno_docs = [[] for i in range(total_annotators)]
        for anno_idx in range(total_annotators):
            print("Ds[anno_idx], should be num_subq: ", len(Ds[anno_idx]))
            for subq_idx in range(len(Ds[anno_idx])):
                D = Ds[anno_idx][subq_idx][0]
                I = Is[anno_idx][subq_idx][0]
                if len(D) == 0:
                    continue
                print("D, should be beamsize: ", D.shape, "I: ", I.shape)
                D = [(0-D[i], i) for i in range(len(D))] #0- so sort is descending instead
                D.sort()
                D = D[:args.topk//len(subqs)+1]

                indices = [idx for (_, idx) in D]
                doc_ids = []
                for idx in indices:
                    doc_ids.append(I[idx])

                D = [0-score for (score, _) in D] #back to positive score
                subq_score = 0
                for score in D:
                    subq_score += score

                anno_scores[anno_idx] += subq_score
                anno_docs[anno_idx] += doc_ids

        max_score = (anno_scores[0], 0)
        for anno_idx in range(total_annotators):
            if anno_scores[anno_idx] > max_score[0]:
                max_score = (anno_scores[anno_idx], anno_idx)

        top_anno_docs = anno_docs[max_score[1]]
       
        retrieved_titles = []
        for doc_id in top_anno_docs:
            if "para_id" in id2doc[str(doc_id)]:
                retrieved_titles.append(id2doc[str(doc_id)]["title"] + "-" + str(id2doc[str(doc_id)]["para_id"]))
        
        retrieval_outputs.append({
            "qid": record["qid"],
            "rp": retrieved_titles,
            "subqs": subqs,
            # "_id": record["qid"],
            # "question": record["question"],
            # "candidate_chains": candidate_chains,
            # "sp": sp_chain,
            # "answer": gold_answers,
            # "type": type_,
            # "coverd_k": covered_k
        })

    if args.save_path != "":
        with open(args.save_path, "w") as out:
            json.dump(retrieval_outputs, out, indent=4)
