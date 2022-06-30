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

def fill_subq(subq, subq_answers):
    # fill in missing spots (ex: "#1") with answers from previous hops
    i = 0
    while i < len(subq):   
        if subq[i] == "#":
            j = i+1
        numstr = " "
        while j<len(subq) and isdigit(sub_q[j])
            numstr += s[j]
            j +=1
      
        if len(numstr) > 0:     
            subq_answers_idx = int(numstr) - 1
            assert(subq_answers_idx < len(subq_answers))
            subq = squbq[:i] + subq_answers[subq_answers_idx] + subq[j:]
            i = j
        else:
            i +=1
    else:
        i +=1

"""
def get_paths(subqs, subq_idx, retrieved_doc_ids, answers):

    id subq_idx >= len(subqs):
        return [[]]

    subq = subqs[subq_idx]
    

    #make query (+contexts of previous hops)
    query_components = [subq] + retrieved_doc_ids
    query = [subq] + [id2doc[str(doc_id)] for doc_id in retrieved_doc_ids]

    # encode query
    query_encodes = tokenizer.encode_plus(query, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
    query_encodes = move_to_cuda(dict(query_encodes))
    q_embed = model.encode_q(query_encodes["input_ids"], query_encodes["attention_mask"], query_encodes.get("token_type_ids", None))
    q_embed_numpy = q_embed.cpu().contiguous().numpy()

    # gets beam_size number of docs that have min dist away from question
    # D, I = distances and IDs of these^ docs    
    # shape = 1 x beam_size 
    D, I = index.search(q_embed_numpy, args.beam_size)           #TODO: figure out how to build this up

    all_query_chains = []
    all_distances = []
    all_ids = []
    for _, doc_id in enumerate(I[0]): #loops thru each doc ID from the search for b_idx^th question
        doc = id2doc[str(doc_id)]["text"]
        if "roberta" in  args.model_name and doc.strip() == "":
            doc = id2doc[str(doc_id)]["title"]
            D[0][_] = float("-inf")

        # TODO: GET ANSWER USING DPR
        # shape = 
        (_D, _I, query_chains) = get_paths(subqs, subq_idx+1, retrieved_doc_ids + [doc_id], answers + [answer])
        all_distances.append(_D)
        all_distances.append(_I)

        for query_chain in query_chains:
            all_query_chains.append(query_components + query_chain)
        
    # shape = 1 x beam_size x 1 x 1 x ...
    dims = []
    for i in range(subq_idx+1, len(subqs)):
        dims.append(i+1)
    D_expanded = np.expand_dims(D, dims)
    I_expanded = np.expand_dims(I, dims)
    D_final = np.stack(all_distances, axis=-1) + D_expanded
    I_final = np.stack(all_ids, axis=-1) + I_expanded
    return (D_final, I_final, all_query_chains)
"""

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
    ds_items = [json.loads(_) for _ in open(args.raw_data).readlines()]

    # # filter
    # if args.only_eval_ans:
    #     ds_items = [_ for _ in ds_items if _["answer"][0] not in ["yes", "no"]]

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
        id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}
    # title2text = {v[0]:v[1] for v in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")
    

    logger.info("Encoding questions and searching")
    # all questions in dataset
    # questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    for record in ds_items:
        
        subqs = ([subq[:-1] for subq in record["decomposition"]])

        metrics = []
        retrieval_outputs = []
    
        path_scores = np.array([0]) #"D_final"
        path_docs = []

        query_objects = [{
            "subq": subqs[0],
            "docs": [],
            "answers": []
        }]

        for subq_index in range(len(subqs)):

            queries = ... #extract from from query_objects
            queries_encodes = tokenizer.batch_encode_plus(queries, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            queries_encodes = move_to_cuda(dict(queries_encodes))
            q_embeds = model.encode_q(queries_encodes["input_ids"], queries_encodes["attention_mask"], queries_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()

            # gets beam_size number of docs that have min dist away from question
            # D, I = distances and IDs of these^ docs    
            # shape = num queries x beam_size 
            # num queries = beam_size ^ (subq_index)
            D, I = index.search(q_embeds_numpy, args.beam_size)
            

            new_query_objects = [] #reset queries. build for next round
            for q_idx in len(query_objects):
                old_query = queries[q_idx]
                for _, doc_id in enumerate(I[q_idx]): #loops thru each doc ID from the search for b_idx^th query
                    doc = id2doc[str(doc_id)]["text"]
                    if "roberta" in  args.model_name and doc.strip() == "":
                        doc = id2doc[str(doc_id)]["title"]
                        D[q_idx][_] = float("-inf")

                    if subq_index+1 < len(subqs):
                        # TODO: obtain answer
                        answer = ...
                        answers = old_query["answers"] + [answer]
                        subq = fill_subq(subqs[subq_idx+1], answers)
                        new_query_obj = {
                            "subq": subq,
                            "docs": old_query["docs"] + [doc_id],
                            "answers": answers
                        }
                        new_queries.append([])

            shape = [1, args.beam_size]
            for i in range(subq_index):
                shape.append(args.beam_size)
            D = D.reshape(shape)
            I = I.reshape(shape)

            # aggregate path scores
            # path_scores[a][b][c] = score of question a picking doc b 
            # (1st hop) + score of query(a+b) picking doc c (2nd hop)
            #path_scores = np.expand_dims(D, axis=2) + D_
            path_scores = np.expand_dims(path_scores, axis=-1) + D
            path_docs.append(I)

            # for idx in range(bsize): # gets top k paths for each question

            search_scores = path_scores[0]


            # i^th row = (a, b) = indices in search scores for i^th best score
            # meaning hop1 = doc a, hop2 = doc b
            # (before transpose: i^th column)
            # ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],
            #                                (args.beam_size, args.beam_size))).transpose()

            ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],
                                        shape[1:])).transpose()


            retrieved_titles = []
            hop1_titles = []
            paths, path_titles = [], []
            for _ in range(args.topk):                     
                path_ids = ranked_pairs[_] #indices in search scores for _th best score

                # hop_1_id = I[0, path_ids[0]] #doc a for hop 1
                # hop_2_id = I_[0, path_ids[0], path_ids[1]] #doc b for hop 2
                hop_ids = []
                for subq_idx in range(len(subqs)): #go through all subqs
                    indices = [0] + path_ids[:subq_idx+1]
                    hop_ids.append(path_docs[subq_idx][indices])

                # retrieved_titles.append(id2doc[str(hop_1_id)]["title"])
                # retrieved_titles.append(id2doc[str(hop_2_id)]["title"])
                # paths.append([str(hop_1_id), str(hop_2_id)])
                # path_titles.append([id2doc[str(hop_1_id)]["title"], id2doc[str(hop_2_id)]["title"]])
                paths.append([str(hop_id) for hop_id in hop_ids])
                path_titles.append([id2doc[str(hop_id)]["title"] for hop_id in hop_ids])
                for hop_id in hop_ids:
                    retrieved_titles.append(id2doc[str(hop_id)]["title"])

                
                # hop1_titles.append(id2doc[str(hop_1_id)]["title"])
            
            # if args.only_eval_ans:
            #     gold_answers = batch_ann[0]["answer"]
            #     concat_p = "yes no "
            #     for p in paths:
            #         concat_p += " ".join([id2doc[doc_id]["title"] + " " + id2doc[doc_id]["text"] for doc_id in p])
            #     metrics.append({
            #         "question": batch_ann[0]["question"],
            #         "ans_recall": int(para_has_answer(gold_answers, concat_p, simple_tokenizer)),
            #         "type": batch_ann[0].get("type", "single")
            #     })
                
            # else:
            sp = record["sp"]

            print("sp: ", sp)

            # assert len(set(sp)) == 2 #commented out bc for stqa it's not 2
            
            question = record["question"]
            type_ = record["type"]
            p_recall, p_em = 0, 0
            sp_covered = [sp_title in retrieved_titles for sp_title in sp]
            if np.sum(sp_covered) > 0:
                p_recall = 1
            if np.sum(sp_covered) == len(sp_covered):
                p_em = 1
            path_covered = [int(set(p) == set(sp)) for p in path_titles]
            path_covered = np.sum(path_covered) > 0
            recall_1 = 0
            covered_1 = [sp_title in hop1_titles for sp_title in sp]
            if np.sum(covered_1) > 0: recall_1 = 1
            metrics.append({
                "question": question,
                "p_recall": p_recall,
                "p_em": p_em,
                "type": type_,
                'recall_1': recall_1,
                'path_covered': int(path_covered)
            })


            # saving when there's no annotations
            candidate_chains = []
            for path in paths:
                # candidate_chains.append([[id2doc[path[0]], id2doc[path[1]]]])
                # unsure why it's wrapped in an extra set of square brackets? i will remove that
                candidate_chains.append([id2doc[hopid] for hop_id in path])
            
            retrieval_outputs.append({
                "_id": batch_ann[idx]["_id"],
                "question": batch_ann[idx]["question"],
                "candidate_chains": candidate_chains,
                # "sp": sp_chain,
                # "answer": gold_answers,
                # "type": type_,
                # "coverd_k": covered_k
            })

    if args.save_path != "":
        with open(args.save_path, "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")

    logger.info(f"Evaluating {len(metrics)} samples...")
    type2items = collections.defaultdict(list)
    for item in metrics:
        type2items[item["type"]].append(item)
    if args.only_eval_ans:
        logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
    else:
        logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in metrics])}')
        logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in metrics])}')
        logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in metrics])}')
        logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in type2items[t]])}')
            logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in type2items[t]])}')
            logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in type2items[t]])}')
            logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in type2items[t]])}')
