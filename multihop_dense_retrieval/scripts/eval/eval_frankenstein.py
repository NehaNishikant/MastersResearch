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

def build_query(query_obj):
    
    q = [query_obj["subq"]]
    for doc_id in query_obj["docs"]:
        doc = id2doc[str(doc_id)]["text"]
        if "roberta" in  args.model_name and doc.strip() == "":
            doc = id2doc[str(doc_id)]["title"]

        q.append(doc)

    if len(q) == 1:
        return q[0]
    return tuple(q)

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
            if is_operation[i] < total_annotators/2:
                subqs.append(record["decomposition"][i])
        
        args.beam_size = 10 # // len(subqs)
        args.topk = 10 # // len(subqs)

        # metrics = []
    
        path_scores = np.array([0]) #"D_final"
        path_docs = []

        query_objects = [{
            "subq": subqs[0],
            "docs": []
            # "answers": []
        }]

        import json
        # f = open("/home/nnishika/mdrout/frank_queries.json", "w")
        # d = {}
        for subq_index in range(len(subqs)):

            # print("subq index: ", subq_index)

            with torch.no_grad():

                #builds and checks queries
                queries = [build_query(query_obj) for query_obj in query_objects]
                # print("queries: ", queries)
                # d[subq_index] = queries

                assert(len(queries) == args.beam_size ** subq_index)
                for q in query_objects:
                    assert(len(q["docs"]) == subq_index)
        
                #encodes queries
                queries_encodes = tokenizer.batch_encode_plus(queries, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
                queries_encodes = move_to_cuda(dict(queries_encodes))
                q_embeds = model.encode_q(queries_encodes["input_ids"], queries_encodes["attention_mask"], queries_encodes.get("token_type_ids", None))
                q_embeds_numpy = q_embeds.cpu().contiguous().numpy()

                # test
                f_embeds = open("encoded_query_frank_"+str(subq_index)+".npy", "wb")
                np.save(f_embeds, q_embeds_numpy)
                f_embeds.close()
                # quit()

                # gets beam_size number of docs that have min dist away from question
                # D, I = distances and IDs of these^ docs    
                # shape = num queries x beam_size 
                # num queries = beam_size ^ (subq_index)
                D, I = index.search(q_embeds_numpy, args.beam_size)

                # print("I: ", I)
                

                new_query_objects = [] #reset queries. build for next round
                for q_idx in range(len(query_objects)):
                    old_query_obj = query_objects[q_idx]
                    for _, doc_id in enumerate(I[q_idx]): #loops thru each doc ID from the search for b_idx^th query
                        doc = id2doc[str(doc_id)]["text"]
                        if "roberta" in  args.model_name and doc.strip() == "":
                            doc = id2doc[str(doc_id)]["title"]
                            D[q_idx][_] = float("-inf")

                        if subq_index+1 < len(subqs):
                            # TODO: obtain answer
                            # answer = ...
                            # answers = old_query_obj["answers"] + [answer]
                            subq = subqs[subq_index+1] #fill_subq(subqs[subq_idx+1], answers)

                            new_query_obj = {
                                "subq": subq,
                                "docs": old_query_obj["docs"] + [doc_id]
                                # "answers": answers
                            }
                            new_query_objects.append(new_query_obj)

                query_objects = new_query_objects

                shape = [1, args.beam_size]
                for i in range(subq_index):
                    shape.append(args.beam_size)
                D = D.reshape(shape)
                I = I.reshape(shape)

                # aggregate path scores
                # path_scores[a][b][c] = score of question a picking doc b 
                # (1st hop) + score of query(a+b) picking doc c (2nd hop)
                # path_scores = np.expand_dims(D, axis=2) + D_

                assert((np.asarray(np.expand_dims(path_scores, axis=-1).shape) == np.asarray(shape[:-1]+[1])).all())
                path_scores = np.expand_dims(path_scores, axis=-1) + D
                assert((np.asarray(path_scores.shape) == np.asarray(D.shape)).all())
                path_docs.append(np.copy(I))

                # for idx in range(bsize): # gets top k paths for each question


        # json.dump(d, f)
        # f.close()


        #path scores assembled for all hops. Now pick the best path.
        search_scores = path_scores[0]
        assert((np.asarray(search_scores.shape) == np.asarray(shape[1:])).all())


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
        ranked_pairs = ranked_pairs[:args.topk]
        assert((np.asarray(ranked_pairs.shape) == np.asarray([args.topk, len(subqs)])).all())
        for path_ids in ranked_pairs:                     
            # ranked_pairs[_] = indices in search scores for _th best score

            # hop_1_id = I[0, path_ids[0]] #doc a for hop 1
            # hop_2_id = I_[0, path_ids[0], path_ids[1]] #doc b for hop 2
            hop_ids = []
            
            for subq_idx in range(len(subqs)): #go through all subqs
                indices = np.array([[x] for x in path_ids[:subq_idx+1]])
                flat_indices = np.ravel_multi_index(indices, path_docs[subq_idx][0].shape)
                hop_ids.append(np.take_along_axis(path_docs[subq_idx][0], flat_indices, axis=None)[0])

            # print("hop ids: ", hop_ids)
            # retrieved_titles.append(id2doc[str(hop_1_id)]["title"])
            # retrieved_titles.append(id2doc[str(hop_2_id)]["title"])
            # paths.append([str(hop_1_id), str(hop_2_id)])
            # path_titles.append([id2doc[str(hop_1_id)]["title"], id2doc[str(hop_2_id)]["title"]])
            paths.append([str(hop_id) for hop_id in hop_ids])
            path_titles.append([id2doc[str(hop_id)]["title"] for hop_id in hop_ids])
            for hop_id in hop_ids:
                if "para_id" in id2doc[str(hop_id)]:
                    retrieved_titles.append(id2doc[str(hop_id)]["title"] + "-" + str(id2doc[str(hop_id)]["para_id"]))
        
            """ 
            print("record: ", record)
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
            """

            # saving when there's no annotations
            candidate_chains = []
            for path in paths:
                # candidate_chains.append([[id2doc[path[0]], id2doc[path[1]]]])
                # unsure why it's wrapped in an extra set of square brackets? i will remove that
                candidate_chains.append([id2doc[hop_id] for hop_id in path])
        
        retrieval_outputs.append({
            "qid": record["qid"],
            "rp": retrieved_titles,
            "subqs": subqs,
            # "_id": record["qid"],
            # "question": record["question"],
            "candidate_chains": candidate_chains,
            # "sp": sp_chain,
            # "answer": gold_answers,
            # "type": type_,
            # "coverd_k": covered_k
        })

    if args.save_path != "":
        with open(args.save_path, "w") as out:
            json.dump(retrieval_outputs, out, indent=4)
            # for l in retrieval_outputs:
            #     out.write(json.dumps(l) + "\n")

    """
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
            logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in type2items[t]])}')"""
