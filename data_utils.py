import pickle as pkl
import json
import os
import csv

def stqa_to_dpr(infile, outfile):

    f = open(infile, 'r')
    data_in = json.load(f)
    
    data_out = {}

    for record in data_in:

        data_out[record['question']] = []

        #if 'albany' in record['question'].lower():
        print(record['question'])
        
        for ctx in record['ctxs']:
            d = {
                "evidence_id": ctx["title"],
                "title": ctx["title"],
                "score": ctx["score"],
                "section": None,
                "content": ctx["text"], 
            }

            data_out[record['question']].append(d)

    f2 = open(outfile, 'w')
    #json.dump(data_out, f2, indent=4)


#stqa_to_dpr('out.json', 'stqa_from_dpr.json')

"""
saves a smaller copy of stqa corpus of up to max_lines passages
"""
def stqa_toy_corp(out, max_lines):
    f_in = open('/projects/tir3/users/nnishika/stqa_corpus.json', 'r')
    f_out = open(out, "w")

    counter = 0
    lines = []
    for line in f_in.readlines():
        lines.append(line)
        counter +=1
        if counter > max_lines:
            break

    f_out.writelines(lines)
    f_in.close()
    f_out.close()

# stqa_toy_corp('stqa_corpus_toy.json', 10)

def get_gold_paras_for_stqa_anno(anno):
    para_names = set()
    for ev in anno:
        for psg_list in ev:
            if isinstance(psg_list, list):
                for psg_name in psg_list:
                    para_names.add(psg_name)

    return list(para_names)

"""
gets gold paragraphs for a stqa record
in the dev dataset.
"""
def get_gold_paras_for_stqa_record(record):
    para_names = set()

    for anno in record["evidence"]:
        for ev in anno:
            for psg_list in ev:
                if isinstance(psg_list, list):
                    for psg_name in psg_list:
                        para_names.add(psg_name)
    
    return list(para_names)

def get_gold_ids_for_updated_stqa_record(record):
    psg_ids = set()

    for anno in record["evidence"]:
        for ev in anno:
            for psg_list in ev:
                if isinstance(psg_list, list):
                    for (psg_name, psg_id) in psg_list:
                        psg_ids.add(psg_id)
    
    return list(psg_ids)

"""
get stqa's dataset in a format to mdr codebase's liking
"""
def stqa_to_mdr():

    f = open("strategyqa/data/strategyqa/dev.json", 'r')
    data_in = json.load(f)
    
    lines = []

    for record in data_in:

       new_record = {
           "question": record["question"],
           "answer": record["answer"],
           "sp": get_gold_paras_for_stqa_record(record),
           "type": None
           }
       lines.append(json.dumps(new_record))

    f_out = open("stqaout/stqa_to_hotpot_noduplicates.json", "w")
    f_out.writelines(lines)

    f.close()
    f_out.close()

# stqa_to_mdr()

"""
get stqa's dataset in a format to mdr codebase's liking
"""
def stqa_decomps_to_mdr():

    f = open("strategyqa/data/strategyqa/dev.json", 'r')
    data_in = json.load(f)[0]
    
    lines = []

    for i in range(len(data_in["decomposition"])):
        subq = data_in["decomposition"][i]

        gold_paras = []
        for anno in data_in["evidence"]:
            ev = anno[i]
            for psg_list in ev:
                if isinstance(psg_list, list):
                    for psg_name in psg_list:
                        gold_paras.append(psg_name)
        if len(gold_paras) > 0:
            new_record = {
                "question": subq,
                "answer": None,
                "sp": gold_paras, 
                "type": None
                }
            lines.append(json.dumps(new_record)+"\n")

    f_out = open("stqaout/stqa_decomps_to_hotpot.json", "w")
    f_out.writelines(lines)

    f.close()
    f_out.close()

# stqa_decomps_to_mdr()

"""
get stqa's corpus in a format to mdr codebase's liking
"""
def stqa_corpus_to_mdr():

    f_in = open('/projects/tir3/users/nnishika/corpus-enwiki-20200511-cirrussearch-parasv2.jsonl', 'r')
    #data = json.load(f_in)

    f_out = open("/projects/tir3/users/nnishika/stqa_corpus.json", "w")

    lines = []
    for line in f_in.readlines():
        v = json.loads(line)
        lines.append(json.dumps({"title": v["title"], "text": v["para"], "id": v["para_id"]})+'\n')

    f_out.writelines(lines)
    f_in.close()
    f_out.close()

# stqa_corpus_to_mdr()

def dpr_to_stqa():

    f_in = open('out-stqa.json', 'r')
    f_out = open("dpr_on_stqa.json", "w")

    data_in = json.load(f_in)
    data_out = {}

    for record in data_in:
        data_out[record['question']] = []
        for ctx in record['ctxs']:
            evidence = {"title": ctx["title"], "content": ctx["text"], "score": ctx["score"]}
            data_out[record['question']].append(evidence)

    f_out.write(json.dumps(data_out, indent=4))

#dpr_to_stqa()

def stqa_stats():

    f = open("stqa_to_hotpot.json", "r")

    f_out = open("stqa_stats.csv", "w")
    writer = csv.writer(f_out)
    writer.writerow(["Question", "Answer", "True Passages", "DPR Passages", "BM25 Passages", "DPR Recall", "BM25 Recall", "DPR correct", "BM25 Correct"])

    f_dpr_passages = open("strategyqa/dpr-retrieved.json")
    dpr_passages = json.load(f_dpr_passages)
    f_bm25_passages = open("strategyqa/retrieved.json")
    bm25_passages=json.load(f_bm25_passages)

    f_dpr_correct = open("strategyqa/dpr_on_stqa_preds.jsonl", "r")
    dpr_correct = f_dpr_correct.readlines()
    f_BM25_correct = open("strategyqa/stqa_preds.jsonl", "r")
    bm25_correct = f_BM25_correct.readlines()

    def is_correct(record, true):
        d = json.loads(record)
        pred = d["label"]        
        if (pred and true) or (not pred and not true):
            return 1
        else:
            return 0

    def recall(relevant_paragraphs, retrieved_paragraphs):
        result = len(set(relevant_paragraphs).intersection(retrieved_paragraphs)) / len(
            relevant_paragraphs
        )
        return result


    i = 0
    for line in f.readlines():
        record = json.loads(line)
        coarse_evidence = []
        for e in record['sp']:
            if str.isnumeric(e[-1]):
                idx = len(e)
                while str.isnumeric(e[idx-1]):
                    idx-=1                    
                    coarse_evidence.append(e[:idx-1]) #-1 to get rid of dash as well
            else:
                coarse_evidence.append(e)

        coarse_bm25 = []
        for e in bm25_passages[record["_id"]]:
            if str.isnumeric(e[-1]):
                idx = len(e)
                while str.isnumeric(e[idx-1]):
                    idx-=1                    
                    coarse_bm25.append(e[:idx-1]) #-1 to get rid of dash as well
            else:
                coarse_bm25.append(e)

        dpr_passages_record = dpr_passages[record["_id"]]
        answer = record["answer"]
        writer.writerow([record["question"], answer, record['sp'], dpr_passages_record, bm25_passages[record["_id"]], recall(coarse_evidence, dpr_passages_record), recall(coarse_evidence, coarse_bm25), is_correct(dpr_correct[i], answer), is_correct(bm25_correct[i], answer)])
        i +=1

# stqa_stats()

"""
outputs title to text dictionary of stqa's corpus.
ARCHIVED.
"""
def title_to_text():

    f_in = open('/projects/tir3/users/nnishika/corpus-enwiki-20200511-cirrussearch-parasv2.jsonl', 'r')

    f_out = open("stqa_title_to_text.json", "w")

    d = {}

    for line in f_in.readlines():
        v = json.loads(line)
        title = v["title"]+"-"+str(v["para_id"])
        text = v["para"]
        d[title] = text

    print(len(d))
    json.dump(d, f_out)

    f_in.close()
    f_out.close()

# title_to_text()

"""
creates a dictionary mapping each stqa train question to
its index in the trainfile (that I have created), which is
in the same order as the stqa train file. ARCHIVED.
"""
def stqa_to_mdr_train_qindex():

    f = open("stqa_to_mdr_trainfile.json", "r")
    data = json.load(f)

    d = {}
    for i in range(len(data)):
        d[data[i]["question"]] = i

    f_out = open("stqa_to_mdr_trainfile_qindex.json", "w")
    json.dump(d, f_out)

    f.close()
    f_out.close()

# stqa_to_mdr_train_qindex()

"""
formats and adds information to the stqa corpus so that we can
finetune mdr on it. adds the pos paras for training. 
ARCHIVED. Wrote iterate_dataset.py instead.
"""
def stqa_to_mdr_train():

    f = open("/projects/tir3/users/nnishika/strategyqa_train.json", "r")
    
    data = json.load(f)

    out = []
    for record in data:
        transformed_record = {}
        transformed_record["question"] = record["question"]
        transformed_record["answers"] = [record["answer"]]
        transformed_record["_id"] = record["qid"]
        transformed_record["type"] = None # for now

        para_names = get_gold_paras_for_stqa_record(record)

        t2t_f = open("stqa_title_to_text.json", "r")
        t2t = json.load(t2t_f)
        pos_paras = []
        for name in para_names:
            pos_paras.append({"title": name, "text": t2t[name]})

        transformed_record["pos_paras"] = pos_paras
        out.append(transformed_record)

    f2 = open("stqa_to_mdr_trainfile.json", "w")
    json.dump(out, f2)

    f.close()
    f2.close()

# stqa_to_mdr_train()


"""
calculates title based (coarse) recall of mdr on stqa
(which was computed using mdr's index which is why it's coaser).
"""
def coarse_recall():

    f_pred = open("mdrout/mdr_stqa_retrieval_top10.json", "r")
    f_true = open("stqaout/stqa_to_hotpot.json", "r")

    # preds = json.load(f_pred)
    true = f_true.readlines()
    print(type(true))

    total_recall = 0
    count = 0
    for line in f_pred.readlines():
        record = json.loads(line)
        for line2 in true:
            record2 = json.loads(line2)
            if record["question"] == record2["question"]:
                true_titles = [(title.split('-'))[:-1][0] for title in record2["sp"]]

                pred_titles = []
                for chain in record["candidate_chains"]:
                    pred_titles += [passage["title"] for passage in chain]

                
                question_recall = 0
                '''for title in pred_titles:
                    if title in true_titles:
                        question_recall +=1
                question_recall /= len(pred_titles)'''
                for title in true_titles:
                    if title in pred_titles:
                        question_recall +=1
                question_recall /= len(true_titles)
                break
                
        total_recall += question_recall
        count +=1
    total_recall /= count
    print(total_recall)

# coarse_recall()

"""
wanted to run stqa's recall@10 script on
mdr on stqa retrieval results. ARCHIVED.
"""
def mdr_for_stqa():

    f_in = open("/home/nnishika/mdrout/mdr_stqa_retrieval_top10.json", "r")
    f_out = open("/home/nnishika/mdrout/mdr_for_stqa.json", "w")


    d = {}
    for line in f_in.readlines():
        record = json.loads(line)
        d[record["question"]] = []
        for i in range(len(record["candidate_chains"])):
            
            for passage in record["candidate_chains"][i]:
                psg_obj = {}
                psg_obj["title"] = passage["title"]
                psg_obj["content"] = passage["text"]
                psg_obj["score"] = 1/(i+1) #for now
                d[record["question"]].append(psg_obj)

    json.dump(d, f_out, indent=4)

# mdr_for_stqa()

"""
turns the stqa_bm25.json into a dictionary of q -> paras.
the in file came from print statements in strateqa_qa_reader.py 
after running the prediction script on stqa's trainfile using
the IR-Q model. ARCHIVED. Wrote iterate_dataset.py instead
"""
def stqa_q_to_para():

    f_in = open("stqaout/temp.json", "r") #open("stqaout/stqa_bm25paras.json", "r")
    f_out = open("stqaout/stqa_q_to_bm25paras.json", "w")

    d = {}
    for line in f_in.readlines():
        record = json.loads(line)
        question = list(record.keys())[0]
        d[question] = record[question]

    print("d: ", d)
    json.dump(d, f_out, indent=4)
    f_in.close()
    f_out.close()

# stqa_q_to_para()

"""
splits stqa's corpus into many parts so I can 
create the index in parts since it always dies
from OOM issues (but doesn't for the toy corpus)
"""
from itertools import islice
def split_corpus(n):

    f_in = open("/projects/tir3/users/nnishika/stqa_corpus.json", "r")
   
    counter = 0
    while True:
        counter +=1

        next_n_lines = list(islice(f_in, n))
        if not next_n_lines:
            break

        pathname ="/projects/tir3/users/nnishika/stqa_corpus_chunks/stqa_corpus_"+str(counter)+".json"
        # create file
        f_out = open(pathname, "w")
        f_out.close()
   
        #keep writing to file
        f_out = open(pathname, "a")
        for line in next_n_lines:
            f_out.write(line)
        f_out.close()

    f_in.close()

# split_corpus(500000)


"""
formats any json file with indents
"""
def format_json(infile):

    f_in = open(infile, "r")
    f_out = open(infile[:-5]+"_formatted.json", "w")

    data = json.load(f_in)
    json.dump(data, f_out, indent=4)
    f_in.close()
    f_out.close()

# format_json("StqaIndexChunk1/id2doc.json")
# format_json("/projects/tir3/users/nnishika/StqaIndex2/id2doc.json")

def format_jsonl(infile):

    f_in = open(infile, "r")
    f_out = open(infile[:-5]+"_formatted.json", "w")

    lines = []
    for line in f_in.readlines():
        d = json.loads(line)
        lines.append(json.dumps(d, indent=4))

    f_out.writelines(lines)

    f_in.close()
    f_out.close()

# format_jsonl("mdrout/mdr_stqa_retrieval_top5.json")

def json_to_jsonlines(infile):

    f_in = open(infile, "r")
    data = json.load(f_in)
    
    lines = []
    for record in data:
        lines.append(json.dumps(record)+"\n")

    f_out = open(infile[:-5]+"_jsonl.jsonl", "w")
    f_out.writelines(lines)

# json_to_jsonlines("stqaout/finetune_mdr/mdr_evalfile.json")
# json_to_jsonlines("stqaout/finetune_mdr/mdr_trainfile.json")

"""
make directories to save all StqaChunk indexes
"""
import shutil
def make_index_dirs():

    # for i in range(4, 75):
    #    os.mkdir("/projects/tir3/users/nnishika/StqaIndexChunk"+str(i))
    for i in range(7, 75):
        shutil.move("/projects/tir3/users/nnishika/StqaIndexChunk"+str(i), "/projects/tir3/users/nnishika/StqaIndexChunks/StqaIndexChunk"+str(i))

# make_index_dirs()

"""
make scripts to encode each chunk of each stqa corpus (for mdr)
"""
def make_chunk_idx_scripts():

    path = "multihop_dense_retrieval/mycommands/encode_corpus_chunks/encode_corpus_chunk"
    f_in = open(path+"4.sh", "r")
    lines = f_in.readlines()

    for counter in range(5, 75):
        #create file
        outfile = path+str(counter)+".sh"
        f_out = open(outfile, "w")
        f_out.close()
        #add to file
        f_out = open(outfile, "a")

        for i in range(len(lines)):
            line = lines[i]
            if i == 22:
                line = line[:80] + str(counter)+".json \\\n"
            elif i == 24:
                line = line[:82] + str(counter)+" \\\n"
            
            f_out.write(line)

        f_out.close()

    f_in.close()

# make_chunk_idx_scripts()

"""
make scripts to encode each chunk of each stqa corpus (for dpr)
"""
def make_chunk_idx_scripts_dpr():

    path = "DPR/mycommands/get_index_chunks/get_index" 
    f_in = open(path+"0.sh", "r")
    lines = f_in.readlines()

    for counter in range(1, 50):
        #create file
        outfile = path+str(counter)+".sh"
        f_out = open(outfile, "w")
        f_out.close()
        #add to file
        f_out = open(outfile, "a")

        for i in range(len(lines)):
            line = lines[i]
            if i == 19:
                # print("here: ", counter)
                line = line[:13] + str(counter)+ line[14:]
            
            f_out.write(line)

        f_out.close()

    f_in.close()

# make_chunk_idx_scripts_dpr()

"""
join all the indexes of each stqa corpus chunk
into one big index
"""
import numpy as np
def join_idxs():

    path = "/projects/tir3/users/nnishika/"
    extended_path = path + "StqaIndexChunks/StqaIndexChunk1"
    
    index = np.load(extended_path + ".npy")
 
    f_1 = open(extended_path + "/id2doc.json", "r") 
    d = json.load(f_1)
    f_1.close()

    idx_offset = len(d)

    for i in range(2, 75):
        extended_path = path + "StqaIndexChunks/StqaIndexChunk" + str(i)
        
        #np_path = extended_path + ".npy" 
        #chunk_idx = np.load(np_path).astype('float32')
        #index = np.vstack((index, chunk_idx))

        f_chunk_dict = open(extended_path +  "/id2doc.json", "r")
        chunk_dict = json.load(f_chunk_dict)
        counter = 0
        for k, v in chunk_dict.items():
            assert(len(v) == 4)
            d[str(int(k)+idx_offset)] = v
            counter +=1
        idx_offset += counter 
        f_chunk_dict.close()

    #save index
    #f_idx = open(path+"StqaIndex/StqaIndex.npy", "wb")
    #np.save(f_idx, index)
    #f_idx.close()

    f_dict = open(path+"StqaIndex/id2doc.json", "w")
    json.dump(d, f_dict)
    f_dict.close()

# join_idxs()

"""
decomps for each question to easily manually look at them. 
goal: make all question decomp yes/no questions so it's easy 
to train a classification model. ARCHIVED.
"""
def get_decomps(infile):

    f_in = open(infile, "r")
    data = json.load(f_in)

    d = {}
    for record in data:
        question = record["question"]
        d[question] = []
        for decomp in record["decomposition"]:
            d[question].append(decomp)


    f_out = open("stqaout/decomps_only.json", "w")
    json.dump(d, f_out, indent=4)

# get_decomps("strategyqa/data/strategyqa/dev.json")


"""
checks if stqa corpus chunks are consistent/well-formed
"""
def sanity_check():
    
    path = "/projects/tir3/users/nnishika/"
    f_corpus = open(path+"stqa_corpus.json", "r")
    print("corpus len: ", len(f_corpus.readlines()))
    f_corpus.close()

    num_docs_across_chunks = 0
    for i in range(1, 75):
        f_chunk = open(path+"stqa_corpus_chunks/stqa_corpus_"+str(i)+".json", "r")
        num_docs_across_chunks += len(f_chunk.readlines())
        f_chunk.close()
    print("num docs across chunks: ", num_docs_across_chunks)

    rows_across_chunks = 0
    ids_across_chunks = 0
    for i in range(1, 75):
        index = np.load(path+"StqaIndexChunks/StqaIndexChunk"+str(i)+".npy")
        chunk_rows = index.shape[0]
        rows_across_chunks += chunk_rows

        f_id2doc = open(path+"StqaIndexChunks/StqaIndexChunk"+str(i)+"/id2doc.json", "r")
        chunk_ids = len(json.load(f_id2doc))
        ids_across_chunks += chunk_ids

        print("chunk "+str(i)+" rows, ids: ", chunk_rows, chunk_ids)

    print("rows, ids across chunks: ", rows_across_chunks, ids_across_chunks)
    
    index_total = np.load(path+"StqaIndex/StqaIndex.npy")
    print("total index shape: ", index_total.shape)
    f_id2doc_total = open(path+"StqaIndex/id2doc.json")
    print("total ids: ", len(json.load(f_id2doc_total)))

# sanity_check()

"""
make a file for all intermediary stqa question decomps
to use in frankenstein
"""

def get_stqa_decomps():

    f_in = open("strategyqa/data/strategyqa/dev.json", "r")
    data = json.load(f_in)

    lines = []
    for record in data:
        decomps = record["decomposition"]
        for i in range(len(decomps)-1):
            obj = {
                    "question": decomps[i],
                    "answers": "N/A",
                    "id": record["qid"]+"_"+str(i)
            }
            lines.append(json.dumps(obj)+"\n")

    #create file
    f_out = open("/projects/tir3/users/nnishika/stqa_intermediate_decomps.jsonl", "w")
    f_out.writelines(lines)

# get_stqa_decomps()


"""
qid to question +metadata dictionary for stqa
"""
def qid_to_q():

    f = open("strategyqa/data/strategyqa/dev.json", "r")
    data = json.load(f)

    d = {}
    for record in data:
        d[record["qid"]] = record

    f_out = open("stqaout/qid_to_question.json", "w")
    json.dump(d, f_out, indent=4)

    f.close()
    f_out.close()

# qid_to_q()


"""
Turns the stqa retrieval similar to standardized format
so I can compare stqa on stqa and mdr on stqa numbers.
"""
def format_stqa_retrieval():

    f_in = open("stqaout/retrieved.json", "r")
    data = json.load(f_in)
    f_in.close()

    f_qid = open("stqaout/qid_to_question.json", "r")
    d_qid = json.load(f_qid)
    f_qid.close()
    
    d_out = []
    for k, v in data.items():

        sp = get_gold_paras_for_stqa_record(d_qid[k])
        rp = v

        d = {}
        d["qid"] = k
        d["sp"] = sp
        d["rp"] = rp
        d_out.append(d)

    f_out = open("stqaout/retrieved_reformatted.json", "w")
    json.dump(d_out, f_out, indent=4)

    f_in.close()
    f_qid.close()
    f_out.close()

# format_stqa_retrieval()

"""
Turns mdr on stqa retrieval into a standardized format
"""
def format_mdr_retrieval(infile):

    f_in = open(infile, "r")
    
    f_qid = open("stqaout/qid_to_question.json", "r")
    d_qid = json.load(f_qid)

    d_out = []
    for line in f_in.readlines():
        record = json.loads(line)

        d = {}
        d["qid"] = record["_id"]
        d["sp"] = get_gold_paras_for_stqa_record(d_qid[record["_id"]]) 
        d["rp"] = []
        for chain in record["candidate_chains"]:
            for passage in chain:
                d["rp"].append(passage["title"]+"-"+str(passage["para_id"]))

        d["rp"] = list(set(d["rp"]))
        d_out.append(d)


    print(infile[:-5]+"_reformatted.json")
    f_out = open(infile[:-5]+"_reformatted.json", "w")
    json.dump(d_out, f_out, indent=4)

    f_in.close()
    f_qid.close()
    f_out.close()

# format_mdr_retrieval("mdrout/mdr_stqa_retrieval_top5.json")
# format_mdr_retrieval("mdrout/mdr_stqa_retrieval_top10.json")

"""
same as above, but the mdr output has already been
formatted, just need to add "sp"
"""
def format_mdr_retrieval_2(infile):

    print("format mdr retrieval 2 (format oracle retrieved)")

    f_in = open(infile, "r")
    data = json.load(f_in)
    f_in.close()
    
    f_qid = open("stqaout/qid_to_question.json", "r")
    d_qid = json.load(f_qid)

    for record in data:
        record["sp"] = get_gold_paras_for_stqa_record(d_qid[record["qid"]])

    f_out = open(infile[:-5]+"_reformatted.json", "w")
    json.dump(data, f_out, indent=4)
    f_out.close()

# format_mdr_retrieval_2("mdrout/frank_stqa_retrieved_noop.json")
# format_mdr_retrieval_2("mdrout/frank_on_stqa/top10/frank_stqa_retrieved_noop_top10.json")
# format_mdr_retrieval_2("mdrout/frank_oracle_top10.json")


"""
gets my version of recall for retrieval files of the aformentioned format
(% of sp covered by rp for each q averaged across all q)
"""
def my_recall(in_file, out_file):

    f_in = open(in_file, "r")
    data = json.load(f_in)

    recall = 0
    for record in data:
        covered = 0
        for true_p in record["sp"]:
            if true_p in record["rp"]:
                covered +=1

        recall += covered/len(record["sp"])

    recall /= len(data)
    f_out = open(out_file, "w")
    f_out.write("My recall: " + str(recall))

    f_in.close()
    f_out.close()

# my_recall("stqaout/retrieved_reformatted.json", "stqaout/my_recall.out")
# my_recall("mdrout/mdr_stqa_retrieval_top5_reformatted.json", "mdrout/my_recall_mdr_on_stqa_top5.out")
# my_recall("mdrout/mdr_stqa_retrieval_top10_reformatted.json", "mdrout/my_recall_mdr_on_stqa_top10.out")
# my_recall("mdrout/frank_stqa_retrieved_noop_reformatted.json", "mdrout/my_recall_frank_on_stqa_top10divsubq")
# my_recall("mdrout/frank_on_stqa/top10/frank_stqa_retrieved_noop_top10_reformatted.json", "mdrout/frank_on_stqa/top10/my_recall_frank_on_stqa_top10.out")
# my_recall("stqaout/stqa_retrieved_from_decomps_7.json", "stqaout/my_recall_stqa_retrieved_from_decomps_7.out")
# my_recall("stqaout/stqa_retrieved_from_decomps_11.json", "stqaout/my_recall_stqa_retrieved_from_decomps_11.out")
# my_recall("out/bm25_decomp_union_frank_top10_retrieved.json", "out/my_recall_bm25_decomp_union_frank_top10.out")
# my_recall("out/bm25_decomp_union_frank_top10divsubq_retrieved.json", "out/my_recall_bm25_decomp_union_frank_top10divsubq.out")
# my_recall("out/mdr_union_frank_top10_retrieved.json", "out/my_recall_mdr_union_frank_top10.out")
# my_recall("out/mdr_union_frank_top10divsubq_retrieved.json", "out/my_recall_mdr_union_frank_top10divsubq.out")
# my_recall("stqaout/stqa_top5_retrieved.json", "stqaout/my_recall_stqa_top5.out")
# my_recall("mdrout/frank_oracle/frank_oracle_top10_reformatted.json", "mdrout/frank_oracle/my_recall_frank_oracle.out")

"""
same as my recall but done separately for each annotator and the highest
is taken
(from stqa's recall@10)
"""
def stqa_recall(in_file, out_file):

    f_in = open(in_file, "r")
    data = json.load(f_in)

    f_qid = open("stqaout/qid_to_question.json", "r")
    d_qid = json.load(f_qid)

    recall = 0
    for record in data:
        stqa_record = d_qid[record["qid"]]

        anno_recall = 0
        for anno in stqa_record["evidence"]:
            covered = 0
            gold_anno = get_gold_paras_for_stqa_anno(anno)
            for true_p in gold_anno:
                if true_p in record["rp"]:
                    covered +=1            

            if len(gold_anno) > 0:
                anno_recall = max(anno_recall, covered/len(gold_anno))

        recall += anno_recall
            
    recall /= len(data)
    f_out = open(out_file, "w")
    f_out.write("Stqa recall: " + str(recall))

    f_in.close()
    f_out.close()

# stqa_recall("stqaout/retrieved_reformatted.json", "stqaout/stqa_recall.out")
# stqa_recall("mdrout/mdr_stqa_retrieval_top5_reformatted.json", "mdrout/stqa_recall_mdr_on_stqa_top5.out")
# stqa_recall("mdrout/mdr_stqa_retrieval_top10_reformatted.json", "mdrout/stqa_recall_mdr_on_stqa_top10.out")
# stqa_recall("mdrout/frank_stqa_retrieved_noop_reformatted.json", "mdrout/stqa_recall_frank_on_stqa_top10divsubq")
# stqa_recall("mdrout/frank_on_stqa/top10/frank_stqa_retrieved_noop_top10_reformatted.json", "mdrout/frank_on_stqa/top10/stqa_recall_frank_on_stqa_top10.out")
# stqa_recall("stqaout/stqa_retrieved_from_decomps_7.json", "stqaout/stqa_recall_stqa_retrieved_from_decomps_7.out")
# stqa_recall("stqaout/stqa_retrieved_from_decomps_11.json", "stqaout/stqa_recall_stqa_retrieved_from_decomps_11.out")
# stqa_recall("out/bm25_decomp_union_frank_top10_retrieved.json", "out/stqa_recall_bm25_decomp_union_frank_top10.out")
# stqa_recall("out/bm25_decomp_union_frank_top10divsubq_retrieved.json", "out/stqa_recall_bm25_decomp_union_frank_top10divsubq.out")
# stqa_recall("out/mdr_union_frank_top10_retrieved.json", "out/stqa_recall_mdr_union_frank_top10.out")
# stqa_recall("out/mdr_union_frank_top10divsubq_retrieved.json", "out/stqa_recall_mdr_union_frank_top10divsubq.out")
# stqa_recall("stqaout/stqa_top5_retrieved.json", "stqaout/stqa_recall_stqa_top5.out")
# stqa_recall("mdrout/frank_oracle/frank_oracle_top10_reformatted.json", "mdrout/frank_oracle/stqa_recall_frank_oracle.out")

"""
% of questions that have at least one of their gold passages
covered by the retrieved passages
from mdr's eval_mhop_retrieval
"""
def mdr_recall(in_file, out_file):

    f_in = open(in_file, "r")
    data = json.load(f_in)

    recall = 0
    for record in data:
        covered = 0
        for true_p in record["sp"]:
            if true_p in record["rp"]:
                covered +=1

        if covered > 0:
            recall += 1 

    recall /= len(data)
    f_out = open(out_file, "w")
    f_out.write("MDR recall: " + str(recall))

    f_in.close()
    f_out.close()

# mdr_recall("stqaout/retrieved_reformatted.json", "stqaout/mdr_recall.out")
# mdr_recall("mdrout/mdr_stqa_retrieval_top5_reformatted.json", "mdrout/mdr_recall_mdr_on_stqa_top5.out")
# mdr_recall("mdrout/mdr_stqa_retrieval_top10_reformatted.json", "mdrout/mdr_recall_mdr_on_stqa_top10.out")
# mdr_recall("mdrout/frank_stqa_retrieved_noop_reformatted.json", "mdrout/mdr_recall_frank_on_stqa_top10divsubq")
# mdr_recall("mdrout/frank_on_stqa/top10/frank_stqa_retrieved_noop_top10_reformatted.json", "mdrout/frank_on_stqa/top10/mdr_recall_frank_on_stqa_top10.out")
# mdr_recall("stqaout/stqa_retrieved_from_decomps_7.json", "stqaout/mdr_recall_stqa_retrieved_from_decomps_7.out")
# mdr_recall("stqaout/stqa_retrieved_from_decomps_11.json", "stqaout/mdr_recall_stqa_retrieved_from_decomps_11.out")
# mdr_recall("out/bm25_decomp_union_frank_top10_retrieved.json", "out/mdr_recall_bm25_decomp_union_frank_top10.out")
# mdr_recall("out/bm25_decomp_union_frank_top10divsubq_retrieved.json", "out/mdr_recall_bm25_decomp_union_frank_top10divsubq.out")
# mdr_recall("out/mdr_union_frank_top10_retrieved.json", "out/mdr_recall_mdr_union_frank_top10.out")
# mdr_recall("out/mdr_union_frank_top10divsubq_retrieved.json", "out/mdr_recall_mdr_union_frank_top10divsubq.out")
# mdr_recall("stqaout/stqa_top5_retrieved.json", "stqaout/mdr_recall_stqa_top5.out")
# mdr_recall("mdrout/frank_oracle/frank_oracle_top10_reformatted.json", "mdrout/frank_oracle/mdr_recall_frank_oracle.out")

# manually fix finetune mdr files
"""
first round of iterate_dataset didn't add the bridge on the finetune files.
adding manually.
"""
def add_bridge(infile):

    f_in = open(infile, "r")
    
    lines = []
    for line in f_in.readlines():
        record = json.loads(line)
        record["type"] = "bridge"
        record["bridge"] = None
        lines.append(json.dumps(record)+"\n")

    f_out = open(infile[:-6]+"2.jsonl", "w")
    f_out.writelines(lines)

    f_in.close()
    f_out.close()

# add_bridge("stqaout/finetune_mdr/mdr_trainfile.jsonl")
# add_bridge("stqaout/finetune_mdr/mdr_evalfile.jsonl")

def reformat_finetune_mdr_file(infile):

    f_in = open(infile, "r")

    lines = []
    for line in f_in.readlines():
        record = json.loads(line)
        pos_paras = []
        neg_paras = []
        for para in record["pos_paras"]:
            pos_paras.append({
                "title": para["title"],
                "text": para["content"],
                "id": int(para["evidence_id"].split('-')[-1])
                })
        for para in record["neg_paras"]:
            neg_paras.append({
                "title": para["title"],
                "text": para["content"],
                "id": int(para["evidence_id"].split('-')[-1])
                })

        record["pos_paras"] = pos_paras
        record["neg_paras"] = neg_paras

        lines.append(json.dumps(record)+"\n")

    f_out = open(infile[:-6]+"_reformatted.jsonl", "w")
    f_out.writelines(lines)

    f_in.close()
    f_out.close()

# reformat_finetune_mdr_file("stqaout/finetune_mdr/mdr_trainfile.jsonl")
# reformat_finetune_mdr_file("stqaout/finetune_mdr/mdr_evalfile.jsonl")


## DEBUGGING FRANKENSTEIN:

"""
testing some retrieved passages
"""
def test():
    f = open("/projects/tir3/users/nnishika/StqaIndex/id2doc.json", "r")
    data = json.load(f)

    for record in data.values():
        # if record[0] == 'Albany, Minnesota' and record[3] == 5:
        #     print("record: ", record)
        # if record[0] == 'Albany, Georgia' and record[3] ==35:
        #     print("record: ", record)
        if record[1] == 'The following schools have distinctions:':
            print("record: ", record)

    #hop1 retrieved
    # print(data["19537564"]) 
    # print(data["3429570"])

    #hop2a
    # print(data['19204826'])
    # print(data['12211297'])

    #hop2b
    # print(data['18648485'])
    # print(data['19204825'])
    f.close()

# test()

"""
test the query embeddings for frankenstein and mhop
from scripts/eval/eval...
"""
def test_embeds():
    
    path = "mdrout/debugging_frank/encoded_query_"
    for i in range(5):
        frank = np.load(path+"frank_"+str(i)+".npy")
        mhop = np.load(path+"mhop"+str(i)+".npy")
    
        # print("frank: ", frank)
        # print("mhop: ", mhop)
        # print(frank==mhop)
        print(np.all(frank==mhop))

# test_embeds()

"""
check metrics from frank on stqa log
"""
def frank_top1():

    f_questions = open("strategyqa/data/strategyqa/dev.json", "r")
    questions = json.load(f_questions)
    f_questions.close()

    f_log = open("multihop_dense_retrieval/log/559603.log", "r") 
    lines = f_log.readlines()[109:]
    f_log.close()

    f_id2doc = open("/projects/tir3/users/nnishika/StqaIndex/id2doc.json", "r")
    id2doc = json.load(f_id2doc)
    f_id2doc.close()

    i = 0
    q = 0
    d = []
    while i<len(lines):
        
        question = questions[q]
        num_subq = len(question["decomposition"])

        i += (num_subq*2)-1
        final_p_id = lines[i][6:-3]
        # print("final_p_id: ", final_p_id)

        final_p = id2doc[final_p_id]
        # print("final_: ", final_p)
        title = final_p[0]+"-"+str(final_p[3])

        d.append({"qid": question["qid"],
            "sp": get_gold_paras_for_stqa_record(question),
            "rp": [title]})

        q+=1
        i+=1

        # if q == 2: #comment out later
        #    break
    
    f_out = open("mdrout/frank_top1_retrieved.json", "w")
    json.dump(d, f_out)
    f_out.close()

# frank_top1()

"""
currently debugging frankenstien by comparing
behavior on queries from frankenstein to
behavior from mhop. extracted queries from frank
and reformatting to plug into mhop
"""
def frank_queries_to_mdr():

    f_in = open("mdrout/debugging_frank/frank_queries.json", "r")
    data = json.load(f_in)
    f_in.close()

    qid = "e0044a7b4d146d611e73"
    lines = []
    for k, v in data.items():
        d = {
            "question": v,
            "_id": qid+"-"+k,
            "answer": None,
            "sp": None,
            "type": None
            }
        lines.append(json.dumps(d)+"\n")

    f_out = open("mdrout/debugging_frank/frank_queries_to_hotpot.json", "w")
    f_out.writelines(lines)
    f_out.close()

# frank_queries_to_mdr()


"""
interpret log files by getting titles from doc ids
printed during evaluation
"""
def see_retrieved_docs(log_file):

    f_log = open(log_file, "r") 
    lines = f_log.readlines()[109:]
    f_log.close()

    f_id2doc = open("/projects/tir3/users/nnishika/StqaIndex/id2doc.json", "r")
    id2doc = json.load(f_id2doc)
    f_id2doc.close()

    i = 0
    subq_num = 0
    while i<len(lines):
        print(subq_num)

        j = i+(2**subq_num)
        lines[i] = lines[i][4:]
        lines[j-1] = lines[j-1][:-1]
        ids = []
        for line in lines[i:j]:
            ids_str = line[2:-2].split()
            print(ids_str)
            ids.append(ids_str[0])
            ids.append(ids_str[1])

        titles = []
        for p_id in ids:
            p = id2doc[p_id]
            titles.append(p[0]+"-"+str(p[3]))
        print(titles)

        subq_num+=1
        i = j + 2

# see_retrieved_docs("mdrout/debugging_frank/mhop_queries_log.log")
# see_retrieved_docs("mdrout/debugging_frank/frank_queries_log.log")


## ...

"""
turn stqa decomps into a stqa-formatted dataset
"""
def get_stqa_decomps():

    f_in = open("strategyqa/data/strategyqa/dev.json", "r")
    data = json.load(f_in)
    f_in.close()

    d = []
    for record in data:
        # get rid of operation hops
        total_annotators = 3
        is_operation = [0 for i in range(len(record["decomposition"]))]
        for anno in record["evidence"]:
            for i in range(len(anno)): #for subq evidence in anno:
                for psg_list in anno[i]:
                    if psg_list == "operation":
                        is_operation[i] += 1

        subqs = []
        new_ev = []
        for i in range(len(record["decomposition"])):
            if is_operation[i] < total_annotators/2:
                subqs.append((record["decomposition"][i], i))
                new_ev.append([[] for anno in range(total_annotators)])
                for anno in range(total_annotators):
                    new_ev[-1][anno].append(record["evidence"][anno][i])
        

        for i in range(len(subqs)):
            (q, decomp_id) = subqs[i]
            d.append({
                "qid": record["qid"]+"-"+str(decomp_id),
                "term": record["term"],
                "description": record["description"],
                "question": q,
                "answer": None,
                "facts": [],
                "decomposition": [q for (q, _) in subqs],
                "evidence": new_ev[i]
                })

    f_out = open("stqaout/stqa_decomps.json", "w")
    json.dump(d, f_out, indent=4)

# get_stqa_decomps() 

"""
got bm25 on stqa decomps to do error analysis
"""
def reformat_stqa_decomp_retrieved():
    print("reformat stqa decomp retrieved")
    # f_in = open("stqaout/stqa_decomps_preds.jsonl", "r")
    f_in = open("stqaout/stqa_top5_preds.jsonl", "r")
    d = []
    for line in f_in.readlines():
        d.append(json.loads(line))
    f_in.close()

    # f_data = open("stqaout/stqa_decomps.json", "r")
    f_data = open("strategyqa/data/strategyqa/dev.json", "r")
    data = json.load(f_data)
    f_data.close()

    for i in range(len(data)):
        assert(data[i]["question"] == d[i]["question"])
        d[i]["qid"] = data[i]["qid"]
        d[i]["sp"] = get_gold_paras_for_stqa_record(data[i])
        d[i]["rp"] = d[i]["bm25"]

    f_out = open("stqaout/stqa_top5_retrieved.json", "w")
    json.dump(d, f_out, indent=4)

# reformat_stqa_decomp_retrieved()

"""
reformat stqa_decomps_retrieved to conslidate decomps of each q
"""
def stqa_retrieved_from_decomp(path, k):
    print("stqa retrieved from decomp , consolidate decomps")
    
    f_in = open("stqaout/stqa_decomps_retrieved.json", "r")
    decomp_data = json.load(f_in)
    f_in.close()

    f_data = open("strategyqa/data/strategyqa/dev.json", "r")
    data = json.load(f_data)
    f_data.close()

    d = []
    counter = 0
    curr_qid = decomp_data[0]["qid"].split('-')[0]
    retrieved = []
    subq_counter = 0
    avg_retrieved = 0
    for record in decomp_data:

        new_qid = record["qid"].split('-')[0]
        if new_qid != curr_qid:
            retrieved = [subq_retrieved[:k//subq_counter] for subq_retrieved in retrieved]
            retrieved = [title for subq_retrieved in retrieved for title in subq_retrieved]
            retrieved = list(set(retrieved))
            avg_retrieved += len(retrieved)
            # print(len(retrieved))
            d.append({
                "qid": curr_qid,
                "sp": get_gold_paras_for_stqa_record(data[counter]),
                "rp": retrieved
                })

            curr_qid = new_qid
            retrieved = []
            counter +=1
            subq_counter = 0
        
        subq_counter +=1
        retrieved.append(record["bm25"])

    print(avg_retrieved/counter)
    f_out = open(path, "w")
    json.dump(d, f_out, indent=4)
    f_out.close()

# 11 and 7 are from the count_frank function below
# stqa_retrieved_from_decomp("stqaout/stqa_retrieved_from_decomps_11.json", 11)
# stqa_retrieved_from_decomp("stqaout/stqa_retrieved_from_decomps_7.json", 7)

"""
compare frank on bm25-decomps
"""
def error_analysis(path1, path2, outpath, name1, name2):

    f_1 = open(path1, "r")
    data_1 = json.load(f_1)
    f_1.close()

    f_2 = open(path2, "r")
    data_2 = json.load(f_2)
    f_2.close()

    f_gold = open("strategyqa/data/strategyqa/dev.json", "r")
    data_gold = json.load(f_gold)
    f_gold.close()

    d = []
    both_count = 0
    onlycount1 = 0
    onlycount2 = 0

    count1 = 0
    count2 = 0
    for i in range(len(data_1)):

        qid = data_1[i]["qid"]
        assert(qid == data_2[i]["qid"])
        assert(qid == data_gold[i]["qid"])

        gold = get_gold_paras_for_stqa_record(data_gold[i])
        list1 = list(set(data_1[i]["rp"]))
        list2 = list(set(data_2[i]["rp"]))
        # assert(len(list1) == len(list2))
        count1 += len(list1)
        count2 += len(list2)

        def intersect_gold(L):
            return [psg for psg in L if psg in gold]
    
        intersection = intersect_gold([psg for psg in list1 if psg in list2]) 
        union =  intersect_gold(list(set(list2 + list1)))
        only1 = intersect_gold([psg for psg in list1 if psg not in list2])
        only2 = intersect_gold([psg for psg in list2 if psg not in list1])

        both_count += len(intersection)
        onlycount1 += len(only1)
        onlycount2 += len(only2)

        d.append({
            "qid": qid,
            "question": data_gold[i]["question"],
            "decomposition": data_gold[i]["decomposition"],
            "gold": gold,
            "intersection": intersection,
            "union": union,
            name1+"_only": only1,
            name2+"_only": only2,
            "total count": len(list(set(list1+list2)))
            })
    
    print("both: ", both_count)
    print(name1+" only: ", onlycount1)
    print(name2+" only: ", onlycount2)
    print(name1+" count: ", count1)
    print(name2+" count: ", count2)
    print(name1+" ratio: ", onlycount1/count1)
    print(name2+" ratio: ", onlycount2/count2)

    f_out = open(outpath, "w")
    json.dump(d, f_out, indent=4)
    f_out.close()

# error_analysis("stqaout/stqa_retrieved_from_decomps_7.json", "mdrout/frank_on_stqa/top10divsubq/frank_stqa_retrieved_noop_reformatted.json", "out/bm25_decomp_frank_error_analysis_top10divsubq.json", "bm25", "frank")
# error_analysis("stqaout/stqa_retrieved_from_decomps_11.json", "mdrout/frank_on_stqa/top10/frank_stqa_retrieved_noop_top10_reformatted.json", "out/bm25_decomp_frank_error_analysis_top10.json", "bm25", "frank")
# error_analysis("mdrout/mdr_on_stqa/top5/mdr_stqa_retrieval_top5_reformatted.json", "mdrout/frank_on_stqa/top10divsubq/frank_stqa_retrieved_noop_reformatted.json", "out/mdr_frank_error_analysis_top10divsubq.json", "mdr", "frank")
# error_analysis("mdrout/mdr_on_stqa/top10/mdr_stqa_retrieval_top10_reformatted.json", "mdrout/frank_on_stqa/top10/frank_stqa_retrieved_noop_top10_reformatted.json", "out/mdr_frank_error_analysis_top10.json", "mdr", "frank")


"""
get union of frank and bm25 in a file to get recall on it
"""
def get_union(inpath, outpath):

    f_in = open("out/"+inpath, "r")
    data = json.load(f_in)
    f_in.close()

    d = []
    count = 0
    for record in data:
        d.append({
            "qid": record["qid"],
            "rp": record["union"],
            "sp": record["gold"]})
        count += record["total count"]

    print("avg unique titles in union: ", count//len(data))

    f_out = open("out/"+outpath, "w")
    json.dump(d, f_out, indent=4)
    f_out.close()

# get_union("bm25_decomp_frank_error_analysis_top10.json", "bm25_decomp_union_frank_top10_retrieved.json")
# get_union("bm25_decomp_frank_error_analysis_top10divsubq.json", "bm25_decomp_union_frank_top10divsubq_retrieved.json")
# get_union("mdr_frank_error_analysis_top10.json", "mdr_union_frank_top10_retrieved.json")
# get_union("mdr_frank_error_analysis_top10divsubq.json", "mdr_union_frank_top10divsubq_retrieved.json")

"""
add doc_id to stqa dataset. need for frankenstein oracle
"""
def add_docid():

    f_in = open("strategyqa/data/strategyqa/dev.json", "r")
    data = json.load(f_in)
    f_in.close()

    f_id = open("/projects/tir3/users/nnishika/StqaIndex/id2doc.json", "r")
    id2doc = json.load(f_id)
    f_id.close()

    title2id = {}
    for k, v in id2doc.items():
        title = v[0]+"-"+str(v[3])
        title2id[title] = k

    for record in data:
        for anno in record["evidence"]:
            for ev in anno:
                for psg_list in ev:
                    if isinstance(psg_list, list):
                        for i in range(len(psg_list)):
                            psg_name = psg_list[i]
                            psg_list[i] = (psg_name, title2id[psg_name])
        # print(record)
        # break
    

    f_out = open("stqaout/updated_dev.json", "w")
    json.dump(data, f_out)
    f_out.close()

# add_docid()


"""
want to count the avg # of UNIQUE docs mdr/frank got so i know how much to retrieve for bm25 to compare apples to apples
"""
def count_unique_docs(path):

    f_in = open(path, "r")
    data = json.load(f_in)
    f_in.close()

    count = 0
    for record in data:
        record["rp"] = list(set(record["rp"]))
        count += len(record["rp"])

    print(count/len(data))
    f_out = open(path, "w")
    json.dump(data, f_out)
    f_out.close()

# count_unique_docs("mdrout/on_stqa/top10/stqa_retrieved_noop_top10_reformatted.json")
# count_unique_docs("mdrout/on_stqa/top10divsubq/stqa_retrieved_noop_reformatted.json")
# count_unique_docs("mdrout/mdr_on_stqa/top10/mdr_stqa_retrieval_top10_reformatted.json")
# count_unique_docs("mdrout/mdr_on_stqa/top5/mdr_stqa_retrieval_top5_reformatted.json")
count_unique_docs("mdrout/frank_oracle/frank_oracle_top10_reformatted.json")

"""
run stqa on hotpotQA. get hotpotQA in the right format
"""
def hotpot_to_stqa():

    f_in = open("multihop_dense_retrieval/data/hotpot/hotpot_qas_val.json", "r")
    data = []
    for line in f_in.readlines():
        data.append(json.loads(line))
    f_in.close()

    d_out = []
    for record in data:
        d = {
                "qid": record["_id"],
                "term": None,
                "description": None,
                "question": record["question"],
                "answer": record["answer"],
                "facts": [],
                "decomposition": [],
                "evidence": record["sp"],
                "type": record["type"]
                }
        d_out.append(d)

    f_out = open("stqaout/hotpot_to_stqa.json", "w")
    json.dump(d_out, f_out, indent=4)
    f_out.close()

# hotpot_to_stqa()

"""
want to run frank on QDMR-high-level
"""
import csv
def break_to_stqa():

    f_in = open("Break-dataset/QDMR-high-level/dev.csv", "r")
    csv_reader = csv.reader(f_in)
    f_in.close()

    d_out = []
    for row in csv_reader:
        d = {
                "qid": row[0],
                "term": None,
                "description": None,
                "question": row[1],
                "answer": None,
                "facts": None,
                # "decomposition": ?,
                # "evidence": ?,
                "type": None
                }
        d_out.append(d)

    f_out = open("Break-dataset/QDMR-high_to_stqa.json", "w")
    json.dump(d_out, f_out)
    f_out.close()

"""
get text overlap to ideally prove that
there is more text overlap between subqs+gold than qs+gold
which implies that BM25 is better for subqs/single hop
"""
import re
import nltk
from nltk.corpus import stopwords

def get_lexical_overlap():

    f_decomp = open("stqaout/stqa_decomps.json", "r")
    decomps = json.load(f_decomp)
    f_decomp.close()

    f_q = open("stqaout/updated_dev.json", "r")
    data = json.load(f_q)
    f_q.close()

    f_id2doc = open("/projects/tir3/users/nnishika/StqaIndex/id2doc.json", "r")
    id2doc = json.load(f_id2doc)
    f_id2doc.close()

    subq_overlap = 0
    q_overlap = 0

    def tokenize(s):
        tokens = s.lower().split()
        tokens = [re.sub(r'\W+', '', token) for token in tokens]
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        # print(tokens)
        return tokens
    
    def get_overlap(query, docs):

        intersection = [t for t in query if t in docs]
        return len(intersection)/len(query)

    i = 0
    counter = 0
    for record in data:
        print(counter)

        qid = record["qid"]
        q = tokenize(record["question"])

        assert(qid == (decomps[i]["qid"].split('-'))[0])

        subqs = []
        while (i < len(decomps) and (decomps[i]["qid"].split('-'))[0] == qid): 
            if i >= len(decomps):
                print(record, i)
            subqs += tokenize(decomps[i]["question"])
            i +=1 

        gold_ids = get_gold_ids_for_updated_stqa_record(record)
        gold_psgs = [id2doc[gold_id][1] for gold_id in gold_ids]
        gold = [t for t_list in [tokenize(psg) for psg in gold_psgs] for t in t_list]

        q_overlap += get_overlap(q, gold)
        subq_overlap += get_overlap(subqs, gold)

        # break
        counter +=1

    subq_overlap /= len(data)
    q_overlap /= len(data)

    print("subq overlap: ", subq_overlap)
    print("q overlap: ", q_overlap)

# get_lexical_overlap()


"""
title to text for stqa corpus
"""
def title_to_text():

    f = open("/projects/tir3/users/nnishika/StqaIndex/id2doc.json", "r")
    id2doc = json.load(f)
    f.close()

    d = {}
    for _, v in id2doc.items():

        title = v[0]+'-'+str(v[3])
        text = v[1]

        d[title] = text
    
    f_out = open("/projects/tir3/users/nnishika/stqa_title_to_text.json", "w")
    json.dump(d, f_out, indent=4)
    f_out.close()

# title_to_text()

"""
test a few questions for error analysis. these are ids where frank got some that mdr didn't
"""
def test_error_analysis(qids):

    f = open("/projects/tir3/users/nnishika/stqa_title_to_text.json", "r")
    title2text = json.load(f)
    f.close()

    f_in = open("out/mdr_frank/top10divsubq/mdr_frank_error_analysis_top10divsubq.json", "r")
    retrieved = json.load(f_in)
    f_in.close()

    f_frank = open("mdrout/frank_on_stqa/top10divsubq/frank_stqa_retrieved_noop_reformatted.json", "r")
    frank = json.load(f_frank)
    f_frank.close()

    f_mdr = open("mdrout/mdr_on_stqa/top5/mdr_stqa_retrieval_top5_reformatted.json", "r")
    mdr = json.load(f_mdr)
    f_mdr.close()

    def get_by_qid(qid, d):
        for record in d:
            if record["qid"] == qid:
                return record

    analysis = []
    count = 0
    for r in retrieved:
        if r["qid"] in qids:

            analysis_record = {}
            analysis_record["retrieved_info"] = r
            analysis_record["frank_all"] = get_by_qid(r["qid"], frank)
            analysis_record["mdr_all"] = get_by_qid(r["qid"], mdr)

            analysis.append(analysis_record)

            count +=1
            if count == len(qids): #done
                break
     
    f_out = open("out/mdr_frank/top10divsubq/subset_finegrain_analysis.json", "w")
    json.dump(analysis, f_out, indent=4)
    f_out.close()

# test_error_analysis(['be5c9933987f046b476e', '1932e05f10680ece229f', '7702ee1e9f757ebffdf1', '11d009721f27a60f9cff', '1e959987a695898808f'])

