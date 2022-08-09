import pickle as pkl
import json
import os
import csv
from rouge import Rouge

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

"""
get stqa's dataset in a format to mdr codebase's liking
"""
def stqa_to_mdr():

    f = open("strategyqa/data/strategyqa/dev.json", 'r')
    data_in = json.load(f)
    
    lines = []

    for record in data_in:
       gold_paras= set()
       for annotator in record["evidence"]:
           for hop in annotator:
               if isinstance(hop, list):
                   for title in hop:
                       gold_paras.add(title[0])

       new_record = {
           "question": record["question"],
           "answer": record["answer"],
           "sp": list(gold_paras),
           "type": None
           }
       lines.append(json.dumps(new_record))

    f_out = open("stqaout/stqa_to_hotpot_noduplicates.json", "w")
    f_out.writelines(lines)

    f.close()
    f_out.close()

# stqa_to_mdr()


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

        para_names = set()
        for anno in record["evidence"]:
            for ev in anno:
                for psg_list in ev:
                    if isinstance(psg_list, list):
                        for psg_name in psg_list:
                            para_names.add(psg_name)

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
truthfully idk. ARCHIVED.
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
WIP
recalculate stats on MDR on stqa dataset because
originally the sp for each question had duplicates
since I wasn't careful when accounting for duplicates
across annotaors when turning stqa dev.json into a
form similar to hotpot for MDR
"""
def unduplicate_mdr_on_stqa():
    f_in = open("mdrout/mdr_on_stqa_raw.out", "r")

    in_lines = f_in.readlines()
    out_lines = []

    for i in range(len(in_lines)):
        if in_lines[i][:2] == "sp":
            #parse
            #TODO: parsing is wrong. split  on "-"
            titles = in_lines[i][7:-2].split(', \'')
            titles = [x.split(', \"') for x in titles]
            titles = [title[:-1] for sublist in titles for title in sublist]

            no_duplicate_titles = list(set(titles))
            titles_to_string = "[" + ', '.join(['\''+title+'\'' for title in titles]) + "]\n"
            out_lines.append("sp:  "+titles_to_string)
        elif in_lines[i][:2] == "rp":
            # print(in_lines[i])
            out_lines.append(in_lines[i])

    f_out = open("mdrout/mdr_retrieved_on_stqa.out", "w")
    f_out.writelines(out_lines)
     
    f_in.close()
    f_out.close()

# unduplicate_mdr_on_stqa()
