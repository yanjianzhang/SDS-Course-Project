from openie import StanfordOpenIE
import pandas as pd
import json
import re
import random
from tqdm import tqdm
import numpy as np

train_data = pd.read_csv('./dataset/task2/dev.tsv', delimiter='\t', header=None)
idx_loc_dict = {}
for i in range(len(train_data)):
    idx_loc_dict[train_data.loc[i,0]] = i
with open('./dataset/stopwords.txt') as f:
    f = f.read().strip().split('\n')
stopwords = f
result_store = {}

def judge_remain_based_on_stopwords(listA, listB):
    calA = 0
    calB = 0
    for word in listA:
        if word in stopwords:
            calA += 1
    for word in listB:
        if word in stopwords:
            calB += 1
    if len(set(listA)) - calA > len(set(listB)) - calB:
        return True
    return False

def if_tuple_similar(tupleA, tupleB):
    mark_dict = {-1:tupleA, 1:tupleB}
    tupleA = tupleA['subject'].split() + tupleA['relation'].split() + tupleA['object'].split()
    tupleB = tupleB['subject'].split() + tupleB['relation'].split() + tupleB['object'].split()
    mark = None
    if len(set(tupleA)) > len(set(tupleB)):
        long = set(tupleA)
        short = set(tupleB)
        mark = -1
    else:
        long = set(tupleB)
        short = set(tupleA)
        mark = 1
    common = long & short
    if len(common)/len(short) > 0.7:
        if judge_remain_based_on_stopwords(long, short) == True:
            return [mark_dict[mark]]
        else:
            return [mark_dict[mark*(-1)]]
    else:
        return [tupleA, tupleB]


def filter(sent, triples):
    old_triples = triples.copy()
    for i in range(len(triples)):
        triples[i] = triples[i]['subject'].split() + triples[i]['relation'].split() + triples[i]['object'].split()
    triple_remain_based_on_len = []
    if len(sent) > 14:
        for triple_idx in range(len(triples)):
            if len(triples) <= len(sent)/2:
                triple_remain_based_on_len.append(old_triples[triple_idx])
    old_triples2 = triple_remain_based_on_len
    triples = old_triples2.copy()
    for i in range(len(triples)):
        triples[i] = triples[i]['subject'].split() + triples[i]['relation'].split() + triples[i]['object'].split()
    old_triples2.sort(key = lambda i: len(i['subject'].split() + i['relation'].split() + i['object'].split()), reverse = True)
    triple_remain_based_on_chutong_and_stopwords = []
    old_triples2_dict = {}
    for i in range(len(old_triples2)):
        old_triples2_dict[i] = old_triples2[i]
    keys = old_triples2_dict.keys()
    iter_len = len(keys)
    iteration = 0
    while iteration < iter_len*5 and len(old_triples2_dict.keys()) >= 2 :
        iteration += 1
        sample = random.sample(old_triples2_dict.keys(), 2)
        key1 = sample[0]
        key2 = sample[1]
        result = if_tuple_similar(old_triples2_dict[key1], old_triples2_dict[key2])
        if len(result) == 1:
            if result == old_triples2_dict[key1]:
                old_triples2_dict.pop(key2)
            else:
                old_triples2_dict.pop(key1)
    triple_remain_based_on_chutong_and_stopwords = list(old_triples2_dict.values())
    
    return triple_remain_based_on_chutong_and_stopwords          


def remove_not(sent, client):
    old_sent = sent
    not_index = None
    if 'not' not in sent and 'n\'t' not in sent:
        return []
    sentlist = sent.split()
    # print(sentlist)
    not_index = []
    for i in range(len(sentlist)):
        if 'not' in sentlist[i]:
            if 'not' == sentlist[i]:
                not_index.append(i)
            else:
                not_index.append(i + 100)
            sentlist[i] = re.sub('not', '', sentlist[i])
            continue
        if 'n\'t' in sentlist[i]:
            if 'n\'t' == sentlist[i]:
                not_index.append(i)
            else:
                not_index.append(i + 100)
            sentlist[i] = re.sub('n\'t', '', sentlist[i])
            continue
    old_sentlist = old_sent.split()
    while '' in sentlist:
        sentlist.remove('')
    sent = ' '.join(sentlist)
    # print('sent: %s' %sent)
    triples = client.annotate(sent)
    optriList = []
    for triple in triples:
        rel = triple['relation'].split()
        rel_insert_idx = []
        obj = triple['object'].split()
        obj_insert_idx = []
        for rel_idx in range(len(rel)):
            if rel[rel_idx] == 'not':
                continue
            for index in not_index:
                if index > 99:
                    index = index - 100
                    not_former_word = old_sentlist[index]
                    not_former_word = re.sub('not','',not_former_word)
                    not_former_word = re.sub('n\'t','',not_former_word)
                    if rel[rel_idx] != old_sentlist[index] and rel[rel_idx] == not_former_word:
                        rel.insert(rel_idx+1, 'not')
                else:
                    not_former_word = old_sentlist[index-1]
                    if rel[rel_idx] == not_former_word:
                        rel.insert(rel_idx+1, 'not')
        triple['relation'] = ' '.join(rel)

        for obj_idx in range(0, len(obj)):
            for index in not_index:
                if index > 99:
                    index = index - 100
                    not_former_word = old_sentlist[index]
                    not_former_word = re.sub('not','',not_former_word)
                    not_former_word = re.sub('n\'t','',not_former_word)
                    if obj[obj_idx] != old_sentlist[index] and obj[obj_idx] == not_former_word:
                        obj.insert(rel_idx+1, 'not')
                else:
                    not_former_word = old_sentlist[index-1]
                    # print(obj[obj_idx])
                    # print(not_former_word)
                    if obj[obj_idx] == not_former_word:
                        obj.insert(obj_idx+1, 'not')
        triple['object'] = ' '.join(obj)
        optriList.append(triple)
    

    # print(triples)

    # print(not_index)
    
    return triples

# with StanfordOpenIE() as client:
#     sent = 'elephants are usually gray while fridges are usually white'
#     print(sent)
#     triples = client.annotate(sent)
#     print(triples)
#     triples = filter(sent, triples)
#     print(triples)

with StanfordOpenIE() as client:
    for i in tqdm(range(len(train_data))):
        # print(i)
        pair_result = [[],[],[],[]]
        question = train_data.loc[i,2]
        result_Q = [triple for triple in client.annotate(question)]
        if len(result_Q) == 0:
            result_Q = remove_not(question, client)
        if len(result_Q) >= 2:
            result_Q = filter(train_data.loc[i,2], result_Q)
        # print(result_Q)
        pair_result[0] = result_Q
        for j in range(3,6):
            optriList = []
            op = train_data.loc[i,j]
            if not isinstance(op, float):
                ac = client.annotate(op)
                if len(ac) == 0:
                    ac = remove_not(op, client)
                if len(ac) >= 2:
                    ac = filter(op, ac)
            else:
                train_data.loc[i,j] = ""
                ac = []
            for triple in ac:
                optriList.append(triple)
            pair_result[j-2] = optriList
        result_store[str(train_data.loc[i,0])] = pair_result

tri_num = {}
new_result = []
for i in result_store:
    temp_list = []
    if len(result_store[i][0]) > 4:
        result_store[i][0] = random.sample(result_store[i][0], k=4)
    if len(result_store[i][0]) in tri_num:
        tri_num[len(result_store[i][0])] += 1
    else:
        tri_num[len(result_store[i][0])] = 1
    for j in range(3):
        if len(result_store[i][j+1]) > 4:
            result_store[i][j+1] = random.sample(result_store[i][j+1], k=4)   
        temp_list.append({'idx': i, 'label': train_data.loc[idx_loc_dict[eval(i)], 1], 'question': train_data.loc[idx_loc_dict[eval(i)],2], \
            'answer': train_data.loc[idx_loc_dict[eval(i)], j+3], 'q_trp':result_store[i][0], 'a_trp':result_store[i][j+1]})
        if len(result_store[i][j+1]) in tri_num:
            tri_num[len(result_store[i][j+1])] += 1
        else:
            tri_num[len(result_store[i][j+1])] = 1
    new_result.append(temp_list)
print(tri_num)

with open('test_dev.json', "w") as fout:
    for d_list in new_result:
        for dic in d_list:
            fout.write(json.dumps(dic, indent=2) + "\n")   


# with open('openIEresult.json','w') as outfile:
#     for idx in result_store:
#         question = train_data.loc[eval(idx), 2]
#         a1 = train_data.loc[eval(idx), 3]
#         a2 = train_data.loc[eval(idx), 4]
#         a3 = train_data.loc[eval(idx), 5]
#         q_dict = {'idx':idx,'question': question, 'q_trp': result_store[idx][0]}
#         a1_dict = {'a1': a1, 'a_trp': result_store[idx][1]}
#         a2_dict = {'a2': a2, 'a_trp': result_store[idx][2]}
#         a3_dict = {'a3': a3, 'a_trp': result_store[idx][3]}
#         outfile.write(json.dumps(q_dict, indent=2) + '\n')
#         outfile.write(json.dumps(a1_dict, indent=2) + '\n')
#         outfile.write(json.dumps(a2_dict, indent=2) + '\n')
#         outfile.write(json.dumps(a3_dict, indent=2) + '\n\n')

# print(result_store)

# with StanfordOpenIE() as client:
#     text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
#     print('Text: %s.' % text)
#     for triple in client.annotate(text):
#         print('|-', triple)


