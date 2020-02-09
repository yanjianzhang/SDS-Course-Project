import argparse
import torch
from transformers import *
from tqdm import tqdm
import os
import pickle
import pandas as pd
import json


# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
# n_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def convert_qa_to_bert_input(tokenizer, question, answer, max_seq_length):
    q_tokens = tokenizer.tokenize(question)
    a_tokens = tokenizer.tokenize(answer)
    tokens = ["[CLS]"] + q_tokens + ["[SEP]"] + a_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * (len(q_tokens) + 2) + [1] * (len(a_tokens) + 1)
    assert len(input_ids) == len(segment_ids) == len(input_ids)

    pad_len = max_seq_length - len(input_ids)
    input_mask = [1] * len(input_ids) + [0] * pad_len
    input_ids += [0] * pad_len
    segment_ids += [0] * pad_len
    span = (len(q_tokens) + 2, len(q_tokens) + 2 + len(a_tokens))
    assert span[1] + 1 == len(tokens)
    return input_ids, input_mask, segment_ids, span



def extract_bert_features(input_file, output_file, cache_file, max_seq_length, device, batch_size):
    config = BertConfig.from_pretrained("./model/pretrained_bert/bert-base-uncased/")
    config.output_attentions = True
    config.attention_probs_dropout_prob = 0.0
    tokenizer = BertTokenizer.from_pretrained("./model/pretrained_bert/bert-base-uncased/", do_lower_case=True)
    model = BertModel.from_pretrained("./model/pretrained_bert/bert-base-uncased/", config=config)
    model.to(device)
    model.eval()

    all_input_ids, all_input_mask, all_segment_ids, all_span, all_idx = [], [], [], [], []
    n = sum(1 for _ in open(input_file, "r"))

    if os.path.isfile(cache_file):
        print("loading cached inputs")
        with open(cache_file, "rb") as fin:
            all_input_ids, all_input_mask, all_segment_ids, all_span, all_idx = pickle.load(fin)
        print("loaded")
    else:
        with open(input_file, "r") as fin:
            for line in tqdm(fin, total=n, desc="Calculating alignments"):
                line = line.strip().split("\t")
                idx = line[0]
                question = line[2]
                answers = line[3:]
                for a_idx in range(len(answers)):
                    answer = answers[a_idx]
                    input_ids, input_mask, segment_ids, span = convert_qa_to_bert_input(tokenizer, question, answer, max_seq_length)
                    all_input_ids.append(input_ids)
                    all_input_mask.append(input_mask)
                    all_segment_ids.append(segment_ids)
                    all_span.append(span)
                    all_idx.append((idx, a_idx))
        with open(cache_file, "wb") as fout:
            pickle.dump((all_input_ids, all_input_mask, all_segment_ids, all_span, all_idx), fout)
            print("Input saved in cache file")

    all_input_ids, all_input_mask, all_segment_ids, all_span = [torch.tensor(x, dtype=torch.long) for x in [all_input_ids, all_input_mask, all_segment_ids, all_span]]

    n = all_input_ids.size(0)

    with torch.no_grad():
        for a in tqdm(range(0, n, batch_size), total=n//batch_size+1, desc="Extracting Attention Scores"):
            attentions_list = []
            idx_list = []
            b = min(n, a + batch_size)
            batch = [x.to(device) for x in [all_input_ids[a:b], all_input_mask[a:b], all_segment_ids[a:b]]]
            idxes = [all_idx[a:b]]
            outputs = model(*batch)
            sequence_output, pooled_output, all_attentions = outputs
            attentions_list.append(all_attentions)
            idx_list.append(idxes)

            with open(output_file+str(a), "wb") as fout:
                pickle.dump((idx_list, attentions_list), fout)
            print("Writing attentions")


def ori_tokens(q_idx, a_idx, return_idx=False):
    tokenizer = BertTokenizer.from_pretrained("./model/pretrained_bert/bert-base-uncased/", do_lower_case=True)
    with open("./features/task2/dev_qa.jsonl", "rb") as f:
        data = pickle.load(f)
    question, answer = data[(q_idx, a_idx)].values()
    q_tokens = tokenizer.tokenize(question)
    a_tokens = tokenizer.tokenize(answer)
    # tokens = ["[CLS]"] + q_tokens + ["[SEP]"] + a_tokens + ["[SEP]"]
    input_ids, input_mask, segment_ids, span = convert_qa_to_bert_input(tokenizer, question, answer, max_seq_length=95)

    if return_idx:
        return len(q_tokens) + 2, q_tokens, a_tokens, segment_ids
    else:
        return q_tokens, a_tokens


def key_words_index(q_idx, file="./features/task1/dev_w_idx.jsonl"):
    with open(file, "rb") as fin:
        dict_json = pickle.load(fin)
    line = dict_json[q_idx]
    key_word_idx = line["idx0"] if line["label"] == 0 else line["idx1"]
    return key_word_idx


def reference_words_idx(q_batch_idx, q_idx, a_idx, key_word_idx, attention, threshold=None, num_top=3):
    # select top 3 words if threshold is None,
    refer_word_idx = []
    for k_idx in json.loads(key_word_idx):
        a = attention[0][-1][int(q_batch_idx), :, int(k_idx) + 1, :] # 12blocks 12heads [heads, length]
        a = torch.mean(a, dim=0)
        e_start_idx, q_tokens, a_tokens, segment_ids = ori_tokens(q_idx, a_idx, return_idx=True)
        # e_start_idx = ori_tokens(q_idx, a_idx, return_idx=True)
        a_cutoff = a[e_start_idx : e_start_idx + len(a_tokens)]
        refer_word_idx.extend(a_cutoff.data.numpy().argsort()[-num_top:][::-1])
    return refer_word_idx


def extract_key_ref_words(key_word_idx, refer_word_idx, original_tokens):
    key_words, ref_words = [], []
    for k in json.loads(key_word_idx):
        print(original_tokens[0])
        print(k)
        key_words.append(original_tokens[0][k])
    for r in refer_word_idx:
        ref_words.append(original_tokens[1][r])
    return key_words, ref_words
        


def reference_words_extraction(attention_file, cache_file):
    # with open(cache_file, "rb") as fin:
    #     all_input_ids, all_input_mask, all_segment_ids, all_span, all_idx = pickle.load(fin)
    a_files = os.walk(attention_file)
    for path, dir_list, file_list in a_files:
        for file_name in file_list:
            with open(os.path.join(path, file_name), "rb") as tmpin:
                idx_list, attentions_list = pickle.load(tmpin)
            idx_list = idx_list[0][0]

            key_ref_words = {}
            for i in tqdm(range(0, len(idx_list)), desc="Extracting Reference Words"):
                q_idx, a_idx = idx_list[i]
                key_word_idx = key_words_index(q_idx=q_idx)
                refer_word_idx = reference_words_idx(q_batch_idx=i, q_idx=q_idx, a_idx=a_idx, key_word_idx=key_word_idx, attention=attentions_list)
                original_tokens = ori_tokens(q_idx=q_idx, a_idx=a_idx)
                key_ref_words[(q_idx, a_idx)] = extract_key_ref_words(key_word_idx=key_word_idx, refer_word_idx=refer_word_idx, original_tokens=original_tokens)
           
            with open(os.path.join("./features/task2/key_ref_words/",file_name), "wb") as tmpout:
                pickle.dump(key_ref_words, tmpout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./dataset/task2/dev.tsv")
    parser.add_argument("--output_file", default="./features/task2/attentions/")
    parser.add_argument("--max_seq_length", type=int, default=95)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--cache_file", default="./features/task2/bert_input.pk")
    parser.add_argument("--feature_file", default="./features/task2/dev.features.pk")
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    # extract_bert_features(input_file=args.input_file, output_file=args.output_file, cache_file=args.cache_file,
    #                       max_seq_length=args.max_seq_length, device=device, batch_size=args.batch_size)
    #
    reference_words_extraction(attention_file=args.output_file, cache_file=args.cache_file)