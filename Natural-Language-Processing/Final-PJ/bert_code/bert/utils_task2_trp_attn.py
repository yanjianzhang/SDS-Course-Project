import torch
import os
from transformers import *
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import math


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 使用FileHandler输出到文�?
fh = logging.FileHandler("./logfile/logBERT.log")
fh.setLevel(logging.INFO)
# 使用StreamHandler输出到屏�?
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)


def record_wrong(wrong_samples, args):
    test_file = os.path.join(args.data_dir, "test" + "." + args.data_name + ".tsv")
    dataset = pd.read_csv(test_file, sep='\t', header=None)

    records = pd.DataFrame()
    for index in wrong_samples:
        line = dataset.loc[dataset[0] == index]
        records = records.append(line)

    save_path = args.wrong_file
    records.to_csv(save_path, sep='\t', index=False, header=False)



def simple_accuracy(preds, labels):
    assert len(preds) == len(labels)
    return (preds == labels).mean()


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # However, since we'd better not to remove tokens of options and questions, you can choose to use a bigger
    # length or only pop from context
    while True:
        if tokens_c:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        else:
            total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if tokens_c:
            if len(tokens_c) > len(tokens_b) or len(tokens_c) > len(tokens_a):
                tokens_c.pop()
        else:
            logger.info('Attention! you are removing from token_b (swag task is ok). '
                        'If you are training ARC and RACE (you are poping question + options), '
                        'you need to try to use a bigger max seq length!')
            tokens_b.pop()


def pad_trp(vec, trp_num):
    for pad_num in range(trp_num - len(vec)):
        vec.append(list(np.zeros(30)))
    # a = torch.cat([torch.tensor(vec, dtype=torch.long), torch.zeros(trp_num - len(vec), dtype=torch.long)], dim=0)
    return vec

def select_field(features, field, type, trp_num=None):
    if type == "sent":
        return [
            [
                sent[field]
                for sent in feature.sent_features
            ]
            for feature in features #[8000, 3, max_len]
        ]

    elif type == "trp":
        trp_list = []
        for feature in features:
            sent_list = []
            for sent in feature.trp_features:
                vec = [
                    trp_id[field] for trp_id in sent
                ]

                vec = pad_trp(vec, trp_num)
                sent_list.append(vec)
            trp_list.append(sent_list)
        return trp_list




class InputExample(object):
    """A single training/test example for multiple choice"""
    def __init__(self, example_id, sent_q, sent_a, trp_q, trp_a, label=None):
        """Constructs an InputExample."""
        self.example_id = example_id
        self.label = label
        self.sent_q = sent_q
        self.sent_a = sent_a
        self.trp_q = trp_q
        self.trp_a = trp_a



class Task2Processor(DataProcessor):
    """
    jsonl data file into examples
    """
    def __init__(self):
        self.label_num = len(self.get_labels())

    def get_train_examples(self, data_dir, data_name):
        data_name = "train" + "." + data_name + ".jsonl"
        logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
        return self._create_examples(lines=self._read_jsonl(os.path.join(data_dir, data_name)))

    def get_test_examples(self, data_dir, data_name):
        data_name = "test" + "." + data_name + ".jsonl"
        logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
        return self._create_examples(lines=self._read_jsonl(os.path.join(data_dir, data_name)))

    def get_dev_examples(self, data_dir, data_name):
        data_name = "dev" + "." + data_name + ".jsonl"
        logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
        return self._create_examples(lines=self._read_jsonl(os.path.join(data_dir, data_name)))

    def get_labels(self):
        return ["A", "B", "C"]

    def _read_jsonl(self, path):
        with open(path, "r") as fin:
            # lines = json.load(fin)
            lines = [json.loads(line) for line in fin]
            print(len(lines))
            lines = list(zip(*(iter(lines),) * self.label_num))
            print(len(lines))
        return lines


    def _create_examples(self, lines):
        examples = []
        for line in lines:
            sent_q, sent_a, trp_q, trp_a = [], [], [], []
            for qa in line:
                sent_q.append(qa["question"])
                sent_a.append(qa["answer"])
                trp_q.append(qa["q_trp"])
                trp_a.append(qa["a_trp"])

            example = InputExample(
                example_id = int(qa["idx"]),
                label = qa["label"],
                sent_q = sent_q, sent_a = sent_a,
                trp_q = trp_q, trp_a = trp_a
            )

            examples.append(example)
        return examples


class InputFeatures(object):
    def __init__(self, example_id, sent_features, trp_features, label):
        self.example_id = example_id
        self.sent_features = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in sent_features
        ]
        self.trp_features = [
            [
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
                for input_ids, attention_mask, token_type_ids in trp_feature
            ]
            for trp_feature in trp_features
        ]
        self.label = label



def convert_examples_to_features(examples, tokenizer,
                                 max_seq_length, max_trp_length,
                                 label_list=None, pad_token=0, pad_token_segment_id=0, mask_padding=0):
    """
    Loads a data file into a list of 'InputFeatures'.
    return:
        a list of task-specific ``InputFeatures`` which can be fed to the model.
        [CLS] + A + [SEP] + B + [SEP]
    """
    def _add_tokens(ex_index, tokens_a, tokens_b, tokens_c, max_length, trp):
        special_tokens_count = 3 if not trp else 4
        _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length - special_tokens_count)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        token_type_ids = [0] * len(tokens)

        # https://github.com/yao8839836/kg-bert/issues/10
        tokens += tokens_b + ["[SEP]"]
        token_type_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            token_type_ids += [0] * (len(tokens_c) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        attention_mask += [mask_padding] * padding_length
        token_type_ids += [pad_token_segment_id] * padding_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % (tokens))
            logger.info("guid: %s" % (example.example_id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        return (input_ids, attention_mask, token_type_ids)



    def _convert_bert_input(ex_index, qa_feat, max_length, trp=False):
        feats = []
        for opt_idx, texts in enumerate(qa_feat):
            question, answer = texts
            if trp:
                feat = []
                for trp_id in range(len(answer)):
                    # add more trp for convinence only take first trp answer[0]
                    tokens_a = tokenizer.tokenize(answer[trp_id]["subject"]) if len(answer) > 0 else tokenizer.tokenize(" ")
                    tokens_b = tokenizer.tokenize(answer[trp_id]["relation"]) if len(answer) > 0 else tokenizer.tokenize(" ")
                    tokens_c = tokenizer.tokenize(answer[trp_id]["object"]) if len(answer) > 0 else tokenizer.tokenize(" ")
                    feat_i = _add_tokens(ex_index, tokens_a, tokens_b, tokens_c, max_length, trp)
                    feat.append(feat_i)
            else:
                tokens_a = tokenizer.tokenize(question)
                # if math.isnan(answer):
                #     tokens_b = tokenizer.tokenize("")
                # else:
                # print("qqqq", question)
                tokens_b = tokenizer.tokenize(answer)
                # print("aaaa", answer)

                tokens_c = None
                feat = _add_tokens(ex_index, tokens_a, tokens_b, tokens_c, max_length, trp)
            feats.append(feat)
            # feats.append((input_ids, attention_mask, token_type_ids))

        return feats



    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        zip_sents = zip(example.sent_q, example.sent_a)
        sent_features = _convert_bert_input(ex_index, zip_sents, max_seq_length)
        zip_trps = zip(example.trp_q, example.trp_a)
        trp_features = _convert_bert_input(ex_index, zip_trps, max_trp_length, trp=True)

        label = label_map[example.label]

        features.append(
            InputFeatures(
                example_id=example.example_id,
                sent_features=sent_features,
                trp_features=trp_features,
                label=label
            )
        )
    return features



def load_examples_and_cache_features(data_dir, data_name, tokenizer, max_seq_length, max_trp_length, trp_num, dev=False, test=False):
    processor = Task2Processor()
    label_list = processor.get_labels()
    if dev:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, "cached_{}_{}".format(cached_mode, str(max_seq_length)))
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        # Datafile -> Examples
        if dev:
            examples = processor.get_dev_examples(data_dir=data_dir, data_name=data_name)
        elif test:
            examples = processor.get_test_examples(data_dir=data_dir, data_name=data_name)
        else:
            examples = processor.get_train_examples(data_dir=data_dir, data_name=data_name)
        # Examples -> Features
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, label_list=label_list,
                                                max_seq_length=max_seq_length, max_trp_length=max_trp_length)
        torch.save(features, cached_features_file)

    # Features -> Dataset
    ## convert features to Tensors
    all_sent_input_ids = torch.tensor(select_field(features, "input_ids", type="sent"), dtype=torch.long)
    all_sent_attention_mask = torch.tensor(select_field(features, "attention_mask", type="sent"), dtype=torch.long)
    all_sent_token_type_ids = torch.tensor(select_field(features, "token_type_ids", type="sent"), dtype=torch.long)
    all_trp_input_ids = torch.tensor(select_field(features, "input_ids", type="trp", trp_num=trp_num), dtype=torch.long)
    all_trp_attention_mask = torch.tensor(select_field(features, "attention_mask", type="trp", trp_num=trp_num), dtype=torch.long)
    all_trp_token_type_ids = torch.tensor(select_field(features, "token_type_ids", type="trp", trp_num=trp_num), dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_guids = torch.tensor([f.example_id for f in features], dtype=torch.long)
    ## Build Dataset
    dataset = TensorDataset(all_sent_input_ids, all_sent_attention_mask, all_sent_token_type_ids,
                            all_trp_input_ids, all_trp_attention_mask, all_trp_token_type_ids,
                            all_labels, all_guids)

    return dataset
























