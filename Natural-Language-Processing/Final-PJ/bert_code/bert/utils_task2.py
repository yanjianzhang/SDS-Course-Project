import torch
import os
from transformers import *
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm
import csv




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
    test_file = os.path.join(args.data_dir, "test" + args.data_name + ".tsv")
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


def select_field(features, field):
    return [
        [
            sent[field]
            for sent in feature.choices_features
        ]
        for feature in features #[8000, 3, max_len]
    ]


class InputExample(object):
    """A single training/test example for multiple choice"""
    def __init__(self, example_id, statement, options, extras=None, label=None):
        """Constructs an InputExample."""
        self.example_id = example_id
        self.statement = statement
        self.options = options
        self.extras = extras  # extras is not None when data_name is not "
        self.label = label


class Task2Processor(DataProcessor):
    """
    tsv origianl data file into examples
    """
    def get_train_examples(self, data_dir, data_name):
        data_name = "train" + data_name + ".tsv"
        logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
        return self._create_examples(lines=self._read_tsv(os.path.join(data_dir, data_name)))

    def get_test_examples(self, data_dir, data_name):
        data_name = "test" + data_name + ".tsv"
        logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
        return self._create_examples(lines=self._read_tsv(os.path.join(data_dir, data_name)))

    def get_dev_examples(self, data_dir, data_name):
        data_name = "dev" + data_name + ".tsv"
        logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
        return self._create_examples(lines=self._read_tsv(os.path.join(data_dir, data_name)))

    def get_labels(self):
        return ["A", "B", "C"]

    def _create_examples(self, lines):
        examples = []
        for line in lines:
            example = InputExample(
                example_id = int(line[0]),
                statement = [line[2], line[2], line[2]],
                options = [line[3], line[4], line[5]],
                # extras = [line[6], line[7], line[8]] if len(line) > 6 else None,
                extras = [line[6], line[6], line[6]] if len(line) > 6 else None,
                label = line[1])
            examples.append(example)
        return examples


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in choices_features
        ]
        self.label = label


def convert_examples_to_features(examples, tokenizer,
                                 max_length, label_list=None,
                                 pad_token=0, pad_token_segment_id=0,
                                 mask_padding=0):
    """
    Loads a data file into a list of 'InputFeatures'.
    return:
        a list of task-specific ``InputFeatures`` which can be fed to the model.
        [CLS] + A + [SEP] + B + [SEP]
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        if example.extras is None:
            zip_texts = zip(example.statement, example.options)
        else:
            zip_texts = zip(example.statement, example.options, example.extras)
        for option_idx, texts in enumerate(zip_texts):
            if len(texts) == 3:
                statement, option, extra = texts
                tokens_b = tokenizer.tokenize(option)
                tokens_c = tokenizer.tokenize(extra)
                # q [SEP] p1p2p3 [SEP] e
                # tokens_b = tokenizer.tokenize(extra)
                # tokens_c = tokenizer.tokenize(option)

                # p1p2p3 [SEP] q [SEP] e
                # tokens_a = tokenizer.tokenize(extra)
                # tokens_b = tokenizer.tokenize(statement)
                # tokens_c = tokenizer.tokenize(option)
            else:
                statement, option = texts
                tokens_b = tokenizer.tokenize(option)
                tokens_c = None

                # q [SEP] p1p2p3 [SEP] e
                # tokens_b = None
                # tokens_c = tokenizer.tokenize(option)

                # p1p2p3 [SEP] q [SEP] e
                # tokens_a = tokenizer.tokenize("")
                # tokens_b = tokenizer.tokenize(statement)
                # tokens_c = tokenizer.tokenize(option)

            tokens_a = tokenizer.tokenize(statement)

            special_tokens_count = 3 if len(texts)==3 else 4
            _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length-special_tokens_count)

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
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            # Save all features as one tuple for a specific option
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % (tokens))
            logger.info("guid: %s" % (example.example_id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label
            )
        )
    return features



def load_examples_and_cache_features(data_dir, data_name, tokenizer, max_seq_length, dev=False, test=False):
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
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, label_list=label_list, max_length=max_seq_length)
        torch.save(features, cached_features_file)

    # Features -> Dataset
    ## convert features to Tensors
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_attention_mask = torch.tensor(select_field(features, "attention_mask"), dtype=torch.long)
    all_token_type_ids = torch.tensor(select_field(features, "token_type_ids"), dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_guids = torch.tensor([f.example_id for f in features], dtype=torch.long)
    ## Build Dataset
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guids)

    return dataset
















