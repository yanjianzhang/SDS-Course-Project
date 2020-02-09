import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
from keras.preprocessing import sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from nltk.translate.bleu_score import sentence_bleu
# device = "cpu"
# with the help of https://www.kaggle.com/kuldeep7688/simple-rnn-using-glove-embeddings-in-pytorch
class TextDataset(torch.utils.data.Dataset):
	'''
	Simple Dataset
	'''
	def __init__(self, data, word_encoder):
		num, statement, explaination, label, options = self.data_split(data, word_encoder)
		self.example_id = num
		self.statement = statement
		self.explain = explaination
		self.label = label
		self.options = options

	def data_split(self, data, word_encoder):
		x1 = []; x2 = []; x3 = []; x4 = []; x5 = []
		label2index = {"A":0, "B":1, "C":2}
		for s in data:
			x1.append(int(s[0]))
			x2.append(s[1])
			x3.append(s[2])
			x4.append(label2index[s[3]])
			x5.append(s[4])
		return x1, x2, x3, x4, x5

	def __len__(self):
		return len(self.example_id)

	def __getitem__(self, idx):
		return [self.example_id[idx],self.statement[idx],self.explain[idx],self.label[idx],self.options[idx]]


class MyCollator(object):
	'''
	Yields a batch from a list of Items
	Args:
	test : Set True when using with test data loader. Defaults to False
	percentile : Trim sequences by this percentile
	'''
	def __init__(self, word_encoder, test=False,percentile=100):
		self.test = test
		self.percentile = percentile
		self.word_encoder = word_encoder
	def __call__(self, batch):

		example_id = [item[0] for item in batch]
		statement = [self.word_encoder.encode(item[1]) for item in batch]
		explain = [self.word_encoder.encode(random.choice(item[2])) for item in batch ]

		label = [item[3] for item in batch ]
		sep_idx = self.word_encoder.encode("<sep>")[0]

		options = [[self.word_encoder.encode("<sep>".join([item[1],item[4][i]])) for item in batch ] for i in range(3) ]

		explain_len = max([len(x) for x in explain])
		# options_len = max([len(x) for s in options for x in s])
		state_len = max([len(x) for x in statement])
		statement = sequence.pad_sequences(statement, maxlen=state_len, padding='post')
		# for i in range(len(batch)):
		# 	explain[i] = s  equence.pad_sequences(explain[i], maxlen=explain_len, padding='post', value=self.word_encoder.encode("<eos>")[0])
		explain = sequence.pad_sequences(explain, maxlen=explain_len, padding='post',
											value=self.word_encoder.encode("<eos>")[0])

		statement = torch.tensor(statement).type(torch.cuda.LongTensor)

		return [example_id,statement,explain,label,options]

class MyCollator_dev(object):
	'''
	Yields a batch from a list of Items
	Args:
	test : Set True when using with test data loader. Defaults to False
	percentile : Trim sequences by this percentile
	'''
	def __init__(self, word_encoder, test=False,percentile=100):
		self.test = test
		self.percentile = percentile
		self.word_encoder = word_encoder
	def __call__(self, batch):

		example_id = [item[0] for item in batch]
		statement = [self.word_encoder.encode(item[1]) for item in batch]
		explain = [item[2] for item in batch]             # the explaination is

		label = [item[3] for item in batch ]
		sep_idx = self.word_encoder.encode("<sep>")[0]

		options = [[self.word_encoder.encode("<sep>".join([item[1],item[4][i]])) for item in batch ] for i in range(3) ]

		explain_len = max([len(x) for x in explain])

		state_len = max([len(x) for x in statement])
		statement = sequence.pad_sequences(statement, maxlen=state_len, padding='post')

		# explain = sequence.pad_sequences(explain, maxlen=explain_len, padding='post',
		# 									value=self.word_encoder.encode("<eos>")[0])
		statement = torch.tensor(statement).type(torch.cuda.LongTensor)
		return [example_id,statement,explain,label,options]

class MyCollator_adv(object):
	'''
	Yields a batch from a list of Items
	Args:
	test : Set True when using with test data loader. Defaults to False
	percentile : Trim sequences by this percentile
	'''
	def __init__(self, word_encoder, test=False,percentile=100):
		self.test = test
		self.percentile = percentile
		self.word_encoder = word_encoder
	def __call__(self, batch):

		example_id = [item[0] for item in batch]
		statement = [self.word_encoder.encode(item[1]) for item in batch]
		explain = [self.word_encoder.encode(random.choice(item[2])) for item in batch ]

		label = [item[3] for item in batch ]
		options = [[([item[1],item[4][i]]) for item in batch] for i in range(3)]
		explain_len = max([len(x) for x in explain])
		state_len = max([len(x) for x in statement])
		statement = sequence.pad_sequences(statement, maxlen=state_len, padding='post')
		explain = sequence.pad_sequences(explain, maxlen=explain_len, padding='post',
											value=self.word_encoder.encode("<eos>")[0])
		statement = torch.tensor(statement).type(torch.cuda.LongTensor)
		return [example_id,statement,explain,label,options]

def bleu(targets,outputs):
	outputs = outputs

	reference = targets
	candidate = outputs
	# score = sentence_bleu(reference, candidate)
	bleu1 = sentence_bleu(reference, candidate, weights=(1.0, 0, 0, 0))
	bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
	bleu3 = sentence_bleu(reference, candidate, weights=(1./3., 1./3., 1./3., 0))
	bleu4 =  sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
	return bleu1, bleu2, bleu3, bleu4