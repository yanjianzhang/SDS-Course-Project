import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.functional import sigmoid
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
	def __init__(self, input_size, embed_size, hidden_size, embedding = None, n_layers=1, dropout=0.5, bidirectional=True):
		super(EncoderRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.embed_size = embed_size
		self.n_layers = n_layers
		self.dropout = dropout
		self.bidirectional = bidirectional
		if embedding:
			self.embedding = embedding.to(device)
		else:
			self.embedding = nn.Embedding(input_size,embed_size).to(device)
		self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidirectional, batch_first= True)

	def forward(self, input_seqs, input_lengths, hidden=None):
		'''
		:param input_seqs:
			Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
		:param input:
			list of sequence length
		:param hidden:
			initial state of GRU
		:returns:
			GRU outputs in shape (T,B,hidden_size(H))
			last hidden stat of RNN(i.e. last output for GRU)
		'''
		embedded = self.embedding(input_seqs)
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first= True, enforce_sorted=False)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first= True)  # unpack (back to padded)
		if self.bidirectional:
			outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
		return outputs, hidden


class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		self.hidden_size = hidden_size
		self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
		self.v = nn.Parameter(torch.rand(hidden_size))
		stdv = 1. / math.sqrt(self.v.size(0))
		self.v.data.normal_(mean=0, std=stdv)

	def forward(self, hidden, encoder_outputs, src_len=None):
		'''
		:param hidden:
			previous hidden state of the decoder, in shape (layers*directions,B,H)
		:param encoder_outputs:
			encoder outputs from Encoder, in shape (T,B,H)
		:param src_len:
			used for masking. NoneType or tensor in shape (B) indicating sequence length
		:return
			attention energies in shape (B,T)
		'''
		max_len = encoder_outputs.size(0)
		this_batch_size = encoder_outputs.size(1)
		H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
		encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
		attn_energies = self.score(H, encoder_outputs)  # compute attention score

		if src_len is not None:
			mask = []
			for b in range(src_len.size(0)):
				mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
			mask = torch.ByteTensor(mask).unsqueeze(1)  # [B,1,T]
			attn_energies = attn_energies.masked_fill(mask, -1e18)
		# print("atten_en", attn_energies.size())
		return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

	def score(self, hidden, encoder_outputs):
		energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
		energy = energy.transpose(2, 1)  # [B*H*T]
		v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
		energy = torch.bmm(v, energy)  # [B*1*T]
		return energy.squeeze(1)  # [B*T]


class BahdanauAttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, embed_size, output_size, embedding = None, n_layers=2, dropout_p=0.1):
		super(BahdanauAttnDecoderRNN, self).__init__()
		# Define parameters
		self.hidden_size = hidden_size
		self.embed_size = embed_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout_p = dropout_p
		# Define layers
		if embedding:
			self.embedding = embedding
		else:
			self.embedding = nn.Embedding(output_size, embed_size)
		self.dropout = nn.Dropout(dropout_p)
		self.attn = Attn('concat', hidden_size)
		self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
		# self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)

	def forward(self, word_input, last_hidden, encoder_outputs):
		'''
		:param word_input:
			word input for current time step, in shape (B)
		:param last_hidden:
			last hidden stat of the decoder, in shape (layers*direction*B*H)
		:param encoder_outputs:
			encoder outputs in shape (T*B*H)
		:return
			decoder output
		Note: we run this one step at a time i.e. you should use a outer loop
			to process the whole sequence
		Tip(update):
		EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
		different from that of DecoderRNN
		You may have to manually guarantee that they have the same dimension outside this function,
		e.g, select the encoder hidden state of the foward/backward pass.
		'''
		# Get the embedding of the current input word (last output word)
		word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1)  # (1,B,V)
		word_embedded = self.dropout(word_embedded)
		# Calculate attention weights and apply to encoder outputs
		attn_weights = self.attn(last_hidden[-1], encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
		context = context.transpose(0, 1)  # (1,B,V)
		# Combine embedded input word and attended context, run through RNN
		rnn_input = torch.cat((word_embedded, context), 2)
		# rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
		output, hidden = self.gru(rnn_input, last_hidden)
		output = output.squeeze(0)  # (1,B,V)->(B,V)
		# context = context.squeeze(0)
		# update: "context" input before final layer can be problematic.
		# output = F.log_softmax(self.out(torch.cat((output, context), 1)))
		# print("output", self.out(output).size())
		output = F.log_softmax(self.out(output), dim=1)
		# Return final output, hidden state
		return output, hidden



class Classifier(nn.Module):
	def __init__(self, hidden_size, classes):
		super(Classifier, self).__init__()
		self.linear = nn.Linear(hidden_size, classes)
		# self.softmax = nn.Softmax(dim = 1)
		self.softmax = nn.Softmax(dim=1)
	def forward(self, input):
		output = self.linear(input)
		return output



class Ensemble_D(nn.Module):
	def __init__(self, input_size, hidden_size, embedding):
		super(Ensemble_D, self).__init__()


		self.state_hidden = None
		self.reason_hidden = None
		if not embedding:
			self.embedding = nn.Embedding(input_size, hidden_size)
		else:
			self.embedding = embedding
		self.state_encoder = EncoderRNN(input_size, hidden_size, embedding=self.embedding)
		self.reason_encoder = EncoderRNN(input_size, hidden_size, embedding=self.embedding)
		self.classifier = Classifier(hidden_size * 2, 2)
	def forward(self, state, reason):
		output, self.state_hidden = self.state_encoder(state)
		output, self.reason_hidden = self.reason_encoder(reason)
		output = self.classifier(torch.cat((self.state_hidden.squeeze(0), self.reason_hidden.squeeze(0)),1))
		return output


class AttnDecoderRNN(nn.Module):
	def __init__(self,
				 hidden_size,
				 output_size,
				 dropout=0.5, embedding = None):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout = dropout
		if embedding:
			self.embedding = embedding
		else:
			self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size*2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.atten_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.attn_score = Attention(self.hidden_size)
	def forward(self, input, hidden, encoder_output):
		embedded = self.embedding(input).view(1, 1, -1)
		# embedded = input
		# embedded = self.embedding(embedded)
		if device == "cuda":
			encoder_output = encoder_output.type(torch.cuda.FloatTensor)
		# encoder_output = encoder_output.unsqueeze(1)
		score = self.attn_score(hidden,encoder_output)
		context = score.unsqueeze(1).bmm(encoder_output.transpose(0, 1)).transpose(0, 1)
		# print(embedded.size(), context.size())
		emb_con = torch.cat((embedded, context), dim=2)

		output, hidden = self.gru(emb_con, hidden)
		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, score

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


class Attention(nn.Module):
	def __init__(self, hidden_dim):
		super(Attention, self).__init__()
		self.hidden_dim = hidden_dim
		self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
		self.v = nn.Parameter(torch.rand(hidden_dim))
		self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

	def forward(self, hidden, encoder_outputs):
		#  encoder_outputs:(seq_len, batch_size, hidden_size)
		#  hidden:(num_layers * num_directions, batch_size, hidden_size)
		max_len = encoder_outputs.size(0)
		h = hidden[-1].repeat(max_len, 1, 1)
		# (seq_len, batch_size, hidden_size)
		attn_energies = self.score(h, encoder_outputs)  # compute attention score
		return F.softmax(attn_energies, dim=1)  # normalize with softmax

	def score(self, hidden, encoder_outputs):
		# (seq_len, batch_size, 2*hidden_size)-> (seq_len, batch_size, hidden_size)

		energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
		# print("energy",energy.size())
		energy = energy.permute(1, 2, 0)  # (batch_size, hidden_size, seq_len)
		v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
		energy = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
		return energy.squeeze(1)  # (batch_size, seq_len)




class MultiModalDataset(Dataset):
	def __init__(self, data, word_encoder):
		from PIL import Image
		from torchvision import transforms
		img_preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		text_input, image_input, label_idx, event_idx  = [], [], [], []
		events = []
		for text, image_url, label, event in data:
			text_input.append(word_encoder.encode(text))
			image = Image.open(image_url)
			image_input.append(img_preprocess(image))
			label_idx.append(1 if label.lower == "true" else 0)
			if event not in events: events.append(event)
			event_idx.append(events.index(event))
		self.text_input, self.image_input, self.label_idx, self.event_idx = text_input, image_input, label_idx, event_idx

	def __len__(self):
		return len(self.text_input)

	def __getitem__(self, idx):
		return [self.text_input[idx],self.image_input[idx],self.label_idx[idx],self.event_idx[idx]]