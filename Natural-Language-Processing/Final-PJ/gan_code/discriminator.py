import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

class Discriminator(nn.Module):

	def __init__(self,gru_hidden_dim, hidden_dim, dropout=0.2):
		super(Discriminator, self).__init__()
		self.gru_hidden_dim = gru_hidden_dim
		self.hidden_dim = hidden_dim
		self.gru2hidden = nn.Linear(gru_hidden_dim, hidden_dim)
		self.dropout_linear = nn.Dropout(p=dropout)
		self.hidden2out = nn.Linear(hidden_dim, 1)
		self.softmax = nn.Softmax(dim=1)

	# def init_hidden(self, batch_size):
	# 	h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))
	#
	# 	if self.gpu:
	# 		return h.cuda()
	# 	else:
	# 		return h

	def forward(self, batch_size, hiddens):
		# state_hidden = state_hidden.permute(1,0,2).view(-1, 2*self.gru_hidden_dim)
		 #                                                                    # (2 * B * H) -> (B * 2 * H)
		#
		# option_hidden = option_hidden.permute(1, 0, 2).view(-1, 2 * self.gru_hidden_dim)
		outs = []
		for i in range(len(hiddens)):
			hidden = hiddens[i]
			out = self.gru2hidden(hidden.squeeze(0))  # batch_size x 4*hidden_dim
			out = torch.tanh(out)
			out = self.dropout_linear(out)
			out = self.hidden2out(out)                                 # batch_size x 1
			# out = torch.sigmoid(out)
			if i == 0:
				outs = out
			else:
				outs = torch.cat((outs, out), 1)

		outs = self.softmax(outs)
		return outs

	def batchClassify(self, inp):
		"""
		Classifies a batch of sequences.

		Inputs: inp
			- inp: batch_size x seq_len

		Returns: out
			- out: batch_size ([0,1] score)
		"""

		h = self.init_hidden(inp.size()[0])
		out = self.forward(inp, h)
		return out.view(-1)

	def batchBCELoss(self, inp, target):
		"""
		Returns Binary Cross Entropy Loss for discriminator.

		 Inputs: inp, target
			- inp: batch_size x seq_len
			- target: batch_size (binary 1/0)
		"""

		loss_fn = nn.BCELoss()
		h = self.init_hidden(inp.size()[0])
		out = self.forward(inp, h)
		return loss_fn(out, target)

