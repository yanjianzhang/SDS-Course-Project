from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers
import Component
import pickle
from torchnlp.word_to_vector import GloVe

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
Unk_token = 3
CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 64

POS_NEG_SAMPLES = 10000

# GEN_EMBEDDING_DIM = 32
# GEN_HIDDEN_DIM = 32
# DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

PRE_TRAIN_DIS = 150
PRE_TRAIN_GEN = 100
ADV_TRAIN_EPOCHS = 30

from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import glove
from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Component import *
import itertools
import torchtext.data as dataTool
import torchtext
from seqGans_util import *
from tqdm import tqdm
teacher_forcing_ratio = 0.5
import random
from keras.preprocessing import sequence
from ranger import Ranger
from rouge import Rouge
from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors
rouge = Rouge()

def train_generator_MLE(state_encoder, reason_decoder, gen_optimizer, epochs):
	"""
	Max Likelihood Pretraining for the generator
	"""
	# state_encoder, reason_decoder, gen_optimizer, train_dataloader, MLE_TRAIN_EPOCHS

	for epoch in tqdm(range(epochs)):
		sys.stdout.flush()
		total_loss = 0
		count = 0
		for train_data in train_loader:
			num, statement, reason, label, options = train_data
			batch_size, input_size = statement.size()
			# reason = [random.choices(r) for r in reason]
			reason = torch.tensor(reason).type(torch.cuda.LongTensor)
			input_lengths = [len(x) for x in statement]


			encoder_outputs, hidden = state_encoder(statement,input_lengths)

			decoder_input = torch.tensor([[word_encoder.encode("<sos>")] for i in range(batch_size)]).type(torch.cuda.LongTensor)


			use_teaching_forcing = True if random.random(
			) < teacher_forcing_ratio else False

			target_length = reason.size()[1]

			encoder_outputs = encoder_outputs.permute(1,0,2) # -> (T*B*H)

			criterion = nn.NLLLoss()
			loss = 0
			reason_permute = reason.permute(1,0)   # (T * B)
			if use_teaching_forcing:
				for di in range(target_length):
					output, hidden = reason_decoder(decoder_input,hidden, encoder_outputs)
					# print("output",output,reason_permute[di])
					loss += criterion(output, reason_permute[di])
					decoder_input = reason_permute[di]  # this point might need change if reason matrix change
			else:
				for di in range(target_length):
					output, hidden = reason_decoder(decoder_input,hidden, encoder_outputs)
					topv, topi = output.topk(1)
					decoder_input = topi.squeeze().detach()
					loss += criterion(output, reason_permute[di])  # (B*V) (B)

			loss.backward()
			gen_optimizer.step()
			count += 1
			total_loss += loss.data.item()
			if epoch %2 == 1:
				pretrained_gen_path = "./model/pretrain_gen"+str(epoch + 1)+".pth"
				state = {"state_encoder": state_encoder.state_dict(), "reason_decoder": reason_decoder.state_dict(),"embed": embedding.state_dict()}
				torch.save(state, pretrained_gen_path)

		# each loss in a batch is loss per sample
		total_loss = total_loss / (count * BATCH_SIZE)

		print('\n average_train_NLL = %.4f' % (total_loss))

def dev_generator(state_encoder, reason_decoder, gen_optimizer, name, max_length = 25):
	"""
	Max Likelihood Pretraining for the generator
	"""
	# state_encoder, reason_decoder, gen_optimizer, train_dataloader, MLE_TRAIN_EPOCHS

	if name == "adv_model":
		epochs = list(range(2, ADV_TRAIN_EPOCHS + 1, 2))
	else:
		epochs = list(range(2, PRE_TRAIN_GEN + 1, 2))

	total_bleu = []
	eos_idx = word_encoder.encode("<eos>")
	for epoch in tqdm(epochs):
		total_loss = 0
		count = 0
		if name == "adv_model":
			adv_model_path = "./model/adv_model" + str(epoch) + ".pth"
			dis_option_encoder.load_state_dict(torch.load(adv_model_path)["dis_option_encoder"])
			embedding.load_state_dict(torch.load(adv_model_path)["embed"])
			dis.load_state_dict(torch.load(adv_model_path)["dis"])
		else:
			pretrained_gen_path = "./model/"+name + str(epoch)+".pth"
			state_encoder.load_state_dict(torch.load(pretrained_gen_path)["state_encoder"])
			reason_decoder.load_state_dict(torch.load(pretrained_gen_path)["reason_decoder"])
			embedding.load_state_dict(torch.load(pretrained_gen_path)["embed"])
		correct_explains = []
		outputs = []
		for train_data in test_loader:
			num, statement, reason, label, options = train_data
			correct_explains += reason
			batch_size, input_size = statement.size()
			reason = [word_encoder.encode(random.choice(r)) for r in reason]
			reason_len = max([len(x) for x in reason])
			reason = sequence.pad_sequences(reason, maxlen=reason_len, padding='post',
											value=word_encoder.encode("<eos>")[0])
			reason = torch.tensor(reason).type(torch.cuda.LongTensor)
			input_lengths = [len(x) for x in statement]

			encoder_outputs, hidden = state_encoder(statement, input_lengths)

			decoder_input = torch.tensor([[word_encoder.encode("<sos>")] for i in range(batch_size)]).type(
				torch.cuda.LongTensor)

			target_length = reason.size()[1]

			encoder_outputs = encoder_outputs.permute(1, 0, 2)  # -> (T*B*H)
			decoder_outputs = [[] for i in range(batch_size)]
			target_length = reason.size()[1]
			loss = 0
			reason_permute = reason.permute(1, 0)  # (T * B)
			for di in range(max_length):
				output, hidden = reason_decoder(decoder_input, hidden, encoder_outputs)
				topv, topi = output.topk(1)
				all_end = True
				for i in range(batch_size):
					# print(topi.squeeze()[i].item())
					if topi.squeeze()[i].item() != eos_idx:
						decoder_outputs[i].append(word_encoder.decode(np.array([topi.squeeze()[i].item()])))
						all_end = False
				if all_end:
					break
				decoder_input = topi.squeeze().detach()

			outputs += decoder_outputs
		scoresSum1, scoresSum2, scoresSum3, scoresSum4 = 0, 0, 0, 0
		n = len(outputs)
		for output, reasons in zip(outputs, correct_explains):
			reasons = [reason.split(" ") for reason in reasons]
			b1, b2, b3, b4 = bleu(reasons, output)
			scoresSum1, scoresSum2, scoresSum3, scoresSum4 = b1 + scoresSum1, b2 + scoresSum2, b3 + scoresSum3, b4 + scoresSum4
		total_bleu.append(((scoresSum1 / n), (scoresSum2 / n), (scoresSum3 / n),(scoresSum4 / n)))
	total_bleu1 = [x[0] for x in total_bleu]
	best_bleu_idx = total_bleu1.index(max(total_bleu1))
	print("best bleu score in dev", total_bleu[best_bleu_idx], "in epoch", epochs[best_bleu_idx])

	best_epoch = epochs[best_bleu_idx]
	print("Evaluating test data BELU in best model")

	epoch = best_epoch

	if name == "adv_model":
		adv_model_path = "./model/adv_model" + str(epoch) + ".pth"
		state_encoder.load_state_dict(torch.load(adv_model_path)["state_encoder"])
		reason_decoder.load_state_dict(torch.load(adv_model_path)["reason_decoder"])
		embedding.load_state_dict(torch.load(adv_model_path)["embed"])
	else:
		pretrained_gen_path = "./model/" + name + str(epoch) + ".pth"
		state_encoder.load_state_dict(torch.load(pretrained_gen_path)["state_encoder"])
		reason_decoder.load_state_dict(torch.load(pretrained_gen_path)["reason_decoder"])
		embedding.load_state_dict(torch.load(pretrained_gen_path)["embed"])

	correct_explains = []
	outputs = []
	for train_data in test_loader:
		num, statement, reason, label, options = train_data
		correct_explains += reason
		batch_size, input_size = statement.size()
		reason = [word_encoder.encode(random.choice(r)) for r in reason]
		reason_len = max([len(x) for x in reason])
		reason = sequence.pad_sequences(reason, maxlen=reason_len, padding='post',
										value=word_encoder.encode("<eos>")[0])
		reason = torch.tensor(reason).type(torch.cuda.LongTensor)
		input_lengths = [len(x) for x in statement]

		encoder_outputs, hidden = state_encoder(statement, input_lengths)

		decoder_input = torch.tensor([[word_encoder.encode("<sos>")] for i in range(batch_size)]).type(
			torch.cuda.LongTensor)

		target_length = reason.size()[1]

		encoder_outputs = encoder_outputs.permute(1, 0, 2)  # -> (T*B*H)
		decoder_outputs = [[] for i in range(batch_size)]
		target_length = reason.size()[1]
		loss = 0
		reason_permute = reason.permute(1, 0)  # (T * B)
		for di in range(max_length):
			output, hidden = reason_decoder(decoder_input, hidden, encoder_outputs)
			topv, topi = output.topk(1)
			all_end = True
			for i in range(batch_size):
				# print(topi.squeeze()[i].item())
				if topi.squeeze()[i].item() != eos_idx:
					decoder_outputs[i].append(word_encoder.decode(np.array([topi.squeeze()[i].item()])))
					all_end = False
			if all_end:
				break
			decoder_input = topi.squeeze().detach()

		outputs += decoder_outputs

	with open("text_result_"+ name +".txt","w") as test_file:
		for output, reasons in zip(outputs, correct_explains):
			test_file.write(" ".join(output)+"\n"+"\t".join(reasons)+"\n")

	scoresSum1, scoresSum2, scoresSum3, scoresSum4 = 0, 0, 0, 0
	n = len(outputs)
	for output, reasons in zip(outputs, correct_explains):
		reasons = [reason.split(" ") for reason in reasons]
		b1, b2, b3, b4 = bleu(reasons, output)
		scoresSum1, scoresSum2, scoresSum3, scoresSum4 = b1 + scoresSum1, b2 + scoresSum2, b3 + scoresSum3, b4 + scoresSum4
	print("total bleu 1 2 3 4 {:.3f},{:.3f},{:.3f},{:.3f}".format((scoresSum1 / n), (scoresSum2 / n), (scoresSum3 / n),(scoresSum4 / n)))
	return best_epoch

def train_discriminator(dis, dis_option_encoder, dis_optimizer, epochs):
	"""
	Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
	Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
	"""

	# generating a small validation set before training (using oracle and generator)

	for epoch in tqdm(range(epochs)):
		# print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
		sys.stdout.flush()
		total_loss = 0
		total_acc = 0
		count = 0
		for train_data in train_loader:
			num, statement, reason, label, options = train_data
			batch_size, input_size = statement.size()
			# state_input_lengths = [len(x) for x in statement]
			option_input_lengths = [[len(x) for x in option]  for option in options]
			option_lens = [max([len(x) for x in option]) for option in options]
			options = [sequence.pad_sequences(option, maxlen=option_len, padding='post') for option, option_len in zip(options,option_lens)]
			options = [torch.tensor(option).type(torch.cuda.LongTensor) for option in options]
			option_hiddens = []
			for i in range(3):
				encoder_outputs, option_hidden = dis_option_encoder(options[i], option_input_lengths[i])
				option_hiddens.append(option_hidden)


			dis_optimizer.zero_grad()
			out = dis(batch_size, option_hiddens)
			loss_fn = nn.CrossEntropyLoss()
			label = torch.tensor(label).type(torch.cuda.LongTensor)
			loss = loss_fn(out, label)
			loss.backward()
			dis_optimizer.step()
			total_loss += loss.data.item()
			total_acc += (out.argmax(1) == label).sum().item()

			sys.stdout.flush()
			count += 1
		if epoch % 2 == 1:
			pretrained_dis_path = "./model/pretrain_dis"+str(epoch+1)+".pth"
			state = {"dis_option_encoder": dis_option_encoder.state_dict(), "dis": dis.state_dict(),'embed': embedding.state_dict()}
			torch.save(state, pretrained_dis_path)

		print('\n average_loss = %.4f, train_acc = %.4f' % (total_loss/(count * BATCH_SIZE), total_acc/(count * BATCH_SIZE)))

	total_loss /= (count * BATCH_SIZE)
	total_acc /= (count * BATCH_SIZE)

	print('\n average_loss = %.4f, train_acc = %.4f' % (total_loss, total_acc))

def dev_discriminator(dis, dis_option_encoder, name):
	"""
	Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
	Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
	"""

	sys.stdout.flush()
	total_accs = []

	if name == "adv_model":
		epochs = list(range(2, ADV_TRAIN_EPOCHS + 1, 2))
	else:
		epochs = list(range(2, PRE_TRAIN_DIS + 1, 2))
	for epoch in tqdm(epochs):
		total_acc = 0
		count = 0
		if name == "adv_model":
			adv_model_path = "./model/adv_model" + str(epoch) + ".pth"
			dis_option_encoder.load_state_dict(torch.load(adv_model_path)["dis_option_encoder"])
			embedding.load_state_dict(torch.load(adv_model_path)["embed"])
			dis.load_state_dict(torch.load(adv_model_path)["dis"])
		else:
			pretrained_dis_path = "./model/"+ name + str(epoch) + ".pth"
			dis_option_encoder.load_state_dict(torch.load(pretrained_dis_path)["dis_option_encoder"])
			embedding.load_state_dict(torch.load(pretrained_dis_path)["embed"])
			dis.load_state_dict(torch.load(pretrained_dis_path)["dis"])
		for dev_data in dev_loader:
			num, statement, reason, label, options = dev_data
			batch_size, input_size = statement.size()
			option_input_lengths = [[len(x) for x in option] for option in options]
			option_lens = [max([len(x) for x in option]) for option in options]
			options = [sequence.pad_sequences(option, maxlen=option_len, padding='post') for option, option_len in
					   zip(options, option_lens)]
			options = [torch.tensor(option).type(torch.cuda.LongTensor) for option in options]
			option_hiddens = []
			for i in range(3):
				encoder_outputs, option_hidden = dis_option_encoder(options[i], option_input_lengths[i])
				option_hiddens.append(option_hidden)

			dis_optimizer.zero_grad()
			out = dis(batch_size, option_hiddens)
			loss_fn = nn.CrossEntropyLoss()
			label = torch.tensor(label).type(torch.cuda.LongTensor)
			loss = loss_fn(out, label)
			dis_optimizer.step()
			total_acc += (out.argmax(1) == label).sum().item()

			sys.stdout.flush()
			count += 1
		total_acc /= count * BATCH_SIZE
		total_accs.append(total_acc)
	best_epoch = epochs[total_accs.index(max(total_accs))]
	print('\n best dev_acc = %.4f' % (max(total_accs)))
	print('\n in epoch',best_epoch)
	print("best epoch in test set")
	count = 0
	if name == "adv_model":
		adv_model_path = "./model/adv_model" + str(best_epoch) + ".pth"
		dis_option_encoder.load_state_dict(torch.load(adv_model_path)["dis_option_encoder"])
		embedding.load_state_dict(torch.load(adv_model_path)["embed"])
		dis.load_state_dict(torch.load(adv_model_path)["dis"])
	else:
		pretrained_dis_path = "./model/" + name + str(best_epoch) + ".pth"
		dis_option_encoder.load_state_dict(torch.load(pretrained_dis_path)["dis_option_encoder"])
		embedding.load_state_dict(torch.load(pretrained_dis_path)["embed"])
		dis.load_state_dict(torch.load(pretrained_dis_path)["dis"])
	for dev_data in test_loader:
		num, statement, reason, label, options = dev_data
		batch_size, input_size = statement.size()
		option_input_lengths = [[len(x) for x in option] for option in options]
		option_lens = [max([len(x) for x in option]) for option in options]
		options = [sequence.pad_sequences(option, maxlen=option_len, padding='post') for option, option_len in
		           zip(options, option_lens)]
		options = [torch.tensor(option).type(torch.cuda.LongTensor) for option in options]
		option_hiddens = []
		for i in range(3):
			encoder_outputs, option_hidden = dis_option_encoder(options[i], option_input_lengths[i])
			option_hiddens.append(option_hidden)

		dis_optimizer.zero_grad()
		out = dis(batch_size, option_hiddens)
		loss_fn = nn.CrossEntropyLoss()
		label = torch.tensor(label).type(torch.cuda.LongTensor)
		loss = loss_fn(out, label)
		dis_optimizer.step()
		total_acc += (out.argmax(1) == label).sum().item()

		sys.stdout.flush()
		count += 1
	total_acc /= count * BATCH_SIZE
	print("test accuracy:", total_acc)
	return best_epoch

def train_generator_PG(dis,  dis_option_encoder, state_encoder, reason_decoder, gen_optimizer, epochs, beta = 15, max_length = 25):
	"""
	The generator is trained using policy gradients, using the reward from the discriminator.
	Training is done for num_batches batches.
	"""
	gen_optimizer = Ranger(itertools.chain(state_encoder.parameters(), reason_decoder.parameters()), lr=1e-3,
						   weight_decay=1e-4)

	for epoch in tqdm(range(epochs)):
		sys.stdout.flush()
		total_mle_loss = 0
		total_label_loss = 0
		total_acc = 0
		count = 0
		for train_data in train_loader_adv:
			loss = 0
			gen_optimizer.zero_grad()
			mle_loss = 0
			label_loss = 0
			num, statement, reason, label, options = train_data
			reason = torch.tensor(reason).type(torch.cuda.LongTensor)
			batch_size, input_size = statement.size()
			state_input_lengths = [len(x) for x in statement]
			falselabel = [[0, 1, 2] for l in label]
			for i in range(len(label)): falselabel[i].remove(label[i])
			choice_label = [random.choice(ls) for ls in falselabel]

			input_lengths = [len(x) for x in statement]
			encoder_outputs, hidden = state_encoder(statement, input_lengths)

			decoder_input = torch.tensor([[word_encoder.encode("<sos>")] for i in range(batch_size)]).type(
				torch.cuda.LongTensor)

			encoder_outputs = encoder_outputs.permute(1, 0, 2)  # -> (T*B*H)
			target_length = reason.size()[1]
			criterion = nn.NLLLoss()
			decoder_outputs = [[] for i in range(batch_size)]
			eos_idx = word_encoder.encode("<eos>")
			reason_permute = reason.permute(1, 0)
			for di in range(max_length):
				output, hidden = reason_decoder(decoder_input, hidden, encoder_outputs)
				if di < target_length:
					mle_loss += criterion(output, reason_permute[di])
				topv, topi = output.topk(1)
				all_end = True
				for i in range(batch_size):
					# print(topi.squeeze()[i].item())
					if topi.squeeze()[i].item() != eos_idx:
						decoder_outputs[i].append(word_encoder.decode(np.array([topi.squeeze()[i].item()])))
						all_end = False
				if all_end:
					break
				decoder_input = topi.squeeze().detach()
			origin_decoder_outputs = decoder_outputs
			decoder_outputs = [" ".join(output) for output in decoder_outputs]


			correct_option = ["<sep>".join(options[label[i]][i]) for i in range(batch_size)]
			correct_option = [word_encoder.encode(option) for option in correct_option]
			for i in range(batch_size):
				options[choice_label[i]][i][1] = decoder_outputs[i]

			option_input_lengths = [[len(x) for x in option] for option in options]
			option_lens = [max(len(option[0]) + len(option[1]) + 1 for option in options[i]) for i in range(3)]
			options = [[word_encoder.encode("<sep>".join([x[0], x[1]])) for x in option] for option in options]
			options = [sequence.pad_sequences(option, maxlen=option_len, padding='post') for option, option_len in
					   zip(options, option_lens)]
			options = [torch.tensor(option).type(torch.cuda.LongTensor) for option in options]

			option_hiddens = []
			for i in range(3):
				encoder_outputs, option_hidden = dis_option_encoder(options[i], option_input_lengths[i])
				option_hiddens.append(option_hidden)

			out = dis(batch_size, option_hiddens)

			for i in range(batch_size):
				for j in range(min(len(origin_decoder_outputs[i]),correct_option[i].size()[0])):
					false_one_hot = torch.zeros(1,3)
					false_one_hot[0][falselabel[i]] = 1
					false_one_hot = false_one_hot.type(torch.cuda.FloatTensor)
					label_loss -= beta * out[i].mul(false_one_hot).sum()


			total_label_loss += label_loss
			total_mle_loss += mle_loss
			loss = label_loss + mle_loss
			loss.backward()
			gen_optimizer.step()
			count += 1
		total_mle_loss = total_mle_loss / (count * BATCH_SIZE)
		total_label_loss = total_label_loss / (count * BATCH_SIZE)

		print('\n average_train_NLL = ', total_mle_loss,' the label loss = ', total_label_loss)

def train_discriminator_adv(dis, dis_option_encoder, state_encoder, reason_decoder, dis_optimizer,  epochs, max_length = 25):
	"""
	Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
	Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
	"""

	# generating a small validation set before training (using oracle and generator)
	dis_optimizer = Ranger(
		itertools.chain(dis_option_encoder.parameters(), dis.parameters()), lr=1e-3,
		weight_decay=1e-4)
	for epoch in tqdm(range(epochs)):
		sys.stdout.flush()
		total_loss = 0
		total_acc = 0
		count = 0
		for train_data in train_loader_adv:
			num, statement, reason, label, options = train_data
			batch_size, input_size = statement.size()
			state_input_lengths = [len(x) for x in statement]
			falselabel = [[0,1,2] for l in label]
			for i in range(len(label)): falselabel[i].remove(label[i])
			choice_label = [random.choice(ls) for ls in falselabel]

			input_lengths = [len(x) for x in statement]
			encoder_outputs, hidden = state_encoder(statement, input_lengths)

			decoder_input = torch.tensor([[word_encoder.encode("<sos>")] for i in range(batch_size)]).type(
				torch.cuda.LongTensor)

			encoder_outputs = encoder_outputs.permute(1, 0, 2)  # -> (T*B*H)

			decoder_outputs = [[] for i in range(batch_size)]
			eos_idx = word_encoder.encode("<eos>")
			for di in range(max_length):
				output, hidden = reason_decoder(decoder_input, hidden, encoder_outputs)
				topv, topi = output.topk(1)
				all_end = True
				for i in range(batch_size):
					# print(topi.squeeze()[i].item())
					if topi.squeeze()[i].item() != eos_idx:
						decoder_outputs[i].append(word_encoder.decode(np.array([topi.squeeze()[i].item()])))
						all_end = False
				if all_end:
					break
				decoder_input = topi.squeeze().detach()
			decoder_outputs = [" ".join(output) for output in decoder_outputs]

			for i in range(batch_size):
				options[choice_label[i]][i][1] = decoder_outputs[i]


			option_input_lengths = [[len(x[0]) + len(x[1]) for x in option] for option in options]
			option_lens = [max(len(option[0])+len(option[1])+1 for option in options[i]) for i in range(3)]
			options = [[word_encoder.encode("<sep>".join([x[0],x[1]])) for x in option] for option in options]
			options = [sequence.pad_sequences(option, maxlen=option_len, padding='post') for option, option_len in zip(options, option_lens) ]
			options = [torch.tensor(option).type(torch.cuda.LongTensor) for option in options]


			option_hiddens = []
			for i in range(3):
				encoder_outputs, option_hidden = dis_option_encoder(options[i], option_input_lengths[i])
				option_hiddens.append(option_hidden)

			dis_optimizer.zero_grad()
			out = dis(batch_size, option_hiddens)
			loss_fn = nn.CrossEntropyLoss()
			label = torch.tensor(label).type(torch.cuda.LongTensor)
			loss = loss_fn(out, label)
			loss.backward()
			dis_optimizer.step()
			total_loss += loss.data.item()
			total_acc += (out.argmax(1) == label).sum().item()

			sys.stdout.flush()
			count += 1

		print('\n average_loss = %.4f, train_acc = %.4f' % (
			total_loss / (count * BATCH_SIZE), total_acc / (count * BATCH_SIZE)))



# MAIN
if __name__ == '__main__':
	# oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
	# oracle.load_state_dict(torch.load(oracle_state_dict_path))
	# oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)

	train_lines = open("./dataset/task2_3/train.tsv", "r",
					   encoding='utf-8').readlines()
	dev_lines = open("./dataset/task2_3/dev.tsv", "r",
					 encoding='utf-8').readlines()
	test_lines = open("./dataset/task2_3/test.tsv", "r",
					  encoding='utf-8').readlines()
	train_data = []
	dev_data = []
	test_data = []
	seqs = []
	for line in train_lines:
		line = line[:-1].split("\t")
		num, statement, explaination, label, options = int(line[0]), line[1], line[2:5], line[5], line[6:9]
		train_data.append((num, statement, explaination, label, options))
		seqs += [statement] + explaination + options
	for line in dev_lines:
		line = line[:-1].split("\t")
		num, statement, explaination, label, options = int(line[0]), line[1], line[2:5], line[5], line[6:9]
		dev_data.append((num, statement, explaination, label, options))
		seqs += [statement] + explaination + options
	for line in test_lines:
		line = line[:-1].split("\t")
		num, statement, explaination, label, options = int(line[0]), line[1], line[2:5], line[5], line[6:9]
		test_data.append((num, statement, explaination, label, options))
		seqs += [statement] + explaination + options

	# print(train_data[0])
	# word_encoder = WhitespaceEncoder(seqs+["<sos>"]+["<eos>"]) # have function of encode() and decoder()
	# pickle.dump(word_encoder,open("./model/word_encoder.p", "wb" ))
	word_encoder = pickle.load(open("./model/word_encoder.p", "rb"))

	hidden_size = 300
	EMBED_DIM = 300
	VOCAB_SIZE = word_encoder.vocab_size
	print("voc",VOCAB_SIZE)
	# exit(88)

	train_dataset = TextDataset(train_data, word_encoder)
	dev_dataset = TextDataset(dev_data, word_encoder)
	test_dataset = TextDataset(test_data, word_encoder)
	collate = MyCollator(word_encoder)
	collate_adv = MyCollator_adv(word_encoder)
	collate_dev = MyCollator_dev(word_encoder)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle=True, collate_fn=collate)
	train_loader_adv = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_adv)
	dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_dev)
	test_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_dev)

	# pretrained_embedding = _PretrainedWordVectors(name = "glove_embed/glove.840B.300d.txt", cache= "glove_embed/glove_cache") # glove.pt become a directory
	# embedding_weights = torch.Tensor(word_encoder.vocab_size, pretrained_embedding.dim)
	# for i, token in enumerate(word_encoder.vocab):
	# 	embedding_weights[i] = pretrained_embedding[token]
	#
	# embedding = nn.Embedding.from_pretrained(embedding_weights)
	# torch.save(embedding, "./glove_embed/glove.embed")
	embedding = torch.load("./glove_embed/glove.embed")
	embedding.weight.requires_grad = True
	state_encoder = EncoderRNN(VOCAB_SIZE, EMBED_DIM, hidden_size=hidden_size, embedding=embedding, bidirectional= True).to(device)
	reason_decoder = BahdanauAttnDecoderRNN(hidden_size, EMBED_DIM,VOCAB_SIZE, embedding=embedding).to(device)



	# GENERATOR MLE TRAINING
	# print('Starting Generator MLE Training...')
	gen_optimizer = Ranger(itertools.chain(state_encoder.parameters(),reason_decoder.parameters()), lr=1e-5,weight_decay= 1e-5)
	# train_generator_MLE(state_encoder, reason_decoder, gen_optimizer,  PRE_TRAIN_GEN)
	print('\nStarting Generator Evaluating...')
	best_epoch = dev_generator(state_encoder, reason_decoder, gen_optimizer, name="pretrain_gen")
	pretrained_gen_path = "./model/pretrain_gen"+str(best_epoch)+".pth"
	state_encoder.load_state_dict(torch.load(pretrained_gen_path)["state_encoder"])
	reason_decoder.load_state_dict(torch.load(pretrained_gen_path)["reason_decoder"])
	embedding.load_state_dict(torch.load(pretrained_gen_path)["embed"])
	state = {"state_encoder": state_encoder.state_dict(), "reason_decoder": reason_decoder.state_dict(),
	         'embed': embedding.state_dict()}
	pretrained_gen_path = "./model/pretrain_gen.pth"
	torch.save(state, pretrained_gen_path)


	# PRETRAIN DISCRIMINATOR
	dis = discriminator.Discriminator(hidden_size,DIS_HIDDEN_DIM).to(
		device)  # use directly as classifier

	dis_option_encoder = EncoderRNN(VOCAB_SIZE, EMBED_DIM, hidden_size=int(hidden_size), embedding=embedding, bidirectional= False).to(device)
	print('\nStarting Discriminator Training...')
	dis_optimizer = Ranger(itertools.chain(dis_option_encoder.parameters(),dis.parameters()), lr=1e-5, weight_decay= 1e-4)
	# train_discriminator(dis, dis_option_encoder, dis_optimizer,  PRE_TRAIN_DIS)
	print('\nStarting Discriminator Evaluating...')
	best_epoch = dev_discriminator(dis,  dis_option_encoder, name="pretrain_dis")
	pretrained_dis_path = "./model/pretrain_dis"+str(best_epoch)+".pth"
	dis_option_encoder.load_state_dict(torch.load(pretrained_dis_path)["dis_option_encoder"])
	dis.load_state_dict(torch.load(pretrained_dis_path)["dis"])
	embedding.load_state_dict(torch.load(pretrained_dis_path)["embed"])
	state = {"dis_option_encoder": dis_option_encoder.state_dict(), "dis": dis.state_dict(),'embed': embedding.state_dict()}
	pretrained_dis_path = "./model/pretrain_dis.pth"
	torch.save(state, pretrained_dis_path)



	# for epoch in range(ADV_TRAIN_EPOCHS):
	# 	print('\n--------\nEPOCH %d\n--------' % (epoch+1))
	# 	# TRAIN GENERATOR
	# 	print('\nAdversarial Training Generator : ')
	# 	sys.stdout.flush()
	# 	train_generator_PG(dis, dis_option_encoder, state_encoder, reason_decoder,  gen_optimizer,  epochs = 10)
	# 	# TRAIN DISCRIMINATOR
	# 	print('\nAdversarial Training Discriminator : ')
	# 	train_discriminator_adv(dis, dis_option_encoder, state_encoder, reason_decoder, dis_optimizer, epochs=10)
	# 	adv_model_path = "./model/adv_model"+str(epoch+1)+".pth"
	# 	if epoch % 2 == 1:
	# 		state = {"state_encoder": state_encoder.state_dict(), "reason_decoder": reason_decoder.state_dict(), "dis_option_encoder": dis_option_encoder.state_dict(), "dis": dis.state_dict(),'embed': embedding.state_dict()}
	# 		torch.save(state, adv_model_path)

	best_epoch_dis = dev_discriminator(dis,  dis_option_encoder, name = "adv_model")
	best_epoch_gen = dev_generator(state_encoder, reason_decoder, gen_optimizer, name = "adv_model")
	pretrained_gen_path = "./model/adv_model" + str(best_epoch_gen) + ".pth"
	pretrained_dis_path = "./model/adv_model" + str(best_epoch_gen) + ".pth"