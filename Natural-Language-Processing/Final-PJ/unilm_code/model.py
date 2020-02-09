import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import init

from torch.nn.functional import softmax

from transformers import *
import numpy as np
import os
from pytorch_pretrained_bert.modeling import  PreTrainedBertModel, BertPreTrainingHeads, BertPreTrainingPairRel
from pytorch_pretrained_bert.loss import LabelSmoothingLoss
import torch.nn.functional as F
import dgl
import dgl.function as fn
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

def weight_init(m):
	'''
	Usage:
		model = Model()
		model.apply(weight_init)
	'''
	if isinstance(m, nn.Conv1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.BatchNorm1d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm2d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm3d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
	elif isinstance(m, nn.LSTM):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.LSTMCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRU):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRUCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)



class TrpAttention(nn.Module):
	def __init__(self):
		pass

	def forward(self):
		pass


class Attention(nn.Module):
	def __init__(self, enc_hid_dim, dec_hid_dim):
		super(Attention, self).__init__()
		self.enc_hid_dim = enc_hid_dim
		self.dec_hid_dim = dec_hid_dim
		self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
		self.v = nn.Parameter(torch.rand(dec_hid_dim))

	def forward(self, hidden, encoder_outputs):
		# hidden = [batch size, dec hid dim]
		# encoder_outputs = [src sent len, batch size, enc hid dim * 2]
		batch_size = encoder_outputs.shape[1]
		src_len = encoder_outputs.shape[0]
		# 重复操作，让隐藏状态的第二个维度和encoder相同
		hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
		# 该函数按指定的向量来重新排列一个数组，在这里是调整encoder输出的维度顺序，在后面能够进行比较
		encoder_outputs = encoder_outputs.permute(1, 0, 2)
		# hidden = [batch size, src sent len, dec hid dim]
		# encoder_outputs = [batch size, src sent len, enc hid dim * 2]
		# 开始计算hidden和encoder_outputs之间的匹配值
		energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
		# energy = [batch size, src sent len, dec hid dim]
		# 调整energy的排序
		energy = energy.permute(0, 2, 1)
		# energy = [batch size, dec hid dim, src sent len]

		# v = [dec hid dim]
		v = self.v.repeat(batch_size, 1).unsqueeze(1)
		# v = [batch_size, 1, dec hid dim] 注意这个bmm的作用，对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操
		attention = torch.bmm(v, energy).squeeze(1)
		# attention=[batch_size, src_len]
		return F.softmax(attention, dim=1)



class Task23Model(BertPreTrainedModel):
	def __init__(self, bert, config, args=None):
		super(Task23Model, self).__init__(config)
		self.args = args
		self.bert = bert
		# self.bert = BertModel(config)

		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.init_weights()
		self.classifier = nn.Linear(config.hidden_size, 1)
		self.trp_attention = self.args.trp_attn
		self.trp_num = self.args.trp_num

		self.sent_dim = config.hidden_size
		self.trp_dim = config.hidden_size
		self.fc_hidden_size = config.fc_hidden_size if hasattr(config, 'fc_hidden_size') else 64
		self.mlp = nn.Sequential(
			nn.Linear(self.sent_dim + self.trp_dim, self.fc_hidden_size * 4),
			nn.BatchNorm1d(self.fc_hidden_size * 4),
			nn.ReLU(),
			nn.Dropout(config.hidden_dropout_prob),
			nn.Linear(self.fc_hidden_size * 4, self.fc_hidden_size),
			nn.BatchNorm1d(self.fc_hidden_size),
			nn.ReLU(),
			nn.Dropout(config.hidden_dropout_prob),
			nn.Linear(self.fc_hidden_size, 1),
		)
		# self.uniLM = BertForPreTrainingLossMask.from_pretrained(**un_inputs)

		# if trp_attention:
		#     self.sent_atrp_att = nn.Linear(self.sent_dim, self.trp_dim)
		#     self.sent_atrp_att.apply(weight_init)
		self.mlp.apply(weight_init)

		# for uniLM
		# self.cls = BertPreTrainingHeads(
		#     config, self.bert.embeddings.word_embeddings.weight, num_labels=2)
		# self.num_sentlvl_labels = 0
		# self.cls2 = None
		# self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
		# self.crit_next_sent = None
		# self.num_labels = 2
		# self.num_rel = 2
		#
		# self.crit_pair_rel = BertPreTrainingPairRel(
		#     config, num_rel=self.num_rel)
		# if hasattr(config, 'label_smoothing') and config.label_smoothing:
		#     self.crit_mask_lm_smoothed = LabelSmoothingLoss(
		#         config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
		# else:
		#     self.crit_mask_lm_smoothed = None
		# self.apply(self.init_bert_weights)
		# self.bert.rescale_some_parameters()

	def bert_features(self, input_ids, attention_masks, token_type_ids):

		input_ids = input_ids.view(-1, input_ids.size(-1))  # [batch_size, num_choices, max_length] --> [batch_size*num_choices, max_length]
		attention_masks = attention_masks.view(-1, attention_masks.size(-1)) if attention_masks is not None else None
		token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
		outputs = self.bert(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
		# outputs[0] [batch_size*num_choices, max_length, hidden_size] ;   outputs[1] [batch_size*num_choices, hidden_size
		pooled_output = outputs[1]  # [batch_size*num_choices, hidden_size]
		pooled_output = self.dropout(pooled_output)
		return pooled_output



	def forward(self,  sent_input_ids, sent_attention_masks, sent_token_type_ids,
				trp_input_ids, trp_attention_masks, trp_token_type_ids,
				labels=None, guids=None, ana_mode=False):
		# print("current gpu", torch.cuda.current_device())
		batch_size = sent_input_ids.shape[0]
		# attention_mask [batch_size, num_choices, max_length]  input_ids [batch_size, num_choices, max_length]
		# token_type_ids [batch_size, num_choices, max_length]  labels [batch_size,]
		num_choices = sent_input_ids.shape[1]
		sent_len = sent_input_ids.shape[2]
		trp_len = trp_input_ids.shape[3]
		trp_num = self.trp_num
		hid_dim = self.sent_dim

		sent_bert_feat = self.bert_features(sent_input_ids, sent_attention_masks, sent_token_type_ids)

		# calculate trp bert features
		old_trp_attn_mask = trp_attention_masks
		trp_input_ids = trp_input_ids.reshape(batch_size, -1, trp_len)
		trp_attention_masks = trp_attention_masks.reshape(batch_size, -1, trp_len)
		trp_token_type_ids = trp_token_type_ids.reshape(batch_size, -1, trp_len)
		all_trp_bert_feat = self.bert_features(trp_input_ids, trp_attention_masks, trp_token_type_ids)  # [batch * num_choice * trp_num, hid_dim]

		# triple attention
		new_trp_attn_mask = old_trp_attn_mask.sum(dim=3) > 0
		new_trp_attn_mask = new_trp_attn_mask.unsqueeze(dim=3).expand(-1, -1, -1, hid_dim).to(dtype=torch.float64)
		all_trp_bert_feat_masked = all_trp_bert_feat.reshape(batch_size, num_choices, trp_num, -1).mul(torch.tensor(new_trp_attn_mask, dtype=torch.float32).to('cuda'))


		if self.trp_attention:
			weights_no = sent_bert_feat.reshape(batch_size, num_choices, 1, -1).expand(-1, -1, trp_num, -1).mul(all_trp_bert_feat_masked).sum(dim=3)
			weights = softmax(weights_no, dim=2)
			avg_trp_bert_feat = weights.unsqueeze(dim=3).expand(-1, -1, -1, hid_dim).mul(all_trp_bert_feat_masked).sum(dim=2)

		else:
			avg_trp_bert_feat = all_trp_bert_feat_masked.mean(dim=2)

		concated = torch.cat((sent_bert_feat, avg_trp_bert_feat.reshape(-1, hid_dim)), 1)
		logits = self.mlp(concated)  # [batch_size*num_choices, 1]
		reshaped_logits = logits.view(-1, num_choices)  # [batch_size, num_choices]

		outputs = (reshaped_logits,)  # add hidden states and attention if they are here  # [batch_size, num_choices]

		if labels is not None: # 无label的test时，preds calculation要改
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(reshaped_logits, labels)
			outputs = (loss,) + outputs

		return outputs  # loss, reshaped_logits, (hidden_states), (attentions)


class BertForPreTrainingLossMask(PreTrainedBertModel):
	"""refer to BertForPreTraining"""

	def __init__(self, config, num_labels=2, num_rel=0, num_sentlvl_labels=0, no_nsp=False):
		super(BertForPreTrainingLossMask, self).__init__(config)
		self.bert = BertModel(config)
		self.cls = BertPreTrainingHeads(
			config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
		self.num_sentlvl_labels = num_sentlvl_labels
		self.cls2 = None
		if self.num_sentlvl_labels > 0:
			self.secondary_pred_proj = nn.Embedding(
				num_sentlvl_labels, config.hidden_size)
			self.cls2 = BertPreTrainingHeads(
				config, self.secondary_pred_proj.weight, num_labels=num_sentlvl_labels)
		self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
		if no_nsp:
			self.crit_next_sent = None
		else:
			self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
		self.num_labels = num_labels
		self.num_rel = num_rel
		if self.num_rel > 0:
			self.crit_pair_rel = BertPreTrainingPairRel(
				config, num_rel=num_rel)
		if hasattr(config, 'label_smoothing') and config.label_smoothing:
			self.crit_mask_lm_smoothed = LabelSmoothingLoss(
				config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
		else:
			self.crit_mask_lm_smoothed = None
		self.apply(self.init_bert_weights)
		self.bert.rescale_some_parameters()

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
				next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None, pair_x=None,
				pair_x_mask=None, pair_y=None, pair_y_mask=None, pair_r=None, pair_pos_neg_mask=None,
				pair_loss_mask=None, masked_pos_2=None, masked_weights_2=None, masked_labels_2=None,
				num_tokens_a=None, num_tokens_b=None, mask_qkv=None):
		if token_type_ids is None and attention_mask is None:
			task_0 = (task_idx == 0)
			task_1 = (task_idx == 1)
			task_2 = (task_idx == 2)
			task_3 = (task_idx == 3)

			sequence_length = input_ids.shape[-1]
			index_matrix = torch.arange(sequence_length).view(
				1, sequence_length).to(input_ids.device)

			num_tokens = num_tokens_a + num_tokens_b

			base_mask = (index_matrix < num_tokens.view(-1, 1)
						 ).type_as(input_ids)
			segment_a_mask = (
				index_matrix < num_tokens_a.view(-1, 1)).type_as(input_ids)

			token_type_ids = (
				task_idx + 1 + task_3.type_as(task_idx)).view(-1, 1) * base_mask
			token_type_ids = token_type_ids - segment_a_mask * \
				(task_0 | task_3).type_as(segment_a_mask).view(-1, 1)

			index_matrix = index_matrix.view(1, 1, sequence_length)
			index_matrix_t = index_matrix.view(1, sequence_length, 1)

			tril = index_matrix <= index_matrix_t

			attention_mask_task_0 = (
				index_matrix < num_tokens.view(-1, 1, 1)) & (index_matrix_t < num_tokens.view(-1, 1, 1))
			attention_mask_task_1 = tril & attention_mask_task_0
			attention_mask_task_2 = torch.transpose(
				tril, dim0=-2, dim1=-1) & attention_mask_task_0
			attention_mask_task_3 = (
				(index_matrix < num_tokens_a.view(-1, 1, 1)) | tril) & attention_mask_task_0

			attention_mask = (attention_mask_task_0 & task_0.view(-1, 1, 1)) | \
							 (attention_mask_task_1 & task_1.view(-1, 1, 1)) | \
							 (attention_mask_task_2 & task_2.view(-1, 1, 1)) | \
							 (attention_mask_task_3 & task_3.view(-1, 1, 1))
			attention_mask = attention_mask.type_as(input_ids)
		sequence_output, pooled_output = self.bert(
			input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)

		def gather_seq_out_by_pos(seq, pos):
			return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

		def gather_seq_out_by_pos_average(seq, pos, mask):
			# pos/mask: (batch, num_pair, max_token_num)
			batch_size, max_token_num = pos.size(0), pos.size(-1)
			# (batch, num_pair, max_token_num, seq.size(-1))
			pos_vec = torch.gather(seq, 1, pos.view(batch_size, -1).unsqueeze(
				2).expand(-1, -1, seq.size(-1))).view(batch_size, -1, max_token_num, seq.size(-1))
			# (batch, num_pair, seq.size(-1))
			mask = mask.type_as(pos_vec)
			pos_vec_masked_sum = (
				pos_vec * mask.unsqueeze(3).expand_as(pos_vec)).sum(2)
			return pos_vec_masked_sum / mask.sum(2, keepdim=True).expand_as(pos_vec_masked_sum)

		def loss_mask_and_normalize(loss, mask):
			mask = mask.type_as(loss)
			loss = loss * mask
			denominator = torch.sum(mask) + 1e-5
			return (loss / denominator).sum()

		if masked_lm_labels is None:
			if masked_pos is None:
				prediction_scores, seq_relationship_score = self.cls(
					sequence_output, pooled_output, task_idx=task_idx)
			else:
				sequence_output_masked = gather_seq_out_by_pos(
					sequence_output, masked_pos)
				prediction_scores, seq_relationship_score = self.cls(
					sequence_output_masked, pooled_output, task_idx=task_idx)
			return prediction_scores, seq_relationship_score

		# masked lm
		sequence_output_masked = gather_seq_out_by_pos(
			sequence_output, masked_pos)
		prediction_scores_masked, seq_relationship_score = self.cls(
			sequence_output_masked, pooled_output, task_idx=task_idx)
		if self.crit_mask_lm_smoothed:
			masked_lm_loss = self.crit_mask_lm_smoothed(
				F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
		else:
			masked_lm_loss = self.crit_mask_lm(
				prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
		masked_lm_loss = loss_mask_and_normalize(
			masked_lm_loss.float(), masked_weights)

		# next sentence
		if self.crit_next_sent is None or next_sentence_label is None:
			next_sentence_loss = 0.0
		else:
			next_sentence_loss = self.crit_next_sent(
				seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))

		if self.cls2 is not None and masked_pos_2 is not None:
			sequence_output_masked_2 = gather_seq_out_by_pos(
				sequence_output, masked_pos_2)
			prediction_scores_masked_2, _ = self.cls2(
				sequence_output_masked_2, None)
			masked_lm_loss_2 = self.crit_mask_lm(
				prediction_scores_masked_2.transpose(1, 2).float(), masked_labels_2)
			masked_lm_loss_2 = loss_mask_and_normalize(
				masked_lm_loss_2.float(), masked_weights_2)
			masked_lm_loss = masked_lm_loss + masked_lm_loss_2

		if pair_x is None or pair_y is None or pair_r is None or pair_pos_neg_mask is None or pair_loss_mask is None:
			return masked_lm_loss, next_sentence_loss

		# pair and relation
		if pair_x_mask is None or pair_y_mask is None:
			pair_x_output_masked = gather_seq_out_by_pos(
				sequence_output, pair_x)
			pair_y_output_masked = gather_seq_out_by_pos(
				sequence_output, pair_y)
		else:
			pair_x_output_masked = gather_seq_out_by_pos_average(
				sequence_output, pair_x, pair_x_mask)
			pair_y_output_masked = gather_seq_out_by_pos_average(
				sequence_output, pair_y, pair_y_mask)
		pair_loss = self.crit_pair_rel(
			pair_x_output_masked, pair_y_output_masked, pair_r, pair_pos_neg_mask)
		pair_loss = loss_mask_and_normalize(
			pair_loss.float(), pair_loss_mask)
		return masked_lm_loss, next_sentence_loss, pair_loss

class NodeApplyModule(nn.Module):

	def __init__(self, in_feats, out_feats, activation):
		super(NodeApplyModule, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.activation = activation

	def forward(self, node):
		h = self.linear(node.data['h'])
		h = self.activation(h)
		return {'h': h}

class GraphConvLayer(nn.Module):

	def __init__(self, in_feats, out_feats, activation):
		super(GraphConvLayer, self).__init__()
		self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

	def forward(self, g, feature):
		g.ndata['h'] = feature
		g.update_all(gcn_msg, gcn_reduce)
		g.apply_nodes(func=self.apply_mod)
		return g.ndata.pop('h')


class GCNEncoder(nn.Module):

	def __init__(self, concept_dim, hidden_dim, output_dim, pretrained_concept_emd, concept_emd=None):
		super(GCNEncoder, self).__init__()

		self.gcn1 = GraphConvLayer(concept_dim, hidden_dim, F.relu)
		self.gcn2 = GraphConvLayer(hidden_dim, output_dim, F.relu)

		if pretrained_concept_emd is not None and concept_emd is None:
			self.concept_emd = nn.Embedding(pretrained_concept_emd.size(0), pretrained_concept_emd.size(1))
			self.concept_emd.weight.data.copy_(pretrained_concept_emd)
		elif pretrained_concept_emd is None and concept_emd is not None:
			self.concept_emd = concept_emd
		else:
			raise ValueError('invalid pretrained_concept_emd/concept_emd')

	def forward(self, g):
		features = self.concept_emd(g.ndata["cncpt_ids"])
		x = self.gcn1(g, features)
		x = self.gcn2(g, x)
		g.ndata['h'] = x
		return g


class GCNSent(nn.Module):

	def __init__(self, sent_dim, fc_hidden_size, concept_dim, graph_hidden_dim, graph_output_dim,
				 pretrained_concept_emd, dropout=0.3):
		super(GCNSent, self).__init__()
		self.sent_dim = sent_dim
		self.fc_hidden_size = fc_hidden_size
		self.concept_dim = concept_dim
		self.graph_hidden_dim = graph_hidden_dim
		self.graph_output_dim = graph_output_dim
		self.pretrained_concept_emd = pretrained_concept_emd
		self.dropout = dropout

		self.graph_encoder = GCNEncoder(concept_dim, graph_hidden_dim, graph_output_dim, pretrained_concept_emd)

		self.mlp = nn.Sequential(
			nn.Linear(self.sent_dim + graph_output_dim, self.fc_hidden_size * 4),
			nn.BatchNorm1d(self.fc_hidden_size * 4),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(self.fc_hidden_size * 4, self.fc_hidden_size),
			nn.BatchNorm1d(self.fc_hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(self.fc_hidden_size, 1),
		)

	def forward(self, sent_vecs, graph):
		node_embed = self.graph_encoder(graph)
		graph_embed = dgl.mean_nodes(node_embed, 'h')
		concated = torch.cat((sent_vecs, graph_embed), 1)
		logits = self.mlp(concated)
		return logits

class Task2Model_graph(BertPreTrainedModel):

	def __init__(self, config, bert = None, args=None):
		super(Task2Model_graph, self).__init__(config)

		self.use_sent = args.use_sent
		self.use_trp = args.use_trp
		self.use_path = args.use_path

		self.device = args.device
		self.bert = bert
		# embedding
		cp_emb, rel_emb = np.load(args.ent_emb), np.load(args.rel_emb)  # [799273,100]   [17, 100]
		concept_num, concept_dim = cp_emb.shape[0] + 1, cp_emb.shape[1]  # add a dummy concept
		cp_emb = torch.tensor(np.insert(cp_emb, 0, np.zeros((1, concept_dim)), 0))
		relation_num, relation_dim = rel_emb.shape[0] * 2 + 1, rel_emb.shape[1]  # for inverse and dummy relations
		rel_emb = np.concatenate((rel_emb, rel_emb), 0)
		rel_emb = torch.tensor(np.insert(rel_emb, 0, np.zeros((1, relation_dim)), 0))
		self.pretrained_concept_emd = cp_emb
		self.pretrained_relation_emd = rel_emb
		self.concept_dim = concept_dim
		self.relation_dim = relation_dim
		self.concept_num = concept_num
		self.relation_num = relation_num

		# bert init
		config.output_hidden_states = True
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.init_weights()
		self.classifier = nn.Linear(config.hidden_size, 1)
		self.trp_num = args.trp_num
		self.sent_dim = config.hidden_size
		self.trp_dim = config.hidden_size
		self.fc_hidden_size =  config.fc_hidden_size if hasattr(config, 'fc_hidden_size') else 64
		self.qas_encoded_dim = args.qas_encoded_dim
		self.lstm_dim = args.lstm_dim
		self.sent_dim = 1024  # bert base, else 1024

		self.features_dim = 0
		if self.use_sent:
			self.features_dim += self.sent_dim
		if self.use_trp:
			self.features_dim += self.trp_dim
		if self.use_path:
			self.features_dim += self.qas_encoded_dim + self.lstm_dim

		self.mlp = nn.Sequential(  # 768 768 128 128
			nn.Linear(self.features_dim, self.fc_hidden_size * 4),
			nn.BatchNorm1d(self.fc_hidden_size * 4),
			nn.ReLU(),
			nn.Dropout(config.hidden_dropout_prob),
			nn.Linear(self.fc_hidden_size * 4, self.fc_hidden_size),
			nn.BatchNorm1d(self.fc_hidden_size),
			nn.ReLU(),
			nn.Dropout(config.hidden_dropout_prob),
			nn.Linear(self.fc_hidden_size, 1),
		)
		self.mlp.apply(weight_init)

		# path init
		self.sent_dim = 1024 # bert base, else 1024
		self.qas_encoded_dim = args.qas_encoded_dim
		self.lstm_dim = args.lstm_dim
		self.lstm_layer_num = args.lstm_layer_num
		self.graph_hidden_dim = args.graph_hidden_dim
		self.graph_output_dim = args.graph_output_dim
		self.bidirect = args.bidirect
		self.num_random_paths = args.num_random_paths
		self.trp_attention = args.trp_attn
		self.path_attention = args.path_attn
		self.qa_attention = args.qa_attn

		self.concept_emd = nn.Embedding(concept_num, concept_dim)
		self.relation_emd = nn.Embedding(relation_num, relation_dim)

		if cp_emb is not None:
			self.concept_emd.weight.data.copy_(cp_emb)
		else:
			bias = np.sqrt(6.0 / self.concept_dim)
			nn.init.uniform_(self.concept_emd.weight, -bias, bias)

		if rel_emb is not None:
			self.relation_emd.weight.data.copy_(rel_emb)
		else:
			bias = np.sqrt(6.0 / self.relation_dim)
			nn.init.uniform_(self.relation_emd.weight, -bias, bias)

		self.lstm = nn.LSTM(input_size=self.graph_output_dim + concept_dim + relation_dim,
							hidden_size=self.lstm_dim,
							num_layers=self.lstm_layer_num,
							bidirectional=self.bidirect,
							dropout=config.hidden_dropout_prob,
							batch_first=True)
		if self.bidirect:
			self.lstm_dim = self.lstm_dim * 2

		self.qas_encoder = nn.Sequential(
			nn.Linear(2 * (concept_dim + self.graph_output_dim) + self.sent_dim, self.qas_encoded_dim * 2),  # binary classification
			nn.Dropout(config.hidden_dropout_prob),
			nn.LeakyReLU(),
			nn.Linear(self.qas_encoded_dim * 2, self.qas_encoded_dim),
			nn.Dropout(config.hidden_dropout_prob),
			nn.LeakyReLU(),
		)

		if self.path_attention:  # TODO: can be optimized by using nn.BiLinaer
			self.qas_pathlstm_att = nn.Linear(self.qas_encoded_dim, self.lstm_dim)  # transform qas vector to query vectors
			self.qas_pathlstm_att.apply(weight_init)

		if self.qa_attention:
			self.sent_ltrel_att = nn.Linear(self.sent_dim, self.qas_encoded_dim)  # transform sentence vector to query vectors
			self.sent_ltrel_att.apply(weight_init)

		self.hidden2output = nn.Sequential(
			nn.Linear(self.qas_encoded_dim + self.lstm_dim + self.sent_dim, 1),  # binary classification
		)

		self.lstm.apply(weight_init)
		self.qas_encoder.apply(weight_init)
		self.hidden2output.apply(weight_init)

		self.graph_encoder = GCNEncoder(self.concept_dim, self.graph_hidden_dim, self.graph_output_dim,
										pretrained_concept_emd=None, concept_emd=self.concept_emd)


	def bert_features(self, input_ids, attention_masks, token_type_ids):
		input_ids = input_ids.view(-1, input_ids.size(-1))  # [batch_size, num_choices, max_length] --> [batch_size*num_choices, max_length]
		attention_masks = attention_masks.view(-1, attention_masks.size(-1)) if attention_masks is not None else None
		token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
		outputs = self.bert(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
		# outputs[0] [batch_size*num_choices, max_length, hidden_size] ;   outputs[1] [batch_size*num_choices, hidden_size
		pooled_output = outputs[1]  # [batch_size*num_choices, hidden_size]
		pooled_output = self.dropout(pooled_output)

		all_hidden_states = outputs[-1] #list len=13
		hidden_layer_1 = self.bert.pooler(all_hidden_states[-1])
		# output_layer_2 = self.bert.pooler(all_hidden_states[-2])

		return pooled_output, hidden_layer_1


	def path_features(self, sent_bert_feat, qa_pairs_batched, cpt_paths_batched, rel_paths_batched,
				graphs, concept_mapping_dicts, qa_path_num_batched, path_len_batched, ana_mode=False):
		s_vec_batched = sent_bert_feat
		output_graphs = self.graph_encoder(graphs)
		new_concept_embed = torch.cat((output_graphs.ndata["h"], s_vec_batched.new_zeros((1, self.graph_output_dim))))  # len(output_concept_embeds) as padding


		final_vecs = []

		if ana_mode:
			path_att_scores = []
			qa_pair_att_scores = []

		n_qa_pairs = [len(t) for t in qa_pairs_batched]  # [17, 17, 26]
		total_qa_pairs = sum(n_qa_pairs) # 60

		s_vec_expanded = s_vec_batched.new_zeros((total_qa_pairs, s_vec_batched.size(1)))  # [total_qa_pairs, 768]
		i = 0
		for n, s_vec in zip(n_qa_pairs, s_vec_batched):  # s_vec_batched  [3, 768]
			j = i + n
			s_vec_expanded[i:j] = s_vec
			i = j

		qa_ids_batched = torch.cat(qa_pairs_batched, 0)  # N x 2     # qa_pairs_batched [ [17,2], [17,2], [26,2] ]  # qa_ids_batched [60,2]
		qa_vecs = self.concept_emd(qa_ids_batched).view(total_qa_pairs, -1)  # [60, 200]
		new_qa_ids = []
		for qa_ids, mdict in zip(qa_pairs_batched, concept_mapping_dicts):
			id_mapping = lambda x: mdict.get(x.item(), len(new_concept_embed) - 1)
			new_qa_ids += [[id_mapping(q), id_mapping(a)] for q, a in qa_ids]
		new_qa_ids = torch.tensor(new_qa_ids).to(self.device) # [60, 2]
		new_qa_vecs = new_concept_embed[new_qa_ids].view(total_qa_pairs, -1)  # [60, 50]
		raw_qas_vecs = torch.cat((qa_vecs, new_qa_vecs, s_vec_expanded), dim=1)  # all the qas triple vectors associated with a statement
			# qa_vecs [60, 200]   # new_qa_vecs [60, 50]  # s_vec_expanded [60, 768]  #[N,1018]

		qas_vecs_batched = self.qas_encoder(raw_qas_vecs) # [N, 128]
		if self.path_attention:
			query_vecs_batched = self.qas_pathlstm_att(qas_vecs_batched)

		flat_cpt_paths_batched = torch.cat(cpt_paths_batched, 0)
		mdicted_cpaths = []
		for cpt_path in flat_cpt_paths_batched:
			mdicted_cpaths.append([id_mapping(c) for c in cpt_path])
		mdicted_cpaths = torch.tensor(mdicted_cpaths).to(self.device)

		new_batched_all_qa_cpt_paths_embeds = new_concept_embed[mdicted_cpaths]
		batched_all_qa_cpt_paths_embeds = self.concept_emd(torch.cat(cpt_paths_batched, 0))  # old concept embed
		batched_all_qa_cpt_paths_embeds = torch.cat((batched_all_qa_cpt_paths_embeds, new_batched_all_qa_cpt_paths_embeds), 2)
		batched_all_qa_rel_paths_embeds = self.relation_emd(torch.cat(rel_paths_batched, 0))  # N_PATHS x D x MAX_PATH_LEN
		batched_all_qa_cpt_rel_path_embeds = torch.cat((batched_all_qa_cpt_paths_embeds,
														batched_all_qa_rel_paths_embeds), 2)

		# if False then abiliate the LSTM
		if True:
			self.lstm.flatten_parameters()
			batched_lstm_outs, _ = self.lstm(batched_all_qa_cpt_rel_path_embeds)
		else:
			batched_lstm_outs = s_vec.new_zeros((batched_all_qa_cpt_rel_path_embeds.size(0),
												 batched_all_qa_cpt_rel_path_embeds.size(1),
												 self.lstm_dim))
		b_idx = torch.arange(batched_lstm_outs.size(0)).to(batched_lstm_outs.device)
		batched_lstm_outs = batched_lstm_outs[b_idx, torch.cat(path_len_batched, 0) - 1, :]

		qa_pair_cur_start = 0
		path_cur_start = 0
		# for each question-answer statement
		for s_vec, qa_ids, cpt_paths, rel_paths, mdict, qa_path_num, path_len in zip(s_vec_batched, qa_pairs_batched, cpt_paths_batched,
																					 rel_paths_batched, concept_mapping_dicts, qa_path_num_batched,
																					 path_len_batched):  # len = batch_size * num_choices
			n_qa_pairs = qa_ids.size(0)
			qa_pair_cur_end = qa_pair_cur_start + n_qa_pairs

			if n_qa_pairs == 0 or False:  # if "or True" then we can do ablation study
				raw_qas_vecs = torch.cat([s_vec.new_zeros((self.concept_dim + self.graph_output_dim) * 2), s_vec], 0).view(1, -1)
				qas_vecs = self.qas_encoder(raw_qas_vecs)
				latent_rel_vecs = torch.cat((qas_vecs, s_vec.new_zeros(1, self.lstm_dim)), dim=1)
			else:
				pooled_path_vecs = []
				qas_vecs = qas_vecs_batched[qa_pair_cur_start:qa_pair_cur_end]
				for j in range(n_qa_pairs):
					if self.path_attention:
						query_vec = query_vecs_batched[qa_pair_cur_start + j]

					path_cur_end = path_cur_start + qa_path_num[j]

					# pooling over all paths for a certain (question concept, answer concept) pair
					blo = batched_lstm_outs[path_cur_start:path_cur_end]
					if self.path_attention:  # TODO: use an attention module for better readibility
						att_scores = torch.mv(blo, query_vec)  # path-level attention scores
						norm_att_scores = F.softmax(att_scores, 0)
						att_pooled_path_vec = torch.mv(blo.t(), norm_att_scores)
						if ana_mode:
							path_att_scores.append(norm_att_scores)
					else:
						att_pooled_path_vec = blo.mean(0)

					path_cur_start = path_cur_end
					pooled_path_vecs.append(att_pooled_path_vec)

				pooled_path_vecs = torch.stack(pooled_path_vecs, 0)
				latent_rel_vecs = torch.cat((qas_vecs, pooled_path_vecs), 1)  # qas and KE-qas

			# pooling over all (question concept, answer concept) pairs
			if self.path_attention:
				sent_as_query = self.sent_ltrel_att(s_vec)  # sent attend on qas
				r_att_scores = torch.mv(qas_vecs, sent_as_query)  # qa-pair-level attention scores
				norm_r_att_scores = F.softmax(r_att_scores, 0)
				if ana_mode:
					qa_pair_att_scores.append(norm_r_att_scores)
				final_vec = torch.mv(latent_rel_vecs.t(), norm_r_att_scores)
			else:
				final_vec = latent_rel_vecs.mean(0).to(self.device)  # mean pooling
			# final_vecs.append(torch.cat((final_vec, s_vec), 0))
			final_vecs.append(final_vec)
			qa_pair_cur_start = qa_pair_cur_end

		# logits = self.hidden2output(torch.stack(final_vecs))
		path_feature = torch.stack(final_vecs)
		if not ana_mode:
			return path_feature
		else:
			return path_feature, path_att_scores, qa_pair_att_scores



	def forward(self, labels, graphs, cpt_paths, rel_paths, qa_pairs,
				concept_mapping_dicts, qa_path_num, path_len,
				sent_input_ids, sent_attention_mask, sent_token_type_ids,
				trp_input_ids, trp_attention_mask, trp_token_type_ids):
		# bert
		# labels = torch.stack([x.squeeze(0) for x in labels[0]])
		sent_input_ids = torch.stack(sent_input_ids)
		sent_attention_mask = torch.stack(sent_attention_mask)
		sent_token_type_ids = torch.stack(sent_token_type_ids)
		trp_input_ids = torch.stack(trp_input_ids)
		trp_attention_mask = torch.stack(trp_attention_mask)
		trp_token_type_ids = torch.stack(trp_token_type_ids)

		batch_size = sent_input_ids.shape[0]
		# attention_mask [batch_size, num_choices, max_length]  input_ids [batch_size, num_choices, max_length]
		# token_type_ids [batch_size, num_choices, max_length]  labels [batch_size,]
		num_choices = sent_input_ids.shape[1]
		sent_len = sent_input_ids.shape[2]
		trp_len = trp_input_ids.shape[3]
		trp_num = self.trp_num
		hid_dim = self.sent_dim

		sent_bert_feat, hidden_layer_1 = self.bert_features(sent_input_ids, sent_attention_mask, sent_token_type_ids)


		# calculate trp bert features
		if self.use_trp:
			old_trp_attn_mask = trp_attention_mask
			trp_input_ids = trp_input_ids.reshape(batch_size, -1, trp_len)
			trp_attention_masks = trp_attention_mask.reshape(batch_size, -1, trp_len)
			trp_token_type_ids = trp_token_type_ids.reshape(batch_size, -1, trp_len)
			all_trp_bert_feat, _ = self.bert_features(trp_input_ids, trp_attention_masks, trp_token_type_ids)  # [batch * num_choice * trp_num, hid_dim]
			# triple attention
			new_trp_attn_mask = old_trp_attn_mask.sum(dim=3) > 0
			new_trp_attn_mask = new_trp_attn_mask.unsqueeze(dim=3).expand(-1, -1, -1, hid_dim).to(dtype=torch.float64)
			all_trp_bert_feat_masked = all_trp_bert_feat.reshape(batch_size, num_choices, trp_num, -1).mul(
				torch.tensor(new_trp_attn_mask, dtype=torch.float32).to('cuda'))

			if self.trp_attention:
				weights_no = sent_bert_feat.reshape(batch_size, num_choices, 1, -1).expand(-1, -1, trp_num, -1).mul(
					all_trp_bert_feat_masked).sum(dim=3)
				weights = softmax(weights_no, dim=2)
				avg_trp_bert_feat = weights.unsqueeze(dim=3).expand(-1, -1, -1, hid_dim).mul(all_trp_bert_feat_masked).sum(
					dim=2)
			else:
				avg_trp_bert_feat = all_trp_bert_feat_masked.mean(dim=2)

		if self.use_path:
			# Path features
			# n * num_qa_pair x 2
			qa_pairs = sum(qa_pairs, [])
			# n * num_path x max_path_len
			cpt_paths = sum(cpt_paths, [])
			rel_paths = sum(rel_paths, [])
			qa_path_num = sum(qa_path_num, [])
			path_len = sum(path_len, [])
			sent_hidden = hidden_layer_1
			path_feature = self.path_features(sent_hidden, qa_pairs, cpt_paths, rel_paths, graphs, concept_mapping_dicts, qa_path_num, path_len)


		concated = sent_bert_feat

		if self.use_trp:
			concated = torch.cat((concated, avg_trp_bert_feat.reshape(-1, hid_dim)), 1)
		if self.use_path:
			concated = torch.cat((concated, path_feature), 1)

		# concated = torch.cat((sent_bert_feat, avg_trp_bert_feat.reshape(-1, hid_dim), path_feature), 1)
		logits = self.mlp(concated)  # [batch_size*num_choices, 1]
		reshaped_logits = logits.view(-1, num_choices)  # [batch_size, num_choices]

		outputs = (reshaped_logits,)  # add hidden states and attention if they are here  # [batch_size, num_choices]

		if labels is not None:  # 无label的test时，preds calculation要改
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(reshaped_logits, labels)
			outputs = (loss,) + outputs

		return outputs  # loss, reshaped_logits, (hidden_states), (attentions)
