import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import init

from torch.nn.functional import softmax

from transformers import *
import numpy as np


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
        # 该函数按指定的向量来重新排列一个数组，在这里是调整encoder输出的维度顺序，在后面能够进行比�?        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        # 开始计算hidden和encoder_outputs之间的匹配�?        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src sent len, dec hid dim]
        # 调整energy的排�?        energy = energy.permute(0, 2, 1)
        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch_size, 1, dec hid dim] 注意这个bmm的作用，对存储在两个批batch1和batch2内的矩阵进行批矩阵乘�?        attention = torch.bmm(v, energy).squeeze(1)
        # attention=[batch_size, src_len]
        return F.softmax(attention, dim=1)



class Task2Model(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(Task2Model, self).__init__(config)

        self.args = args
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.trp_attention = self.args.trp_attn
        self.trp_num = self.args.trp_num

        self.sent_dim = config.hidden_size
        self.trp_dim = config.hidden_size
        self.fc_hidden_size = config.fc_hidden_size
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

        # if trp_attention:
        #     self.sent_atrp_att = nn.Linear(self.sent_dim, self.trp_dim)
        #     self.sent_atrp_att.apply(weight_init)
        self.mlp.apply(weight_init)

    def bert_features(self, input_ids, attention_masks, token_type_ids):
        input_ids = input_ids.view(-1, input_ids.size(-1))  # [batch_size, num_choices, max_length] --> [batch_size*num_choices, max_length]
        attention_masks = attention_masks.view(-1, attention_masks.size(-1)) if attention_masks is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        outputs = self.bert(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        # outputs[0] [batch_size*num_choices, max_length, hidden_size] ;   outputs[1] [batch_size*num_choices, hidden_size
        pooled_output = outputs[1]  # [batch_size*num_choices, hidden_size]
        pooled_output = self.dropout(pooled_output)
        return pooled_output



    def forward(self, sent_input_ids, sent_attention_masks, sent_token_type_ids,
                trp_input_ids, trp_attention_masks, trp_token_type_ids,
                labels=None, guids=None, ana_mode=False):

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

        # trp_attention_masks_ = trp_attention_masks.sum(dim=3) > 0
        # trp_attention_masks_ = trp_attention_masks_.unsqueeze(dim=3).expand(-1, -1, -1, 768).to(dtype=torch.float64)
        #
        # all_trp_bert_feat_ = all_trp_bert_feat.reshape(6, 3, 4, -1).mul(torch.tensor(trp_attention_masks_, dtype=torch.float32).to('cuda'))

        if self.trp_attention:
            weights_no = sent_bert_feat.reshape(batch_size, num_choices, 1, -1).expand(-1, -1, trp_num, -1).mul(all_trp_bert_feat_masked).sum(dim=3)
            weights = softmax(weights_no, dim=2)
            avg_trp_bert_feat = weights.unsqueeze(dim=3).expand(-1, -1, -1, hid_dim).mul(all_trp_bert_feat_masked).sum(dim=2)

            # weights_no = sent_bert_feat.reshape(6, 3, 1, -1).expand(-1, -1, 4, -1).mul(all_trp_bert_feat_masked).sum(dim=3)
            # weights = softmax(weights_no, dim=2)
            # avg_trp_bert_feat = weights.unsqueeze(dim=3).expand(-1, -1, -1, 768).mul(all_trp_bert_feat_masked).sum(dim=2)
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




        # if ana_mode:
        #     trp_att_scores = []

        # sent_feat = self.feature_extractor(sent_input_ids, sent_attention_mask, sent_token_type_ids, labels)
            #
            # # 多个triple在explanation中，要加权求和，每个triple单独过一下Bert
            # trp_feats = []
            # for trp in trp_bert_input:
            #     trp_input_ids, trp_attention_mask, trp_token_type_ids = trp_bert_input # only the explanation's triples as features
            #     trp_feat = self.feature_extractor(trp_input_ids, trp_attention_mask, trp_token_type_ids)
            #     trp_feats.append(trp_feat)
            # if self.trp_attn:
            #     sent_feat_as_query = self.sent_atrp_att(sent_bert_input)
            #     per_trp_att_scores = torch.mv(trp_feats, sent_feat_as_query)
            #     norm_per_trp_att_scores = F.softmax(per_trp_att_scores, 0)
            #     if ana_mode:
            #         trp_att_scores.append(norm_per_trp_att_scores)
            #     trp_vec = torch.mv(trp_feats, norm_per_trp_att_scores)
            #     # trp_attn_vec = self.trp_attn(trp_vecs, sent_feat)
            # else:
            #     trp_vec = trp_feats.mean(0) # mean pooling
            # trp_vecs.append()
            #
            # concated = torch.cat((sent_feat, trp_attn_vec), 1)
            # logits = self.mlp(concated) #?? num_choice的维度去了哪�?            #
            # logits = self.classifier(pooled_output)  # [batch_size*num_choices, 1]
            # reshaped_logits = logits.view(-1, num_choices)  # [batch_size, num_choices]
            #
            # outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here  # [batch_size, num_choices]
            #
            # if labels is not None:  # 无label的test时，preds calculation要改
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(reshaped_logits, labels)
            #     outputs = (loss,) + outputs
            #
            # return outputs  # loss, reshaped_logits, (hidden_states), (attentions)
            #
            #





# class BertForMultipleChoice(BertPreTrainedModel):
#     r"""
#         **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
#             Labels for computing the multiple choice classification loss.
#             Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
#             of the input tensors. (see `input_ids` above)
#
#     Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#         **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#             Classification loss.
#         **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
#             of the input tensors. (see `input_ids` above).
#             Classification scores (before SoftMax).
#         **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
#             list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
#             of shape ``(batch_size, sequence_length, hidden_size)``:
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         **attentions**: (`optional`, returned when ``config.output_attentions=True``)
#             list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
#
#     Examples::
#
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
#         choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
#         input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
#         labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)
#         loss, classification_scores = outputs[:2]
#
#     """
#     def __init__(self, config):
#         super(BertForMultipleChoice, self).__init__(config)
#
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#
#         self.init_weights()
#
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None, labels=None):        # attention_mask [16, 3, 55]  input_ids [16, 3, 55]  token_type_ids [16, 3, 55] labels [16]
#         num_choices = input_ids.shape[1]  # = 3
#
#         input_ids = input_ids.view(-1, input_ids.size(-1))   # 1, 3, 128   [48, 55]
#         attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None    # [48, 55]
#         token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None    # [48, 55]
#         position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
#
#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask)    # outputs[0] [48, 55, 768]   outputs[1] [48, 768]
#
#         pooled_output = outputs[1]  # [48, 768]
#
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)  # [48, 1]
#         reshaped_logits = logits.view(-1, num_choices)  #？[16, 3]
#
#         outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here  # [16, 3]
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             outputs = (loss,) + outputs
#
#         return outputs  # loss, reshaped_logits, (hidden_states), (attentions)










