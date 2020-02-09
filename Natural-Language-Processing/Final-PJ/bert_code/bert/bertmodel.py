import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import *


class BertForTask1(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTask1.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTask1, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, guids=None):
        # labels [batch]只有0/1   input_ids/attention_mask/token_type_ids [batch, choice, max_length]
        num_sents = input_ids.size(1)
        assert num_sents == 2

        results = {}

        # Calculate each sent separately, and return loss * logits tuple containing 2 sent result
        logits_list = []
        loss_list = []
        for i in range(num_sents):
            input_ids_ = input_ids[:, i, :].squeeze(dim=1)
            attention_mask_ = attention_mask[:, i, :].squeeze(dim=1) if attention_mask is not None else None  # [batch, 1, max_length] --> [batch, max_length]
            token_type_ids_ = token_type_ids[:, i, :].squeeze(dim=1) if token_type_ids is not None else None

            outputs = self.bert(input_ids_,
                                attention_mask=attention_mask_,
                                token_type_ids=token_type_ids_)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)  # [batch,2]
            # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            logits_list.append(logits)

            if labels is not None: # Training的时候执行，计算loss
                loss_fct = CrossEntropyLoss()
                label_ = 1 - labels if i == 0 else labels # 原始label=0，则sent0错了，sent0_label=1，sent1_label=0.   原始label=1，则sent1错了，sent1_label=1, sent0_label=0
                loss = loss_fct(logits.view(-1, self.num_labels), label_.view(-1))  # 计算loss value, num_labels=22
                loss_list.append(loss)

        loss_sum = (loss_list[0] + loss_list[1]) / 2
        results.update({"logits": logits_list, "loss": loss_sum})

        return results




class BertForTask2(BertPreTrainedModel):
    """
    Labels: torch.LongTensor of shape (batch_size,)
        Labels for computing the multiple choice classification loss.
        Indices should be in [0, ..., num_choices] where num_choices is the size of the second dimension of input tensors.

    Outputs:
        loss: torch.FloatTensor of shape (1,), classification loss.
        classification_scores: torch.FloatTensor of shape (batch_size, num_choices), classification scores before Softmax.
        hidden_states: only returned when config.output_hidden_states=True. torch.FloatTensor.
            (one for the output of each layer + output of the embedding) of shape (batch_size, sequence_length, hidden_size).
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions: only returned when config.output_attention=True.
            (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
            Attention weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForTask2.from_pretrained("bert-base-uncased")
        choices = ["I eat apples.", "I eat stones."]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # (Batch_size 1, 2 choices)
        labels = torch.tensor(1).unsqueeze(0)  # (Batch_size 1,)
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]
    """

    def __init__(self, config):
        super(BertForTask2, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, guids=None):
        # attention_mask [batch_size, num_choices, max_length]  input_ids [batch_size, num_choices, max_length]
        # token_type_ids [batch_size, num_choices, max_length]  labels [batch_size,]
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))  # [batch_size, num_choices, max_length] --> [batch_size*num_choices, max_length]
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs[0] [batch_size*num_choices, max_length, hidden_size] ;   outputs[1] [batch_size*num_choices, hidden_size]

        pooled_output = outputs[1]  # [batch_size*num_choices, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size*num_choices, 1]
        reshaped_logits = logits.view(-1, num_choices)  # [batch_size, num_choices]

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here  # [batch_size, num_choices]

        if labels is not None: # 无label的test时，preds calculation要改
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # loss, reshaped_logits, (hidden_states), (attentions)


















