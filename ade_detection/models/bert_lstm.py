from transformers import BertPreTrainedModel, BertForTokenClassification, AutoModel
import torch.nn.functional as F
from torchcrf import CRF
from torch import nn
import torch
log_soft = F.log_softmax

import logging
logger = logging.getLogger(__name__)


class Bert_LSTM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.bert_out_to_labels = nn.Linear(config.hidden_size, self.num_labels)
        
        # ultimo embedding di LSTM
        self.final_lstm = nn.LSTM(self.num_labels,
                                  self.num_labels,
                                  2,
                                  dropout=config.dropout,
                                  batch_first=True)
        self.init_weights()
        self.bert = AutoModel.from_pretrained(config.model, config=config)

    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.bert(input_ids, attention_mask)
        bert_output = outputs[0]
        bert_output = self.dropout(bert_output)
        probs = self.bert_out_to_labels(bert_output)
        logits = log_soft(probs, 2)
        attention_mask = attention_mask.type(torch.uint8)
        output = None
        
        lstm_out, _ = self.final_lstm(logits)
        lstm_out = lstm_out.contiguous()
        
        preds = lstm_out.argmax(axis=-1)
        
        outputs = (preds,)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lstm_out.view(-1, lstm_out.shape[2]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
    