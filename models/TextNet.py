import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import OrderedDict 

import tokenization
from .BERT import BertModel, BertConfig

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class EncoderText(nn.Module):
    """
    """
    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.opt = opt
        bert_config = BertConfig.from_json_file(opt.bert_config_file)
        self.bert = BertModel(bert_config)
        ckpt = torch.load(opt.init_checkpoint, map_location='cpu')
        embed_size = opt.embed_size
        self.bert.load_state_dict(ckpt)
        freeze_layers(self.bert)

        # 1D-CNN
        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = opt.embed_size
        bert_hid = bert_config.hidden_size
        self.fc = nn.Linear(bert_hid, opt.embed_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, bert_hid)) for K in Ks])
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.mapping = nn.Linear(len(Ks)*out_channel, opt.embed_size)

    def forward(self, input_ids, attention_mask, token_type_ids, lengths):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        x = all_encoder_layers[-1].unsqueeze(1)  
        if self.training:
            bert_emb = all_encoder_layers[-1].detach().mean(dim=1)
            bert_emb = F.normalize(bert_emb, dim=-1)
        x_emb = self.fc(all_encoder_layers[-1])
        x1 = F.relu(self.convs1[0](x)).squeeze(3)  
        x2 = F.relu(self.convs1[1](F.pad(x, (0, 0, 0, 1)))).squeeze(3)
        x3 = F.relu(self.convs1[2](F.pad(x, (0, 0, 1, 1)))).squeeze(3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.transpose(1, 2)  
        word_emb = self.mapping(x)
        word_emb = word_emb + x_emb
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in [x1, x2, x3]]
        x = torch.cat(x, 1)

        txt_emb = self.mapping(x)
        txt_emb = txt_emb + x_emb.mean(1)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)
        word_emb = F.normalize(word_emb, p=2, dim=-1)
        if self.training:
            return (txt_emb, bert_emb), word_emb
        else:
            return txt_emb, word_emb

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in own_state.items():
            if name in state_dict:
                new_state[name] = state_dict[name]
            else:
                new_state[name] = param
        super(EncoderText, self).load_state_dict(new_state)

