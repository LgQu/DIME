import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  

class AttentionLayer(nn.Module):
    def __init__(self, embed_size, h, is_share=False, drop=0.0):
        super(AttentionLayer, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear] 
        else:
            self.linears = clones(nn.Linear(embed_size, embed_size), 3)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)

    def forward(self, inp, mask=None):
        nbatches = inp.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (inp, inp, inp))]
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)     
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value) 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden, drop=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden)
        self.fc2 = nn.Linear(hidden, embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class SelfAttention(nn.Module):
    def __init__(self, embed_size, hid_size, h, drop=0.0):
        super(SelfAttention, self).__init__()
        self.h = h
        self.att_layer = AttentionLayer(embed_size, h, drop=drop)
        self.feed_forward_layer = FeedForward(embed_size, hid_size, drop=drop)
        self.dropout = nn.Dropout(drop)

    def forward(self, local_emb, mask=None):
        x = self.att_layer(local_emb, mask=mask)  
        self_att_emb = local_emb + self.dropout(x)
        x = self.feed_forward_layer(self_att_emb)      
        fw_emb = self_att_emb + self.dropout(x)

        return fw_emb
