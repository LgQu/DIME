import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math

from models.InteractionModule import InteractionModule
from loss import ContrastiveLoss
from models.TextNet import EncoderText
from models.VisNet import EncoderImage

class DIME(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.txt_enc = EncoderText(opt)
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size, opt.direction, 
                                    opt.finetune, use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm, drop=opt.drop)
        self.itr_module = InteractionModule(opt)
        if torch.cuda.is_available(): 
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.itr_module.cuda()
            cudnn.benchmark = True
        
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt, margin=opt.margin,
                                            measure=opt.measure,
                                            max_violation=opt.max_violation)
  
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.itr_module.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.itr_module.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.itr_module.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.itr_module.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.itr_module.eval()

    def forward_emb(self, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        images, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        if torch.cuda.is_available():
            images = images.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        # Forward
        stc_emb, wrd_emb = self.txt_enc(input_ids, attention_mask, token_type_ids, lengths)
        img_emb, rgn_emb = self.img_enc(images)
        return img_emb, rgn_emb, stc_emb, wrd_emb

    def train_emb(self, epoch, batch_data, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        images, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        img_emb, self_att_emb, stc_emb, word_emb = self.forward_emb(batch_data)
        stc_emb, bert_emb = stc_emb
        if img_emb.size(0) < stc_emb.size(0): # extraStc
            bert_emb = bert_emb[:img_emb.size(0)]

        sim_mat, sim_paths = self.itr_module(self_att_emb, img_emb, word_emb, stc_emb, lengths)
        retrieval_loss = self.criterion(sim_mat)
        self.logger.update('Rank', retrieval_loss.item(), img_emb.size(0))
        sim_label = bert_emb.matmul(bert_emb.t())   
        path_sim_loss = ((sim_paths - sim_label) ** 2).sum() / self.opt.batch_size
        self.logger.update('Path', path_sim_loss.item(), img_emb.size(0)) 
        loss = retrieval_loss + path_sim_loss * self.opt.trade_off
        self.logger.update('Le', loss.item(), img_emb.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            if isinstance(self.params[0], dict):
                params = []
                for p in self.params:
                    params.extend(p['params'])
                clip_grad_norm(params, self.grad_clip)
            else:
                clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
