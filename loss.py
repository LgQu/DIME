from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.opt = opt
        self.max_violation = max_violation
    
    def forward(self, scores):
        if self.opt.extra_stc > 0:
            return self.forward_extraStc(scores)
        elif self.opt.extra_img > 0:
            return self.forward_extraImg(scores)
        else:
            return self.forward_(scores)

    def forward_(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            
        return cost_s.sum() + cost_im.sum()

    def forward_extraStc(self, scores):
        n_img, n_stc = scores.size()
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1) 
        d1 = diagonal 
        d2 = diagonal.t() 

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0) 
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores[:, :n_img] - d2).clamp(min=0)  

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5  
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
            extra_zeros = torch.zeros((n_img, n_stc - n_img)).byte().cuda()

        mask_s = torch.cat([I, extra_zeros], dim=1)
        cost_s = cost_s.masked_fill_(mask_s, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    def forward_extraImg(self, scores):
        n_img, n_stc = scores.size()
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(1), 1)   
        d1 = diagonal 
        d2 = diagonal.t() 

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores[:n_stc, :] - d1).clamp(min=0)  
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)   

        # clear diagonals
        mask = torch.eye(scores.size(1)) > .5 
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
            extra_zeros = torch.zeros((n_img - n_stc, n_stc)).byte().cuda()

        mask_im = torch.cat([I, extra_zeros], dim=0)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(mask_im, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            # print(cost_im.max(0)[1])
            cost_im = cost_im.max(0)[0]
            
        return cost_s.sum() + cost_im.sum()
