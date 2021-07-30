import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .SelfAttention import SelfAttention
from .Router import Router
from models.Refinement import Refinement


class RectifiedIdentityCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(RectifiedIdentityCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)

    def forward(self, x):
        path_prob = self.router(x)
        emb = self.keep_mapping(x)

        return emb, path_prob

class IntraModelReasoningCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(IntraModelReasoningCell, self).__init__()
        self.opt = opt
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)
        self.sa = SelfAttention(opt.embed_size, opt.hid_IMRC, opt.num_head_IMRC)

    def forward(self, inp, stc_lens=None):
        path_prob = self.router(inp)
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            x = inp.view(-1, n_local, dim)
        else:
            x = inp

        sa_emb = self.sa(x)
        if inp.dim() == 4:
            sa_emb = sa_emb.view(n_img, n_stc, n_local, -1)
        return sa_emb, path_prob

class CrossModalRefinementCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(CrossModalRefinementCell, self).__init__()
        self.direction = opt.direction
        self.refine = Refinement(opt.embed_size, opt.raw_feature_norm_CMRC, opt.lambda_softmax_CMRC, opt.direction) 
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)

    def forward(self, rgn, img, wrd, stc, stc_lens):
        if self.direction == 'i2t':
            l_emb = rgn
        else:
            l_emb = wrd

        path_prob = self.router(l_emb)
        rf_pairs_emb = self.refine(rgn, img, wrd, stc, stc_lens)
        return rf_pairs_emb, path_prob

class GlobalLocalGuidanceCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(GlobalLocalGuidanceCell, self).__init__()
        self.opt = opt
        self.direction = self.opt.direction
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)
        self.fc_1 = nn.Linear(opt.embed_size, opt.embed_size)
        self.fc_2 = nn.Linear(opt.embed_size, opt.embed_size)

    def regulate(self, l_emb, g_emb_expand):
        l_emb_mid = self.fc_1(l_emb)
        x = l_emb_mid * g_emb_expand
        x = F.normalize(x, dim=-2)
        ref_l_emb = (1 + x) * l_emb
        return ref_l_emb

    def forward_i2t(self, rgn, img, wrd, stc, stc_lens):
        n_img = rgn.size(0)
        n_rgn = rgn.size(-2)
        n_stc = stc.size(0) 
        ref_rgns = []
        for i in range(n_stc):
            if rgn.dim() == 4:
                query = rgn[:, i, :, :]
            else:
                query = rgn
            
            stc_i = stc[i].unsqueeze(0).unsqueeze(1).contiguous()
            stc_i_expand = stc_i.expand(n_img, n_rgn, -1)
            ref_rgn = self.regulate(query, stc_i_expand)
            ref_rgn = ref_rgn.unsqueeze(1)
            ref_rgns.append(ref_rgn)
        
        ref_rgns = torch.cat(ref_rgns, dim=1)  
        return ref_rgns

    def forward_t2i(self, rgn, img, wrd, stc, stc_lens):
        # print(rgn.size())
        n_img = rgn.size(0)
        n_rgn = rgn.size(-2)
        n_stc = stc.size(0) 
        n_wrd = wrd.size(-2)
        ref_wrds = []
        for i in range(n_stc):
            if wrd.dim() == 4:
                wrd_i = wrd[:, i, :, :]
                wrd_i_expand = wrd_i
            else:
                wrd_i = wrd[i]
                wrd_i_expand = wrd_i.unsqueeze(0).expand(n_img, -1, -1)
            img_expand = img.unsqueeze(1).expand(-1, n_wrd, -1)


            ref_wrd = self.regulate(wrd_i_expand, img_expand)
            ref_wrd = ref_wrd.unsqueeze(1)
            ref_wrds.append(ref_wrd)
        
        ref_wrds = torch.cat(ref_wrds, dim=1)   # (n_img, n_stc, n_wrd, d) 
        return ref_wrds

    def forward(self, rgn, img, wrd, stc, stc_lens):
        if self.direction == 'i2t':
            path_prob = self.router(rgn)

            ref_emb = self.forward_i2t(rgn, img, wrd, stc, stc_lens)
            
        else:
            path_prob = self.router(wrd)

            ref_emb = self.forward_t2i(rgn, img, wrd, stc, stc_lens)
        return ref_emb, path_prob



