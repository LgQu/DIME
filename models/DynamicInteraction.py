import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle

from models.Cells import RectifiedIdentityCell, IntraModelReasoningCell, GlobalLocalGuidanceCell, CrossModalRefinementCell

def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DynamicInteraction_Layer0(nn.Module):
    def __init__(self, opt, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.opt = opt
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.ric = RectifiedIdentityCell(opt, num_out_path)
        self.imrc = IntraModelReasoningCell(opt, num_out_path)
        self.glgc = GlobalLocalGuidanceCell(opt, num_out_path)
        self.cmrc = CrossModalRefinementCell(opt, num_out_path)

    def forward(self, rgn, img, wrd, stc, stc_lens):
        if self.opt.direction == 'i2t':
            aggr_res_lst = self.forward_i2t(rgn, img, wrd, stc, stc_lens)
        else:
            aggr_res_lst = self.forward_t2i(rgn, img, wrd, stc, stc_lens)
        return aggr_res_lst
         
    def forward_i2t(self, rgn, img, wrd, stc, stc_lens):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(rgn)
        emb_lst[1], path_prob[1] = self.glgc(rgn, img, wrd, stc, stc_lens)
        emb_lst[2], path_prob[2] = self.imrc(rgn)
        emb_lst[3], path_prob[3] = self.cmrc(rgn, img, wrd, stc, stc_lens)

        gate_mask = (sum(path_prob) < self.threshold).float()
        all_path_prob = torch.stack(path_prob, dim=2)  
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze3d(path_prob[j][:, i])
                if emb_lst[j].dim() == 3:
                    cur_emb = emb_lst[j].unsqueeze(1)
                else:   # 4
                    cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb.unsqueeze(1)
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob

    def forward_t2i(self, rgn, img, wrd, stc, stc_lens):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(wrd)
        emb_lst[1], path_prob[1] = self.glgc(rgn, img, wrd, stc, stc_lens)
        emb_lst[2], path_prob[2] = self.imrc(wrd, stc_lens=stc_lens)
        emb_lst[3], path_prob[3] = self.cmrc(rgn, img, wrd, stc, stc_lens)

        gate_mask = (sum(path_prob) < self.threshold).float() 
        all_path_prob = torch.stack(path_prob, dim=2)  
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0] 
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze2d(path_prob[j][:, i]).unsqueeze(0) 
                if emb_lst[j].dim() == 3:
                    cur_emb = emb_lst[j].unsqueeze(0)
                else:  
                    cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb.unsqueeze(0)
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob

class DynamicInteraction_Layer(nn.Module):
    def __init__(self, opt, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.opt = opt
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.ric = RectifiedIdentityCell(opt, num_out_path)
        self.glgc = GlobalLocalGuidanceCell(opt, num_out_path)
        self.imrc = IntraModelReasoningCell(opt, num_out_path)
        self.cmrc = CrossModalRefinementCell(opt, num_out_path)
        

    def forward(self, ref_emb, rgn, img, wrd, stc, stc_lens):
        if self.opt.direction == 'i2t':
            aggr_res_lst = self.forward_i2t(ref_emb, rgn, img, wrd, stc, stc_lens)
        else:
            aggr_res_lst = self.forward_t2i(ref_emb, rgn, img, wrd, stc, stc_lens)
        
        return aggr_res_lst

    def forward_i2t(self, ref_rgn, rgn, img, wrd, stc, stc_lens):
        assert len(ref_rgn) == self.num_cell and ref_rgn[0].dim() == 4
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(ref_rgn[0])
        emb_lst[1], path_prob[1] = self.glgc(ref_rgn[1], img, wrd, stc, stc_lens)
        emb_lst[2], path_prob[2] = self.imrc(ref_rgn[2])
        emb_lst[3], path_prob[3] = self.cmrc(ref_rgn[3], img, wrd, stc, stc_lens)

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float() 
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_rgn[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=3) 
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()  
            all_path_prob = torch.stack(path_prob, dim=3)   
            
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, :, i] for i in range(all_path_prob.size(3))]
            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, :, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:, :, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob

    def forward_t2i(self, ref_wrd, rgn, img, wrd, stc, stc_lens):
        assert len(ref_wrd) == self.num_cell and ref_wrd[0].dim() == 4
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(ref_wrd[0])
        emb_lst[1], path_prob[1] = self.glgc(rgn, img, ref_wrd[1], stc, stc_lens)
        emb_lst[2], path_prob[2] = self.imrc(ref_wrd[2], stc_lens)
        emb_lst[3], path_prob[3] = self.cmrc(rgn, img, ref_wrd[3], stc, stc_lens)
        
        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float() 
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_wrd[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=3)  
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()   
            all_path_prob = torch.stack(path_prob, dim=3)   
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, :, i] for i in range(all_path_prob.size(3))]

            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, :, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:, :, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob

    


