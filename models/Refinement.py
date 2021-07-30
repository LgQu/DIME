import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)   #(n, d, qL)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)   #(n, cL, qL)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=-1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        # attn = l2norm(attn, 2)
        attn = F.normalize(attn, dim=2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous() #(n, qL, cL)
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)    #(n*qL, cL)
    attn = F.softmax(attn*smooth, dim=-1)                #(n*qL, cL)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)   #(n, qL, cL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()    #(n, cL, qL)

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)   #(n, d, cL)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)    #(n, d, qL)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)    #(n, qL, d)

    return weightedContext, attnT


class Refinement(nn.Module):
    def __init__(self, embed_size, raw_feature_norm, lambda_softmax, direction):
        super(Refinement, self).__init__()
        self.raw_feature_norm = raw_feature_norm
        self.lambda_softmax = lambda_softmax
        self.direction = direction

        self.fc_scale = nn.Linear(embed_size, embed_size)
        self.fc_shift = nn.Linear(embed_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, embed_size)
        self.fc_2 = nn.Linear(embed_size, embed_size)

    def refine(self, query, weiContext):
        scaling = F.tanh(self.fc_scale(weiContext))
        shifting = self.fc_shift(weiContext)  
        modu_res = self.fc_2(F.relu(self.fc_1(query * scaling + shifting))) 
        ref_q = modu_res + query

        return ref_q

    def forward_i2t(self, rgn, wrd, cap_lens):
        ref_imgs = []
        n_image = rgn.size(0)
        n_caption = wrd.size(0) 

        for i in range(n_caption):
            if rgn.dim() == 4:
                query = rgn[:, i, :, :]  # (n_img, r_rgn, d)
            else:
                query = rgn
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = wrd[i, :n_word, :].unsqueeze(0).contiguous()
            # (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            weiContext, attn = func_attention(query, cap_i_expand, self.raw_feature_norm, smooth=self.lambda_softmax)
            ref_img = self.refine(query, weiContext)
            ref_img = ref_img.unsqueeze(1)
            ref_imgs.append(ref_img)

        ref_imgs = torch.cat(ref_imgs, dim=1)   #(n_img, n_stc, n_rgn, d)
        return ref_imgs       

    def forward_t2i(self, rgn, wrd, cap_lens):
        ref_wrds = []
        n_image = rgn.size(0)
        n_caption = len(cap_lens)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            if wrd.dim() == 4:
                # cap_i_expand = wrd[:, i, :n_word, :]
                cap_i_expand = wrd[:, i, :, :]
            else:
                # cap_i = wrd[i, :n_word, :].unsqueeze(0).contiguous()
                cap_i = wrd[i, :, :].unsqueeze(0).contiguous()
                cap_i_expand = cap_i.repeat(n_image, 1, 1) # (n_image, n_word, d)
            weiContext, attn = func_attention(cap_i_expand, rgn, self.raw_feature_norm, smooth=self.lambda_softmax)
            ref_wrd = self.refine(cap_i_expand, weiContext)
            ref_wrd = ref_wrd.unsqueeze(1)
            ref_wrds.append(ref_wrd)

        ref_wrds = torch.cat(ref_wrds, dim=1)   #(n_img, n_stc, n_rgn, d)
        return ref_wrds

    def forward(self, rgn, img, wrd, stc, stc_lens):
        if self.direction == 'i2t':
            ref_emb = self.forward_i2t(rgn, wrd, stc_lens) #(n_img, n_stc, n_rgn, d)
        else:
            ref_emb = self.forward_t2i(rgn, wrd, stc_lens) #(n_img, n_stc, n_wrd, d)

        return ref_emb

    

