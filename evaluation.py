from __future__ import print_function
import os, sys
import pickle

import torch
import numpy
from data import get_test_loader
import time
import numpy as np
from model import DIME
from collections import OrderedDict
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()): 
            if i > 0:
                s += '  '
            if(k == 'lr'):
                v = '{:.3e}'.format(v.val)
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems(): 
            tb_logger.log_value(prefix + k, v.val, step=step)

def encode_data(model, data_loader, opt, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()
    end = time.time()

    max_n_word = 0
    l_idx = 0
    
    for i, batch_data in enumerate(data_loader):
        images, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        max_n_word = max(max_n_word, max(lengths))
    

    # numpy array to keep all the embeddings
    is_init = True
    for i, batch_data in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        images, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
    
        # compute the embeddings
        img_emb, self_att_emb, cap_emb, word_emb = model.forward_emb(batch_data, volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if is_init:
            is_init = False
            rgn = np.zeros((len(data_loader.dataset), self_att_emb.size(1), self_att_emb.size(2)), dtype=np.float32)
            img = np.zeros((len(data_loader.dataset), img_emb.size(1)), dtype=np.float32)
            wrd = np.zeros((len(data_loader.dataset), max_n_word, word_emb.size(2)), dtype=np.float32)
            stc = np.zeros((len(data_loader.dataset), cap_emb.size(1)), dtype=np.float32)
            stc_lens = np.zeros(len(data_loader.dataset), dtype=np.int)

        rgn[ids] = self_att_emb.detach().cpu().numpy().copy()
        img[ids] = img_emb.data.cpu().numpy().copy()
        cur_max_len = word_emb.size(1)
        wrd[ids, :cur_max_len, :] = word_emb.data.cpu().numpy().copy()
        stc[ids] = cap_emb.data.cpu().numpy().copy()
        # preserve the lengths of sentences
        stc_lens[ids] = np.asarray(lengths, dtype=np.int)

        del batch_data

    return rgn, img, wrd, stc, stc_lens


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = DIME(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    rgn, img, wrd, stc, stc_lens = encode_data(model, data_loader, opt)

    if not fold5:
        # no cross-validation, full evaluation
        embs = (rgn, img, wrd, stc)
        r, rt, sims = i2t(model, embs, stc_lens, opt,  return_ranks=True)
        ri, rti = t2i(model, sims, opt, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            if opt.itr == 'stc_rgn' or opt.itr == 'stc_rgn_max':
                embs = (rgn[i * 5000:(i + 1) * 5000], None, None, stc[i * 5000:(i + 1) * 5000])
            elif opt.itr == 'img_wrd':
                embs = (None, img[i * 5000:(i + 1) * 5000], wrd[i * 5000:(i + 1) * 5000], None)
            elif opt.itr == 'rgn_wrd':
                embs = (rgn[i * 5000:(i + 1) * 5000], img[i * 5000:(i + 1) * 5000], wrd[i * 5000:(i + 1) * 5000], \
                        stc[i * 5000:(i + 1) * 5000])
            else:
                embs = (None, img[i * 5000:(i + 1) * 5000], None, stc[i * 5000:(i + 1) * 5000])

            r, rt0, sims = i2t(model, embs, opt, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(model, sims, opt, return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')

def calItr(model, embs, stc_lens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    rgn, img, wrd, stc = embs
    n_img = len(rgn) 
    n_stc = len(wrd) 

    t0 = time.time()
    n_im_shard = (n_img-1) // shard_size + 1
    n_cap_shard = (n_stc-1) // shard_size + 1
    d = np.zeros((n_img, n_stc))
    if sys.stdout.isatty():
        pbar = tqdm(total=(n_im_shard * n_cap_shard))

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), n_img)
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), n_stc)
            cur_stc_lens = stc_lens[cap_start: cap_end]
            with torch.no_grad():
                rgn_block = torch.from_numpy(rgn[im_start:im_end]).cuda()
                stc_block = torch.from_numpy(stc[cap_start:cap_end]).cuda()
                img_block = torch.from_numpy(img[im_start:im_end]).cuda()
                wrd_block = torch.from_numpy(wrd[cap_start:cap_end]).cuda()
                sim = model.itr_module(rgn_block, img_block, wrd_block, stc_block, cur_stc_lens)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()

            if sys.stdout.isatty():
                pbar.update(1)
    if sys.stdout.isatty():
        pbar.close()
    print('Calculate similarity matrix elapses: {:.3f}s'.format(time.time() - t0))
    return d


def i2t(model, embs, stc_lens, opt, npts=None, return_ranks=False):
    t0 = time.time()
    rgn, img, wrd, stc = embs
    rgn = numpy.array([rgn[i] for i in range(0, len(rgn), 5)])
    img = numpy.array([img[i] for i in range(0, len(img), 5)])
    npts = len(rgn)

    embs = (rgn, img, wrd, stc)
    sims = calItr(model, embs, stc_lens, opt, shard_size=int(opt.batch_size * 2))
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), sims
    else:
        return (r1, r5, r10, medr, meanr), sims

def t2i(model, sims, opt, npts=None, return_ranks=False):
    t0 = time.time()
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

