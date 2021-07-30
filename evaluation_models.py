from __future__ import print_function
import os, sys
import pickle

import torch
import numpy
from data import get_test_loader
import time
import numpy as np
from tqdm import tqdm
import copy

from model import DIME
from collections import OrderedDict
from misc.utils import print_options
from evaluation import encode_data, calItr
from evaluation import i2t as i2t_single
from evaluation import t2i as t2i_single



def evalrank_single(model_path, test_opt=None, data_path=None, split='test', fold5=False, data_loader=None):
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print('Best model 1: Epoch = {}, Eiters = {}'.format(checkpoint['epoch'], checkpoint['Eiters']))
    print('best_rsum: {:.2f}, best_r1: {:.2f}'.format(checkpoint['best_rsum'], checkpoint['best_r1']))
    if data_path is not None:
        opt.data_path = data_path

    model = DIME(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    if data_loader is None:
        data_loader = get_test_loader(split, opt.data_name, opt.batch_size, opt.workers, opt)

    print('Computing results...')
    t0 = time.time()

    rgn, img, wrd, stc, stc_lens = encode_data(model, data_loader, opt)
    print('encode_data elapses: {:.2f}'.format(time.time() - t0))

    if not fold5:
        # no cross-validation, full evaluation
        embs = (rgn, img, wrd, stc)
        r, rt, sims = i2t_single(model, embs, stc_lens, opt, return_ranks=True)
        ri, rti = t2i_single(model, sims, opt, return_ranks=True)
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

            cur_stc_lens = stc_lens[i * 5000:(i + 1) * 5000]
            r, rt0, sims = i2t_single(model, embs, cur_stc_lens, opt, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i_single(model, sims, opt, return_ranks=True)
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
        print("rsum: %.1f" % (mean_metrics[0] + mean_metrics[1] + mean_metrics[2] + mean_metrics[5] + mean_metrics[6] + mean_metrics[7]))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
        return data_loader

def evalrank_ensemble(model_path, model_path2, data_path=None, split='dev', fold5=False):
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print('Best model 1: Epoch = {}, Eiters = {}'.format(checkpoint['epoch'], checkpoint['Eiters']))
    print('best_rsum: {:.2f}, best_r1: {:.2f}'.format(checkpoint['best_rsum'], checkpoint['best_r1']))
    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']
    print('Best model 2: Epoch = {}, Eiters = {}'.format(checkpoint2['epoch'], checkpoint2['Eiters']))
    print('best_rsum: {:.2f}, best_r1: {:.2f}'.format(checkpoint2['best_rsum'], checkpoint2['best_r1']))

    if data_path is not None:
        opt.data_path = data_path

    # print_options(opt)
    model = DIME(opt)
    model2 = DIME(opt2)

    # load model state
    model.load_state_dict(checkpoint['model'])
    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, opt.batch_size, opt.workers, opt)

    print('Computing results...')
    t0 = time.time()

    rgn, img, wrd, stc, stc_lens = encode_data(model, data_loader, opt)
    rgn2, img2, wrd2, stc2, stc_lens2 = encode_data(model2, data_loader, opt2)
    print('encode_data elapses: {:.2f}'.format(time.time() - t0))
    if not fold5:
        # no cross-validation, full evaluation
        embs = (rgn, img, wrd, stc)
        embs2 = (rgn2, img2, wrd2, stc2)
        r, rt, sims = i2t(model, model2, embs, embs2, stc_lens, stc_lens2, opt,  return_ranks=True)
        ri, rti = t2i(sims, opt, return_ranks=True)
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
                embs2 = (rgn2[i * 5000:(i + 1) * 5000], None, None, stc2[i * 5000:(i + 1) * 5000])
            elif opt.itr == 'img_wrd':
                embs = (None, img[i * 5000:(i + 1) * 5000], wrd[i * 5000:(i + 1) * 5000], None)
                embs2 = (None, img2[i * 5000:(i + 1) * 5000], wrd2[i * 5000:(i + 1) * 5000], None)
            elif opt.itr == 'rgn_wrd':
                embs = (rgn[i * 5000:(i + 1) * 5000], img[i * 5000:(i + 1) * 5000], wrd[i * 5000:(i + 1) * 5000], \
                        stc[i * 5000:(i + 1) * 5000])
                embs2 = (rgn2[i * 5000:(i + 1) * 5000], img2[i * 5000:(i + 1) * 5000], wrd2[i * 5000:(i + 1) * 5000], \
                        stc2[i * 5000:(i + 1) * 5000])
            else:
                embs = (None, img[i * 5000:(i + 1) * 5000], None, stc[i * 5000:(i + 1) * 5000])
                embs2 = (None, img2[i * 5000:(i + 1) * 5000], None, stc2[i * 5000:(i + 1) * 5000])

            r, rt0, sims = i2t(model, model2, embs, embs2, stc_lens, stc_lens2, opt,  return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(sims, opt, return_ranks=True)
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

def i2t(model, model2, embs, embs2, stc_lens, stc_lens2, opt, npts=None, return_ranks=False):
    t0 = time.time()

    rgn, img, wrd, stc = embs
    rgn2, img2, wrd2, stc2 = embs2

    rgn = numpy.array([rgn[i] for i in range(0, len(rgn), 5)])
    img = numpy.array([img[i] for i in range(0, len(img), 5)])
    npts = len(rgn)
    rgn2 = numpy.array([rgn2[i] for i in range(0, len(rgn2), 5)])
    img2 = numpy.array([img2[i] for i in range(0, len(img2), 5)])

    embs = (rgn, img, wrd, stc)
    embs2 = (rgn2, img2, wrd2, stc2)
    sims = calItr(model, embs, stc_lens, opt, shard_size=opt.batch_size * 2)
    sims2 = calItr(model2, embs2, stc_lens2, opt, shard_size=opt.batch_size * 2)
    sims = (sims + sims2) / 2

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


def t2i(sims, opt, npts=None, return_ranks=False):
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

