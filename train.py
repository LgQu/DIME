import pickle
import os, sys
import time
import shutil
import numpy as np
import torch

import data
from model import DIME
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
import logging
import tensorboard_logger as tb_logger
import argparse
import evaluation_models
from misc.utils import print_options


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_name', default='runs/runX', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--data_path', default='/data', help='path to datasets')
    parser.add_argument('--data_name', default='precomp', help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument("--bert_path", default='./', type=str, help="The BERT model path.")
    parser.add_argument('--max_words', default=32, type=int, help='maximum number of words in a sentence.')
    parser.add_argument('--extra_stc', type=int, default=0, help='Sample (extra_stc * bs) extra sentences.')
    parser.add_argument('--extra_img', type=int, default=0, help='Sample (extra_stc * bs) extra images.')
    # Optimization
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true', default=True, help='Use max instead of sum in the rank loss.')
    parser.add_argument('--trade_off', default=0.5, type=float, help='Trade-off parameter for path regularization.')
    # Base
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    # parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
    parser.add_argument('--embed_size', default=256, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--direction', type=str, default='i2t',help='Version of model, i2t | t2i')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=2000, type=int, help='Number of steps to run validation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune the image encoder.')
    # parser.add_argument('--cnn_type', default='vgg19', help="""The CNN used for image encoder(e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true', help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true', help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true', help='Ensure the training is always done in train mode (Not recommended).')
    # DIME
    parser.add_argument('--num_head_IMRC', type=int, default=16, help='Number of heads in Intra-Modal Reasoning Cell')
    parser.add_argument('--hid_IMRC', type=int, default=512, help='Hidden size of FeedForward in Intra-Modal Reasoning Cell')
    parser.add_argument('--raw_feature_norm_CMRC', default="clipped_l2norm", help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax_CMRC', default=4., type=float, help='Attention softmax temperature.')
    parser.add_argument('--hid_router', type=int, default=512, help='Hidden size of MLP in routers')
    opt = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)
    # BERT Configuration
    opt.vocab_file = opt.bert_path + '/vocab.txt' 
    opt.bert_config_file = opt.bert_path + '/bert_config.json'
    opt.init_checkpoint = opt.bert_path + '/pytorch_model.bin'
    opt.do_lower_case = True
    print_options(opt)
    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, opt.batch_size, opt.workers, opt)

    model = DIME(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            best_r1 = checkpoint['best_r1']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            # validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    best_r1 = 0

    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        best_rsum, best_r1 = train(opt, train_loader, model, epoch, val_loader, best_rsum, best_r1)
        # evaluate on validation set
        rsum, r1 = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        best_r1 = max(r1, best_r1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'best_r1': best_r1, 
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader, best_rsum, best_r1):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()

    for i, train_data in enumerate(train_loader):
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end, 1)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(epoch, train_data)

        # measure elapsed time
        batch_time.update(time.time() - end, 1)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            # validate(opt, val_loader, model)

            # evaluate on validation set
            rsum, r1 = validate(opt, val_loader, model)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            best_r1 = max(r1, best_r1)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'best_r1': best_r1, 
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, prefix=opt.logger_name + '/')
    
    return best_rsum, best_r1

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    t0 = time.time()
    embs = encode_data(model, val_loader, opt, opt.log_step, logging.info)
    print('encode_data elapses: {:.2f}s'.format(time.time() - t0))

    # caption retrieval
    (r1, r5, r10, medr, meanr), sims = i2t(model, embs[:-1], embs[-1], opt)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(model, sims, opt)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r1i + r5i 

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, r1

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every lr_update epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
