import os
import cv2
import torch
import shutil
import logging
import datetime
import numpy as np
import torch
import torch.distributed as dist

from collections import OrderedDict
from utils.colormap import colormap
from utils.radam import RAdam


def reduce_tensor_dict(tensor_dict, world_size, mode='mean'):
    """
    average tensor dict over different GPUs
    """
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            tensor_dict[key] = reduce_tensor(tensor, world_size, mode)
    return tensor_dict


def reduce_tensor(tensor, world_size, mode='mean'):
    """
    average tensor over different GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= world_size
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt


def get_logger(save_dir):
    assert(save_dir != "")
    exp_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(save_dir, exp_string+'.log')

    logger = logging.getLogger("LOGGING")
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s-%(message)s")

    # log file stream
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def get_optimizers(conf, model):
    nets = ['encoder', 'decoder']
    # if conf.model.arch.use_attention:
    #     nets += ['attention']
   
    optimizers = []
    for net in nets:
        parameters = getattr(model.module, net).parameters()
        if conf.train.optim == "Adam":
            optimizer = torch.optim.Adam(parameters, conf.train.lr, 
                                         betas=(conf.train.beta1, conf.train.beta2), eps=conf.train.eps, 
                                         weight_decay=conf.train.weight_decay)
        elif conf.train.optim == "RAdam":
            optimizer = RAdam(parameters, conf.train.lr, 
                                         betas=(conf.train.beta1, conf.train.beta2), eps=1e-8,
                                         weight_decay=conf.train.weight_decay)
        else:
            raise NotImplementedError
        optimizers.append((net, optimizer))
    return optimizers


def adjust_learning_rate(conf, optimizers, epoch):
    for submodule, optimizer in optimizers:
        for param_group in optimizer.param_groups:
            if epoch < conf.train.warmup_step:
                lr = (epoch+1) / conf.train.warmup_step * conf.train.lr
            else:
                lr = max(conf.train.lr * (0.1 ** (epoch // conf.train.decay_step)), conf.train.min_lr)
            param_group['lr'] = lr


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def copy_weight(dst_dict, src_dict, key):
    ws = src_dict[key] 
    wd = dst_dict[key]
    if len(ws.shape) == 4:
        cout1, cin1, kh, kw = ws.shape
        cout2, cin2, kh, kw = wd.shape
        weight = torch.zeros((cout2, cin2, kh, kw)).float().to(ws.device)
        weight[:cout1, 0:cin1] = ws
        src_dict[key] = weight
    else:
        cout1, = ws.shape
        cout2, = wd.shape
        weight = torch.zeros((cout2,)).float().to(ws.device)
        weight[:cout1] = ws
        src_dict[key] = weight
    return src_dict


def remove_mismatch_weight(dst_dict, src_dict):
    new_dict = OrderedDict()
    mismatched = []
    for k,v in src_dict.items():
        if k in dst_dict and v.shape != dst_dict[k].shape:
            mismatched.append((k, v.shape, dst_dict[k].shape))
            continue
        new_dict[k] = v
    print(mismatched)
    return new_dict


def zero_weight(dst_dict, src_dict, key):
    src_dict[key] = torch.zeros_like(dst_dict[key])
    return src_dict


def remove_prefix(state_dict):
    new_state_dict = OrderedDict() 
    for k, v in state_dict.items():
        new_k = '.'.join(k.split('.')[1:])
        new_state_dict[new_k] = v
    return new_state_dict


def save_checkpoint(save_dir, model, epoch, best_sad, logger, mname="latest", best=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_out_path = "{}/ckpt_{}.pth".format(save_dir, mname)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.module.state_dict(),
        'best_sad': best_sad
    }, model_out_path)
    if best:
        shutil.copyfile(model_out_path, os.path.join(save_dir, 'ckpt_best.pth'))
    logger.info("Checkpoint saved to {}".format(model_out_path))


def log_time(batch_time, data_time):
    msg = '\n'
    msg += ('\tTime Batch {batch_time.val:.2f}({batch_time.avg:.2f}) '
                   'Data {data_time.val:.2f}({data_time.avg:.2f})').format(
                     batch_time=batch_time, data_time=data_time)
    return msg


def log_loss(loss_keys, losses):
    msg = ''
    for loss_key in loss_keys:
        msg += '\n\t{0:15s} {loss.val:.4f} ({loss.avg:.4f})'.format(loss_key, loss=losses[loss_key])
    return msg


def log_dict(info_dict):
    msg = ''
    for key, val in info_dict.items():
        msg += '\n\t{0:s} = {1:.4f}'.format(key, val.item()) 
    return msg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
