import os
import cv2
import time
import glob
import random
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import networks.resnet as resnet_models

from collections import OrderedDict
from pprint import pprint
from networks.model import build_model
from utils.config import load_config
from utils.util import *
from data.util import *


#########################################################################################
#   args
#########################################################################################
parser = argparse.ArgumentParser(description='SIM')
parser.add_argument('-c', '--config', type=str, metavar='FILE', help='path to config file')
parser.add_argument('-p', '--phase', type=str, metavar='PHASE', help='train or test')


def build_classifier(args, logger):
    logger.info("=> creating classifier '{}'".format(args.arch))
    model = resnet_models.__dict__[args.arch](args.n_channel, num_classes=args.num_classes, pretrained=False)
    if os.path.isfile(args.resume_checkpoint):
        logger.info("=> loading checkpoint '{}'".format(args.resume_checkpoint))
        checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
        state_dict = remove_prefix(checkpoint['state_dict'])
        model.load_state_dict(state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.resume_checkpoint))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume_checkpoint))
        exit()
    return model


def build_sim_model(args, logger):
    model = build_model(args)

    if args.pretrain_checkpoint and os.path.isfile(args.pretrain_checkpoint):
        logger.info("Pretrain: loading '{}'".format(args.pretrain_checkpoint))
        ckpt = torch.load(args.pretrain_checkpoint, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        n_keys = len(ckpt.keys())
        ckpt = remove_mismatch_weight(model.state_dict(), ckpt)
        n_keys_rest = len(ckpt.keys())
        logger.info("Remove %d mismatched keys" % (n_keys - n_keys_rest))

        model.load_state_dict(ckpt, strict=False)
        logger.info("Pretrain: loaded '{}'".format(args.pretrain_checkpoint))
    else:
        logger.info("Pretrain: no checkpoint found at '{}'".format(args.pretrain_checkpoint))
    
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        logger.info("Resume: loading '{}'".format(args.resume_checkpoint))
        ckpt = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt, strict=True)
        logger.info("Resume: loaded '{}'".format(args.resume_checkpoint))
    else:
        logger.info("Resume: no checkpoint found at '{}'".format(args.resume_checkpoint))

    return model


def extract_semantic_trimap(model, image, trimap, thresh=0.3, return_cam=False):
    model.eval()
    with torch.no_grad():
        N, C, H, W = image.shape
        output, cam, feats = model(torch.cat([image, trimap/255.], dim=1))
        cam = F.interpolate(cam, (H, W), mode='bilinear')
        if return_cam: return cam, output

        cam_norm = (cam - cam.min()) / (cam.max() - cam.min())
        semantic_trimap = cam_norm * (trimap==128).float()
        torch.cuda.empty_cache()
    return semantic_trimap


def extract_semantic_trimap_whole(args, model, image, trimap, thresh=0.1):
    step = args.load_size
    N, C, H, W = image.shape
    cam = torch.zeros((N, args.num_classes, H, W)).to(image.device)
    weight = torch.zeros((N, args.num_classes, H, W)).to(image.device)
    for step in [320, 800]:
        xs = list(range(0, W-step, step//2)) + [W-step]
        ys = list(range(0, H-step, step//2)) + [H-step]
        for i in ys:
            for j in xs:
                patcht = trimap[:,:,i:i+step,j:j+step]
                if (patcht == 128).sum() == 0: continue
                patchi = image[:,:,i:i+step, j:j+step]
                patchc, out = extract_semantic_trimap(model, patchi, patcht, return_cam=True)
                cam[:,:,i:i+step,j:j+step] += patchc
                weight[:,:,i:i+step,j:j+step] += 1
    cam = cam / torch.clamp_min(weight,1)
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min())
    smap = cam_norm * (trimap == 128).float()
    return smap


def run_discriminator(args, discriminator, pred, gt):
    g_fg_trans = gt['fg_trans'] # (N, 3, H, W)
    g_alpha = gt['alpha'] # (N, 1, H, W)
    p_alpha = pred['alpha'] # (N, 1, H, W)

    with torch.no_grad():
        if args.n_channel == 4:
            # keep consistent with training
            real_inputs = torch.cat([g_fg_trans, g_alpha], dim=1)
            fake_inputs = torch.cat([g_fg_trans, p_alpha], dim=1)
        else:
            real_inputs = g_alpha.repeat((1,3,1,1))
            fake_inputs = p_alpha.repeat((1,3,1,1))

        real_ret,_,real_feats = discriminator(real_inputs)
        fake_ret,_,fake_feats = discriminator(fake_inputs)

    out = {
        "real_ret": real_ret, 
        "fake_ret": fake_ret,
        "real_feats": real_feats,
        "fake_feats": fake_feats,
    }
    return out


def load_sim_samples(cfg):
    names = ['defocus', 'fire', 'fur', 'glass_ice', 'hair_easy', 'hair_hard', 
             'insect', 'motion', 'net', 'plant_flower', 'plant_leaf', 'plant_tree', 
             'plastic_bag', 'sharp', 'smoke_cloud', 'spider_web', 'texture_holed', 
             'texture_smooth', 'water_drop', 'water_spray']
    name2class = {name:idx for idx, name in enumerate(names)}

    data_dir = cfg.data.test_dir
    merged = sorted(glob.glob("%s/*/merged/*.png" % data_dir))
    trimap = sorted(glob.glob("%s/*/trimap/*.png" % data_dir))
    alpha = sorted(glob.glob("%s/*/alpha/*.png" % data_dir))

    print('Found %d samples' % len(merged))

    filenames = []
    target = []
    names = []
    for fp in merged:
        splits = fp.split('/')
        filenames.append(splits[-3] + "_" + splits[-1])
        names.append(splits[-3])
        target.append(name2class[splits[-3]])
    print('Found %d samples' % len(alpha))
    return zip(alpha, trimap, merged, names, filenames, target)


def load_adobe_samples(cfg):
    data_dir = cfg.data.test_dir
    merged = sorted(glob.glob("%s/merged/*.png" % data_dir))
    trimap = sorted(glob.glob("%s/trimap/*.png" % data_dir))
    alpha = sorted(glob.glob("%s/alpha/*.png" % data_dir))

    filenames = []
    target = []
    names = []
    for fp in merged:
        splits = fp.split('/')
        filenames.append(splits[-1])
        names.append(None)
        target.append(None)
    print('Found %d samples' % len(alpha))
    return zip(alpha, trimap, merged, names, filenames, target)


def preprocess(alpha_path, trimap_path, image_path, stride=8):
    alpha = cv2.imread(alpha_path, 0)
    trimap = cv2.imread(trimap_path, 0)
    image = cv2.imread(image_path)

    h, w = image.shape[:2]
    pad_h = (h // stride + 1) * stride - h
    pad_w = (w // stride + 1) * stride - w

    trimap = cv2.copyMakeBorder(trimap, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    alpha = cv2.copyMakeBorder(alpha, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    image_scale, image_trans = transform(image, scale=255.)
    trimap_tensor = torch.from_numpy(trimap).unsqueeze(0)
    alpha_tensor = torch.from_numpy(alpha/255.).unsqueeze(0)

    trimap_2chn = trimap_to_2chn(trimap)
    trimap_clks = trimap_to_clks(trimap_2chn, 320)
    trimap_2chn = torch.from_numpy(trimap_2chn.transpose((2,0,1)))
    trimap_clks = torch.from_numpy(trimap_clks.transpose((2,0,1)))

    inputs = {
        "alpha": alpha_tensor.unsqueeze(0),
        "trimap": trimap_tensor.unsqueeze(0),
        "image_scale": image_scale.unsqueeze(0),
        "image_trans": image_trans.unsqueeze(0),
        "trimap_2chn": trimap_2chn.unsqueeze(0), 
        "trimap_clks": trimap_clks.unsqueeze(0), 
        "origin_image": image,
        "origin_alpha": alpha,
        "origin_h": h,
        "origin_w": w
    } 
    return inputs


def save_prediction(pred, save_path):
    p_a = pred[0,0].data.cpu().numpy() * 255
    cv2.imwrite(save_path, p_a)


def run(cfg, model, classifier, logger):
    batch_time = AverageMeter()
    total_sad = AverageMeter()

    sad_list = {}

    model.eval()

    end = time.time()

    if cfg.task == 'SIM':
        samples = load_sim_samples(cfg)
    elif cfg.task == 'Adobe':
        samples = load_adobe_samples(cfg)
    else:
        raise NotImplementedError

    samples = list(samples)

    with torch.no_grad():
        for idx, (alpha_path, trimap_path, image_path, name, filename, target) in enumerate(samples):
            inputs = preprocess(alpha_path, trimap_path, image_path)

            trimap = inputs['trimap'].float().to(device)
            alpha = inputs['alpha'].float().to(device)
            image_scale = inputs['image_scale'].float().to(device)
            image_trans = inputs['image_trans'].float().to(device)
            trimap_2chn = inputs['trimap_2chn'].float().to(device)
            trimap_clks = inputs['trimap_clks'].float().to(device)

            oh = inputs['origin_h']
            ow = inputs['origin_w']

            semantic_trimap = extract_semantic_trimap_whole(cfg.classifier, classifier, image_trans, trimap)
            out = model(image_scale, trimap_2chn, image_trans, trimap_clks, semantic_trimap, is_training=False)
            out['alpha'] = torch.clamp(out['alpha'], 0, 1)

            pred = out['alpha']
            pred[trimap==0] = 0
            pred[trimap==255] = 1

            pred = pred[:,:,0:oh,0:ow]
            trimap = trimap[:,:,0:oh,0:ow]
            alpha = alpha[:,:,0:oh,0:ow]

            save_prediction(pred, os.path.join(cfg.log.visualize_path, filename))

            sad = ((pred - alpha) * (trimap==128).float()).abs().sum() / 1000.
            total_sad.update(sad)
            if name is not None:
                if name not in sad_list: 
                    sad_list[name] = []
                sad_list[name].append(sad.item())
            msg = 'Test: [{0}/{1}] SAD {sad:.4f}'.format(idx, len(samples), sad=sad)
            logger.info(msg)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            del trimap, alpha, pred
            del out, inputs
            del image_scale, image_trans, trimap_2chn, trimap_clks

    msg = 'Test: Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
    logger.info(msg)

    for key in sorted(sad_list.keys()):
        logger.info("{} {:.4f}".format(key, np.array(sad_list[key]).mean()))

    logger.info("MeanSAD: {sad.avg:.3f}".format(sad=total_sad))

    return total_sad.avg


def main():
    global args, device

    args = parser.parse_args()
    cfg = load_config(args.config)

    cfg.version = args.config.split('/')[-1].split('.')[0]
    cfg.phase = args.phase


    if cfg.is_default:
        raise ValueError("No .toml config loaded.")

    USE_CUDA = torch.cuda.is_available()

    cfg.log.logging_path = os.path.join(cfg.log.logging_path, cfg.version)
    cfg.log.visualize_path = os.path.join(cfg.log.visualize_path, cfg.version)

    os.makedirs(cfg.log.logging_path, exist_ok=True)
    os.makedirs(cfg.log.visualize_path, exist_ok=True)

    logger = get_logger(cfg.log.logging_path)

    pprint(cfg, stream=open(os.path.join(cfg.log.logging_path, "cfg.json"), 'w'))

    classifier = build_classifier(cfg.classifier, logger) 
    model = build_sim_model(cfg.model, logger)

    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    model.cuda()
    classifier.cuda()
    run(cfg, model, classifier, logger)


if __name__ == '__main__':
    main()
