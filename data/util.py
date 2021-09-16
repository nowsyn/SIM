import os
import cv2
import numpy as np
import torch


def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)


def get_fname(x):
    return os.path.splitext(os.path.basename(x))[0]


def gen_trimap(alpha, ksize=3, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad


def transform(image, scale=255.):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    image_scale = image / scale
    image_trans = (image_scale - mean) / std
    image_scale = torch.from_numpy(image_scale.transpose(2,0,1)).float()
    image_trans = torch.from_numpy(image_trans.transpose(2,0,1)).float()
    return image_scale, image_trans


def trimap_to_2chn(trimap):
    h, w = trimap.shape[:2]
    trimap_2chn = np.zeros((h, w, 2), dtype=np.float32)
    trimap_2chn[:,:,0] = (trimap == 0)
    trimap_2chn[:,:,1] = (trimap == 255)
    return trimap_2chn


def trimap_to_clks(trimap, L=320):
    h, w = trimap.shape[:2]
    clicks = np.zeros((h, w, 6), dtype=np.float32)
    for k in range(2):
        if (np.count_nonzero(trimap[:, :, k]) > 0):
            dt_mask = -dt(1 - trimap[:, :, k])**2
            clicks[:, :, 3*k] = np.exp(dt_mask / (2 * ((0.02 * L)**2)))
            clicks[:, :, 3*k+1] = np.exp(dt_mask / (2 * ((0.08 * L)**2)))
            clicks[:, :, 3*k+2] = np.exp(dt_mask / (2 * ((0.16 * L)**2)))
    return clicks


def composite(bg, fg, alpha):
    # bg: [h, w, 3], fg: [h, w, 3], alpha: [h, w]
    h, w ,c = fg.shape
    bh, bw, bc = bg.shape
    wratio = float(w) / bw
    hratio = float(h) / bh
    ratio = wratio if wratio > hratio else hratio     
    if ratio > 1:
        new_bw = int(bw * ratio + 1.0)
        new_bh = int(bh * ratio + 1.0)
        bg = cv2.resize(bg, (new_bw, new_bh), cv2.INTER_LINEAR)
    bg = bg[0:h, 0:w, :]
    alpha_f = alpha[:,:,None] / 255.
    comp = (fg*alpha_f + bg*(1.-alpha_f)).astype(np.uint8)
    return comp, bg
