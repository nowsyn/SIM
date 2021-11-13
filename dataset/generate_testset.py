import os
import sys
import cv2
import numpy as np

from scipy.ndimage import morphology


names = ['defocus', 'fire', 'fur', 'glass_ice', 'hair_easy', 'hair_hard', 
         'insect', 'motion', 'net', 'plant_flower', 'plant_leaf', 'plant_tree', 
         'plastic_bag', 'sharp', 'smoke_cloud', 'spider_web', 'texture_holed', 
         'texture_smooth', 'water_drop', 'water_spray']
name2class = {name:idx for idx, name in enumerate(names)}


def gen_trimap(alpha, ksize=3, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


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
        bg = cv2.resize(bg, (new_bw, new_bh), cv2.INTER_LANCZOS4)
    bg = bg[0:h, 0:w, :]
    alpha_f = alpha[:,:,None] / 255.
    comp = (fg*alpha_f + bg*(1.-alpha_f)).astype(np.uint8)
    return comp, bg


def read_and_resize(fg_path, alpha_path, max_size=1920, min_size=800):
    fg = cv2.imread(fg_path)
    alpha = cv2.imread(alpha_path, 0)
    if max_size > 0 and min_size > 0:
        h, w = alpha.shape[:2]
        r = max_size / max(h,w)
        if r < 1:
            th, tw = h*r, w*r
        else:
            th, tw = h, w
        r = min_size / min(th, tw)
        if r > 1:
            th, tw = int(th*r), int(tw*r)
        else:
            th, tw = int(th), int(tw)
        if th!=h or tw!=w:
            alpha = cv2.resize(alpha, (tw,th), cv2.INTER_LANCZOS4)
            fg = cv2.resize(fg, (tw,th), cv2.INTER_LANCZOS4)
    return fg, alpha


def read_and_composite(bg_path, fg_path, alpha_path, max_size=1920, min_size=800):
    fg, alpha = read_and_resize(fg_path, alpha_path, max_size, min_size)
    bg = cv2.imread(bg_path)
    comp, bg = composite(bg, fg, alpha)
    return alpha, fg, bg, comp


def load_test_samples(test_fg_dir, test_bg_dir, filelist, sv_test_fg_dir):
    with open(filelist) as f:
        lines = f.read().splitlines()
        for line in lines:
            name, fg_name, bg_name = line.split(':')
            print(name, fg_name)
            alpha_file = os.path.join(test_fg_dir, name, "alpha", fg_name)
            fg_file = os.path.join(test_fg_dir, name, "fg", fg_name)
            bg_file = os.path.join(test_bg_dir, bg_name)
            filename = name + "_" + fg_name + "_" + bg_name
            alpha, fg, bg, comp = read_and_composite(bg_file, fg_file, alpha_file)

            trimap = gen_trimap(alpha)
            image_dir = os.path.join(sv_test_fg_dir, name, "merged")
            trimap_dir = os.path.join(sv_test_fg_dir, name, "trimap")
            alpha_dir = os.path.join(sv_test_fg_dir, name, "alpha")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(trimap_dir, exist_ok=True)
            os.makedirs(alpha_dir, exist_ok=True)

            cv2.imwrite(os.path.join(image_dir, fg_name[:-4]+"_"+bg_name[:-4]+".png"), comp)
            cv2.imwrite(os.path.join(trimap_dir, fg_name[:-4]+"_"+bg_name[:-4]+".png"), trimap)
            cv2.imwrite(os.path.join(alpha_dir, fg_name[:-4]+"_"+bg_name[:-4]+".png"), alpha)


if __name__ == "__main__":
    test_fg_dir = "PATH/TO/SIMD/ROOT/DIR"
    test_bg_dir = "PATH/TO/VOC/IMAGE/DIR"
    filelist_test = "SIMD_composition_test_filelist.txt"
    sv_test_fg_dir = "PATH/TO/SAVE/DIR"
    load_test_samples(test_fg_dir, test_bg_dir, filelist_test, sv_test_fg_dir)
