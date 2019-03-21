import sys
sys.path.append("..")

import os
import random
import time

import numpy as np
import cv2
import h5py


from utils import *
from ply_io import *


def load_filelist(top_folder, fname, shuffle_flist=True):
    fpath = os.path.join(top_folder, fname)
    
    f = open(fpath, 'r')
    lines = f.readlines()
    f.close()

    ret_fpaths = []
    for L in lines:

        # subset_name, h5_fname = L.split()
        # ret_fpaths.append(os.path.join(top_folder, subset_name, h5_fname))

        # assumes that each line is a full path to the .h5 files
        # edit this if the data is stored in a location which is not part of the paths from the input file list
        ret_fpaths.append(L.strip())

    if shuffle_flist: random.shuffle(ret_fpaths) ## shuffle so that we do not go through an entire area of images before reaching another area
    return ret_fpaths


### LR -> left depth + normals
def load_LR_norm_hdf5(ipath, data_aug_params=None):
    h5f = h5py.File(ipath,'r')
    img_L = h5f['color_left'][:]
    img_R = h5f['color_right'][:]
    img_D = h5f['depth_left'][:]
    img_N = h5f['normals_left'][:]
    h5f.close()

    if data_aug_params != None:
        rows, cols = img_D.shape
        if data_aug_params['rgb2bgr_prob'] > 0.0:
            coin_toss_color = random.uniform(0, 1.0) ## apply a random transformation with certain probablity
            if coin_toss_color > data_aug_params['rgb2bgr_prob']:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                img_R = cv2.cvtColor(img_R, cv2.COLOR_RGB2BGR)

        if data_aug_params['jpeg_degrade'] > 0.0:
            coin_toss_color = random.uniform(0.0, 1.0) ## apply a random transformation with certain probablity
            if coin_toss_color > data_aug_params['jpeg_degrade']:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.choice(list(range(70,95)))]
                res, encimg_L = cv2.imencode('.jpg', img_L, encode_param)
                res, encimg_R = cv2.imencode('.jpg', img_R, encode_param)
                img_L = cv2.imdecode(encimg_L, 1)
                img_R = cv2.imdecode(encimg_R, 1)

        ## TODO: add in crop % and range from 50 to 95?
        has_cut = False
        if data_aug_params['cut_half_bot'] == True and not has_cut:
            coin_toss_cut_bot = random.uniform(0.0, 1.0)
            if coin_toss_cut_bot > data_aug_params['cut_half_bot_prob']:
                img_L[rows//2:][:] = np.array([0,0,0])
                img_R[rows//2:][:] = np.array([0,0,0])
                img_D[rows//2:][:] = 0
                img_N[rows//2:][:] = np.array([0,0,0])
                has_cut = True

        if data_aug_params['cut_half_top'] == True and not has_cut: # don't cut if we've already done so - prevent feeding in blank images
            coin_toss_cut_top = random.uniform(0.0, 1.0)
            if coin_toss_cut_top > data_aug_params['cut_half_top_prob']:
                img_L[:rows//2][:] = np.array([0,0,0])
                img_R[:rows//2][:] = np.array([0,0,0])
                img_D[:rows//2][:] = 0
                img_N[:rows//2][:] = np.array([0,0,0])
                has_cut = True

    return img_L/255, img_R/255, img_D, img_N


def load_LR_norm_hdf5_batch(input_img_lists, data_aug_params=None):
    imgs_train_L = []
    imgs_train_R = []
    imgs_gt_depth = []
    imgs_gt_norms = []

    for f in input_img_lists:
        img_L, img_R, img_D, img_N  = load_LR_norm_hdf5(f, data_aug_params)
        imgs_train_L.append(img_L)
        imgs_train_R.append(img_R)
        imgs_gt_depth.append(img_D)
        imgs_gt_norms.append(img_N)

    return imgs_train_L, imgs_train_R, imgs_gt_depth, imgs_gt_norms


def data_batcher(batch_size, filelist, shuffle_per_run=True):
    ''' Generic data batcher where the input is a list of file paths to individual files and a batch size. '''
    n_files = len(filelist)
    idx = 0
    while True:
        idx_start = idx * batch_size
        idx_end = (idx+1) * batch_size

        idx_diff = 0
        if idx_end >= n_files: 
            idx_diff = idx_end - (n_files-1)
            idx_end = n_files-1
            idx = 0

        fpaths = filelist[idx_start:idx_end]
        if idx_diff > 0: # some non-zero difference was left over ==> shuffle the file list because we've reach the end and looped back over
            fpaths += filelist[0:idx_diff]
            if shuffle_per_run: random.shuffle(filelist)

        idx += 1

        yield fpaths

