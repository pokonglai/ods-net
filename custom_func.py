
import keras
from keras import backend as K

import tensorflow as tf
import numpy as np


def berHu_loss_elementwise(y_true, y_pred):
    ''' As described in deeper depth: https://arxiv.org/pdf/1606.00373.pdf '''
    ret_loss = 0
    y_diff = y_true - y_pred
    y_diff_abs = K.abs(y_diff)
    c = (1.0/5.0)*K.max(y_diff_abs)

    L2_berHu = (K.pow(y_diff_abs, 2) + c**2) / (2*c)
    berHu_tensor = tf.where(K.less_equal(y_diff_abs, c), y_diff_abs, L2_berHu)

    n_pixels = tf.to_float(tf.size(y_true))
    ret_loss = K.sum(berHu_tensor) / n_pixels

    return ret_loss

def berHu_loss_elementwise_w_border(y_true, y_pred):
    ''' Proposed Lpano as described in our paper. '''
    ret_loss = 0

    y_diff = y_true - y_pred
    y_diff_abs = K.abs(y_diff)
    c = (1.0/5.0)*K.max(y_diff_abs)

    L2_berHu = (K.pow(y_diff_abs, 2) + c**2) / (2*c)
    berHu_tensor = tf.where(K.less_equal(y_diff_abs, c), y_diff_abs, L2_berHu)

    ## regular reverse huber
    n_pixels = tf.to_float(tf.size(y_true))
    berHu_overall = K.sum(berHu_tensor) / n_pixels

    ## add extra weight to the borders
    ## build boolean mask by combining boundary conditions for the rows and cols
    shape = tf.shape(berHu_tensor)
    bs, R, C, chans = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), tf.range(shape[2]), tf.range(shape[3])) ## batch_size, width, height, channels
    row_lines = tf.logical_or(K.less_equal(R, 16), K.greater_equal(R, 112)) # these numbers will need to change when moving to larger image sizes or different border width
    col_lines = tf.logical_or(K.less_equal(C, 16), K.greater_equal(C, 240))
    border_mask = tf.logical_or(row_lines, col_lines)
    border_berHu_vals = tf.boolean_mask(berHu_tensor, border_mask)

    n_border_pixels = tf.to_float(tf.size(border_berHu_vals))
    berHu_border = K.sum(border_berHu_vals) / n_border_pixels

    lambda_frac = 0.5 # default paper amount
    # lambda_frac = 2.0 
    # lambda_frac = 20.0
    ret_loss = berHu_overall + lambda_frac*berHu_border 

    return ret_loss
    

def rmse(y_true, y_pred):
    ''' Root mean squared error '''
    return K.sqrt( K.mean( K.square(y_pred - y_true) ) ) 

def rmlse(y_true, y_pred):
    ''' Root mean log squared error '''
    return K.sqrt( K.mean( K.log( K.square(y_pred - y_true) ) ) )

def get_median(v):
    ''' Given a tensor v, return the median value of the entire tensor. '''
    v = tf.reshape(v, [-1])
    m = tf.shape(v)[0]//2
    return tf.nn.top_k(v, m).values[m-1]

def median_dist(y_true, y_pred):
    ''' Median absolute difference in depth values '''
    return get_median(K.abs(y_pred - y_true))

def log10(x):
    numer = tf.log(x)
    denom = tf.log(tf.constant(10, dtype=numer.dtype))
    return numer / denom

def log10_error(y_true, y_pred):
    ''' Average absolute difference between the log of the prediction and ground truth '''
    out = tf.maximum(y_pred, 1.0/255.0)
    gt = tf.maximum(y_true, 1.0/255.0)
    return K.mean(K.abs(log10(out) - log10(gt)))
    