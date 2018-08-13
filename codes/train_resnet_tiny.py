#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: tiny-CAM-resnet.py

# Written by Junsuk Choe <skykite@yonsei.ac.kr>
# CAM / Grad-CAM implementation based on Pre-activation ResNet


import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
import random

from PIL import Image
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils import viz
from tensorpack.tfutils.tower import get_current_tower_context

from utils import *
from utils_loc import *
from utils_args import *
from models_resnet import *

class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.uint8, [None, args.final_size, args.final_size, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label'),
                tf.placeholder(tf.int32, [None], 'xa'),
                tf.placeholder(tf.int32, [None], 'ya'),
                tf.placeholder(tf.int32, [None], 'xb'),
                tf.placeholder(tf.int32, [None], 'yb')]

    def build_graph(self, image, label, xa, ya, xb, yb):
        image = image_preprocess(image, bgr=True) # image = (image - image_mean) / image_std
        label_onehot = tf.one_hot(label,200)

        ctx = get_current_tower_context()
        isTrain = ctx.is_training
        
        cfg = {
            18: ([2, 2, 2, 2]),
            34: ([3, 4, 6, 3]),
        }
        defs = cfg[DEPTH]
        with argscope(Conv2D, use_bias=False,
                            kernel_initializer=
                                tf.variance_scaling_initializer(
                                    scale=2.0, mode='fan_out')):
            convmaps = Conv2D('conv0', image, 64, 7, strides=1)
            convmaps = BatchNorm('bnfirst', convmaps)
            convmaps = tf.nn.relu(convmaps, 'relufirst')
            #convmaps = MaxPooling('pool0', convmaps, 3, strides=2, padding='SAME') # 32x32
            convmaps = preresnet_group(
                    'group0', convmaps, 64, defs[0], 1, isTrain, args.sn) # 32x32
            convmaps = preresnet_group(
                    'group1', convmaps, 128, defs[1], 2, isTrain, args.sn) # 16x16
            convmaps = preresnet_group(
                    'group2', convmaps, 256, defs[2], 2, isTrain, args.sn) # 8x8
            convmaps_target = preresnet_group(
                    'group3new', convmaps, 512, defs[3], 1, isTrain, args.sn)
            convmaps_gap = tf.reduce_mean(convmaps_target, [1,2], name='gap')
            logits = FullyConnected('linearnew', convmaps_gap, 200)

        weights = tf.identity(w, name='linearweight')
        activation_map = tf.identity(convmaps_target, name='actmap')
        y_c = tf.reduce_sum(tf.multiply(logits, label_onehot), axis=1)
        target_conv_layer_grad = tf.identity(
                    tf.gradients(y_c, convmaps_target)[0], name='grad')

        loss = compute_loss_and_error(logits, label)
        wd_cost = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        return tf.add_n([loss, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        gradprocs = [gradproc.ScaleGradient(
            [('conv0.*', 0.1), ('group[0-2].*', 0.1)])]
        return optimizer.apply_grad_processors(opt, gradprocs)


if __name__ == '__main__':
    args = get_args()
    
    DEPTH = args.depth
    nr_gpu = get_nr_gpu()
    TOTAL_BATCH_SIZE = int(args.batch)
    BATCH_SIZE = TOTAL_BATCH_SIZE // nr_gpu
    args.batch = BATCH_SIZE
    model = Model()
    if args.cam:
        args.batch = int(100)
        args.load = 'train_log/' + args.logdir + '/checkpoint'
        cam(model, args, gradcam=False)
        sys.exit()
    if args.gradcam:
        args.batch = int(100)
        args.load = 'train_log/' + args.logdir + '/checkpoint'
        cam(model, args, gradcam=True)
        sys.exit()

    logdir = 'train_log/' + args.logdir
    logger.set_logger_dir(logdir)
    config = get_config(model, args)
    if args.load:
        config.session_init = get_model_loader(args.load)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
