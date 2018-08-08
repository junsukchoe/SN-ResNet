#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: main.py

# This code is mainly borrowed from the official example codes of tensorpack library.
# https://github.com/ppwwyyxx/tensorpack/tree/master/examples

# Revised by Junsuk Choe <skykite@yonsei.ac.kr>
# SN-VGG (GAP)


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
from tensorpack.tfutils import optimizer, gradproc, get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils import viz
from tensorpack.tfutils.tower import get_current_tower_context

from utils import *
from utils_args import *
from models_vgg import *

class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.uint8, [None, args.final_size, args.final_size, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image_preprocess(image, bgr=True) # image = (image - image_mean) / image_std

        if args.sn:
            sn = True
        else:
            sn= False
        
        logits = vgg_gap(image, sn)
        loss = compute_loss_and_error(logits, label)
        wd_cost = regularize_cost('.*/W', l2_regularizer(5e-4), name='l2_regularize_loss')
        
        add_moving_summary(loss, wd_cost)

        return tf.add_n([loss, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        return opt

def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))

if __name__ == '__main__':

    # For Multi-GPU
    args = get_args()
    nr_gpu = get_nr_gpu()
    TOTAL_BATCH_SIZE = int(args.batch)
    BATCH_SIZE = TOTAL_BATCH_SIZE // nr_gpu
    args.batch = BATCH_SIZE

    # Model init
    model = Model()

    if args.eval:
        args.batch = 128
        args.load = 'train_log/' + args.logdir + '/checkpoint'
        ds = get_data('val', args)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
        sys.exit()

    logdir = 'train_log/' + args.logdir
    logger.set_logger_dir(logdir)
    config = get_config(model, args)
    if args.load:
        config.session_init = get_model_loader(args.load)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
