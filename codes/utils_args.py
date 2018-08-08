#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: utils_args.py

# Parsing arguments code for SN-VGG (GAP).


import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod
import argparse
import os

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger

def get_args():
    parser = argparse.ArgumentParser()
    
    # Common - Required
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--logdir', help='log directory name')

    # Common - Default
    parser.add_argument('--epoch', help='max epoch', type=int, default=105)
    parser.add_argument('--steps', help='steps_per_epoch', default=5000, type=int)
    parser.add_argument('--final-size', type=int, default=224)

    # Training
    parser.add_argument('--batch', help='enter batch size to use', default=256)
    parser.add_argument('--sn', action='store_true')

    # Data Augmentation
    parser.add_argument('--GR', action='store_true')
    
    # Test
    parser.add_argument('--load', help='load model')
    parser.add_argument('--eval', help='test', action='store_true')


    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args
