# -*- coding: utf-8 -*-
# File: models_vgg.py

# Model architecture code for SN-VGG (GAP).

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import *
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected)

from tensorpack.tfutils.tower import get_current_tower_context
from ops import *

def vgg_gap(image, option):

    l = tf.nn.relu(Spec_Conv2D('conv1_1', image, 64, sn=option.sn), name='conv1_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv1_2', l, 64, sn=option.sn), name='conv1_2_relu')
    l = tf.nn.max_pool(l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    l = tf.nn.relu(Spec_Conv2D('conv2_1', l, 128, sn=option.sn), name='conv2_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv2_2', l, 128, sn=option.sn), name='conv2_2_relu')
    l = tf.nn.max_pool(l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')

    l = tf.nn.relu(Spec_Conv2D('conv3_1', l, 256, sn=option.sn), name='conv3_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv3_2', l, 256, sn=option.sn), name='conv3_2_relu')
    l = tf.nn.relu(Spec_Conv2D('conv3_3', l, 256, sn=option.sn), name='conv3_3_relu')
    l = tf.nn.max_pool(l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')

    l = tf.nn.relu(Spec_Conv2D('conv4_1', l, 512, sn=option.sn), name='conv4_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv4_2', l, 512, sn=option.sn), name='conv4_2_relu')
    l = tf.nn.relu(Spec_Conv2D('conv4_3', l, 512, sn=option.sn), name='conv4_3_relu')
    l = tf.nn.max_pool(l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')

    l = tf.nn.relu(Spec_Conv2D('conv5_1', l, 512, sn=option.sn), name='conv5_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv5_2', l, 512, sn=option.sn), name='conv5_2_relu')
    l = tf.nn.relu(Spec_Conv2D('conv5_3', l, 512, sn=option.sn), name='conv5_3_relu')
    l = tf.nn.relu(Spec_Conv2D('conv5_4', l, 1024, sn=option.sn), name='conv5_4_relu')

    l = tf.reduce_mean(l, axis=[1,2]) # GAP
    logits = Spec_FullyConnected('linear', l, 1000, sn=option.sn)

    return logits

def vgg_gap_tiny(image, option):

    l = tf.nn.relu(Spec_Conv2D('conv1_1', image, 64, sn=option.sn), name='conv1_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv1_2', l, 64, sn=option.sn), name='conv1_2_relu')
    l = tf.nn.relu(Spec_Conv2D('conv1_3', l, 64, sn=option.sn), name='conv1_3_relu')
    l = tf.nn.max_pool(l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    l = tf.nn.relu(Spec_Conv2D('conv2_1', l, 128, sn=option.sn), name='conv2_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv2_2', l, 128, sn=option.sn), name='conv2_2_relu')
    l = tf.nn.relu(Spec_Conv2D('conv2_3', l, 128, sn=option.sn), name='conv2_3_relu')
    l = tf.nn.max_pool(l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')

    l = tf.nn.relu(Spec_Conv2D('conv3_1', l, 256, sn=option.sn), name='conv3_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv3_2', l, 256, sn=option.sn), name='conv3_2_relu')
    l = tf.nn.relu(Spec_Conv2D('conv3_3', l, 256, sn=option.sn), name='conv3_3_relu')
    l = tf.nn.relu(Spec_Conv2D('conv3_4', l, 256, sn=option.sn), name='conv3_4_relu')
    l = tf.nn.max_pool(l, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')

    l = tf.nn.relu(Spec_Conv2D('conv4_1', l, 512, sn=option.sn), name='conv4_1_relu')
    l = tf.nn.relu(Spec_Conv2D('conv4_2', l, 512, sn=option.sn), name='conv4_2_relu')
    l = tf.nn.relu(Spec_Conv2D('conv4_3', l, 512, sn=option.sn), name='conv4_3_relu')
    l = tf.nn.relu(Spec_Conv2D('conv4_4', l, 1024, sn=option.sn), name='conv4_4_relu')

    l = tf.reduce_mean(l, axis=[1,2]) # GAP
    logits = Spec_FullyConnected('linear', l, 200, sn=option.sn)

    return logits