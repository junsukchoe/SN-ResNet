# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from ops import *


def resnet_shortcut(l, n_out, stride, sn):
    n_in = l.get_shape().as_list()[3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Spec_Conv2D('convshortcut', l, n_out, 1, stride, sn)
    else:
        return l


def apply_preactivation(l, preact, isTrain):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = batch_norm_resnet(l, isTrain, 'prebn')
        l = tf.nn.relu(l, 'preact')
    else:
        shortcut = l
    return l, shortcut


def preresnet_basicblock(l, ch_out, stride, preact, isTrain, sn):
    l, shortcut = apply_preactivation(l, preact, isTrain)
    l = Spec_Conv2D('conv1', l, ch_out, 3, stride, sn)
    l = batch_norm_resnet(l, isTrain, 'bn1')
    l = tf.nn.relu(l, 'relu1')
    l = Spec_Conv2D('conv2', l, ch_out, 3, 1, sn)
    return l + resnet_shortcut(shortcut, ch_out, stride, sn)


def preresnet_group(name, l, features, count, stride, isTrain, sn):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = preresnet_basicblock(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu', isTrain, sn)
        # end of each group need an extra activation
        l = batch_norm_resnet(l, isTrain, 'bnlast')
        l = tf.nn.relu(l, 'relulast')
    return l

