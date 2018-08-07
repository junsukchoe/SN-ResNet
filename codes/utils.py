#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: utils.py

# Written by Junsuk Choe <skykite@yonsei.ac.kr>
# Utility code for Wearkly-Supervised Object Localization (WSOL).


import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
import random
from abc import abstractmethod

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger

def get_data(train_or_test, option):
    isTrain = train_or_test == 'train'

    datadir = option.data
    ds = dataset.ILSVRC12(datadir, train_or_test, shuffle=isTrain)
    augmentors = fbresnet_augmentor(isTrain, option=option)
    augmentors.append(imgaug.ToUint8())
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(25, multiprocessing.cpu_count()))
    ds = BatchData(ds, int(option.batch), remainder=not isTrain)
    return ds

def get_config(model, option):
    dataset_train = get_data('train', option)
    dataset_val = get_data('val', option)

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]),
                every_k_epochs=2),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-3), (55, 1e-4), (75, 1e-5), (95, 1e-6)]),
        ],
        steps_per_epoch=option.steps,
        max_epoch=option.epoch,
    )

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08, crop_area_max=1.0,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, self.crop_area_max) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain, option):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        if option.GR:
            print ('Use GoogleNetResize Augmentation')
            augmentors = [
                GoogleNetResize(target_shape=option.final_size)
            ]
        else:
            print ('Do not use GR')
            augmentors = [
                    imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
                    imgaug.CenterCrop((option.final_size, option.final_size))
            ]
        
        basic = [
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                imgaug.Contrast((0.6, 1.4), clip=False),
                imgaug.Saturation(0.4, rgb=False),
                # rgb-bgr conversion for the constants copied from fb.resnet.torch
                imgaug.Lighting(0.1,
                                eigval=np.asarray(
                                    [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                eigvec=np.array(
                                    [[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]],
                                    dtype='float32')[::-1, ::-1]
                                )]),
            imgaug.Flip(horiz=True),
        ]

        augmentors.extend(basic)

    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((option.final_size, option.final_size)),
        ]
    return augmentors


def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)
        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image

        
def compute_loss_and_error(logits, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    loss = tf.reduce_mean(loss, name='xentropy-loss')

    def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
        with tf.name_scope('prediction_incorrect'):
            x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
        return tf.cast(x, tf.float32, name=name)

    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
    return loss