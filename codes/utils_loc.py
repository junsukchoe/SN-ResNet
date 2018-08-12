#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: utils_loc.py

# Written by Junsuk Choe <skykite@yonsei.ac.kr>
# Function Code for visualizing heatmap of learned CNNs.
# Including CAM and Grad-CAM.

import cv2
import sys
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils import viz

from utils import get_data

def cam(model, option, gradcam=False):
    model_file = option.load
    data_dir = option.data
    valnum = option.valnum

    ds = get_data('val', option)

    if gradcam:
        pred_config = PredictConfig(
            model=model,
            session_init=get_model_loader(model_file),
            input_names=['input', 'label','xa','ya','xb','yb'],
            output_names=['wrong-top1', 'actmap', 'grad'],
            return_input=True
        )
    else:
        pred_config = PredictConfig(
            model=model,
            session_init=get_model_loader(model_file),
            input_names=['input', 'label','xa','ya','xb','yb'],
            output_names=['wrong-top1', 'actmap', 'linearnew/W'],
            return_input=True
        )
    
    meta = dataset.tinyImagenetHaSMeta(dir=option.data).get_synset_words_1000(option.dataname)
    meta_labels = dataset.tinyImagenetHaSMeta(dir=option.data).get_synset_1000(option.dataname)

    pred = SimpleDatasetPredictor(pred_config, ds)
    
    cnt = 0
    cnt_false = 0
    hit_known = 0
    hit_top1 = 0
    for inp, outp in pred.get_result():
        images, labels, gxa, gya, gxb, gyb = inp

        if gradcam:
            wrongs, convmaps, grads_val = outp
            batch = wrongs.shape[0]
            NUMBER,CHANNEL,HEIGHT,WIDTH = np.shape(convmaps)  
            grads_val = np.transpose(grads_val, [0,2,3,1])
            W = np.mean(grads_val, axis=(1,2))
        else:
            wrongs, convmaps, W = outp
            batch = wrongs.shape[0]     
            NUMBER,CHANNEL,HEIGHT,WIDTH = np.shape(convmaps)       

        for i in range(batch):
            # generating heatmap
            #if wrongs[i]:
            #    cnt += 1
            #    continue
            if gradcam:
                weight = W[i]   # c x 1
            else:
                weight = W[:, [labels[i]]].T
            convmap = convmaps[i, :, :, :]  # c x h x w
            mergedmap = np.matmul(weight, convmap.reshape((CHANNEL, -1))).reshape(HEIGHT, WIDTH)
            #mergedmap = np.maximum(mergedmap, 0)
            if gradcam:
                mergedmap = np.maximum(mergedmap, 0)
            mergedmap = cv2.resize(mergedmap, (option.final_size, option.final_size))
            heatmap = viz.intensity_to_rgb(mergedmap, normalize=True)
            blend = images[i] * 0.5 + heatmap * 0.5
            
            # initialization for boundary box
            bbox_img = images[i]
            bbox_img = bbox_img.astype('uint8')
            heatmap = heatmap.astype('uint8')
            blend = blend.astype('uint8')
            
            # thresholding heatmap
            gray_heatmap = cv2.cvtColor(heatmap,cv2.COLOR_RGB2GRAY)
            th_value = np.max(gray_heatmap)*0.2
            #th_value = 0
            _, thred_gray_heatmap = cv2.threshold(gray_heatmap,int(th_value),255,cv2.THRESH_TOZERO)
            _, contours, _ = cv2.findContours(thred_gray_heatmap, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            # calculate bbox coordinates
            rect = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                rect.append([x,y,w,h])
            
            x,y,w,h = large_rect(rect)
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(bbox_img, (gxa[i], gya[i]), (gxb[i], gyb[i]), (0, 0, 255), 2)
            
            # calculate IOU
            estimated_box = [x,y,x+w,y+h]
            gt_box = [gxa[i],gya[i],gxb[i],gyb[i]]
            IOU_ = bb_IOU(estimated_box, gt_box)
            
            if IOU_ > 0.5:
                hit_known = hit_known + 1
                
            if IOU_ > 0.5 and not wrongs[i]:
                hit_top1 = hit_top1 + 1
                
            if wrongs[i]:
                cnt_false += 1
            
            concat = np.concatenate((bbox_img, heatmap, blend), axis=1)

            classname = meta[meta_labels[labels[i]]].split(',')[0]
            
            dirname = 'result/{}'.format(option.logdir)
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            
            if cnt < 500:
                cv2.imwrite('result/{}/cam{}-{}.jpg'.format(option.logdir, cnt, classname), concat)
                
            cnt += 1
            if cnt == valnum:
                fname = 'result/{}/Loc.txt'.format(option.logdir)
                f = open(fname, 'w')
                acc_known = hit_known/cnt
                acc_top1 = hit_top1/cnt
                top1_acc = 1 - cnt_false / (cnt)
                line = 'GT-known Loc: {}\nTop-1 Loc: {}\nTop-1 Acc: {}'.format(acc_known,acc_top1,top1_acc)
                f.write(line)
                f.close()
                return


def bb_IOU(boxA, boxB):
    # This is from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    for i in range(len(rect)):
        area = rect[i][2]*rect[i][3]
        if large_area < area:
            large_area = area
            target = i
        
    x = rect[target][0]
    y = rect[target][1]
    w = rect[target][2]
    h = rect[target][3]
    
    return x, y, w, h