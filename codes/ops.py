# -*- coding: utf-8 -*-
# File: ops.py

# Ops code for SN-VGG (GAP).

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import *

def GroupNorm(x, group, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')

def Spec_FullyConnected(name, 
            input_, output_dim, use_bias=True, 
            bias_start=0., stddev=0.01, sn=True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable("W",
                    [shape[1], output_dim], tf.float32,
                    tf.random_normal_initializer(stddev=stddev))
        if use_bias:
            bias = tf.get_variable("b", [output_dim],
                                    initializer=tf.constant_initializer(bias_start))
            if sn==True:
                mul = tf.matmul(input_, spectral_norm(w)) + bias
            else:
                mul = tf.matmul(input_, w) + bias
        else:
            if sn==True:
                mul = tf.matmul(input_, spectral_norm(w))
            else:
                mul = tf.matmul(input_, w)
        return mul

def Spec_Conv2D(name, 
            input_, output_dim, kernel_shape=3, stride=1, 
            use_bias=True, stddev=0.02, sn=True, padding='SAME'):
    if sn:
        print('Spectral Normalization Activated.')
    with tf.variable_scope(name):
        w = tf.get_variable('W', 
                [kernel_shape, kernel_shape, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        if sn==True:
            conv = tf.nn.conv2d(input_, 
                spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, 
                w, strides=[1, stride, stride, 1], padding=padding)

        biases = tf.get_variable('b', 
            [output_dim], initializer=tf.constant_initializer(0.0))
        
        if use_bias == True:
            conv = tf.nn.bias_add(conv, biases)

        return conv

def spectral_norm(input_):
  """Performs Spectral Normalization on a weight tensor."""
  if len(input_.shape) < 2:
    raise ValueError(
        "Spectral norm can only be applied to multi-dimensional tensors")

  # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
  # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
  # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
  # (KH * KW * C_in, C_out), and similarly for other layers that put output
  # channels as last dimension.
  # n.b. this means that w here is equivalent to w.T in the paper.
  w = tf.reshape(input_, (-1, input_.shape[-1]))

  # Persisted approximation of first left singular vector of matrix `w`.

  u_var = tf.get_variable(
      input_.name.replace(":", "") + "/u_var",
      shape=(w.shape[0], 1),
      dtype=w.dtype,
      initializer=tf.random_normal_initializer(),
      trainable=False)
  u = u_var

  # Use power iteration method to approximate spectral norm.
  # The authors suggest that "one round of power iteration was sufficient in the
  # actual experiment to achieve satisfactory performance". According to
  # observation, the spectral norm become very accurate after ~20 steps.

  power_iteration_rounds = 1
  for _ in range(power_iteration_rounds):
    # `v` approximates the first right singular vector of matrix `w`.
    v = tf.nn.l2_normalize(
        tf.matmul(tf.transpose(w), u), axis=None, epsilon=1e-12)
    u = tf.nn.l2_normalize(tf.matmul(w, v), axis=None, epsilon=1e-12)

  # Update persisted approximation.
  with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
    u = tf.identity(u)

  # The authors of SN-GAN chose to stop gradient propagating through u and v.
  # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
  # seem to hinder either so it's kept in order to be a faithful implementation.
  u = tf.stop_gradient(u)
  v = tf.stop_gradient(v)

  # Largest singular value of `w`.
  norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
  norm_value.shape.assert_is_fully_defined()
  norm_value.shape.assert_is_compatible_with([1, 1])

  w_normalized = w / norm_value

  # Unflatten normalized weights to match the unnormalized tensor.
  w_tensor_normalized = tf.reshape(w_normalized, input_.shape)
  return w_tensor_normalized