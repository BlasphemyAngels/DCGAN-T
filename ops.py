#########################################################################
# File Name: ops.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-09-16 23:01:11
# Last modified:2017-09-16 23:01:25
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

import math
import tensorflow as tf


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [input_.shape[1], output_size], tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev))

        bias = tf.get_variable(
            "Bais", [output_size], tf.float32,
            initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias


class batch_norm(object):
    def __init__(self, decay=0.9, epsilon=1e-5, name="batch_norm"):
        self.decay = decay
        self.epsilon = epsilon
        self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
            x, decay=self.decay, scale=True, epsilon=self.epsilon,
            updates_collections=None, is_training=train, scope=self.name)


def deconv2d(input_, output_size, k_h=5, k_w=5, s_h=2,
             s_w=2, name="deconv2d", stddev=0.02):
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_h, k_w, output_size.shape[-1]], dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = rf.nn.conv2d_transpose(
            input_, w, output_size, strides=[1, s_h, s_w, 1], padding="SAME")
        bias = tf.get_variable(
            "bias", [output_size[-1]], tf.float32,
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, bias)
        return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def conv2d(input_, output_size, k_h=5, k_w=5, s_h=2, s_w=2,
           stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_h, k_w, input_.shape[-1], output_size], dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            input_, w, strides=[1, d_h, d_w, 1], padding="SAME", name=name)
        bias = tf.get_variable(
            "bias", [output_size], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        conv = tf.bias_add(conv, bias)

        return conv
