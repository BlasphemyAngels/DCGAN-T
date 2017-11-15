#########################################################################
# File Name: read_data.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-09-25 22:00:06
# Last modified: 2017-09-25 22:00:06
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

import tensorflow as tf


def read_and_decode(filename, height, width, c_dim, epochs):
    f_queue = tf.train.string_input_producer([filename], num_epochs=epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(f_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
        })
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [height, width, c_dim])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img


def get_one_batch(filename, batch_size, height, width, c_dim, epochs):
    img = read_and_decode(filename, height, width, c_dim, epochs)

    return tf.train.shuffle_batch(
        [img], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
