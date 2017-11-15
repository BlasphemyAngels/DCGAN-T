#########################################################################
# File Name: test_data.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-10-13 22:23:42
# Last modified: 2017-10-13 22:23:42
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

import tensorflow as tf
from read_data import read_and_decode
from read_data import get_one_batch

if __name__ == '__main__':
    img_batch = get_one_batch("./data/new/train.tfrecords", 64, 96, 96, 3)
    tf.global_variables_initializer()
    with tf.Session() as sess:
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            print(img_batch)
            val = sess.run([img_batch])
            print(val[0][0].shape)
