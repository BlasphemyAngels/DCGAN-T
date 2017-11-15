#########################################################################
# File Name: main.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-09-19 19:42:13
# Last modified: 2017-09-19 19:42:13
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

import tensorflow as tf
from model import DCGAN

flags = tf.app.flags
flags.DEFINE_integer("input_height", 96, "the height of the input image")
flags.DEFINE_integer("input_width", 96, "the width of the input image")
flags.DEFINE_float("beta1", 0.5, "the beta1 for adam")
flags.DEFINE_float("learning_rate", 0.0002, "the learning rate of the model")
flags.DEFINE_integer(
    "epoch", 1000, "the number of the epoches about the model")
flags.DEFINE_string("dataset_dir", "data", "The path of the dataset.")
flags.DEFINE_string("dataset_h_dir", "new",
                    "The new dir of the handled data")

FLAGS = flags.FLAGS


def main(_):
    #  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #  run_config = tf.ConfigProto(gpu_options=gpu_options)
    #  run_config.gpu_options.allow_growth = True
    #  with tf.Session(config=run_config) as sess:
    with tf.Session() as sess:
        dcgan = DCGAN(sess, FLAGS.input_height, FLAGS.input_width)
        dcgan.build_model()
        dcgan.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
