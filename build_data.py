#########################################################################
# File Name: build_data.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-09-15 22:19:10
# Last modified:2017-09-15 22:19:13
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

import os
import sys
from PIL import Image
import tensorflow as tf
from glob import glob

tf.flags.DEFINE_string("dataset_dir", "data", "The path of the dataset.")
tf.flags.DEFINE_string("dataset", "faces", "The name of the dataset.")
tf.flags.DEFINE_string("image_fname_pattern", "*.jpg",
                       "The pattern of the image name")
tf.flags.DEFINE_float("train_data_rate", 0.8,
                      "The rate of the train data process")
tf.flags.DEFINE_float("val_data_rate", 0.1,
                      "The rate of the validate data process")
tf.flags.DEFINE_float("test_data_rate", 0.1,
                      "The rate of the test data process")
tf.flags.DEFINE_string("dataset_h_dir", "new",
                       "The new dir of the handled data")

FLAGS = tf.flags.FLAGS


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def remove(dire):
    for f in os.listdir(dire):
        os.remove(os.path.join(dire, f))


def write(filename, image_names):
    writer = tf.python_io.TFRecordWriter(os.path.join(
        "./", FLAGS.dataset_dir, FLAGS.dataset_h_dir, filename))
    for image_name in image_names:
        image = Image.open(image_name, mode="r")
        example = tf.train.Example(features=tf.train.Features(
            feature={"image": bytes_feature(image.tobytes())}))
        writer.write(example.SerializeToString())
    writer.close()


def main(_):
    dataset = glob(os.path.join("./", FLAGS.dataset_dir,
                                FLAGS.dataset, FLAGS.image_fname_pattern))
    if len(dataset) == 0:
        print("The dataset doesn't exists!")
        sys.exit(1)
    train_num = int(len(dataset) * FLAGS.train_data_rate)
    val_num = int(len(dataset) * FLAGS.val_data_rate)
    test_num = int(len(dataset) * FLAGS.test_data_rate)

    if not os.path.exists(
            os.path.join("./", FLAGS.dataset_dir, FLAGS.dataset_h_dir)):
        os.mkdir(os.path.join("./", FLAGS.dataset_dir, FLAGS.dataset_h_dir))
    remove(os.path.join("./", FLAGS.dataset_dir, FLAGS.dataset_h_dir))

    train = dataset[:train_num]
    validate = dataset[train_num:train_num + val_num]
    test = dataset[train_num + val_num:]

    write("train.tfrecords", train)
    write("validate.tfrecords", validate)
    write("test.tfrecords", test)


if __name__ == '__main__':
    tf.app.run()
