#########################################################################
# File Name: model.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-09-16 22:47:50
# Last modified:2017-09-16 22:47:55
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

import os
import time
import tensorflow as tf
import numpy as np
from ops import *
from read_data import get_one_batch
from six.moves import xrange


class DCGAN(object):
    def __init__(self, sess, output_height, output_width,
                 gf_dim=64, df_dim=64, z_dim=100, c_dim=3, batch_size=64):
        self._output_height = output_height
        self._output_width = output_width
        self._gf_dim = gf_dim
        self._df_dim = df_dim
        self._z_dim = z_dim
        self._c_dim = c_dim
        self.batch_size = batch_size
        self.sess = sess

    def generator(self, z):
        s_h, s_w = self._output_height, self._output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        self.z_ = linear(z, self._gf_dim * 8 * s_h16 * s_h16, "g_h0_lin")

        self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self._gf_dim * 8])
        h0 = tf.nn.relu(batch_norm(name="g_bn0")(self.h0))

        self.h1 = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self._gf_dim * 4], name="g_h1")
        h1 = tf.nn.relu(batch_norm(name="g_bn1")(self.h1))

        self.h2 = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self._gf_dim * 2], name="g_h2")
        h2 = tf.nn.relu(batch_norm(name="g_bn2")(self.h2))

        self.h3 = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self._gf_dim], name="g_h3")
        h3 = tf.nn.relu(batch_norm(name="g_bn3")(self.h3))

        h4 = deconv2d(
            h3, [self.batch_size, s_h, s_w, self._c_dim], name="g_h4")

        return tf.nn.tanh(h4)

    def discriminator(self, image, name="discriminator", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self._df_dim, name="d_h0_conv"))

            h1 = lrelu(batch_norm(name="d_bn1")(
                conv2d(h0, self._df_dim * 2, name="d_h1_conv")))

            h2 = lrelu(batch_norm(name="d_bn2")(
                conv2d(h1, self._df_dim * 4, name="d_h2_conv")))

            h3 = lrelu(batch_norm(name="d_bn3")(
                conv2d(h2, self._df_dim * 8, name="d_h3_conv")))

            h4 = linear(tf.reshape(
                h3, [self.batch_size, -1]), 1, name="d_h4_lin")

            return tf.nn.sigmoid(h4), h4

    def build_model(self):
        image_dims = [self._output_height, self._output_width, self._c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name="real_images")

        self.z = tf.placeholder(dtype=tf.float32, shape=[
                                None, self._z_dim], name="z")
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)

        self.g_sum = tf.summary.image("g", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_loss_fake_sum = tf.summary.scalar(
            "d_loss_fake", self.d_loss_fake)
        self.d_loss_real_sum = tf.summary.scalar(
            "d_loss_real", self.d_loss_real)

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if "d_" in var.name]
        self.g_vars = [var for var in t_vars if "g_" in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optm = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate, beta1=config.beta1).minimize(
            loss=self.d_loss, var_list=self.d_vars)
        g_optm = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate, beta1=config.beta1).minimize(
            loss=self.g_loss, var_list=self.g_vars)

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.g_sum,
             self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./log", self.sess.graph)
        counter = 1
        img_batch = get_one_batch(
            os.path.join(
                config.dataset_dir,
                config.dataset_h_dir,
                "train.tfrecords"),
            self.batch_size,
            self._output_height, self._output_width, self._c_dim,
            config.epoch)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        epoch = 0
        start_time = time.time()
        try:
            while not coord.should_stop():
                batch_z = np.random.uniform(
                    -1, 1, [self.batch_size, self._z_dim])
                img = self.sess.run(img_batch)
                _, summary_str = self.sess.run(
                    [d_optm, self.d_sum],
                    feed_dict={self.inputs: img, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                _, summary_str = self.sess.run(
                    [g_optm, self.g_sum],
                    feed_dict={self.z: batch_z})

                self.writer.add_summary(summary_str, counter)
                _, summary_str = self.sess.run(
                    [g_optm, self.g_sum],
                    feed_dict={self.z: batch_z})

                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: img})
                errG = self.g_loss.eval({self.z: batch_z})
                epoch += 1
                print("---%.8f, %.8f" % (errD_fake, errG))
                if epoch % config.epoch == 0:
                    print("Epoch [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch // config.epoch,
                             time.time() - start_time,
                             errD_real + errD_fake,
                             errG))
                    start_time = time.time()
        except tf.errors.OutOfRangeError:
            print("Epoch = %d" % epoch)
            print("Epochs Complete")
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)
