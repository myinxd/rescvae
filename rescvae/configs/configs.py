# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
# MIT liscence

"""Configurations for the DNNAE network."""

import tensorflow as tf

class config_mnist(object):
    rs = 28
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,rs,rs,1), name='x_in')
    outputs = tf.placeholder(dtype=tf.float32, shape=(None,rs,rs,1), name='x_out')
    numclass = 10
    conditions = tf.placeholder(
        dtype=tf.float32, shape=(None,numclass), name='conditions')
    conditions_input = tf.placeholder(
        dtype=tf.float32, shape=(None,rs,rs,numclass))
    cflag = True
    z_length = 128
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    blocks = [
        [(8, 4, 2)],
        [(16, 8, 2)]]
    loss_mse = True
    epocheps = 1e-5

class config_train(object):
    valrate = 0.3
    batchsize = 64
    epochs = 50
    lr_init = 0.0001
    decay_rate = 0.9
    keep_prob = 0.5
    print_step = 1


class config_rg(object):
    rs = 40
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,rs,rs,1), name='x_in')
    outputs = tf.placeholder(dtype=tf.float32, shape=(None,rs,rs,1), name='x_out')
    numclass = 2
    conditions = tf.placeholder(
        dtype=tf.float32, shape=(None,numclass), name='conditions')
    conditions_input = tf.placeholder(
        dtype=tf.float32, shape=(None,rs,rs,numclass))
    cflag = True
    z_length = 32
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    blocks = [
        [(16, 8, 2)],
        [(32, 16, 2)],
        [(64, 32, 2)]]
    loss_mse = False
    epocheps = 1e-5

class config_train_rg(object):
    valrate = 0.2
    batchsize = 100
    epochs = 200
    lr_init = 0.0001
    decay_rate = 0.9
    keep_prob = 0.5
    print_step = 1