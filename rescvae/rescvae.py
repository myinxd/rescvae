# Copyright (C) 2018 zhixian MA <zx@mazhixian.me>
# MIT liscence

'''
Conditional variational autoencoder, redisual case.

Characters
==========
1. selectable loss function
2. configurable with configurations
3. extendable to general vae
4. early stoppable by setting the epoch eps.
'''

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm

from .utils import utils
from .block import Block


class rescvae():
    """
    The conditional variational autoendocer class, residual case.

    inputs
    ======
    configs: object class
        configurations for the residual convolutional network
        configs.inputs: placeholder of the network's input,
                       whose shape is (None, rows, cols) or (rows*cols).
        configs.conditions: placeholder of the conditions w.r.t. the samples,
                            whose shape is (None, rows, cols, numclass) or
                            (rows*cols, numclass).
        configs.output: placeholder of the network's output of the
                       same shape as the inputs.
        configs.blocks: a list of the blocks to form the residual network
        # configs.actfun: a list of activation functions for the layers
        # configs.batchflag: a list of batchly normalization flags for the layers
        configs.cflag: a flag of whether it is cvae or vae, default as False
        configs.z_length: length of the latent z
        configs.keep_rate: keep rate for training the network
        configs.init_lr: initialized learning rate
        configs.decay_rate: initialized decay rate for the learning rate adjustment
        configs.numclass: number of classes to be classified
        configs.mseflag: a flag of whether to use the mse to form the loss function

    methods
    =======
    rescvae_build: build the network
    rescvae_train: train the network
    rescvae_test: test the network
    get_batch: get training batches
    """

    def __init__(self, configs):
        """Initializer"""
        self.inputs = configs.inputs
        self.outputs = configs.outputs
        self.cflag = configs.cflag
        self.keep_prob = configs.keep_prob
        self.numclass = configs.numclass
        self.z_length = configs.z_length
        self.rs = configs.rs
        self.epocheps = configs.epocheps
        try:
            self.mseflag = configs.loss_mse
        except:
            self.mseflag = True
        # get input shape
        self.input_shape = self.inputs.get_shape().as_list()
        if len(self.input_shape) == 2:
            self.net = tf.reshape(
                self.inputs, [-1, self.rs, self.rs, 1])
            self.output_flatten = tf.reshape(
                self.outputs, [-1, self.rs*self.rs])
        elif len(self.input_shape) == 4:
            self.net = self.inputs
            self.input_flatten = tf.reshape(
                self.inputs, [-1, self.rs*self.rs])
            self.output_flatten = tf.reshape(
                self.outputs, [-1, self.rs*self.rs])
        else:
            print("Something wrong with the input shape, please check.")
        # condition flag
        if self.cflag:
            # conditional VAE
            self.blocks_en = configs.blocks
            self.odd_flags = []
            self.conditions = configs.conditions
            self.conditions_input = configs.conditions_input
            self.net = tf.concat(
                [self.net, self.conditions_input], axis=-1)
            self.z = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='z')
            self.mu = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_avg')
            self.sigma = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_std')
        else:
            # VAE
            self.blocks_en = configs.blocks
            self.odd_flags = []
            self.z = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='z')
            self.mu = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_avg')
            self.sigma = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_std')
        # batch normalization
        self.is_training = tf.placeholder(tf.bool, name='is_training')


    #====================
    # Build the network #
    #====================
    def vae_build(self):
        """Build the general vae network"""
        self.netprinter = []
        self.netprinter.append(["Input layer",
                                self.net.get_shape().as_list()])
        # The encoder part
        with tf.name_scope("vae"):
            with tf.name_scope("block_en"):
                for i, block in enumerate(self.blocks_en):
                    block_params = utils.get_block_en(block, is_training=self.is_training)
                    print(block_params)
                    block_obj = Block(
                        inputs=self.net,
                        block_params=block_params,
                        is_training=self.is_training,
                        encode_flag=True,
                        scope='block_en_'+str(i)
                    )
                    self.netprinter.append(["Block_en_"+str(i),
                                            self.net.get_shape().as_list()])
                    self.net, odd_flag = block_obj.get_block()
                    self.odd_flags.append(odd_flag)
                # get shape of the last block
                encode_last_block_shape = self.net.get_shape()
            # flatten
            with tf.name_scope('flatten_en'):
                self.net = layers.flatten(self.net)
                self.netprinter.append(["En_flatten", self.net.get_shape().as_list()])
                self.flatten_length = int(self.net.get_shape()[-1])

            # output
            with tf.name_scope("en_output"):
                self.mu = fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                self.sigma=fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                # softplus
                self.sigma=1e-6 + tf.nn.softplus(self.sigma)
                self.netprinter.append(["En_mu", self.mu.get_shape().as_list()])
                self.netprinter.append(["En_sigma", self.sigma.get_shape().as_list()])

            # Reparameterization to obtain z
            with tf.name_scope("reparameterization"):
                self.epsilon = tf.random_normal(tf.shape(self.mu))
                self.z = self.mu + tf.multiply(self.epsilon, self.sigma)
                self.netprinter.append(["z", self.z.get_shape().as_list()])

            # The decoder subnet
            self.net = self.z
            # deflatten
            with tf.name_scope("flatten_de"):
                self.net = fully_connected(
                    inputs=self.net,
                    num_outputs=self.flatten_length,
                    activation_fn=tf.nn.relu
                )
                self.net = tf.nn.dropout(
                    x=self.net,
                    keep_prob=self.keep_prob,
                    name="drop_de")
                self.netprinter.append(["De_flatten", self.net.get_shape().as_list()])
            # flatten to convolve
            with tf.name_scope("flatten_to_conv"):
                self.net = tf.reshape(
                    self.net,
                    [-1,
                     int(encode_last_block_shape[1]),
                     int(encode_last_block_shape[2]),
                     int(encode_last_block_shape[3])]
                )
                self.netprinter.append(["De_flatten_to_conv",
                                        self.net.get_shape().as_list()])
            # the decoder
            # block reverse
            block_list, block_stride = utils.get_block_list(
                self.blocks_en, input_depth=1)
            self.blocks_de = utils.get_block_de_params(
                block_list, block_stride)
            blocks_de_tmp = self.blocks_de.copy()
            self.odd_flags_de = utils.get_odd_flags(
                self.odd_flags, blocks=blocks_de_tmp)
            with tf.name_scope("block_de"):
                for i, block in enumerate(self.blocks_de):
                    if i == len(self.blocks_de):
                        block_params = utils.get_block_de(
                            block, is_training=self.is_training, outflag=True)
                    else:
                        block_params = utils.get_block_de(
                            block, is_training=self.is_training, outflag=False)
                    print(block_params)
                    block_obj = Block(
                        inputs=self.net,
                        block_params=block_params,
                        is_training=self.is_training,
                        encode_flag=False,
                        scope="block_de_"+str(i),
                        odd_flags=self.odd_flags_de[i]
                    )
                    self.net = block_obj.get_block()
                    self.netprinter.append(["Block_de_"+str(i),
                                           self.net.get_shape().as_list()])
                # Output
                self.netprinter.append(
                    ["Output layer", self.net.get_shape().as_list()])
                self.outputs_de = self.net


    def cvae_build(self):
        """Build the general cvae network"""
        self.netprinter = []
        self.netprinter.append(["Input layer",
                                self.net.get_shape().as_list()])
        # The encoder part
        with tf.name_scope("cvae"):
            with tf.name_scope("block_en"):
                for i, block in enumerate(self.blocks_en):
                    block_params = utils.get_block_en(block, is_training=self.is_training)
                    print(block_params)
                    block_obj = Block(
                        inputs=self.net,
                        block_params=block_params,
                        is_training=self.is_training,
                        encode_flag=True,
                        scope='block_en_'+str(i),
                    )
                    self.netprinter.append(["Block_en_"+str(i),
                                            self.net.get_shape().as_list()])
                    self.net, odd_flag = block_obj.get_block()
                    self.odd_flags.append(odd_flag)
                # get shape of the last block
                encode_last_block_shape = self.net.get_shape()
            # flatten
            with tf.name_scope('flatten_en'):
                self.net = layers.flatten(self.net)
                self.netprinter.append(["En_flatten", self.net.get_shape().as_list()])
                self.flatten_length = int(self.net.get_shape()[-1])

            # output
            with tf.name_scope("en_output"):
                self.mu = fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                self.sigma=fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                # softplus
                self.sigma=1e-6 + tf.nn.softplus(self.sigma)
                self.netprinter.append(["En_mu", self.mu.get_shape().as_list()])
                self.netprinter.append(["En_sigma", self.sigma.get_shape().as_list()])

            # Reparameterization to obtain z
            with tf.name_scope("reparameterization"):
                self.epsilon = tf.random_normal(tf.shape(self.mu))
                self.z = self.mu + tf.multiply(self.epsilon, self.sigma)
                self.netprinter.append(["z", self.z.get_shape().as_list()])

            # The decoder subnet
            self.net = tf.concat(
                [self.z, self.conditions], axis=1)
            # deflatten
            with tf.name_scope("flatten_de"):
                self.net = fully_connected(
                    inputs=self.net,
                    num_outputs=self.flatten_length,
                    activation_fn=tf.nn.relu
                )
                self.net = tf.nn.dropout(
                    x=self.net,
                    keep_prob=self.keep_prob,
                    name="drop_de")
                self.netprinter.append(["De_flatten", self.net.get_shape().as_list()])
            # flatten to convolve
            with tf.name_scope("flatten_to_conv"):
                self.net = tf.reshape(
                    self.net,
                    [-1,
                     int(encode_last_block_shape[1]),
                     int(encode_last_block_shape[2]),
                     int(encode_last_block_shape[3])]
                )
                self.netprinter.append(["De_flatten_to_conv",
                                        self.net.get_shape().as_list()])
            # the decoder
            # block reverse
            block_list, block_stride = utils.get_block_list(
                self.blocks_en, input_depth=1)
            self.blocks_de = utils.get_block_de_params(
                block_list, block_stride)
            blocks_de_tmp = self.blocks_de.copy()
            self.odd_flags_de = utils.get_odd_flags(
                self.odd_flags, blocks=blocks_de_tmp)
            with tf.name_scope("block_de"):
                for i, block in enumerate(self.blocks_de):
                    if i == len(self.blocks_de)-1:
                        block_params = utils.get_block_de(
                            block, is_training=self.is_training, outflag=True)
                    else:
                        block_params = utils.get_block_de(
                            block, is_training=self.is_training, outflag=False)
                    print(block_params)
                    block_obj = Block(
                        inputs=self.net,
                        block_params=block_params,
                        is_training=self.is_training,
                        encode_flag=False,
                        scope="block_de_"+str(i),
                        odd_flags=self.odd_flags_de[i]
                    )
                    self.net = block_obj.get_block()
                    self.netprinter.append(["Block_de_"+str(i),
                                           self.net.get_shape().as_list()])
                # Output
                self.netprinter.append(
                    ["Output layer", self.net.get_shape().as_list()])
                # Reshape the output layer
                self.output_de = tf.nn.sigmoid(self.net)
            with tf.name_scope('flatten_en'):
                self.output_flatten_de = layers.flatten(self.output_de)
                self.netprinter.append(["Flatten", self.net.get_shape().as_list()])


    def net_print(self):
        """Print the network"""
        print("Layer ID    Layer type    Layer shape")
        for i, l in enumerate(self.netprinter):
            print(i, l[0], l[1])


    def get_loss(self):
        """Get loss function
        # https://blog.csdn.net/ljhandlwt/article/details/77334450
        """
        with tf.name_scope("loss"):
            eps = 1e-6
            self.output_clip = tf.clip_by_value(self.output_flatten_de, eps, 1.0-eps)
            self.loss_ce = - tf.reduce_sum(
                self.output_flatten * tf.log(eps+self.output_clip) +
                (1.0 - self.output_flatten) * tf.log(eps+1.0 - self.output_clip), 1)
            self.loss_mse = tf.reduce_sum(
                tf.squared_difference(self.output_flatten_de, self.output_flatten), 1)
            if not self.mseflag:
                self.loss_recon = self.loss_ce
            else:
                self.loss_recon = self.loss_mse
            # latend loss
            self.loss_latent = 0.5 * tf.reduce_sum(
                tf.square(self.mu) + tf.square(self.sigma) -
                tf.log(1e-8 + tf.square(self.sigma)) - 1, 1)
            # combine
            self.loss = tf.reduce_mean(self.loss_recon + self.loss_latent)


    def get_opt(self):
        """Training optimizer"""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("train_ops"):
                self.train_ops = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def get_learning_rate(self):
        """Get the exponentially decreased learning rate."""
        self.init_lr = tf.placeholder(tf.float32, name="init_lr")
        self.global_step = tf.placeholder(tf.float32, name="global_step")
        self.decay_step = tf.placeholder(tf.float32, name="decay_step")
        self.decay_rate = tf.placeholder(tf.float32, name="decay_rate")
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.init_lr ,
                global_step=self.global_step,
                decay_steps=self.decay_step,
                decay_rate=self.decay_rate,
                staircase=False,
                name=None)


    #====================
    # Methods for RG #
    #====================
    def cvae_train(self, data, train_configs, labels):
        """Train the cvae network"""

        self.get_learning_rate()
        self.get_loss()
        self.get_opt()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # get validation
        data_trn,data_val = utils.gen_validation(
            data, valrate=train_configs.valrate, label=labels)

        numbatch_trn = len(data_trn["data"]) // train_configs.batchsize
        numbatch_val = len(data_val["data"]) // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_recon_trn = np.zeros(x_epoch.shape)
        y_loss_recon_val = np.zeros(x_epoch.shape)
        y_loss_latent_trn = np.zeros(x_epoch.shape)
        y_loss_latent_val = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_recon    Val_loss_recon    Trn_loss_latent    Val_loss_latent"
              % (timestamp))

        last_recon_trn = 10000
        last_recon_val = 10000
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_recon_all = 0.0
            loss_val_recon_all = 0.0
            loss_trn_latent_all = 0.0
            loss_val_latent_all = 0.0
            loss_val_all = 0.0
            loss_trn_all = 0.0

            indices_trn = utils.gen_BatchIterator_label(
                data_trn['data'],
                data_trn['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_trn in range(numbatch_trn):
                idx_trn = indices_trn[i_trn*train_configs.batchsize:
                                      (i_trn+1)*train_configs.batchsize]
                data_trn_batch = np.reshape(
                    data_trn['data'][idx_trn], (-1,self.input_shape[1],self.input_shape[2],1))
                label_trn_input = utils.condition_reshape(
                    label=data_trn['label'][idx_trn],
                    numclass=self.numclass,
                    imgshape=(self.input_shape[1], self.input_shape[2]))
                train_dict = {
                    self.inputs: data_trn_batch,
                    self.conditions: data_trn['label'][idx_trn],
                    self.conditions_input: label_trn_input,
                    self.outputs: data_trn_batch,
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_recon, loss_trn_latent, loss_trn = self.sess.run(
                    [self.train_ops, self.loss_recon,
                     self.loss_latent, self.loss],
                    feed_dict=train_dict)
                loss_trn_recon_all += loss_trn_recon.mean()
                loss_trn_latent_all += loss_trn_latent.mean()
                loss_trn_all += loss_trn.mean()

            y_loss_recon_trn[i] = loss_trn_recon_all / numbatch_trn
            y_loss_latent_trn[i] = loss_trn_latent_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn
            # validation
            indices_val = utils.gen_BatchIterator_label(
                data_val['data'],
                data_val['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_val in range(numbatch_val):
                idx_val = indices_val[i_val*train_configs.batchsize:
                                      (i_val+1)*train_configs.batchsize]
                data_val_batch = np.reshape(
                    data_val['data'][idx_val], (-1,self.input_shape[1],self.input_shape[2],1))
                label_val_input = utils.condition_reshape(
                    label=data_val['label'][idx_val],
                    numclass=self.numclass,
                    imgshape=(self.input_shape[1], self.input_shape[2]))
                val_dict = {
                    self.inputs: data_val_batch,
                    self.conditions: data_val['label'][idx_val],
                    self.conditions_input: label_val_input,
                    self.outputs: data_val_batch,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_recon, loss_val_latent, loss_val = self.sess.run(
                    [self.loss_recon, self.loss_latent, self.loss],
                    feed_dict=val_dict)
                loss_val_recon_all += loss_val_recon.mean()
                loss_val_latent_all += loss_val_latent.mean()
                loss_val_all += loss_val.mean()

            y_loss_recon_val[i] = loss_val_recon_all / numbatch_val
            y_loss_latent_val[i] = loss_val_latent_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val
            # print results
            if i % train_configs.print_step == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %6d    %14.5f    %14.5f    %15.5f    %15.5f' % (
                    timestamp, i,
                    y_loss_recon_trn[i], y_loss_recon_val[i],
                    y_loss_latent_trn[i], y_loss_latent_val[i]
                    ))


            # stop condition
            diff_trn = np.abs(last_recon_trn - y_loss_recon_trn[i])
            diff_val = np.abs(last_recon_val - y_loss_recon_val[i])
            last_recon_trn = y_loss_recon_trn[i]
            last_recon_val = y_loss_recon_val[i]
            if diff_trn <= self.epocheps:
                break;
        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_recon": y_loss_recon_trn,
            "trn_loss_latent": y_loss_latent_trn,
            "trn_loss": y_loss_trn,
            "val_loss_recon": y_loss_recon_val,
            "val_loss_latent": y_loss_latent_val,
            "val_loss": y_loss_val,
            "epochs": i+1}


    def vae_train(self, data, train_configs, labels=None):
        """Train the vae network"""

        self.get_learning_rate()
        self.get_loss()
        self.get_opt()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # get validation
        data_trn,data_val = utils.gen_validation(
            data, valrate=train_configs.valrate, label=labels)

        numbatch_trn = len(data_trn["data"]) // train_configs.batchsize
        numbatch_val = len(data_val["data"]) // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_recon_trn = np.zeros(x_epoch.shape)
        y_loss_recon_val = np.zeros(x_epoch.shape)
        y_loss_latent_trn = np.zeros(x_epoch.shape)
        y_loss_latent_val = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_recon    Val_loss_recon    Trn_loss_latent    Val_loss_latent"
              % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_recon_all = 0.0
            loss_val_recon_all = 0.0
            loss_trn_latent_all = 0.0
            loss_val_latent_all = 0.0
            loss_val_all = 0.0
            loss_trn_all = 0.0

            indices_trn = utils.gen_BatchIterator_label(
                data_trn['data'],
                data_trn['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_trn in range(numbatch_trn):
                idx_trn = indices_trn[i_trn*train_configs.batchsize:
                                      (i_trn+1)*train_configs.batchsize]
                train_dict = {
                    self.inputs: data_trn['data'][idx_trn],
                    self.outputs: data_trn['data'][idx_trn],
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_recon, loss_trn_latent, loss_trn = self.sess.run(
                    [self.train_ops, self.loss_recon,
                     self.loss_latent, self.loss],
                    feed_dict=train_dict)
                loss_trn_recon_all += loss_trn_recon.mean()
                loss_trn_latent_all += loss_trn_latent.mean()
                loss_trn_all += loss_trn.mean()

            y_loss_recon_trn[i] = loss_trn_recon_all / numbatch_trn
            y_loss_latent_trn[i] = loss_trn_latent_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn
            # validation
            indices_val = utils.gen_BatchIterator_label(
                data_val['data'],
                data_val['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_val in range(numbatch_val):
                idx_val = indices_val[i_val*train_configs.batchsize:
                                      (i_val+1)*train_configs.batchsize]
                val_dict = {
                    self.inputs: data_val['data'][idx_val],
                    self.outputs: data_val['data'][idx_val],
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_recon, loss_val_latent, loss_val = self.sess.run(
                    [self.loss_recon, self.loss_latent, self.loss],
                    feed_dict=val_dict)
                loss_val_recon_all += loss_val_recon.mean()
                loss_val_latent_all += loss_val_latent.mean()
                loss_val_all += loss_val.mean()

            y_loss_recon_val[i] = loss_val_recon_all / numbatch_val
            y_loss_latent_val[i] = loss_val_latent_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val
            # print results
            if i % train_configs.print_step == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %6d    %14.5f    %14.5f    %15.5f    %15.5f' %
                      (timestamp, i, y_loss_recon_trn[i], y_loss_recon_val[i],
                       y_loss_latent_trn[i], y_loss_latent_val[i]))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_recon": y_loss_recon_trn,
            "trn_loss_latent": y_loss_latent_trn,
            "trn_loss": y_loss_trn,
            "val_loss_recon": y_loss_recon_val,
            "val_loss_latent": y_loss_latent_val,
            "val_loss": y_loss_val}


    #====================
    # Methods for MNIST #
    #====================
    def vae_train_mnist(self, mnist, train_configs):
        """Train the network on MNIST data"""

        self.get_learning_rate()
        self.get_loss()
        self.get_opt()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        numbatch_trn = mnist.train.images.shape[0] // train_configs.batchsize
        numbatch_val = mnist.validation.images.shape[0] // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_recon_trn = np.zeros(x_epoch.shape)
        y_loss_recon_val = np.zeros(x_epoch.shape)
        y_loss_latent_trn = np.zeros(x_epoch.shape)
        y_loss_latent_val = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_recon    Val_loss_recon    Trn_loss_latent    Val_loss_latent"
              % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_recon_all = 0.0
            loss_val_recon_all = 0.0
            loss_trn_latent_all = 0.0
            loss_val_latent_all = 0.0
            loss_val_all = 0.0
            loss_trn_all = 0.0

            for i_trn in range(numbatch_trn):
                data_trn, label_trn = mnist.train.next_batch(
                      batch_size=train_configs.batchsize)
                data_trn = np.reshape(data_trn, (-1,28,28,1))
                train_dict = {
                    self.inputs: data_trn,
                    self.outputs: data_trn,
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_recon, loss_trn_latent, loss_trn = self.sess.run(
                    [self.train_ops, self.loss_recon,
                     self.loss_latent, self.loss],
                    feed_dict=train_dict)
                loss_trn_recon_all += loss_trn_recon.mean()
                loss_trn_latent_all += loss_trn_latent.mean()
                loss_trn_all += loss_trn.mean()

            y_loss_recon_trn[i] = loss_trn_recon_all / numbatch_trn
            y_loss_latent_trn[i] = loss_trn_latent_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn

            # validation
            for i_val in range(numbatch_val):
                data_val, label_val = mnist.validation.next_batch(
                      batch_size=train_configs.batchsize)
                data_val = np.reshape(data_val, (-1,28,28,1))
                val_dict = {
                    self.inputs: data_val,
                    self.outputs: data_val,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_recon, loss_val_latent, loss_val = self.sess.run(
                    [self.loss_recon, self.loss_latent, self.loss],
                    feed_dict=val_dict)
                loss_val_recon_all += loss_val_recon.mean()
                loss_val_latent_all += loss_val_latent.mean()
                loss_val_all += loss_val.mean()
                # print(loss_val_recon_all, loss_val_latent_all, loss_val_all)

            y_loss_recon_val[i] = loss_val_recon_all / numbatch_val
            y_loss_latent_val[i] = loss_val_latent_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val
            # print results
            if i % train_configs.print_step == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %6d    %14.5f    %14.5f    %15.5f    %15.5f' % (
                    timestamp, i,
                    y_loss_recon_trn[i], y_loss_recon_val[i],
                    y_loss_latent_trn[i], y_loss_latent_val[i]
                    ))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_recon": y_loss_recon_trn,
            "trn_loss_latent": y_loss_latent_trn,
            "trn_loss": y_loss_trn,
            "val_loss_recon": y_loss_recon_val,
            "val_loss_latent": y_loss_latent_val,
            "val_loss": y_loss_val}


    def cvae_train_mnist(self, mnist, train_configs):
        """Train the network on MNIST data"""

        self.get_learning_rate()
        self.get_loss()
        self.get_opt()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        numbatch_trn = mnist.train.images.shape[0] // train_configs.batchsize
        numbatch_val = mnist.validation.images.shape[0] // train_configs.batchsize

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_recon_trn = np.zeros(x_epoch.shape)
        y_loss_recon_val = np.zeros(x_epoch.shape)
        y_loss_latent_trn = np.zeros(x_epoch.shape)
        y_loss_latent_val = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_recon    Val_loss_recon    Trn_loss_latent    Val_loss_latent"
              % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_recon_all = 0.0
            loss_val_recon_all = 0.0
            loss_trn_latent_all = 0.0
            loss_val_latent_all = 0.0
            loss_val_all = 0.0
            loss_trn_all = 0.0

            for i_trn in range(numbatch_trn):
                data_trn, label_trn = mnist.train.next_batch(
                      batch_size=train_configs.batchsize)
                data_trn = np.reshape(data_trn, (-1,28,28,1))
                label_trn_input = utils.condition_reshape(
                    label=label_trn,
                    numclass=self.numclass,
                    imgshape=(self.input_shape[1], self.input_shape[2]))
                train_dict = {
                    self.inputs: data_trn,
                    self.conditions: label_trn,
                    self.conditions_input: label_trn_input,
                    self.outputs: data_trn,
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_recon, loss_trn_latent, loss_trn = self.sess.run(
                    [self.train_ops, self.loss_recon,
                     self.loss_latent, self.loss],
                    feed_dict=train_dict)
                loss_trn_recon_all += loss_trn_recon.mean()
                loss_trn_latent_all += loss_trn_latent.mean()
                loss_trn_all += loss_trn.mean()

            y_loss_recon_trn[i] = loss_trn_recon_all / numbatch_trn
            y_loss_latent_trn[i] = loss_trn_latent_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn

            # validation
            for i_val in range(numbatch_val):
                data_val, label_val = mnist.validation.next_batch(
                      batch_size=train_configs.batchsize)
                data_val = np.reshape(data_val, (-1,28,28,1))
                label_val_input = utils.condition_reshape(
                    label=label_val,
                    numclass=self.numclass,
                    imgshape=(self.input_shape[1], self.input_shape[2]))
                val_dict = {
                    self.inputs: data_val,
                    self.conditions: label_val,
                    self.conditions_input: label_val_input,
                    self.outputs: data_val,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_recon, loss_val_latent, loss_val = self.sess.run(
                    [self.loss_recon, self.loss_latent, self.loss],
                    feed_dict=val_dict)
                loss_val_recon_all += loss_val_recon.mean()
                loss_val_latent_all += loss_val_latent.mean()
                loss_val_all += loss_val.mean()

            y_loss_recon_val[i] = loss_val_recon_all / numbatch_val
            y_loss_latent_val[i] = loss_val_latent_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val
            # print results
            if i % train_configs.print_step == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %6d    %14.5f    %14.5f    %15.5f    %15.5f' % (
                    timestamp, i,
                    y_loss_recon_trn[i], y_loss_recon_val[i],
                    y_loss_latent_trn[i], y_loss_latent_val[i]
                    ))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_recon": y_loss_recon_trn,
            "trn_loss_latent": y_loss_latent_trn,
            "trn_loss": y_loss_trn,
            "val_loss_recon": y_loss_recon_val,
            "val_loss_latent": y_loss_latent_val,
            "val_loss": y_loss_val}

    #========================
    # Save and test methods #
    #========================
    def loss_save(self,savepath):
        """Save the loss and accuracy staffs"""
        import pickle
        with open(savepath, 'wb') as fp:
            pickle.dump(self.train_dict, fp)


    def cvae_test(self, data, labels):
        """Test the network"""
        test_loss = self.sess.run(
            self.loss_recon,
            feed_dict={
                self.inputs: data,
                self.conditions: labels,
                self.outputs: data,
                self.is_training: False,
                self.keep_prob: 1.0})

        return test_loss.mean()
