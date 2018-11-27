# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
# MIT liscence

"""Utilities for the ResCVAE"""

import os
import pickle
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.layers as layers


def get_timestamp():
    """Get time at present"""
    import time
    timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    return timestamp


def get_batch_norm(inputs,is_training,scope=None):
    """Do batch normalization"""
    bn_out = layers.batch_norm(
        inputs=inputs,
        center=True,
        scale=True,
        is_training=is_training,
        scope=scope)
    return bn_out


def get_block_en(resnet_classic_param, is_training=None):
    """Generate encoder block parameters according to the
       classic list like structure.

    input
    =====
    resnet_classic_param: list
        a list composed of bottleneck configuration
    is_training: tf.placeholder
        a placeholder for batch_normalization

    output
    ======
    block_params: list
        block parameters generated for class Block
    """
    block_params = []
    bottle_conf = namedtuple(
        'bottle_conf',
        ['depth3','depth1','stride'])

    for i, bottle in enumerate(resnet_classic_param):
        bottle = bottle_conf._make(bottle)
        bottle_params = []
        # layer 1
        bottle_params.append(
            ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
        # layer 2
        bottle_params.append(
            ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
        # layer 3
        if i == len(resnet_classic_param)-1:
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
        else:
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))

        block_params.append(bottle_params)

    return block_params


def get_block_list(blocks_params, input_depth):
    """Transfer the blocks parameters into list"""
    block_list = [input_depth]
    block_stride = []
    for block in blocks_params:
        for bottle in block:
            bottle_in = bottle[1]
            bottle_out = bottle[0]
            bottle_stride = bottle[2]
            block_list.append(bottle_in)
            block_list.append(bottle_in)
            block_list.append(bottle_out)
            block_stride.append(bottle_stride)

    return block_list, block_stride


def get_block_de_params(block_list, block_stride):
    """Generate decoder block paramters from reversed encoder block list"""
    blocks_de = []
    # reverse
    block_list = block_list[0:-1]
    block_list.reverse()
    num_bottles = len(block_list) // 3
    block_stride.reverse()
    # Whether 3x
    if num_bottles != len(block_stride):
        print("Number of bottles should be equal to number of strides, please check...")
        return None
    else:
        block = []
        for i in range(num_bottles):
            bottle = [(block_list[i*3+2], block_list[i*3+0], block_stride[i])]
            block += bottle
            if i + 1 == num_bottles or block_stride[i+1] > 1: # or 的判断有先后顺序
                blocks_de.append(block)
                block = []

    return blocks_de


def get_block_de(resnet_classic_param, is_training=None, outflag=False):
    """Generate decoder block parameters according to the
       classic list like structure.

    input
    =====
    resnet_classic_param: list
        a list composed of bottleneck configuration
    is_training: tf.placeholder
        a placeholder for batch_normalization

    output
    ======
    block_params: list
        block parameters generated for class Block
    """
    block_params = []
    bottle_conf = namedtuple(
        'bottle_conf',
        ['depth3','depth1','stride'])

    for i, bottle in enumerate(resnet_classic_param):
        bottle = bottle_conf._make(bottle)
        bottle_params = []
        # layer 1
        bottle_params.append(
            ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
        # layer 2
        bottle_params.append(
            ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
        # layer 3
        if i == len(resnet_classic_param)-1:
            # Batch_norm
            if outflag:
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))
            else:
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
        else:
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))
        block_params.append(bottle_params)

    return block_params


def get_block(resnet_classic_param, encode_flag=True, is_training=None):
    """Generate block parameters according to the
       classic list like structure.

    input
    =====
    resnet_classic_param: list
        a list composed of bottleneck configuration
    encode_flag: bool
        It true, encoder blosk; if false, decoder block
    is_training: tf.placeholder
        a placeholder for batch_normalization

    output
    ======
    block_params: list
        block parameters generated for class Block
    """
    block_params = []
    bottle_conf = namedtuple(
        'bottle_conf',
        ['depth3','depth1','stride'])

    if encode_flag:
        for i, bottle in enumerate(resnet_classic_param):
            bottle = bottle_conf._make(bottle)
            bottle_params = []
            # layer 1
            bottle_params.append(
                ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
            # layer 2
            bottle_params.append(
                ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
            # layer 3
            if i == len(resnet_classic_param)-1:
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
            else:
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))

            block_params.append(bottle_params)
    else:
        for i, bottle in enumerate(resnet_classic_param):
            bottle = bottle_conf._make(bottle)
            bottle_params = []
            # layer 1
            bottle_params.append(
                ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
            # layer 2
            bottle_params.append(
                ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
            # layer 3
            if i == len(resnet_classic_param)-1:
                # Batch_norm
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
            else:
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))
            block_params.append(bottle_params)

    return block_params


def get_odd_flags(odd_flags, blocks, flag_input=False):
    """Reverse the odd_flag list from the encoder"""
    odd_list = [flag_input]
    for odd_block in odd_flags:
        odd_list += odd_block
    # reverse
    odd_list.reverse()
    odd_de = blocks.copy()
    i = 0
    for m, odd_block in enumerate(odd_de):
        odd_de[m] = blocks[m].copy()
        for n, bottle in enumerate(odd_block):
            odd_de[m][n] = odd_list[i+1]
            i += 1

    return odd_de


def gen_validation(data, valrate = 0.2, label=None):
    """Separate the dataset into training and validation subsets.
    inputs
    ======
    data: np.ndarray
        The input data, 4D matrix
    label: np.ndarray or list, opt
        The labels w.r.t. input data, optional
    outputs
    =======
    data_train: {"data": , "label": }
    data_val: {"data":, "label":}
    """
    if label is None:
        label_train = None
        label_val = None
        idx = np.random.permutation(len(data))
        num_val = int(len(data)*valrate)
        data_val = {"data": data[idx[0:num_val]],
                    "label": label_val}
        # train
        data_train = {"data": data[idx[num_val:]],
                      "label": label_train}
    else:
        # Training and validation
        idx = np.random.permutation(len(data))
        num_val = int(len(data)* valrate)
        data_val = {"data": data[idx[0:num_val]],
                    "label": label[idx[0:num_val]]}
        # train
        data_train = {"data": data[idx[num_val:]],
                      "label": label[idx[num_val:]]}

    return data_train,data_val


def gen_BatchIterator(data, batch_size=100, shuffle=True):
    """
    Return the next 'batch_size' examples
    from the X_in dataset
    Reference
    =========
    [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
    Input
    =====
    data: 4d np.ndarray
        The samples to be batched
    batch_size: int
        Size of a single batch.
    shuffle: bool
        Whether shuffle the indices.
    Output
    ======
    Yield a batch generator
    """
    if shuffle:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt]


def gen_BatchIterator_label(data, label, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset
        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        data: 4d np.ndarray
            The samples to be batched
        label: np.ndarray
            The labels to be batched w.r.t. data
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.
        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
        else:
            indices = np.arange(len(data))
        return indices


def vec2onehot(label,numclass):
    label_onehot = np.zeros((len(label),numclass))
    for i,l in enumerate(label):
        label_onehot[i, int(l)] = 1

    return label_onehot


def save_net(sess, namepath, netpath, savedict):
    """Save the net"""
    import pickle
    import sys
    sys.setrecursionlimit(1000000)

    with open(namepath, 'wb') as fp:
        pickle.dump(savedict, fp)

    # save the net
    saver = tf.train.Saver()
    saver.save(sess, netpath)


def loss_eval(net, data, labels, numclass=2, rs=40):
    """Evaluate the network performance on test samples"""
    loss = np.zeros([labels.shape[0]])
    for i in range(len(loss)):
        label_input = condition_reshape(
            label=labels[i,np.newaxis], numclass=numclass, imgshape=(rs, rs))
        img_est = net.sess.run(
            net.output_flatten_de,
            feed_dict={
                net.inputs: data[i].reshape(-1, rs, rs, 1),
                net.conditions: labels[i].reshape([1, labels[i].shape[0]]),
                net.conditions_input: label_input,
                net.is_training: False,
                net.keep_prob: 1.0})
        img_est = (img_est - img_est.min()) / (img_est.max() - img_est.min())
        loss[i] = np.mean((data[i] - img_est)**2)
        # loss[i] = np.sum((data[i] - img_est)**2) / (40**2)
    # print(data[i].shape)
    # evaluation
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)

    return loss,loss_mean,loss_std


def loss_eval_vae(net, data, labels=None):
    """Evaluate the network performance on test samples"""
    loss = np.zeros([labels.shape[0]])
    for i in range(len(loss)):
        img_est = net.sess.run(
            net.outputs_de,
            feed_dict={
                net.inputs: data[i].reshape([1, data[i].shape[0]]),
                net.is_training: False,
                net.keep_prob: 1.0})
        img_est = (img_est - img_est.min()) / (img_est.max() - img_est.min())
        loss[i] = np.mean((data[i].reshape([1, data[i].shape[0]]) - img_est)**2)
    # evaluation
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)

    return loss,loss_mean,loss_std


def loss_save(net,savepath):
    """Save the loss and accuracy staffs"""
    import pickle
    with open(savepath, 'wb') as fp:
        pickle.dump(net.train_dict, fp)


def load_net(namepath):
    """
    Load the cae network
    reference
    =========
    [1] https://www.cnblogs.com/azheng333/archive/2017/06/09/6972619.html
    input
    =====
    namepath: str
        Path to save the trained network
    output
    ======
    sess: tf.Session()
        The restored session
    names: dict
        The dict saved variables names
    """
    try:
        fp = open(namepath,'rb')
    except:
        return None

    names = pickle.load(fp)

    # load the net
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, names['netpath'])

    return sess, names

def load_raw_image(names, folder, rs=80):
    def gen_norm(img):
        return (img-img.min())/(img.max() - img.min())
    from astropy.io import fits
    img_all = np.zeros((len(names),rs,rs))
    for i, name in enumerate(names):
        filepath = os.path.join(folder, name+".fits")
        with fits.open(filepath) as h:
            img = h[0].data
            img_shape = img.shape
            r_c = img_shape[0] // 2
            c_c = img_shape[1] // 2
            r_h = rs // 2
            c_h = rs // 2
            img = img[r_c-r_h:r_c+r_h, c_c-c_h:c_c+c_h]
            img_all[i, :, :] = gen_norm(img)

    return img_all


def condition_reshape(label, numclass, imgshape):
    """Expand the label, i.e., the condition, to matrices as
       the same shape as the image.
    """
    label = np.array(label)
    rows, cols = label.shape
    conmat = np.zeros(
        shape=(rows,imgshape[0], imgshape[1], numclass))
    if cols == numclass:
        for i in range(rows):
            for j in range(cols):
                conmat[i,:,:,j] = label[i,j]
    elif cols == 1:
        for i in range(rows):
            conmat[i,:,:,j] = label[i,0]

    return conmat
