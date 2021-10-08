# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt

from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.residual_block import ResidualBlock
from pydl.nn.nn import NN
from pydl.training.momentum import Momentum


def get_data(file_path):
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                  'data_batch_5', 'test_batch']
    meta_file = 'batches.meta'

    data = list()
    labels = list()
    class_names = None

    for df in data_files:
        with open(file_path + df, 'rb') as batch_file:
            data_dict = pickle.load(batch_file, encoding='bytes')
            data.append(np.reshape(data_dict[b'data'], newshape=(-1, 3, 32, 32)))
            labels.append(data_dict[b'labels'])

    with open(file_path + meta_file, 'rb') as meta_f:
        meta_dict = pickle.load(meta_f, encoding='bytes')
        class_names = meta_dict[b'label_names']
        class_names = [c_name.decode("utf-8") for c_name in class_names]

    return np.vstack(data), np.hstack(labels), class_names


def main():
    X, y, class_names = get_data('data/CIFAR-10/')
    K = np.max(y) + 1

    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    print("K: ", K)

    # plot first few images
    fig = plt.figure()
    for i, r in enumerate(np.random.randint(0, y.size, 9)):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(np.transpose(X[r], axes=[1, 2, 0]))
        plt.title(class_names[y[r]])

    # show the figure
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    w_scale = 1e-0
    xv = True
    bn = True

    l1 = Conv(X, receptive_field=(3, 3), num_filters=16, zero_padding=1, stride=1, name='Conv-1',
              weight_scale=w_scale, xavier=xv, activation_fn='ReLU', batchnorm=bn)

    # Residual Block 1A
    rcp_field_1A = [3, 3]
    num_filters_1A = [16, 16]
    activation_fn_1A = ['Relu', 'Relu']
    stride_1A = 1
    conv_layers_1A = [l1]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_1A, num_filters_1A, activation_fn_1A)):
        pad = int((rcp - 1) / 2)
        conv = Conv(conv_layers_1A[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_1A if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-1A_%d' % (i + 1))
        conv_layers_1A.append(conv)

    res_block_1A = ResidualBlock(l1, conv_layers_1A[1:], activation_fn='Relu', name='ResBlock-1A')

    # Residual Block 1B
    rcp_field_1B = [3, 3]
    num_filters_1B = [16, 16]
    activation_fn_1B = ['Relu', 'Relu']
    stride_1B = 1
    conv_layers_1B = [res_block_1A]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_1B, num_filters_1B, activation_fn_1B)):
        pad = int((rcp - 1) / 2)
        conv = Conv(conv_layers_1B[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_1B if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-1B_%d' % (i + 1))
        conv_layers_1B.append(conv)

    res_block_1B = ResidualBlock(res_block_1A, conv_layers_1B[1:], activation_fn='Relu',
                                 name='ResBlock-1B')

    # Residual Block 1C
    rcp_field_1C = [3, 3]
    num_filters_1C = [16, 16]
    activation_fn_1C = ['Relu', 'Relu']
    stride_1C = 1
    conv_layers_1C = [res_block_1B]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_1C, num_filters_1C, activation_fn_1C)):
        pad = int((rcp - 1) / 2)
        conv = Conv(conv_layers_1C[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_1C if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-1C_%d' % (i + 1))
        conv_layers_1C.append(conv)

    res_block_1C = ResidualBlock(res_block_1B, conv_layers_1C[1:], activation_fn='Relu',
                                 name='ResBlock-1C')

    # Residual Block 2A
    rcp_field_2A = [3, 3]
    num_filters_2A = [32, 32]
    activation_fn_2A = ['Relu', 'Relu']
    stride_2A = 2
    conv_layers_2A = [res_block_1C]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_2A, num_filters_2A, activation_fn_2A)):
        pad = (0, 1) if i == 0 else int((rcp - 1) / 2)
        conv = Conv(conv_layers_2A[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_2A if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-2A_%d' % (i + 1))
        conv_layers_2A.append(conv)

    res_block_2A = ResidualBlock(res_block_1C, conv_layers_2A[1:], activation_fn='Relu',
                                 name='ResBlock-2A')

    # Residual Block 2B
    rcp_field_2B = [3, 3]
    num_filters_2B = [32, 32]
    activation_fn_2B = ['Relu', 'Relu']
    stride_2B = 1
    conv_layers_2B = [res_block_2A]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_2B, num_filters_2B, activation_fn_2B)):
        pad = int((rcp - 1) / 2)
        conv = Conv(conv_layers_2B[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_2B if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-2B_%d' % (i + 1))
        conv_layers_2B.append(conv)

    res_block_2B = ResidualBlock(res_block_2A, conv_layers_2B[1:], activation_fn='Relu',
                                 name='ResBlock-2B')

    # Residual Block 2C
    rcp_field_2C = [3, 3]
    num_filters_2C = [32, 32]
    activation_fn_2C = ['Relu', 'Relu']
    stride_2C = 1
    conv_layers_2C = [res_block_2B]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_2C, num_filters_2C, activation_fn_2C)):
        pad = int((rcp - 1) / 2)
        conv = Conv(conv_layers_2C[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_2C if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-2C_%d' % (i + 1))
        conv_layers_2C.append(conv)

    res_block_2C = ResidualBlock(res_block_2B, conv_layers_2C[1:], activation_fn='Relu',
                                 name='ResBlock-2C')

    # Residual Block 3A
    rcp_field_3A = [3, 3]
    num_filters_3A = [64, 64]
    activation_fn_3A = ['Relu', 'Relu']
    stride_3A = 2
    conv_layers_3A = [res_block_2C]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_3A, num_filters_3A, activation_fn_3A)):
        pad = (0, 1) if i == 0 else int((rcp - 1) / 2)
        conv = Conv(conv_layers_3A[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_3A if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-3A_%d' % (i + 1))
        conv_layers_3A.append(conv)

    res_block_3A = ResidualBlock(res_block_2C, conv_layers_3A[1:], activation_fn='Relu',
                                 name='ResBlock-3A')

    # Residual Block 3B
    rcp_field_3B = [3, 3]
    num_filters_3B = [64, 64]
    activation_fn_3B = ['Relu', 'Relu']
    stride_3B = 1
    conv_layers_3B = [res_block_3A]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_3B, num_filters_3B, activation_fn_3B)):
        pad = int((rcp - 1) / 2)
        conv = Conv(conv_layers_3B[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_3B if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-3B_%d' % (i + 1))
        conv_layers_3B.append(conv)

    res_block_3B = ResidualBlock(res_block_3A, conv_layers_3B[1:], activation_fn='Relu',
                                 name='ResBlock-3B')

    # Residual Block 3C
    rcp_field_3C = [3, 3]
    num_filters_3C = [64, 64]
    activation_fn_3C = ['Relu', 'Relu']
    stride_3C = 1
    conv_layers_3C = [res_block_3B]
    for i, (rcp, n_filters, actv_fn) in \
            enumerate(zip(rcp_field_3C, num_filters_3C, activation_fn_3C)):
        pad = int((rcp - 1) / 2)
        conv = Conv(conv_layers_3C[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                    zero_padding=pad, stride=(stride_3C if i == 0 else 1), batchnorm=True,
                    activation_fn=actv_fn, name='Conv-3C_%d' % (i + 1))
        conv_layers_3C.append(conv)

    res_block_3C = ResidualBlock(res_block_3B, conv_layers_3C[1:], activation_fn='Relu',
                                 name='ResBlock-3C')

    avg_pool_layer = Pool(res_block_3C, receptive_field=None, stride=1, pool='AVG', name='AvgPool')

    output_layer = FC(avg_pool_layer, num_neurons=K, weight_scale=w_scale, xavier=xv,
                      activation_fn='SoftMax', name="Output-Layer")

    layers = [l1, res_block_1A, res_block_1B, res_block_1C, res_block_2A, res_block_2B,
              res_block_2C, res_block_3A, res_block_3B, res_block_3C, avg_pool_layer, output_layer]

    nn = NN(X, layers)

    momentum = Momentum(nn, step_size=1e-1, mu=0.9, reg_lambda=1e-4, train_size=50000,
                        test_size=10000)
    momentum.train(X, y, normalize='mean', shuffle=False, batch_size=128, epochs=1000, log_freq=-1,
                   plot='MNIST - ResNet - Momentum')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
