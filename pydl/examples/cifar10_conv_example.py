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
import matplotlib.image as mpimg

from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.nn import NN
from pydl.training.training import Adam
from pydl.training.training import SGD
from pydl import conf


def get_data(file_path):
    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                  'data_batch_5', 'test_batch']
    meta_file = 'batches.meta'

    data = list()
    labels = list()
    class_names = None

    for df in data_files:
        with open(file_path+df, 'rb') as batch_file:
            data_dict = pickle.load(batch_file, encoding='bytes')
            data.append(np.reshape(data_dict[b'data'], newshape=(-1, 3, 32, 32)))
            labels.append(data_dict[b'labels'])

    with open(file_path+meta_file, 'rb') as meta_f:
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

    w_scale = 1.0
    dp = None
    bn = True

    l1 = Conv(X, receptive_field=(3,3), num_filters=32, zero_padding=1, stride=1, name='Conv-1',
              weight_scale=w_scale, xavier=True, activation_fn='ReLU', batchnorm=bn, dropout=dp)

    l2 = Pool(l1, receptive_field=(2,2), stride=2, name='MaxPool-2')

    l3 = Conv(l2, receptive_field=(3,3), num_filters=32, zero_padding=1, stride=1, name='Conv-3',
              weight_scale=w_scale, xavier=True, activation_fn='ReLU', batchnorm=bn, dropout=dp)

    l4 = Pool(l3, receptive_field=(2,2), stride=1, name='MaxPool-4')

    l5 = Conv(l4, receptive_field=(3,3), num_filters=16, zero_padding=1, stride=1, name='Conv-5',
              weight_scale=w_scale, xavier=True, activation_fn='ReLU', batchnorm=bn, dropout=dp)

    l6 = Pool(l5, receptive_field=(2,2), stride=1, name='MaxPool-6')

    l7 = FC(l6, num_neurons=128, weight_scale=w_scale, xavier=True, activation_fn='ReLU',
            batchnorm=bn, dropout=0.9, name="FC-7")

    l8 = FC(l7, num_neurons=64, weight_scale=w_scale, xavier=True, activation_fn='ReLU',
            batchnorm=bn, dropout=0.9, name="FC-8")

    l9 = FC(l8, num_neurons=K, weight_scale=w_scale, xavier=True, activation_fn='SoftMax',
             name="Output-Layer")

    layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9]

    nn = NN(X, layers)
    adam = Adam(nn, step_size=1e-2, beta_1=0.9,  beta_2=0.999, reg_lambda=1e-1, train_size=50000,
                test_size=10000)
    adam.train(X, y, normalize='mean', shuffle=False, batch_size=256, epochs=10000, log_freq=1,
               plot='MNIST - Adam - Dropout')


    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
