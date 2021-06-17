# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.nn import NN
from pydl.training.training import Adam
from pydl import conf

def main():
    mnist = fetch_openml('mnist_784')
    X = np.array(mnist.data, dtype=conf.dtype).reshape(-1, 1, 28, 28)
    y = np.array(mnist.target, dtype=np.int)
    K = np.max(y) + 1

    # plot first few images
    fig = plt.figure()
    for i, r in enumerate(np.random.randint(0, y.size, 9)):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(X[r], cmap=plt.get_cmap('gray'))
        plt.title(str(y[r]))

    # show the figure
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    w_scale = 1.0
    dp = 0.9
    bn = True

    l1 = Conv(X, receptive_field=(3,3), num_filters=16, zero_padding=1, stride=1, name='Conv-1',
              weight_scale=w_scale, xavier=True, activation_fn='ReLU', batchnorm=bn, dropout=dp)

    l2 = Pool(l1, receptive_field=(2,2), stride=2, name='MaxPool-2')

    l3 = Conv(l2, receptive_field=(3,3), num_filters=16, zero_padding=1, stride=1, name='Conv-3',
              weight_scale=w_scale, xavier=True, activation_fn='ReLU', batchnorm=bn, dropout=dp)

    l4 = Pool(l3, receptive_field=(2,2), stride=1, name='MaxPool-4')

    l5 = Conv(l4, receptive_field=(3,3), num_filters=8, zero_padding=1, stride=1, name='Conv-5',
              weight_scale=w_scale, xavier=True, activation_fn='ReLU', batchnorm=bn, dropout=dp)

    l6 = Pool(l5, receptive_field=(2,2), stride=1, name='MaxPool-6')

    l7 = FC(l6, num_neurons=64, weight_scale=w_scale, xavier=True, activation_fn='ReLU',
            batchnorm=bn, dropout=0.9, name="FC-7")

    l8 = FC(l7, num_neurons=32, weight_scale=w_scale, xavier=True, activation_fn='ReLU',
            batchnorm=bn, dropout=0.9, name="FC-8")

    l9 = FC(l8, num_neurons=K, weight_scale=w_scale, xavier=True, activation_fn='SoftMax',
             name="Output-Layer")

    layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9]

    nn = NN(X, layers)
    adam = Adam(nn, step_size=1e-3, beta_1=0.9,  beta_2=0.999, reg_lambda=1e-1, train_size=60000,
                test_size=10000)
    adam.train(X, y, normalize='mean', shuffle=False, batch_size=32, epochs=100, log_freq=1,
               plot='MNIST - Adam - Dropout')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
