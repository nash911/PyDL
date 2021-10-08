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
from pydl.nn.nn import NN
from pydl.training.adam import Adam
from pydl import conf


def main():
    mnist = fetch_openml('mnist_784')
    X = np.array(mnist.data, dtype=conf.dtype)
    y = np.array(mnist.target, dtype=np.int)
    K = np.max(y) + 1

    # plot first few images
    fig = plt.figure()
    for i, r in enumerate(np.random.randint(0, y.size, 9)):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(np.reshape(X[r], (-1, 28)), cmap=plt.get_cmap('gray'))
        plt.title(str(y[r]))

    # show the figure
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    weight_scale = 1.0

    l1 = FC(X, num_neurons=200, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='Tanh', batchnorm=True)
    l2 = FC(l1, num_neurons=100, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='Tanh', batchnorm=True)
    l3 = FC(l2, num_neurons=50, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='Tanh', batchnorm=True)
    l4 = FC(l3, num_neurons=25, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='Tanh', batchnorm=True)
    l5 = FC(l4, num_neurons=15, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='Tanh', batchnorm=True)
    l6 = FC(l5, num_neurons=K, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='SoftMax')
    layers = [l1, l2, l3, l4, l5, l6]

    nn = NN(X, layers)
    adam = Adam(nn, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-1, train_size=60000,
                test_size=10000)
    adam.train(X, y, normalize='pca', dims=0.97, shuffle=False, epochs=10000, plot='MNIST - Adam',
               log_freq=1)

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
