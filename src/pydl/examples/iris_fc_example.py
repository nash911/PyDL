# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
from sklearn import datasets

from pydl.nn.layers import FC
from pydl.nn.nn import NN
from pydl.training.sgd import SGD
from pydl.training.momentum import Momentum
from pydl.training.rmsprop import RMSprop
from pydl.training.adam import Adam


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    K = np.max(y) + 1

    # Data Stats
    print("Data Size: ", X.shape[0])
    print("Feature Size: ", X.shape[1])
    print("Data Min: ", np.min(X, axis=0))
    print("Data Max: ", np.max(X, axis=0))
    print("Data Range: ", np.max(X, axis=0) - np.min(X, axis=0))
    print("Data Mean: ", np.mean(X, axis=0))
    print("Data STD: ", np.std(X, axis=0))

    # SoftMax Cross Entropy
    l1 = FC(X, num_neurons=int(X.shape[-1] * 2), bias=True, weight_scale=1.0, xavier=False,
            activation_fn='ReLU')
    l2 = FC(l1, num_neurons=int(X.shape[-1]), bias=True, weight_scale=0.01, xavier=False,
            activation_fn='Tanh')
    l3_a = FC(l2, num_neurons=K, bias=True, weight_scale=1.0, xavier=False, activation_fn='SoftMax')
    layers = [l1, l2, l3_a]
    nn_a = NN(X, layers)

    # SGD
    sgd = SGD(nn_a, step_size=1e-3, reg_lambda=1e-2)
    sgd.train(X, y, normalize='pca', dims=2, epochs=10000, y_onehot=False, plot='SGD - PCA')

    # Momentum
    nn_a.reinitialize_network()
    momentum = Momentum(nn_a, step_size=1e-3, mu=0.5, reg_lambda=1e-2)
    momentum.train(X, y, normalize='pca', dims=2, epochs=10000, y_onehot=False,
                   plot='Momentum - PCA')

    # RMSprop
    nn_a.reinitialize_network()
    rms = RMSprop(nn_a, step_size=1e-3, beta=0.9, reg_lambda=1e-2)
    rms.train(X, y, normalize='pca', dims=2, epochs=10000, y_onehot=False, plot='RMSprop - PCA')

    # Adam
    nn_a.reinitialize_network()
    adam = Adam(nn_a, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-2)
    adam.train(X, y, normalize='pca', dims=2, epochs=10000, y_onehot=False, plot='Adam - PCA')

    # Sigmoid Cross Entropy
    l1 = FC(X, num_neurons=int(X.shape[-1] * 2), bias=True, weight_scale=1.0, xavier=True,
            activation_fn='ReLU')
    l2 = FC(l1, num_neurons=int(X.shape[-1]), bias=True, weight_scale=1.0, xavier=True,
            activation_fn='Tanh')
    l3_b = FC(l2, num_neurons=K, bias=True, weight_scale=1.0, xavier=True, activation_fn='Sigmoid')
    layers = [l1, l2, l3_b]
    nn_b = NN(X, layers)

    # SGD
    sgd = SGD(nn_b, step_size=1e-3, reg_lambda=1e-2)
    sgd.train(X, y, normalize='mean', epochs=10000, y_onehot=False, plot='SGD - Mean Normalized')

    # Momentum
    nn_b.reinitialize_network()
    momentum = Momentum(nn_b, step_size=1e-3, mu=0.5, reg_lambda=1e-2)
    momentum.train(X, y, normalize='mean', epochs=10000, y_onehot=False,
                   plot='Momentum - Mean Normalized')

    # RMSprop
    nn_b.reinitialize_network()
    rms = RMSprop(nn_b, step_size=1e-3, beta=0.9, reg_lambda=1e-2)
    rms.train(X, y, normalize='mean', epochs=10000, y_onehot=False, plot='RMSprop - Mean Norm')

    # Adam
    nn_b.reinitialize_network()
    adam = Adam(nn_b, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-2)
    adam.train(X, y, normalize='mean', epochs=10000, y_onehot=False, plot='Adam - Mean Norm')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
