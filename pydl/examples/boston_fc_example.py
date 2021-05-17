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
from pydl.training.training import SGD
from pydl.training.training import Momentum
from pydl.training.training import RMSprop
from pydl.training.training import Adam
from pydl import conf

def main():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    # Data Stats
    print("Data Size: ", X.shape[0])
    print("Feature Size: ", X.shape[1])
    print("Data Min: ", np.min(X, axis=0))
    print("Data Max: ", np.max(X, axis=0))
    print("Data Range: ", np.max(X, axis=0) - np.min(X, axis=0))
    print("Data Mean: ", np.mean(X, axis=0))
    print("Data STD: ", np.std(X, axis=0))

    # Regression NN
    l1 = FC(X, num_neurons=int(X.shape[-1]*5), bias=True, weight_scale=1.0, xavier=True,
            dropout=0.5, activation_fn='ReLU')
    l2 = FC(l1, num_neurons=int(l1.shape[-1]), bias=True, weight_scale=1.0, xavier=True,
            dropout=1.0, activation_fn='Tanh')
    l3 = FC(l2, num_neurons=1, bias=True, weight_scale=1.0, xavier=True, activation_fn='Linear')
    layers = [l1, l2, l3]
    nn = NN(X, layers)

    # SGD
    sgd = SGD(nn, step_size=1e-3, reg_lambda=1e-1, regression=True)
    sgd.train(X, y, normalize='mean', epochs=50000, plot='SGD')

    # Adam
    nn.reinitialize_network()
    adam = Adam(nn, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-1, regression=True)
    adam.train(X, y, normalize='mean', epochs=50000, plot='Adam')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
