# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from pydl.nn.layers import FC
from pydl.nn.rnn import RNN
from pydl.nn.nn import NN
from pydl.training.momentum import Momentum
from pydl import conf

np.random.seed(11421111)


def generate_sine_data():
    T = 20
    L = 2084
    N = 3
    A = np.random.rand(1, N)

    x = np.array(range(L)).reshape(L, 1) + np.random.randint(-4 * T, 4 * T, N).reshape(1, N)
    data = A * np.sin(x / 1.0 / T).astype(conf.dtype)

    return data


def main():
    seq_len = 2
    weight_scale = 1e-2

    X = generate_sine_data()
    print("X.shape: ", X.shape)

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for c in range(X.shape[-1]):
        plt.plot(X[:, c], linestyle='-', color=colors[c])

    # show the figure
    plt.draw()
    plt.waitforbuttonpress(10)
    plt.close()

    l1 = RNN(X, num_neurons=5, bias=True, seq_len=seq_len, weight_scale=weight_scale, xavier=True,
             activation_fn='Tanh', tune_internal_states=True, name="RNN-1")
    l2 = FC(l1, num_neurons=X.shape[-1], bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='Linear', name="Output-Layer")
    layers = [l1, l2]

    nn = NN(None, layers)

    momentum = Momentum(nn, step_size=1e-2, mu=0.5, reg_lambda=0, train_size=80, test_size=20,
                        regression=True)
    momentum.train_recurrent(X, batch_size=seq_len, epochs=100, sample_length=1000, log_freq=1,
                             normalize=None, fit_test_data=True, plot='Sinusoid-RNN - Momentum')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
