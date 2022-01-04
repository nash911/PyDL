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
from pydl.nn.lstm import LSTM
from pydl.nn.nn import NN
from pydl.training.momentum import Momentum
from pydl import conf

np.random.seed(11421111)


def generate_sine_data():
    T = 160
    L = 4096
    N = 3
    C = 10
    A = np.random.rand(N, 1, C)
    B = np.random.rand(N, 1, C)
    A0 = np.random.rand(1, N) / 2.0
    offset = np.random.randint(-5, 6, (1, N))
    scale = np.arange(0, 1, (1.0 / L)).reshape(-1, 1)

    x = (np.tile(np.array(range(L)), (N, 1)).reshape(N, L, 1) +
         np.random.randint(-4 * T, 4 * T, (N, C)).reshape(N, 1, C)) * \
        np.array(range(1, C + 1)).reshape(1, C)

    data = np.sum(A * np.sin(x / 1.0 / T).astype(conf.dtype) +
                  B * np.cos(x / 1.0 / T).astype(conf.dtype), axis=-1).transpose()
    data += A0
    data += offset

    # Scale the signals exponentially
    data *= np.square(scale)

    print("data.shape: ", data.shape)

    return data


def main():
    seq_len = 8
    num_neurons = 4
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

    l1 = LSTM(X, num_neurons=num_neurons, bias=1.0, seq_len=seq_len, weight_scale=weight_scale,
              xavier=True, tune_internal_states=True, name="LSTM-1")
    l2 = FC(l1, num_neurons=X.shape[-1], bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='Linear', name="Output-Layer")
    layers = [l1, l2]

    nn = NN(None, layers)

    momentum = Momentum(nn, step_size=1e-2, mu=0.5, reg_lambda=0, train_size=70, test_size=30,
                        regression=True)
    momentum.train_recurrent(X, batch_size=seq_len, epochs=500, sample_length=1000, log_freq=1,
                             normalize='mean', data_diff=True, fit_test_data=True,
                             plot='Sinusoid-LSTM - Momentum')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
