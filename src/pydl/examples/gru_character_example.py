# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np

from pydl.nn.layers import FC
from pydl.nn.gru import GRU
from pydl.nn.nn import NN
from pydl.training.rmsprop import RMSprop
from pydl import conf

np.random.seed(11421111)


def get_data(file_path, seq_len):
    data = open(file_path, 'r').read()
    unique_chars = list(set(data))
    K = len(unique_chars)
    X = np.zeros((1, K), dtype=conf.dtype)

    return data, X, K


def main():
    seq_len = 50
    weight_scale = 1e-2

    data, X, K = get_data('data/paulgraham_essays.txt', seq_len)

    print("X.shape: ", X.shape)
    print("K: ", K)

    l1 = GRU(X, num_neurons=200, bias=1.0, seq_len=seq_len, weight_scale=weight_scale, xavier=True,
             tune_internal_states=True, name="RNN-1")
    l2 = FC(l1, num_neurons=K, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='SoftMax', name="Output-Layer")
    layers = [l1, l2]

    nn = NN(None, layers)

    rms = RMSprop(nn, step_size=1e-3, beta=0.9, reg_lambda=0, train_size=90, test_size=10)
    rms.train_recurrent(data, batch_size=seq_len, epochs=10000, sample_length=1000, temperature=0.5,
                        log_freq=1, plot='Character-GRU - RMSprop - Tune Hidden')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
