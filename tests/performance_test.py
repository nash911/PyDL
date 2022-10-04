# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import unittest
import numpy as np
import time

from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.rnn import RNN
from pydl.nn.lstm import LSTM
from pydl.nn.nn import NN


class PerformanceTest(unittest.TestCase):
    def performance_test_convolution_pooling(self):
        def test(nn, inp, i):
            start = time.time()
            nn_out = nn.forward(inp)
            end = time.time()
            print("%d - Forward Time: %.8f" % (i, (end - start) * 16))
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            _ = nn.backward(inp_grad)
            print("%d - Backward Time: %.8f" % (i, (time.time() - end) * 16))
            nn.update_weights(1e-4)

        print("CNN - Performance Test")
        for _ in range(1):
            # Inputs
            batch_size = 16
            depth = 3
            num_rows = 32
            num_cols = 32
            X = np.empty((batch_size, depth, num_rows, num_cols))
            bn = False

            # Case-1
            # ------
            # NN Architecture
            # Layer 1 - Convolution
            num_kernals_1 = 16
            rec_h_1 = 5
            rec_w_1 = 5
            pad_1 = 2
            stride_1 = 1
            w_1 = np.random.randn(num_kernals_1, depth, rec_h_1, rec_w_1)
            b_1 = np.random.uniform(-1, 1, (num_kernals_1))
            dp1 = None
            l1 = Conv(X, weights=w_1, bias=b_1, zero_padding=pad_1, stride=stride_1,
                      activation_fn='ReLU', name='Conv-1', batchnorm=bn, dropout=dp1)

            # Layer 2 - MaxPool
            rec_h_2 = 2
            rec_w_2 = 2
            stride_2 = 2
            l2 = Pool(l1, receptive_field=(rec_h_2, rec_w_2), stride=stride_2, name='MaxPool-2')

            # Layer 3 - Convolution
            num_kernals_3 = 20
            rec_h_3 = 5
            rec_w_3 = 5
            pad_3 = 2
            stride_3 = 1
            w_3 = np.random.randn(num_kernals_3, num_kernals_1, rec_h_3, rec_w_3)
            b_3 = np.random.uniform(-1, 1, (num_kernals_3))
            dp3 = None
            l3 = Conv(l2, weights=w_3, bias=b_3, zero_padding=pad_3, stride=stride_3,
                      activation_fn='ReLU', name='Conv-3', batchnorm=bn, dropout=dp3)

            # Layer 4 - MaxPool
            rec_h_4 = 2
            rec_w_4 = 2
            stride_4 = 2
            l4 = Pool(l3, receptive_field=(rec_h_4, rec_w_4), stride=stride_4, pool='MAX',
                      name='MaxPool-4')

            # Layer 5 - Convolution
            num_kernals_5 = 20
            rec_h_5 = 5
            rec_w_5 = 5
            pad_5 = 2
            stride_5 = 1
            w_5 = np.random.randn(num_kernals_5, num_kernals_3, rec_h_5, rec_w_5)
            b_5 = np.random.uniform(-1, 1, (num_kernals_5))
            dp5 = None
            l5 = Conv(l4, weights=w_5, bias=b_5, zero_padding=pad_5, stride=stride_5,
                      activation_fn='ReLU', name='Conv-5', batchnorm=bn, dropout=dp5)

            # Layer 6 - MaxPool
            rec_h_6 = 2
            rec_w_6 = 2
            stride_6 = 2
            l6 = Pool(l5, receptive_field=(rec_h_6, rec_w_6), stride=stride_6, pool='AVG',
                      name='MaxPool-6')

            # Layer 7 - SoftMax
            w_7 = np.random.randn(np.prod(l6.shape[1:]), 10)
            b_7 = np.random.uniform(-1, 1, (10))
            l7 = FC(l6, num_neurons=10, weights=w_7, bias=b_7, activation_fn='SoftMax',
                    name='Softmax-7')

            layers = [l1, l2, l3, l4, l5, l6, l7]
            nn = NN(X, layers)

            for i in range(5):
                X = np.random.uniform(-1, 1, (batch_size, depth, num_rows, num_cols))
                test(nn, X, i)

    def performance_test_rnn(self):
        def test(nn, inp, i):
            start = time.time()
            nn_out = nn.forward(inp)
            end = time.time()
            print("%d - Forward Time: %.8f" % (i, (end - start) * 16))
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            _ = nn.backward(inp_grad)
            print("%d - Backward Time: %.8f" % (i, (time.time() - end) * 16))
            nn.update_weights(1e-4)

        print("RNN - Performance Test")
        for _ in range(1):
            # Layer 1 - RNN
            # -------------
            seq_len = 100
            inp_feat = 400
            num_neur_rnn_1 = 300
            X = np.empty((seq_len, inp_feat))
            wh_1 = np.random.randn(num_neur_rnn_1, num_neur_rnn_1) * 0.01
            wx_1 = np.random.randn(X.shape[-1], num_neur_rnn_1) * 0.01
            w_1 = {'hidden': wh_1, 'inp': wx_1}
            b_1 = np.random.uniform(-1, 1, (num_neur_rnn_1)) * 0.01
            l1 = RNN(X, num_neur_rnn_1, w_1, b_1, seq_len=seq_len, activation_fn='Sigmoid',
                     name='RNN-1')

            # Layer 2 - RNN
            # -------------
            num_neur_rnn_2 = 150
            wh_2 = np.random.randn(num_neur_rnn_2, num_neur_rnn_2) * 0.01
            wx_2 = np.random.randn(l1.shape[-1], num_neur_rnn_2) * 0.01
            w_2 = {'hidden': wh_2, 'inp': wx_2}
            b_2 = np.random.uniform(-1, 1, (num_neur_rnn_2)) * 0.01
            l2 = RNN(l1, num_neur_rnn_2, w_2, b_2, seq_len=seq_len, activation_fn='Sigmoid',
                     name='RNN-2')

            # Layer 3 - RNN
            # -------------
            num_neur_rnn_3 = 100
            wh_3 = np.random.randn(num_neur_rnn_3, num_neur_rnn_3) * 0.01
            wx_3 = np.random.randn(l2.shape[-1], num_neur_rnn_3) * 0.01
            w_3 = {'hidden': wh_3, 'inp': wx_3}
            b_3 = np.random.uniform(-1, 1, (num_neur_rnn_3)) * 0.01
            l3 = RNN(l2, num_neur_rnn_3, w_3, b_3, seq_len=seq_len, activation_fn='Sigmoid',
                     name='RNN-3')

            # Layer 4 - FC-Softmax
            # --------------------
            num_neur_out = 200
            w_4 = np.random.randn(l3.shape[-1], num_neur_out) * 0.01
            b_4 = np.random.uniform(-1, 1, (num_neur_out)) * 0.01
            l4 = FC(l3, num_neur_out, w_4, b_4, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3, l4]
            nn = NN(X, layers)

            for i in range(5):
                X = np.random.uniform(-1, 1, (seq_len, inp_feat)) * 0.01
                test(nn, X, i)

    def performance_test_lstm(self):
        def test(nn, inp, i):
            start = time.time()
            nn_out = nn.forward(inp)
            end = time.time()
            print("%d - Forward Time: %.8f" % (i, (end - start) * 16))
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            _ = nn.backward(inp_grad)
            print("%d - Backward Time: %.8f" % (i, (time.time() - end) * 16))
            nn.update_weights(1e-4)

        print("LSTM - Performance Test")
        for _ in range(1):
            # Layer 1 - LSTM
            # --------------
            seq_len = 100
            inp_feat = 400
            num_neur_lstm_1 = 300
            X = np.empty((seq_len, inp_feat))
            w_1 = np.random.randn((num_neur_lstm_1 + X.shape[-1]), (4 * num_neur_lstm_1)) * 0.01
            b_1 = np.random.uniform(-1, 1, (4 * num_neur_lstm_1)) * 0.01
            l1 = LSTM(X, num_neur_lstm_1, w_1, b_1, seq_len, name='LSTM-1')

            # Layer 2 - LSTM
            # --------------
            num_neur_lstm_2 = 150
            w_2 = np.random.randn((num_neur_lstm_2 + l1.shape[-1]), (4 * num_neur_lstm_2)) * 0.01
            b_2 = np.random.uniform(-1, 1, (4 * num_neur_lstm_2)) * 0.01
            l2 = LSTM(l1, num_neur_lstm_2, w_2, b_2, seq_len, name='LSTM-2')

            # Layer 3 - LSTM
            # --------------
            num_neur_lstm_3 = 100
            w_3 = np.random.randn((num_neur_lstm_3 + l2.shape[-1]), (4 * num_neur_lstm_3)) * 0.01
            b_3 = np.random.uniform(-1, 1, (4 * num_neur_lstm_3)) * 0.01
            l3 = LSTM(l2, num_neur_lstm_3, w_3, b_3, seq_len, name='LSTM-3')

            # Layer 4 - FC-Softmax
            # --------------------
            num_neur_out = 200
            w_4 = np.random.randn(l3.shape[-1], num_neur_out) * 0.01
            b_4 = np.random.uniform(-1, 1, (num_neur_out)) * 0.01
            l4 = FC(l3, num_neur_out, w_4, b_4, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3, l4]
            nn = NN(X, layers)

            for i in range(5):
                X = np.random.uniform(-1, 1, (seq_len, inp_feat)) * 0.01
                test(nn, X, i)


if __name__ == '__main__':
    unittest.main()
