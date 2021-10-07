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
import numpy.testing as npt
import itertools
from collections import OrderedDict
import copy

from pydl.nn.lstm import LSTM
from pydl import conf


class TestLSTM(unittest.TestCase):
    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-6
        tol = 8

        def test(inp, num_neur, w, bias, seq_len, inp_grad, p=None, mask=None,
                 architecture_type='many_to_many'):
            if type(bias) == int:
                bias = np.ones(4 * num_neur) * bias
            lstm = LSTM(inp, num_neur, w, bias, seq_len=seq_len, dropout=p,
                        architecture_type=architecture_type)
            _ = lstm.forward(inp, mask=mask)
            inputs_grad = lstm.backward(inp_grad)
            weights_grad = lstm.weights_grad
            bias_grad = lstm.bias_grad

            # Weights finite difference gradients
            weights_finite_diff = np.empty(weights_grad.shape)
            for i in range(weights_grad.shape[0]):
                for j in range(weights_grad.shape[1]):
                    w_delta = np.zeros_like(w)
                    w_delta[i, j] = self.delta
                    lstm.weights = w + w_delta
                    lhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                    lstm.weights = w - w_delta
                    rhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    weights_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            lstm.weights = w

            # Bias finite difference gradients
            bias_finite_diff = np.empty(bias_grad.shape)
            for i in range(bias_grad.shape[0]):
                bias_delta = np.zeros(bias.shape, dtype=conf.dtype)
                bias_delta[i] = self.delta
                lstm.bias = bias + bias_delta
                lhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                lstm.bias = bias - bias_delta
                rhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                lhs_sum = np.zeros_like(list(lhs.values())[0])
                rhs_sum = np.zeros_like(list(rhs.values())[0])
                for k in list(lhs.keys()):
                    if k > 0:
                        lhs_sum += lhs[k] * inp_grad[k]
                        rhs_sum += rhs[k] * inp_grad[k]
                bias_finite_diff[i] = \
                    np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            lstm.bias = bias

            # Inputs finite difference gradients
            inputs_grad = np.vstack(reversed(list(inputs_grad.values())))
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                    i_delta[i, j] = self.delta
                    lhs = copy.deepcopy(lstm.forward(inp + i_delta, mask=mask))
                    rhs = copy.deepcopy(lstm.forward(inp - i_delta, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    inputs_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)), keepdims=False)

            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=tol)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=tol)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=tol)

            # # Weights gradient check
            # grad_diff = (abs(weights_grad - weights_finite_diff) /
            #              (abs(weights_grad + weights_finite_diff) + 1e-64))
            # error_threshold = np.ones_like(grad_diff) * 1e-6
            # npt.assert_array_less(grad_diff, error_threshold)
            #
            # # Inputs gradient check
            # grad_diff = (abs(inputs_grad - inputs_finite_diff) /
            #              (abs(inputs_grad + inputs_finite_diff) + 1e-64))
            # error_threshold = np.ones_like(grad_diff) * 1e-6
            # npt.assert_array_less(grad_diff, error_threshold)

        # Combinatorial Test Cases
        # ------------------------
        sequence_length = [1, 2, 3, 11]
        feature_size = [1, 2, 3, 11]
        num_neurons = [1, 2, 3, 11]
        bias = [10, 'rand']
        one_hot = [True, False]
        scale = [1e-2, 1e+0]
        unit_inp_grad = [True, False]
        dropout = [False]
        architecture_type = ['many_to_many', 'many_to_one']
        repeat = list(range(1))

        for seq_len, feat, neur, b, oh, scl, unit, dout, a_type, r in \
            list(itertools.product(sequence_length, feature_size, num_neurons, bias, one_hot, scale,
                                   unit_inp_grad, dropout, architecture_type, repeat)):

            # Initialize inputs
            if oh:
                X = np.zeros((seq_len, feat), dtype=conf.dtype)
                rnd_idx = np.random.randint(feat, size=seq_len)
                X[range(seq_len), rnd_idx] = 1
            else:
                X = np.random.uniform(-scl, scl, (seq_len, feat))

            # Initialize weights and bias
            w = np.random.rand((neur + feat), (4 * neur)) * scl
            if b == 'rand':
                bias = np.random.rand(4 * neur) * scl
            else:
                bias = b

            # Initialize input gradients
            inp_grad = OrderedDict()
            if a_type == 'many_to_many':
                for s in range(1, seq_len + 1):
                    inp_grad[s] = np.ones((1, neur), dtype=conf.dtype) if unit else \
                        np.random.uniform(-1, 1, (1, neur))
            else:
                inp_grad[seq_len] = np.ones((1, neur), dtype=conf.dtype) if unit else \
                    np.random.uniform(-1, 1, (1, neur))

            # Set dropout mask
            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(seq_len, neur) < p, dtype=conf.dtype)
            else:
                p = None
                mask = None

            test(X, neur, w, bias, seq_len, inp_grad, p, mask, a_type)


if __name__ == '__main__':
    unittest.main()
