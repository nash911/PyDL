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

        def test(inp, num_neur, w, bias, seq_len, inp_grad, init_cell_state=None,
                 init_hidden_state=None, p=None, mask=None, architecture_type='many_to_many'):
            if type(bias) == int:
                bias = np.ones(4 * num_neur) * bias
            lstm = LSTM(inp, num_neur, w, bias, seq_len=seq_len, dropout=p,
                        tune_internal_states=(False if init_hidden_state is None else True),
                        architecture_type=architecture_type)
            if init_cell_state is not None:
                lstm.init_cell_state = init_cell_state
                lstm.init_hidden_state = init_hidden_state
                lstm.reset_internal_states()
            _ = lstm.forward(inp, mask=mask)
            inputs_grad = lstm.backward(inp_grad)
            weights_grad = lstm.weights_grad
            bias_grad = lstm.bias_grad
            cell_grad = lstm.cell_state_grad
            hidden_grad = lstm.hidden_state_grad

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

            if init_cell_state is not None:
                # Initial cell state finite difference gradients
                cell_finite_diff = np.empty(cell_grad.shape)
                for i in range(init_cell_state.shape[0]):
                    for j in range(init_cell_state.shape[1]):
                        h_delta = np.zeros(init_cell_state.shape, dtype=conf.dtype)
                        h_delta[i, j] = self.delta
                        lstm.init_cell_state = init_cell_state + h_delta
                        lstm.reset_internal_states()
                        lhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                        lstm.init_cell_state = init_cell_state - h_delta
                        lstm.reset_internal_states()
                        rhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                        lhs_sum = np.zeros_like(list(lhs.values())[0])
                        rhs_sum = np.zeros_like(list(rhs.values())[0])
                        for k in list(lhs.keys()):
                            if k > 0:
                                lhs_sum += lhs[k] * inp_grad[k]
                                rhs_sum += rhs[k] * inp_grad[k]
                        cell_finite_diff[i, j] = \
                            np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)), keepdims=False)
                lstm.init_cell_state = init_cell_state
                lstm.reset_internal_states()

            if init_hidden_state is not None:
                # Initial hidden state finite difference gradients
                hidden_finite_diff = np.empty(hidden_grad.shape)
                for i in range(init_hidden_state.shape[0]):
                    for j in range(init_hidden_state.shape[1]):
                        h_delta = np.zeros(init_hidden_state.shape, dtype=conf.dtype)
                        h_delta[i, j] = self.delta
                        lstm.init_hidden_state = init_hidden_state + h_delta
                        lstm.reset_internal_states()
                        lhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                        lstm.init_hidden_state = init_hidden_state - h_delta
                        lstm.reset_internal_states()
                        rhs = copy.deepcopy(lstm.forward(inp, mask=mask))
                        lhs_sum = np.zeros_like(list(lhs.values())[0])
                        rhs_sum = np.zeros_like(list(rhs.values())[0])
                        for k in list(lhs.keys()):
                            if k > 0:
                                lhs_sum += lhs[k] * inp_grad[k]
                                rhs_sum += rhs[k] * inp_grad[k]
                        hidden_finite_diff[i, j] = \
                            np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)), keepdims=False)
                lstm.init_hidden_state = init_hidden_state
                lstm.reset_internal_states()

            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=tol)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=tol)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=tol)
            if init_cell_state is not None:
                npt.assert_almost_equal(cell_grad, cell_finite_diff, decimal=tol)
            if init_hidden_state is not None:
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)

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
        sequence_length = [1, 2, 3, 4]
        reduce_size = [0, 1]
        feature_size = [1, 2, 3, 4]
        num_neurons = [1, 2, 3, 4]
        bias = [10, 'rand']
        one_hot = [True, False]
        scale = [1e-2, 1e+0]
        unit_inp_grad = [True, False]
        dropout = [True, False]
        architecture_type = ['many_to_many', 'many_to_one']
        tune_internal_states = [True, False]
        repeat = list(range(1))

        for seq_len, r_size, feat, neur, b, oh, scl, unit, dout, a_type, tune, r in \
            list(itertools.product(sequence_length, reduce_size, feature_size, num_neurons, bias,
                                   one_hot, scale, unit_inp_grad, dropout, architecture_type,
                                   tune_internal_states, repeat)):

            batch_size = seq_len - (r_size if seq_len > 1 else 0)

            # Initialize inputs
            if oh:
                X = np.zeros((batch_size, feat), dtype=conf.dtype)
                rnd_idx = np.random.randint(feat, size=batch_size)
                X[range(batch_size), rnd_idx] = 1
            else:
                X = np.random.uniform(-scl, scl, (batch_size, feat))

            # Initialize weights and bias
            w = np.random.rand((neur + feat), (4 * neur)) * scl
            if b == 'rand':
                bias = np.random.rand(4 * neur) * scl
            else:
                bias = b
            init_c_state = np.random.rand(1, neur) if tune else None
            init_h_state = np.random.rand(1, neur) if tune else None

            # Initialize input gradients
            inp_grad = OrderedDict()
            if a_type == 'many_to_many':
                for s in range(1, batch_size + 1):
                    inp_grad[s] = np.ones((1, neur), dtype=conf.dtype) if unit else \
                        np.random.uniform(-1, 1, (1, neur))
            else:
                inp_grad[batch_size] = np.ones((1, neur), dtype=conf.dtype) if unit else \
                    np.random.uniform(-1, 1, (1, neur))

            # Set dropout mask
            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(batch_size, neur) < p, dtype=conf.dtype)
            else:
                p = None
                mask = None

            test(X, neur, w, bias, seq_len, inp_grad, init_c_state, init_h_state, p, mask, a_type)


if __name__ == '__main__':
    unittest.main()
