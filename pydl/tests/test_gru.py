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

from pydl.nn.gru import GRU
from pydl import conf

np.random.seed(11421111)


class TestGRU(unittest.TestCase):
    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-6
        tol = 8

        def test(inp, num_neur, w, bias, seq_len, inp_grad, init_hidden_state=None, p=None,
                 mask=None, architecture_type='many_to_many'):
            if type(bias) == int:
                bias = {'gates': np.ones(2 * num_neur) * bias,
                        'candidate': np.ones(num_neur) * bias}

            gru = GRU(inp, num_neur, w, bias, seq_len=seq_len, dropout=p,
                      tune_internal_states=(False if init_hidden_state is None else True),
                      architecture_type=architecture_type)
            if init_hidden_state is not None:
                gru.init_hidden_state = init_hidden_state
                gru.reset_internal_states()
            _ = gru.forward(inp, mask=mask)
            inputs_grad = gru.backward(inp_grad)
            weights_grad = gru.weights_grad
            bias_grad = gru.bias_grad
            hidden_grad = gru.hidden_state_grad

            gates_weights_grad = weights_grad[:, :num_neur]
            candidate_weights_grad = weights_grad[:, -num_neur:]
            gates_bias_grad = bias_grad[:num_neur]
            candidate_bias_grad = bias_grad[-num_neur:]

            gates_w = w['gates']
            cand_w = w['candidate']
            weights = np.concatenate((gates_w, cand_w), axis=-1)

            gates_bias = bias['gates']
            cand_bias = bias['candidate']
            bias = np.concatenate((gates_bias, cand_bias))

            # Gates weight finite difference gradients
            gates_weight_finite_diff = np.empty(gates_weights_grad.shape)
            for i in range(gates_weights_grad.shape[0]):
                for j in range(gates_weights_grad.shape[1]):
                    w_delta = np.zeros_like(gates_w)
                    w_delta[i, j] = self.delta
                    gru.gates_weights = gates_w + w_delta
                    lhs = copy.deepcopy(gru.forward(inp, mask=mask))
                    gru.gates_weights = gates_w - w_delta
                    rhs = copy.deepcopy(gru.forward(inp, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    gates_weight_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            gru.weights = weights

            # Candidate weights finite difference gradients
            cand_weight_finite_diff = np.empty(candidate_weights_grad.shape)
            for i in range(candidate_weights_grad.shape[0]):
                for j in range(candidate_weights_grad.shape[1]):
                    w_delta = np.zeros_like(cand_w)
                    w_delta[i, j] = self.delta
                    gru.candidate_weights = cand_w + w_delta
                    lhs = copy.deepcopy(gru.forward(inp, mask=mask))
                    gru.candidate_weights = cand_w - w_delta
                    rhs = copy.deepcopy(gru.forward(inp, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    cand_weight_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            gru.weights = weights

            # Gates bias finite difference gradients
            gates_bias_finite_diff = np.empty(gates_bias_grad.shape)
            for i in range(gates_bias_grad.shape[0]):
                bias_delta = np.zeros(gates_bias.shape, dtype=conf.dtype)
                bias_delta[i] = self.delta
                gru.gates_bias = gates_bias + bias_delta
                lhs = copy.deepcopy(gru.forward(inp, mask=mask))
                gru.gates_bias = gates_bias - bias_delta
                rhs = copy.deepcopy(gru.forward(inp, mask=mask))
                lhs_sum = np.zeros_like(list(lhs.values())[0])
                rhs_sum = np.zeros_like(list(rhs.values())[0])
                for k in list(lhs.keys()):
                    if k > 0:
                        lhs_sum += lhs[k] * inp_grad[k]
                        rhs_sum += rhs[k] * inp_grad[k]
                gates_bias_finite_diff[i] = \
                    np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            gru.bias = bias

            # Candidate bias finite difference gradients
            cand_bias_finite_diff = np.empty(candidate_bias_grad.shape)
            for i in range(candidate_bias_grad.shape[0]):
                bias_delta = np.zeros(cand_bias.shape, dtype=conf.dtype)
                bias_delta[i] = self.delta
                gru.candidate_bias = cand_bias + bias_delta
                lhs = copy.deepcopy(gru.forward(inp, mask=mask))
                gru.candidate_bias = cand_bias - bias_delta
                rhs = copy.deepcopy(gru.forward(inp, mask=mask))
                lhs_sum = np.zeros_like(list(lhs.values())[0])
                rhs_sum = np.zeros_like(list(rhs.values())[0])
                for k in list(lhs.keys()):
                    if k > 0:
                        lhs_sum += lhs[k] * inp_grad[k]
                        rhs_sum += rhs[k] * inp_grad[k]
                cand_bias_finite_diff[i] = \
                    np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            gru.bias = bias

            # Inputs finite difference gradients
            inputs_grad = np.vstack(reversed(list(inputs_grad.values())))
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                    i_delta[i, j] = self.delta
                    lhs = copy.deepcopy(gru.forward(inp + i_delta, mask=mask))
                    rhs = copy.deepcopy(gru.forward(inp - i_delta, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    inputs_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)), keepdims=False)

            if init_hidden_state is not None:
                # Initial hidden state finite difference gradients
                hidden_finite_diff = np.empty(hidden_grad.shape)
                for i in range(init_hidden_state.shape[0]):
                    for j in range(init_hidden_state.shape[1]):
                        h_delta = np.zeros(init_hidden_state.shape, dtype=conf.dtype)
                        h_delta[i, j] = self.delta
                        gru.init_hidden_state = init_hidden_state + h_delta
                        gru.reset_internal_states()
                        lhs = copy.deepcopy(gru.forward(inp, mask=mask))
                        gru.init_hidden_state = init_hidden_state - h_delta
                        gru.reset_internal_states()
                        rhs = copy.deepcopy(gru.forward(inp, mask=mask))
                        lhs_sum = np.zeros_like(list(lhs.values())[0])
                        rhs_sum = np.zeros_like(list(rhs.values())[0])
                        for k in list(lhs.keys()):
                            if k > 0:
                                lhs_sum += lhs[k] * inp_grad[k]
                                rhs_sum += rhs[k] * inp_grad[k]
                        hidden_finite_diff[i, j] = \
                            np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)), keepdims=False)
                gru.init_hidden_state = init_hidden_state
                gru.reset_internal_states()

            npt.assert_almost_equal(gates_weights_grad, gates_weight_finite_diff, decimal=tol)
            npt.assert_almost_equal(candidate_weights_grad, cand_weight_finite_diff, decimal=tol)
            npt.assert_almost_equal(gates_bias_grad, gates_bias_finite_diff, decimal=tol)
            npt.assert_almost_equal(candidate_bias_grad, cand_bias_finite_diff, decimal=tol)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=tol)
            if init_hidden_state is not None:
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)

            # # Weights gradient check
            # tol = 1e-6
            # grad_diff = (abs(gates_weights_grad - gates_weight_finite_diff) /
            #              (abs(gates_weights_grad + gates_weight_finite_diff) + 1e-64))
            # error_threshold = np.ones_like(grad_diff) * tol
            # npt.assert_array_less(grad_diff, error_threshold)
            #
            # grad_diff = (abs(candidate_weights_grad - cand_weight_finite_diff) /
            #              (abs(candidate_weights_grad + cand_weight_finite_diff) + 1e-64))
            # error_threshold = np.ones_like(grad_diff) * tol
            # npt.assert_array_less(grad_diff, error_threshold)
            #
            # grad_diff = (abs(gates_bias_grad - gates_bias_finite_diff) /
            #              (abs(gates_bias_grad + gates_bias_finite_diff) + 1e-64))
            # error_threshold = np.ones_like(grad_diff) * tol
            # npt.assert_array_less(grad_diff, error_threshold)
            #
            # grad_diff = (abs(candidate_bias_grad - cand_bias_finite_diff) /
            #              (abs(candidate_bias_grad + cand_bias_finite_diff) + 1e-64))
            # error_threshold = np.ones_like(grad_diff) * tol
            # npt.assert_array_less(grad_diff, error_threshold)
            #
            # # Inputs gradient check
            # grad_diff = (abs(inputs_grad - inputs_finite_diff) /
            #              (abs(inputs_grad + inputs_finite_diff) + 1e-64))
            # error_threshold = np.ones_like(grad_diff) * tol
            # npt.assert_array_less(grad_diff, error_threshold)
            #
            # # Hidden gradient check
            # if init_hidden_state is not None:
            #     grad_diff = (abs(hidden_grad - hidden_finite_diff) /
            #                  (abs(hidden_grad + hidden_finite_diff) + 1e-64))
            #     error_threshold = np.ones_like(grad_diff) * tol
            #     npt.assert_array_less(grad_diff, error_threshold)

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
            w = OrderedDict()
            w['gates'] = np.random.rand((neur + feat), (2 * neur)) * scl
            w['candidate'] = np.random.rand((neur + feat), neur) * scl
            if b == 'rand':
                bias = OrderedDict()
                bias['gates'] = np.random.rand(2 * neur) * scl
                bias['candidate'] = np.random.rand(neur) * scl
            else:
                bias = b

            init_h_state = np.random.rand(1, neur) if tune else None

            # Initialize input gradients
            inp_grad = OrderedDict()
            if a_type == 'many_to_many':
                for s in range(1, batch_size + 1):
                    inp_grad[s] = np.ones((1, neur), dtype=conf.dtype) if unit else \
                        np.random.uniform(-1, 1, (1, neur)) * scl
            else:
                inp_grad[batch_size] = np.ones((1, neur), dtype=conf.dtype) if unit else \
                    np.random.uniform(-1, 1, (1, neur)) * scl

            # Set dropout mask
            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(batch_size, neur) < p, dtype=conf.dtype)
            else:
                p = None
                mask = None

            test(X, neur, w, bias, seq_len, inp_grad, init_h_state, p, mask, a_type)


if __name__ == '__main__':
    unittest.main()
