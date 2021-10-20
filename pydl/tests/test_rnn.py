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

from pydl.nn.rnn import RNN
from pydl import conf


class TestRNN(unittest.TestCase):
    def test_score_fn(self):
        def test(inp, w, seq_len, true_out, bias=False):
            num_neur = w['hidden'].shape[0]
            rnn = RNN(inp, num_neur, w, bias, seq_len)
            out_rnn = np.zeros((1, num_neur), dtype=conf.dtype)
            for _ in range(seq_len):
                out_rnn = rnn.score_fn({'h': out_rnn, 'inp': inp})
            npt.assert_almost_equal(out_rnn, true_out, decimal=5)

        # Manually calculated
        # -------------------
        X = np.ones((1, 3), dtype=conf.dtype)
        wh = np.ones((7, 7), dtype=conf.dtype)
        wx = np.random.rand(3, 7)
        w = {'hidden': wh, 'inp': wx}
        bias = np.random.rand(7)
        true_out = np.array([np.sum(wx) + np.sum(bias)] * 7).reshape(1, -1) + \
            np.sum(wx, axis=0, keepdims=True) + bias
        test(X, w, seq_len=2, true_out=true_out, bias=bias)

        # Combinatorial Test Cases
        # ------------------------
        feature_size = [1, 2, 3, 5, 6, 11]
        num_neurons = [1, 2, 3, 5, 6, 11]
        scale = [1e-6, 1e-3, 1e-1, 1e-0, 2, 3, 10]
        batch = 1

        for feat, neur, scl in list(itertools.product(feature_size, num_neurons, scale)):
            X = np.ones((batch, feat), dtype=conf.dtype)
            wh = np.ones((neur, neur), dtype=conf.dtype)
            wx = np.random.rand(feat, neur) * scl
            w = {'hidden': wh, 'inp': wx}
            bias = np.random.rand(neur) * scl
            true_out = np.array([np.sum(wx) + np.sum(bias)] * neur).reshape(1, -1) + \
                np.sum(wx, axis=0, keepdims=True) + bias
            test(X, w, seq_len=2, true_out=true_out, bias=bias)

    def test_forward(self):
        def test(inp, w, seq_len, true_out, bias=False, init_h_state=None, actv_fn='Sigmoid',
                 p=None, mask=None, architecture_type='many_to_many'):
            num_neur = w['hidden'].shape[0]
            rnn = RNN(inp, num_neur, w, bias, seq_len=seq_len, activation_fn=actv_fn,
                      architecture_type=architecture_type, dropout=p, tune_internal_states=True)
            rnn.init_hidden_state = init_h_state
            out_rnn = rnn.forward(inp, mask=mask)

            # Check if the output has the right keys
            npt.assert_equal(out_rnn.keys(), true_out.keys())

            for k, v in out_rnn.items():
                npt.assert_almost_equal(v, true_out[k], decimal=5)

        # Combinatorial Test Cases
        # ------------------------
        sequence_length = [1, 2, 3, 5, 6, 11]
        reduce_size = [0, 1]
        feature_size = [1, 2, 3, 5, 6, 11]
        num_neurons = [1, 2, 3, 5, 6, 11]
        one_hot = [True, False]
        scale = [1e-6, 1e-3, 1e-1, 1e-0, 2]
        dropout = [True, False]
        architecture_type = ['many_to_many', 'many_to_one']

        for seq_len, r_size, feat, neur, oh, scl, dout, a_type in list(itertools.product(
                sequence_length, reduce_size, feature_size, num_neurons, one_hot, scale, dropout,
                architecture_type)):
            batch_size = seq_len - (r_size if seq_len > 1 else 0)
            if oh:
                X = np.zeros((batch_size, feat), dtype=conf.dtype)
                rnd_idx = np.random.randint(feat, size=batch_size)
                X[range(batch_size), rnd_idx] = 1
            else:
                X = np.random.uniform(-scl, scl, (batch_size, feat))

            wh = np.random.rand(neur, neur) * scl
            wx = np.random.rand(feat, neur) * scl
            w = {'hidden': wh, 'inp': wx}
            bias = np.random.rand(neur) * scl

            # Linear
            h_init = np.array(np.random.rand(1, neur), dtype=conf.dtype) * scl
            h = np.copy(h_init)
            true_out_linear = OrderedDict()
            p = None
            mask = None
            for i, x in enumerate(X):
                h = np.matmul(h, wh) + np.matmul(x.reshape(1, -1), wx) + bias
                if dout:
                    if p is None:
                        p = np.random.rand()
                        mask = list()
                    mask.append(np.array(np.random.rand(*h.shape) < p, dtype=conf.dtype) / p)
                    layer_out = h * mask[-1]
                else:
                    layer_out = h
                if a_type == 'many_to_one':
                    if i == batch_size - 1:
                        true_out_linear = OrderedDict()
                        true_out_linear[i + 1] = layer_out
                else:
                    true_out_linear[i + 1] = layer_out
            test(X, w, seq_len, true_out_linear, bias, h_init, actv_fn='Linear', p=p, mask=mask,
                 architecture_type=a_type)

            # Sigmoid
            h_init = np.array(np.random.rand(1, neur), dtype=conf.dtype) * scl
            h = np.copy(h_init)
            true_out_sigmoid = OrderedDict()
            p = None
            mask = None
            for i, x in enumerate(X):
                score = np.matmul(h, wh) + np.matmul(x.reshape(1, -1), wx) + bias
                h = 1.0 / (1.0 + np.exp(-score))
                if dout:
                    if p is None:
                        p = np.random.rand()
                        mask = list()
                    mask.append(np.array(np.random.rand(*h.shape) < p, dtype=conf.dtype))
                    layer_out = h * mask[-1]
                else:
                    layer_out = h
                if a_type == 'many_to_one':
                    if i == batch_size - 1:
                        true_out_sigmoid = OrderedDict()
                        true_out_sigmoid[i + 1] = layer_out
                else:
                    true_out_sigmoid[i + 1] = layer_out
            test(X, w, seq_len, true_out_sigmoid, bias, h_init, actv_fn='Sigmoid', p=p, mask=mask,
                 architecture_type=a_type)

            # Tanh
            h_init = np.array(np.random.rand(1, neur), dtype=conf.dtype) * scl
            h = np.copy(h_init)
            true_out_tanh = OrderedDict()
            p = None
            mask = None
            for i, x in enumerate(X):
                score = np.matmul(h, wh) + np.matmul(x.reshape(1, -1), wx) + bias
                h = (2.0 / (1.0 + np.exp(-2.0 * score))) - 1.0
                if dout:
                    if p is None:
                        p = np.random.rand()
                        mask = list()
                    mask.append(np.array(np.random.rand(*h.shape) < p, dtype=conf.dtype))
                    layer_out = h * mask[-1]
                else:
                    layer_out = h
                if a_type == 'many_to_one':
                    if i == batch_size - 1:
                        true_out_tanh = OrderedDict()
                        true_out_tanh[i + 1] = layer_out
                else:
                    true_out_tanh[i + 1] = layer_out
            test(X, w, seq_len, true_out_tanh, bias, h_init, actv_fn='Tanh', p=p, mask=mask,
                 architecture_type=a_type)

            # ReLU
            h_init = np.array(np.random.rand(1, neur), dtype=conf.dtype) * scl
            h = np.copy(h_init)
            true_out_relu = OrderedDict()
            p = None
            mask = None
            for i, x in enumerate(X):
                score = np.matmul(h, wh) + np.matmul(x.reshape(1, -1), wx) + bias
                h = np.maximum(0, score)
                if dout:
                    if p is None:
                        p = np.random.rand()
                        mask = list()
                    mask.append(np.array(np.random.rand(*h.shape) < p, dtype=conf.dtype) / p)
                    layer_out = h * mask[-1]
                else:
                    layer_out = h
                if a_type == 'many_to_one':
                    if i == batch_size - 1:
                        true_out_relu = OrderedDict()
                        true_out_relu[i + 1] = layer_out
                else:
                    true_out_relu[i + 1] = layer_out
            test(X, w, seq_len, true_out_relu, bias, h_init, actv_fn='ReLU', p=p, mask=mask,
                 architecture_type=a_type)

            # SoftMax
            h_init = np.array(np.random.rand(1, neur), dtype=conf.dtype) * scl
            h = np.copy(h_init)
            true_out_softmax = OrderedDict()
            p = None
            mask = None
            for i, x in enumerate(X):
                score = np.matmul(h, wh) + np.matmul(x.reshape(1, -1), wx) + bias
                unnorm_prob = np.exp(score)
                h = unnorm_prob / np.sum(unnorm_prob, axis=-1, keepdims=True)
                if dout:
                    if p is None:
                        p = np.random.rand()
                        mask = list()
                    mask.append(np.array(np.random.rand(*h.shape) < p, dtype=conf.dtype))
                    layer_out = h * mask[-1]
                else:
                    layer_out = h
                if a_type == 'many_to_one':
                    if i == batch_size - 1:
                        true_out_softmax = OrderedDict()
                        true_out_softmax[i + 1] = layer_out
                else:
                    true_out_softmax[i + 1] = layer_out
            test(X, w, seq_len, true_out_softmax, bias, h_init, actv_fn='Softmax', p=p, mask=mask,
                 architecture_type=a_type)

    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-6
        tol = 8

        def test(inp, w, seq_len, inp_grad, bias=False, init_hidden_state=None, actv_fn='Sigmoid',
                 p=None, mask=None, architecture_type='many_to_many'):
            num_neur = w['hidden'].shape[0]
            wh = w['hidden']
            wx = w['inp']
            rnn = RNN(inp, num_neur, w, bias, seq_len=seq_len, activation_fn=actv_fn,
                      architecture_type=architecture_type, tune_internal_states=True, dropout=p)
            rnn.init_hidden_state = init_hidden_state
            _ = rnn.forward(inp, mask=mask)
            inputs_grad = rnn.backward(inp_grad)
            hidden_weights_grad = rnn.hidden_weights_grad
            input_weights_grad = rnn.input_weights_grad
            bias_grad = rnn.bias_grad
            hidden_grad = rnn.hidden_state_grad

            # Hidden weights finite difference gradients
            hidden_weights_finite_diff = np.empty(hidden_weights_grad.shape)
            for i in range(hidden_weights_grad.shape[0]):
                for j in range(hidden_weights_grad.shape[1]):
                    w_delta = np.zeros_like(wh)
                    w_delta[i, j] = self.delta
                    rnn.hidden_weights = wh + w_delta
                    lhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                    rnn.hidden_weights = wh - w_delta
                    rhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    hidden_weights_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            rnn.hidden_weights = wh

            # Input weights finite difference gradients
            input_weights_finite_diff = np.empty(input_weights_grad.shape)
            for i in range(input_weights_grad.shape[0]):
                for j in range(input_weights_grad.shape[1]):
                    w_delta = np.zeros_like(wx)
                    w_delta[i, j] = self.delta
                    rnn.input_weights = wx + w_delta
                    lhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                    rnn.input_weights = wx - w_delta
                    rhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    input_weights_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            rnn.input_weights = wx

            # Bias finite difference gradients
            bias_finite_diff = np.empty(bias_grad.shape)
            for i in range(bias_grad.shape[0]):
                bias_delta = np.zeros(bias.shape, dtype=conf.dtype)
                bias_delta[i] = self.delta
                rnn.bias = bias + bias_delta
                lhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                rnn.bias = bias - bias_delta
                rhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                lhs_sum = np.zeros_like(list(lhs.values())[0])
                rhs_sum = np.zeros_like(list(rhs.values())[0])
                for k in list(lhs.keys()):
                    if k > 0:
                        lhs_sum += lhs[k] * inp_grad[k]
                        rhs_sum += rhs[k] * inp_grad[k]
                bias_finite_diff[i] = \
                    np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)))
            rnn.bias = bias

            # Initial hidden state finite difference gradients
            hidden_finite_diff = np.empty(hidden_grad.shape)
            for i in range(init_hidden_state.shape[0]):
                for j in range(init_hidden_state.shape[1]):
                    h_delta = np.zeros(init_hidden_state.shape, dtype=conf.dtype)
                    h_delta[i, j] = self.delta
                    rnn.init_hidden_state = init_hidden_state + h_delta
                    lhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                    rnn.init_hidden_state = init_hidden_state - h_delta
                    rhs = copy.deepcopy(rnn.forward(inp, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    hidden_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)), keepdims=False)
            rnn.init_hidden_state = init_hidden_state

            # Inputs finite difference gradients
            inputs_grad = np.vstack(reversed(list(inputs_grad.values())))
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                    i_delta[i, j] = self.delta
                    lhs = copy.deepcopy(rnn.forward(inp + i_delta, mask=mask))
                    rhs = copy.deepcopy(rnn.forward(inp - i_delta, mask=mask))
                    lhs_sum = np.zeros_like(list(lhs.values())[0])
                    rhs_sum = np.zeros_like(list(rhs.values())[0])
                    for k in list(lhs.keys()):
                        if k > 0:
                            lhs_sum += lhs[k] * inp_grad[k]
                            rhs_sum += rhs[k] * inp_grad[k]
                    inputs_finite_diff[i, j] = \
                        np.sum(((lhs_sum - rhs_sum) / (2 * self.delta)), keepdims=False)

            npt.assert_almost_equal(hidden_weights_grad, hidden_weights_finite_diff, decimal=tol)
            npt.assert_almost_equal(input_weights_grad, input_weights_finite_diff, decimal=tol)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=tol)
            npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)

            if not actv_fn == 'ReLU':
                npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=tol)

            # if not actv_fn == 'Softmax':
            #     # Hidden weights gradient check
            #     grad_diff = (abs(hidden_weights_grad - hidden_weights_finite_diff) /
            #                  (abs(hidden_weights_grad + hidden_weights_finite_diff) + 1e-64))
            #     error_threshold = np.ones_like(grad_diff) * 1e-5
            #     npt.assert_array_less(grad_diff, error_threshold)
            #
            #     # Input weights gradient check
            #     grad_diff = (abs(input_weights_grad - input_weights_finite_diff) /
            #                  (abs(input_weights_grad + input_weights_finite_diff) + 1e-64))
            #     error_threshold = np.ones_like(grad_diff) * 1e-5
            #     npt.assert_array_less(grad_diff, error_threshold)
            #
            #     # Inputs gradient check
            #     grad_diff = (abs(inputs_grad - inputs_finite_diff) /
            #                  (abs(inputs_grad + inputs_finite_diff) + 1e-64))
            #     error_threshold = np.ones_like(grad_diff) * 1e-5
            #     npt.assert_array_less(grad_diff, error_threshold)

        # Combinatorial Test Cases
        # ------------------------
        sequence_length = [1, 2, 3, 11]
        reduce_size = [0, 1]
        feature_size = [1, 2, 3, 11]
        num_neurons = [1, 2, 3, 11]
        one_hot = [True, False]
        scale = [1e-2]
        unit_inp_grad = [True, False]
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'ReLU', 'Softmax']
        dropout = [True, False]
        architecture_type = ['many_to_many', 'many_to_one']
        repeat = list(range(1))

        for seq_len, r_size, feat, neur, oh, scl, unit, actv, dout, a_type, r in \
            list(itertools.product(sequence_length, reduce_size, feature_size, num_neurons, one_hot,
                                   scale, unit_inp_grad, activation_fn, dropout, architecture_type,
                                   repeat)):
            batch_size = seq_len - (r_size if seq_len > 1 else 0)

            # Initialize inputs
            if oh:
                X = np.zeros((batch_size, feat), dtype=conf.dtype)
                rnd_idx = np.random.randint(feat, size=batch_size)
                X[range(batch_size), rnd_idx] = 1
            else:
                X = np.random.uniform(-scl, scl, (batch_size, feat))

            # Initialize weights and bias
            wh = np.random.rand(neur, neur) * scl
            wx = np.random.rand(feat, neur) * scl
            w = {'hidden': wh, 'inp': wx}
            bias = np.random.rand(neur) * scl
            init_h_state = np.random.rand(1, neur) * scl

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
                if actv in ['Linear', 'ReLU']:
                    mask /= p
            else:
                p = None
                mask = None

            test(X, w, seq_len, inp_grad, bias, init_h_state, actv, p, mask, a_type)


if __name__ == '__main__':
    unittest.main()
