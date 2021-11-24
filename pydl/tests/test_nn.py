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
from collections import OrderedDict
import itertools

from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.residual_block import ResidualBlock
from pydl.nn.rnn import RNN
from pydl.nn.lstm import LSTM
from pydl.nn.gru import GRU
from pydl.nn.nn import NN
from pydl import conf

np.random.seed(11421111)


class TestNN(unittest.TestCase):
    def test_forward(self):
        def test(inp, layers, true_out):
            nn = NN(inp, layers)
            nn_out = nn.forward(inp)
            npt.assert_almost_equal(nn_out, true_out, decimal=5)

        # NN Architecture
        # Layer 1 - Sigmoid
        X = np.random.uniform(-1, 1, (10, 25))
        w_1 = np.random.randn(X.shape[-1], 19)
        b_1 = np.random.uniform(-1, 1, (19))
        l1_score = np.matmul(X, w_1) + b_1
        l1_out = 1.0 / (1.0 + np.exp(-(l1_score)))

        # Layer 2
        w_2 = np.random.randn(l1_out.shape[-1], 15)
        b_2 = np.random.uniform(-1, 1, (15))
        l2_score = np.matmul(l1_out, w_2) + b_2
        l2_out = np.maximum(0, l2_score)

        # Layer 3
        w_3 = np.random.randn(l2_out.shape[-1], 11)
        b_3 = np.random.uniform(-1, 1, (11))
        l3_score = np.matmul(l2_out, w_3) + b_3
        l3_out = (2.0 / (1.0 + np.exp(-2.0 * (l3_score)))) - 1.0

        # Layer 4
        w_4 = np.random.randn(l3_out.shape[-1], 9)
        b_4 = np.random.uniform(-1, 1, (9))
        l4_score = np.matmul(l3_out, w_4) + b_4
        l4_out = np.maximum(0, l4_score)

        # Layer 4
        w_5 = np.random.randn(l4_out.shape[-1], 9)
        b_5 = np.random.uniform(-1, 1, (9))
        l5_score = np.matmul(l4_out, w_5) + b_5
        l5_out = np.exp(l5_score) / np.sum(np.exp(l5_score), axis=-1, keepdims=True)

        l1 = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Sigmoid')
        l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='ReLU')
        l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='Tanh')
        l4 = FC(l3, w_4.shape[-1], w_4, b_4, activation_fn='ReLU')
        l5 = FC(l4, w_5.shape[-1], w_5, b_5, activation_fn='SoftMax')

        inp_list = list([X, l1_out, l2_out, l3_out, l4_out])
        out_list = list([l1_out, l2_out, l3_out, l4_out, l5_out])
        layers_list = list([l1, l2, l3, l4, l5])
        for s in range(5):
            for e in range(s + 1, 6):
                layers = layers_list[s:e]
                true_out = out_list[e - 1]
                test(inp_list[s], layers, true_out)

    def test_backward_fc(self):
        self.delta = 1e-3

        def test(inp, layers):
            nn = NN(inp, layers)
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)

            for layer in layers:
                w = layer.weights
                weights_grad = layer.weights_grad

                # Weights finite difference gradients
                weights_finite_diff = np.empty(weights_grad.shape)
                for i in range(weights_grad.shape[0]):
                    for j in range(weights_grad.shape[1]):
                        w_delta = np.zeros(w.shape, dtype=conf.dtype)
                        w_delta[i, j] = self.delta
                        layer.weights = w + w_delta
                        lhs = nn.forward(inp)
                        # layer_out_lhs = layer.output
                        layer.weights = w - w_delta
                        rhs = nn.forward(inp)
                        # layer_out_rhs = layer.output
                        weights_finite_diff[i, j] = \
                            np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                        # if layer.activation.lower() == 'relu':
                        #     # Replace finite-diff gradients calculated close to 0 with NN
                        #     # calculated gradients to pass assertion test
                        #     mask = np.array(np.logical_xor(layer_out_lhs > 0, layer_out_rhs > 0),
                        #                     dtype=conf.dtype)
                        #     if np.sum(mask, keepdims=False) > 0.0:
                        #         weights_finite_diff[i, j] = weights_grad[i, j]
                npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=3)
                layer.weights = w

                if layer.has_batchnorm:
                    bn = layer.batchnorm
                    gamma = bn.gamma
                    gamma_grad = bn.gamma_grad
                    beta = bn.beta
                    beta_grad = bn.beta_grad

                    # Gamma finite difference gradients
                    gamma_finite_diff = np.empty(gamma_grad.shape)
                    for i in range(gamma_grad.shape[0]):
                        g_delta = np.zeros(gamma.shape, dtype=conf.dtype)
                        g_delta[i] = self.delta
                        bn.gamma = gamma + g_delta
                        lhs = nn.forward(inp)
                        bn.gamma = gamma - g_delta
                        rhs = nn.forward(inp)
                        gamma_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)
                    bn.gamma = gamma
                    npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)

                    # Beta finite difference gradients
                    beta_finite_diff = np.empty(beta_grad.shape)
                    for i in range(beta_grad.shape[0]):
                        b_delta = np.zeros(beta.shape, dtype=conf.dtype)
                        b_delta[i] = self.delta
                        bn.beta = beta + b_delta
                        lhs = nn.forward(inp)
                        bn.beta = beta - b_delta
                        rhs = nn.forward(inp)
                        beta_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)
                    bn.beta = beta
                    npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                    i_delta[i, j] = self.delta
                    inputs_finite_diff[i, j] = np.sum(((nn.forward(inp + i_delta) -
                                                       nn.forward(inp - i_delta)) /
                                                       (2 * self.delta)) * inp_grad)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

        for _ in range(1):
            # NN Architecture
            # Layer 1
            batch_size = 10
            X = np.random.uniform(-1, 1, (batch_size, 25))
            w_1 = np.random.randn(X.shape[-1], 30)
            b_1 = np.random.uniform(-1, 1, (30))

            # Layer 2
            w_2 = np.random.randn(w_1.shape[-1], 23)
            b_2 = np.random.uniform(-1, 1, (23))

            # Layer 3
            w_3 = np.random.randn(w_2.shape[-1], 16)
            b_3 = np.random.uniform(-1, 1, (16))

            # Layer 4
            w_4 = np.random.randn(w_3.shape[-1], 19)
            b_4 = np.random.uniform(-1, 1, (19))

            # Layer 5
            w_5 = np.random.randn(w_4.shape[-1], 11)
            b_5 = np.random.uniform(-1, 1, (11))

            # Layer 6
            w_6 = np.random.randn(w_5.shape[-1], 9)
            b_6 = np.random.uniform(-1, 1, (9))

            # Layer 7
            w_7 = np.random.randn(w_6.shape[-1], 7)
            b_7 = np.random.uniform(-1, 1, (7))

            # Case-1
            # ------
            l1_a = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh')
            l2_a = FC(l1_a, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid')
            l3_a = FC(l2_a, w_3.shape[-1], w_3, b_3, activation_fn='Tanh')
            l4_a = FC(l3_a, w_4.shape[-1], w_4, b_4, activation_fn='Linear')
            l5_a = FC(l4_a, w_5.shape[-1], w_5, b_5, activation_fn='Sigmoid')
            l6_a = FC(l5_a, w_6.shape[-1], w_6, b_6, activation_fn='Linear')
            l7_a = FC(l6_a, w_7.shape[-1], w_7, b_7, activation_fn='SoftMax')

            layers_a = [l1_a, l2_a, l3_a, l4_a, l5_a, l6_a, l7_a]
            test(X, layers_a)

            # Case-2: With BatchNorm
            # ----------------------
            l1_b = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh', batchnorm=False)
            l2_b = FC(l1_b, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid', batchnorm=True)
            l3_b = FC(l2_b, w_3.shape[-1], w_3, b_3, activation_fn='Tanh', batchnorm=True)
            l4_b = FC(l3_b, w_4.shape[-1], w_4, b_4, activation_fn='Linear', batchnorm=False)
            l5_b = FC(l4_b, w_5.shape[-1], w_5, b_5, activation_fn='Sigmoid', batchnorm=False)
            l6_b = FC(l5_b, w_6.shape[-1], w_6, b_6, activation_fn='Linear', batchnorm=True)
            l7_b = FC(l6_b, w_7.shape[-1], w_7, b_7, activation_fn='SoftMax', batchnorm=False)

            layers_b = [l1_b, l2_b, l3_b, l4_b, l5_b, l6_b, l7_b]
            test(X, layers_b)

            # Case-3: With Dropout
            # --------------------
            # Layer-1
            dp1 = np.random.rand()
            l1_c = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh', batchnorm=False,
                      dropout=dp1)
            mask_l1 = np.array(np.random.rand(batch_size, w_1.shape[-1]) < dp1, dtype=conf.dtype)
            l1_c.dropout_mask = mask_l1

            # Layer-2
            dp2 = np.random.rand()
            l2_c = FC(l1_c, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid', batchnorm=True,
                      dropout=dp2)
            mask_l2 = np.array(np.random.rand(batch_size, w_2.shape[-1]) < dp2, dtype=conf.dtype)
            l2_c.dropout_mask = mask_l2

            # Layer-3
            l3_c = FC(l2_c, w_3.shape[-1], w_3, b_3, activation_fn='Tanh', batchnorm=True)

            # Layer-4
            dp4 = np.random.rand()
            l4_c = FC(l3_c, w_4.shape[-1], w_4, b_4, activation_fn='Linear', batchnorm=False,
                      dropout=dp4)
            mask_l4 = np.array(np.random.rand(batch_size, w_4.shape[-1]) < dp4, dtype=conf.dtype)
            l4_c.dropout_mask = mask_l4

            # Layer-5
            l5_c = FC(l4_c, w_5.shape[-1], w_5, b_5, activation_fn='Sigmoid', batchnorm=False)

            # Layer-6
            dp6 = np.random.rand()
            l6_c = FC(l5_c, w_6.shape[-1], w_6, b_6, activation_fn='Tanh', batchnorm=True,
                      dropout=dp6)
            mask_l6 = np.array(np.random.rand(batch_size, w_6.shape[-1]) < dp6, dtype=conf.dtype)
            l6_c.dropout_mask = mask_l6

            # Layer-7
            l7_c = FC(l6_c, w_7.shape[-1], w_7, b_7, activation_fn='SoftMax', batchnorm=False)

            layers_c = [l1_c, l2_c, l3_c, l4_c, l5_c, l6_c, l7_c]
            test(X, layers_c)

    def fc_layer_grads_test(self, nn, layer, inp, inp_grad, delta):
        w = layer.weights
        b = layer.bias

        weights_grad = layer.weights_grad
        bias_grad = layer.bias_grad

        # Weights finite difference gradients
        weights_finite_diff = np.empty(weights_grad.shape)
        for i in range(weights_grad.shape[0]):
            for j in range(weights_grad.shape[1]):
                w_delta = np.zeros(w.shape, dtype=conf.dtype)
                w_delta[i, j] = delta
                layer.weights = w + w_delta
                lhs = nn.forward(inp)
                # layer_out_lhs = layer.output
                layer.weights = w - w_delta
                rhs = nn.forward(inp)
                # layer_out_rhs = layer.output
                weights_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)

                # if layer.activation.lower() == 'relu':
                #     # Replace finite-diff gradients calculated close to 0 with NN calculated
                #     # gradients to pass assertion test
                #     mask = np.array(np.logical_xor(layer_out_lhs > 0, layer_out_rhs > 0),
                #                     dtype=conf.dtype)
                #     if np.sum(mask, keepdims=False) > 0.0:
                #         weights_finite_diff[i, j] = weights_grad[i, j]
                #
                #         # # DEBUGGER - Measure number of finite-diff gradients calculated
                #         # # close to 0
                #         # ratio_incorrect = np.sum(mask) / mask.size
                #         # if ratio_incorrect > 0.0:
                #         #     print("Weights Finite-Diff Grad - Incorrect: %f  - Size: %d" %
                #         #           (ratio_incorrect * 100.0, lhs.size))
        try:
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=3)
        except AssertionError:
            print("AssertionError Weights: ", layer.name)
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=3)
        layer.weights = w

        # Bias finite difference gradients
        bias_finite_diff = np.empty(bias_grad.shape)
        for i in range(b.shape[0]):
            b_delta = np.zeros(b.shape, dtype=conf.dtype)
            b_delta[i] = delta
            layer.bias = b + b_delta
            lhs = nn.forward(inp)
            layer.bias = b - b_delta
            rhs = nn.forward(inp)
            bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)

        try:
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=3)
        except AssertionError:
            print("AssertionError Bias: ", layer.name)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=3)
        layer.bias = b

        if layer.has_batchnorm:
            bn = layer.batchnorm
            gamma = bn.gamma
            gamma_grad = bn.gamma_grad
            beta = bn.beta
            beta_grad = bn.beta_grad

            # Gamma finite difference gradients
            gamma_finite_diff = np.empty(gamma_grad.shape)
            for i in range(gamma_grad.shape[0]):
                g_delta = np.zeros(gamma.shape, dtype=conf.dtype)
                g_delta[i] = delta
                bn.gamma = gamma + g_delta
                lhs = nn.forward(inp)
                bn.gamma = gamma - g_delta
                rhs = nn.forward(inp)
                gamma_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            bn.gamma = gamma
            try:
                npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)
            except AssertionError:
                print("AssertionError BachNorm Gamma: ", layer.name)
                npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)

            # Beta finite difference gradients
            beta_finite_diff = np.empty(beta_grad.shape)
            for i in range(beta_grad.shape[0]):
                b_delta = np.zeros(beta.shape, dtype=conf.dtype)
                b_delta[i] = delta
                bn.beta = beta + b_delta
                lhs = nn.forward(inp)
                bn.beta = beta - b_delta
                rhs = nn.forward(inp)
                beta_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            bn.beta = beta
            try:
                npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)
            except AssertionError:
                print("AssertionError BachNorm Beta: ", layer.name)
                npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)

    def conv_layer_grads_test(self, nn, layer, inp, inp_grad, delta):
        w = layer.weights
        b = layer.bias

        weights_grad = layer.weights_grad
        bias_grad = layer.bias_grad

        # Weights finite difference gradients
        weights_finite_diff = np.empty(weights_grad.shape)
        for i in range(weights_grad.shape[0]):
            for j in range(weights_grad.shape[1]):
                for k in range(weights_grad.shape[2]):
                    for m in range(weights_grad.shape[3]):
                        w_delta = np.zeros(w.shape, dtype=conf.dtype)
                        w_delta[i, j, k, m] = delta
                        layer.weights = w + w_delta
                        lhs = nn.forward(inp)
                        layer.weights = w - w_delta
                        rhs = nn.forward(inp)
                        weights_finite_diff[i, j, k, m] = \
                            np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)

                        # if layer.activation.lower() is 'relu':
                        #     print("Inside\n")
                        #     # Replace finite-diff gradients calculated close to 0 with NN
                        #     # calculated gradients to pass assertion test
                        #     mask = np.array(np.logical_xor(layer_out_lhs > 0, layer_out_rhs > 0),
                        #                     dtype=conf.dtype)
                        #     if np.sum(mask, keepdims=False) > 0.0:
                        #         weights_finite_diff[i, j] = weights_grad[i, j]
        try:
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=3)
        except AssertionError:
            print("AssertionError Weights: ", layer.name)
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=3)
        layer.weights = w

        # Bias finite difference gradients
        bias_finite_diff = np.empty(bias_grad.shape)
        for i in range(b.shape[0]):
            b_delta = np.zeros(b.shape, dtype=conf.dtype)
            b_delta[i] = delta
            layer.bias = b + b_delta
            lhs = nn.forward(inp)
            layer.bias = b - b_delta
            rhs = nn.forward(inp)
            bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)

        try:
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=3)
        except AssertionError:
            print("AssertionError Bias: ", layer.name)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=3)
        layer.bias = b

        if layer.has_batchnorm:
            bn = layer.batchnorm
            gamma = bn.gamma
            gamma_grad = bn.gamma_grad
            beta = bn.beta
            beta_grad = bn.beta_grad

            # Gamma finite difference gradients
            gamma_finite_diff = np.empty(gamma_grad.shape)
            for i in range(gamma_grad.shape[0]):
                for j in range(gamma_grad.shape[1]):
                    for k in range(gamma_grad.shape[2]):
                        g_delta = np.zeros(gamma.shape, dtype=conf.dtype)
                        g_delta[i, j, k] = delta
                        bn.gamma = gamma + g_delta
                        lhs = nn.forward(inp)
                        bn.gamma = gamma - g_delta
                        rhs = nn.forward(inp)
                        gamma_finite_diff[i, j, k] = \
                            np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            bn.gamma = gamma
            try:
                npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)
            except AssertionError:
                print("AssertionError BachNorm Gamma: ", layer.name)
                npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)

            # Beta finite difference gradients
            beta_finite_diff = np.empty(beta_grad.shape)
            for i in range(beta_grad.shape[0]):
                for j in range(beta_grad.shape[1]):
                    for k in range(beta_grad.shape[2]):
                        b_delta = np.zeros(beta.shape, dtype=conf.dtype)
                        b_delta[i, j, k] = delta
                        bn.beta = beta + b_delta
                        lhs = nn.forward(inp)
                        bn.beta = beta - b_delta
                        rhs = nn.forward(inp)
                        beta_finite_diff[i, j, k] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            bn.beta = beta
            try:
                npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)
            except AssertionError:
                print("AssertionError BachNorm Beta: ", layer.name)
                npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)

    def rnn_layer_grads_test(self, nn, layer, inp, inp_grad, delta):
        tol = 7

        wh = layer.hidden_weights
        wx = layer.input_weights
        bias = layer.bias
        init_hidden_state = np.copy(layer.init_hidden_state)

        hidden_weights_grad = layer.hidden_weights_grad
        input_weights_grad = layer.input_weights_grad
        bias_grad = layer.bias_grad
        hidden_grad = layer.hidden_state_grad

        # Hidden weights finite difference gradients
        hidden_weights_finite_diff = np.empty(hidden_weights_grad.shape)
        for i in range(hidden_weights_grad.shape[0]):
            for j in range(hidden_weights_grad.shape[1]):
                w_delta = np.zeros_like(wh)
                w_delta[i, j] = delta
                layer.hidden_weights = wh + w_delta
                lhs = nn.forward(inp)
                layer.hidden_weights = wh - w_delta
                rhs = nn.forward(inp)
                hidden_weights_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(hidden_weights_grad, hidden_weights_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Hidden Weights: ", layer.name)
            npt.assert_almost_equal(hidden_weights_grad, hidden_weights_finite_diff, decimal=tol)
        layer.hidden_weights = wh

        # Input weights finite difference gradients
        input_weights_finite_diff = np.empty(input_weights_grad.shape)
        for i in range(input_weights_grad.shape[0]):
            for j in range(input_weights_grad.shape[1]):
                w_delta = np.zeros_like(wx)
                w_delta[i, j] = delta
                layer.input_weights = wx + w_delta
                lhs = nn.forward(inp)
                layer.input_weights = wx - w_delta
                rhs = nn.forward(inp)
                input_weights_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(input_weights_grad, input_weights_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Input Weights: ", layer.name)
            npt.assert_almost_equal(input_weights_grad, input_weights_finite_diff, decimal=tol)
        layer.input_weights = wx

        # Bias finite difference gradients
        bias_finite_diff = np.empty(bias_grad.shape)
        for i in range(bias_grad.shape[0]):
            bias_delta = np.zeros_like(bias)
            bias_delta[i] = delta
            layer.bias = bias + bias_delta
            lhs = nn.forward(inp)
            layer.bias = bias - bias_delta
            rhs = nn.forward(inp)
            bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Bias: ", layer.name)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=tol)
        layer.bias = bias

        if layer.tune_internal_states:
            # Initial hidden state finite difference gradients
            hidden_finite_diff = np.empty(hidden_grad.shape)
            for i in range(hidden_grad.shape[0]):
                for j in range(hidden_grad.shape[1]):
                    h_delta = np.zeros_like(init_hidden_state)
                    h_delta[i, j] = delta
                    layer.init_hidden_state = init_hidden_state + h_delta
                    layer.reset_internal_states()
                    lhs = nn.forward(inp)
                    layer.init_hidden_state = init_hidden_state - h_delta
                    layer.reset_internal_states()
                    rhs = nn.forward(inp)
                    hidden_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            try:
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)
            except AssertionError:
                print("AssertionError Initial Hidden State: ", layer.name)
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)
            layer.init_hidden_state = init_hidden_state
            layer.reset_internal_states()

    def lstm_layer_grads_test(self, nn, layer, inp, inp_grad, delta):
        tol = 7

        w = layer.weights
        bias = layer.bias
        init_cell_state = np.copy(layer.init_cell_state)
        init_hidden_state = np.copy(layer.init_hidden_state)

        weights_grad = layer.weights_grad
        bias_grad = layer.bias_grad
        cell_grad = layer.cell_state_grad
        hidden_grad = layer.hidden_state_grad

        # Weights finite difference gradients
        weights_finite_diff = np.empty(weights_grad.shape)
        for i in range(weights_grad.shape[0]):
            for j in range(weights_grad.shape[1]):
                w_delta = np.zeros_like(w)
                w_delta[i, j] = delta
                layer.weights = w + w_delta
                lhs = nn.forward(inp)
                layer.weights = w - w_delta
                rhs = nn.forward(inp)
                weights_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Weights: ", layer.name)
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=tol)
        layer.weights = w

        # Bias finite difference gradients
        bias_finite_diff = np.empty(bias_grad.shape)
        for i in range(bias_grad.shape[0]):
            bias_delta = np.zeros_like(bias)
            bias_delta[i] = delta
            layer.bias = bias + bias_delta
            lhs = nn.forward(inp)
            layer.bias = bias - bias_delta
            rhs = nn.forward(inp)
            bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Bias: ", layer.name)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=tol)
        layer.bias = bias

        if layer.tune_internal_states:
            # Initial cell state finite difference gradients
            cell_finite_diff = np.empty(cell_grad.shape)
            for i in range(cell_grad.shape[0]):
                for j in range(cell_grad.shape[1]):
                    h_delta = np.zeros_like(init_cell_state)
                    h_delta[i, j] = delta
                    layer.init_cell_state = init_cell_state + h_delta
                    layer.reset_internal_states()
                    lhs = nn.forward(inp)
                    layer.init_cell_state = init_cell_state - h_delta
                    layer.reset_internal_states()
                    rhs = nn.forward(inp)
                    cell_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            try:
                npt.assert_almost_equal(cell_grad, cell_finite_diff, decimal=tol)
            except AssertionError:
                print("AssertionError Initial Cell State: ", layer.name)
                npt.assert_almost_equal(cell_grad, cell_finite_diff, decimal=tol)
            layer.init_cell_state = init_cell_state
            layer.reset_internal_states()

        if layer.tune_internal_states:
            # Initial hidden state finite difference gradients
            hidden_finite_diff = np.empty(hidden_grad.shape)
            for i in range(hidden_grad.shape[0]):
                for j in range(hidden_grad.shape[1]):
                    h_delta = np.zeros_like(init_hidden_state)
                    h_delta[i, j] = delta
                    layer.init_hidden_state = init_hidden_state + h_delta
                    layer.reset_internal_states()
                    lhs = nn.forward(inp)
                    layer.init_hidden_state = init_hidden_state - h_delta
                    layer.reset_internal_states()
                    rhs = nn.forward(inp)
                    hidden_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            try:
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)
            except AssertionError:
                print("AssertionError Initial Hidden State: ", layer.name)
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)
            layer.init_hidden_state = init_hidden_state
            layer.reset_internal_states()

    def gru_layer_grads_test(self, nn, layer, inp, inp_grad, delta):
        tol = 7

        gates_w = layer.gates_weights
        candidate_w = layer.candidate_weights
        gates_b = layer.gates_bias
        candidate_b = layer.candidate_bias
        init_hidden_state = np.copy(layer.init_hidden_state)

        gates_weights_grad = layer.gates_weights_grad
        candidate_weights_grad = layer.candidate_weights_grad
        gates_bias_grad = layer.gates_bias_grad
        candidate_bias_grad = layer.candidate_bias_grad
        hidden_grad = layer.hidden_state_grad

        # Gates weights finite difference gradients
        gates_weights_finite_diff = np.empty(gates_weights_grad.shape)
        for i in range(gates_weights_grad.shape[0]):
            for j in range(gates_weights_grad.shape[1]):
                w_delta = np.zeros_like(gates_w)
                w_delta[i, j] = delta
                layer.gates_weights = gates_w + w_delta
                lhs = nn.forward(inp)
                layer.gates_weights = gates_w - w_delta
                rhs = nn.forward(inp)
                gates_weights_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(gates_weights_grad, gates_weights_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Gates Weights: ", layer.name)
            npt.assert_almost_equal(gates_weights_grad, gates_weights_finite_diff, decimal=tol)
        layer.gates_weights = gates_w

        # Candidate weights finite difference gradients
        candidate_weights_finite_diff = np.empty(candidate_weights_grad.shape)
        for i in range(candidate_weights_grad.shape[0]):
            for j in range(candidate_weights_grad.shape[1]):
                w_delta = np.zeros_like(candidate_w)
                w_delta[i, j] = delta
                layer.candidate_weights = candidate_w + w_delta
                lhs = nn.forward(inp)
                layer.candidate_weights = candidate_w - w_delta
                rhs = nn.forward(inp)
                candidate_weights_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(candidate_weights_grad, candidate_weights_finite_diff,
                                    decimal=tol)
        except AssertionError:
            print("AssertionError Candidate Weights: ", layer.name)
            npt.assert_almost_equal(candidate_weights_grad, candidate_weights_finite_diff,
                                    decimal=tol)
        layer.candidate_weights = candidate_w

        # Gates bias finite difference gradients
        gates_bias_finite_diff = np.empty(gates_bias_grad.shape)
        for i in range(gates_bias_grad.shape[0]):
            b_delta = np.zeros_like(gates_b)
            b_delta[i] = delta
            layer.gates_bias = gates_b + b_delta
            lhs = nn.forward(inp)
            layer.gates_bias = gates_b - b_delta
            rhs = nn.forward(inp)
            gates_bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(gates_bias_grad, gates_bias_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Gates Bias: ", layer.name)
            npt.assert_almost_equal(gates_bias_grad, gates_bias_finite_diff, decimal=tol)
        layer.gates_bias = gates_b

        # Candidate bias finite difference gradients
        candidate_bias_finite_diff = np.empty(candidate_bias_grad.shape)
        for i in range(candidate_bias_grad.shape[0]):
            b_delta = np.zeros_like(candidate_b)
            b_delta[i] = delta
            layer.candidate_bias = candidate_b + b_delta
            lhs = nn.forward(inp)
            layer.candidate_bias = candidate_b - b_delta
            rhs = nn.forward(inp)
            candidate_bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(candidate_bias_grad, candidate_bias_finite_diff, decimal=tol)
        except AssertionError:
            print("AssertionError Candidate Bias: ", layer.name)
            npt.assert_almost_equal(candidate_bias_grad, candidate_bias_finite_diff, decimal=tol)
        layer.candidate_bias = candidate_b

        if layer.tune_internal_states:
            # Initial hidden state finite difference gradients
            hidden_finite_diff = np.empty(hidden_grad.shape)
            for i in range(hidden_grad.shape[0]):
                for j in range(hidden_grad.shape[1]):
                    h_delta = np.zeros_like(init_hidden_state)
                    h_delta[i, j] = delta
                    layer.init_hidden_state = init_hidden_state + h_delta
                    layer.reset_internal_states()
                    lhs = nn.forward(inp)
                    layer.init_hidden_state = init_hidden_state - h_delta
                    layer.reset_internal_states()
                    rhs = nn.forward(inp)
                    hidden_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            try:
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)
            except AssertionError:
                print("AssertionError Initial Hidden State: ", layer.name)
                npt.assert_almost_equal(hidden_grad, hidden_finite_diff, decimal=tol)
            layer.init_hidden_state = init_hidden_state
            layer.reset_internal_states()

    def inputs_1D_grad_test(self, nn, inp, inp_grad, inputs_grad, delta):
        if type(inputs_grad) is OrderedDict:
            seq_len = len(inputs_grad)
            inputs_grad = np.vstack([inputs_grad[t] for t in range(1, seq_len)])

        inputs_finite_diff = np.empty(inputs_grad.shape)
        for i in range(inputs_grad.shape[0]):
            for j in range(inputs_grad.shape[1]):
                i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                i_delta[i, j] = delta
                inputs_finite_diff[i, j] = np.sum(((nn.forward(inp + i_delta) -
                                                   nn.forward(inp - i_delta)) /
                                                   (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)
        except AssertionError:
            print("AssertionError - 1D Inputs")
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

    def inputs_3D_grad_test(self, nn, inp, inp_grad, inputs_grad, delta):
        inputs_finite_diff = np.empty(inputs_grad.shape)
        for i in range(inputs_grad.shape[0]):
            for j in range(inputs_grad.shape[1]):
                for k in range(inputs_grad.shape[2]):
                    for m in range(inputs_grad.shape[3]):
                        i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                        i_delta[i, j, k, m] = delta
                        inputs_finite_diff[i, j, k, m] = np.sum(((nn.forward(inp + i_delta) -
                                                                 nn.forward(inp - i_delta)) /
                                                                 (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)
        except AssertionError:
            print("AssertionError - 3D Inputs")
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

    def test_backward_FC(self):
        self.delta = 1e-3

        def test(inp, layers):
            nn = NN(inp, layers)
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)

            for layer in layers:
                self.fc_layer_grads_test(nn, layer, inp, inp_grad, self.delta)

            # Inputs finite difference gradients
            self.inputs_1D_grad_test(nn, inp, inp_grad, inputs_grad, self.delta)

        for _ in range(1):
            # NN Architecture
            # Layer 1
            batch_size = 10
            X = np.random.uniform(-1, 1, (batch_size, 25))
            w_1 = np.random.randn(X.shape[-1], 30)
            b_1 = np.random.uniform(-1, 1, (30))

            # Layer 2
            w_2 = np.random.randn(w_1.shape[-1], 23)
            b_2 = np.random.uniform(-1, 1, (23))

            # Layer 3
            w_3 = np.random.randn(w_2.shape[-1], 16)
            b_3 = np.random.uniform(-1, 1, (16))

            # Layer 4
            w_4 = np.random.randn(w_3.shape[-1], 19)
            b_4 = np.random.uniform(-1, 1, (19))

            # Layer 5
            w_5 = np.random.randn(w_4.shape[-1], 11)
            b_5 = np.random.uniform(-1, 1, (11))

            # Layer 6
            w_6 = np.random.randn(w_5.shape[-1], 9)
            b_6 = np.random.uniform(-1, 1, (9))

            # Layer 7
            w_7 = np.random.randn(w_6.shape[-1], 7)
            b_7 = np.random.uniform(-1, 1, (7))

            # Case-1
            # ------
            l1_a = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh', name='FC-1')
            l2_a = FC(l1_a, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid', name='FC-2')
            l3_a = FC(l2_a, w_3.shape[-1], w_3, b_3, activation_fn='Tanh', name='FC-3')
            l4_a = FC(l3_a, w_4.shape[-1], w_4, b_4, activation_fn='Linear', name='FC-4')
            l5_a = FC(l4_a, w_5.shape[-1], w_5, b_5, activation_fn='Sigmoid', name='FC-5')
            l6_a = FC(l5_a, w_6.shape[-1], w_6, b_6, activation_fn='Linear', name='FC-6')
            l7_a = FC(l6_a, w_7.shape[-1], w_7, b_7, activation_fn='SoftMax', name='FC-7-Out')

            layers_a = [l1_a, l2_a, l3_a, l4_a, l5_a, l6_a, l7_a]
            test(X, layers_a)

            # Case-2: With BatchNorm
            # ----------------------
            l1_b = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh', batchnorm=False,
                      name='FC-1')
            l2_b = FC(l1_b, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid', batchnorm=True,
                      name='FC-2')
            l3_b = FC(l2_b, w_3.shape[-1], w_3, b_3, activation_fn='Tanh', batchnorm=True,
                      name='FC-3')
            l4_b = FC(l3_b, w_4.shape[-1], w_4, b_4, activation_fn='Linear', batchnorm=False,
                      name='FC-4')
            l5_b = FC(l4_b, w_5.shape[-1], w_5, b_5, activation_fn='Sigmoid', batchnorm=False,
                      name='FC-5')
            l6_b = FC(l5_b, w_6.shape[-1], w_6, b_6, activation_fn='Linear', batchnorm=True,
                      name='FC-6')
            l7_b = FC(l6_b, w_7.shape[-1], w_7, b_7, activation_fn='SoftMax', batchnorm=False,
                      name='FC-7-Out')

            layers_b = [l1_b, l2_b, l3_b, l4_b, l5_b, l6_b, l7_b]
            test(X, layers_b)

            # Case-3: With Dropout
            # --------------------
            # Layer-1
            dp1 = np.random.rand()
            l1_c = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh', batchnorm=False,
                      dropout=dp1, name='FC-1')
            mask_l1 = np.array(np.random.rand(batch_size, w_1.shape[-1]) < dp1, dtype=conf.dtype)
            l1_c.dropout_mask = mask_l1

            # Layer-2
            dp2 = np.random.rand()
            l2_c = FC(l1_c, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid', batchnorm=True,
                      dropout=dp2, name='FC-2')
            mask_l2 = np.array(np.random.rand(batch_size, w_2.shape[-1]) < dp2, dtype=conf.dtype)
            l2_c.dropout_mask = mask_l2

            # Layer-3
            l3_c = FC(l2_c, w_3.shape[-1], w_3, b_3, activation_fn='Tanh', batchnorm=True,
                      name='FC-3')

            # Layer-4
            dp4 = np.random.rand()
            l4_c = FC(l3_c, w_4.shape[-1], w_4, b_4, activation_fn='Linear', batchnorm=False,
                      dropout=dp4, name='FC-4')
            mask_l4 = np.array(np.random.rand(batch_size, w_4.shape[-1]) < dp4, dtype=conf.dtype)
            l4_c.dropout_mask = mask_l4

            # Layer-5
            l5_c = FC(l4_c, w_5.shape[-1], w_5, b_5, activation_fn='Sigmoid', batchnorm=False,
                      name='FC-5')

            # Layer-6
            dp6 = np.random.rand()
            l6_c = FC(l5_c, w_6.shape[-1], w_6, b_6, activation_fn='Tanh', batchnorm=True,
                      dropout=dp6, name='FC-6')
            mask_l6 = np.array(np.random.rand(batch_size, w_6.shape[-1]) < dp6, dtype=conf.dtype)
            l6_c.dropout_mask = mask_l6

            # Layer-7
            l7_c = FC(l6_c, w_7.shape[-1], w_7, b_7, activation_fn='SoftMax', batchnorm=False,
                      name='FC-7-Out')

            layers_c = [l1_c, l2_c, l3_c, l4_c, l5_c, l6_c, l7_c]
            test(X, layers_c)

    def test_backward_convolution_pooling(self):
        self.delta = 1e-6

        def test(nn, layers, inp):
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)

            for layer in layers:
                if layer.type == 'FC_Layer':
                    self.fc_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                elif layer.type == 'Convolution_Layer':
                    self.conv_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                elif layer.type == 'Pooling_Layer':
                    continue

            # Inputs finite difference gradients
            self.inputs_3D_grad_test(nn, inp, inp_grad, inputs_grad, self.delta)

        for _ in range(1):
            # Inputs
            batch_size = 8
            depth = 3
            num_rows = 32
            num_cols = 32
            X = np.empty((batch_size, depth, num_rows, num_cols))

            # Case-1
            # ------
            # NN Architecture
            # Layer 1 - Convolution
            num_kernals_1 = 6
            rec_h_1 = 5
            rec_w_1 = 5
            pad_1 = 0
            stride_1 = 3
            w_1 = np.random.randn(num_kernals_1, depth, rec_h_1, rec_w_1)
            b_1 = np.random.uniform(-1, 1, (num_kernals_1))
            dp1 = np.random.rand()

            l1 = Conv(X, weights=w_1, bias=b_1, zero_padding=pad_1, stride=stride_1,
                      activation_fn='ReLU', name='Conv-1', batchnorm=True, dropout=dp1)
            mask_l1 = np.array(np.random.rand(batch_size, *l1.shape[1:]) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1

            # Layer 2 - Convolution
            num_kernals_2 = 5
            rec_h_2 = 3
            rec_w_2 = 3
            pad_2 = (0, 0)
            stride_2 = 1
            w_2 = np.random.randn(num_kernals_2, num_kernals_1, rec_h_2, rec_w_2)
            b_2 = np.random.uniform(-1, 1, (num_kernals_2))
            dp2 = np.random.rand()

            l2 = Conv(l1, weights=w_2, bias=b_2, zero_padding=pad_2, stride=stride_2,
                      activation_fn='ReLU', name='Conv-2', batchnorm=True, dropout=dp2)
            mask_l2 = np.array(np.random.rand(batch_size, *l2.shape[1:]) < dp2, dtype=conf.dtype)
            l2.dropout_mask = mask_l2

            # Layer 3 - MaxPool
            rec_h_3 = 2
            rec_w_3 = 2
            stride_3 = 2
            l3 = Pool(l2, receptive_field=(rec_h_3, rec_w_3), stride=stride_3, pool='MAX',
                      name='MaxPool-3')

            # Layer 4 - Convolution
            num_kernals_4 = 4
            rec_h_4 = 3
            rec_w_4 = 3
            pad_4 = (1, 2)
            stride_4 = 1
            w_4 = np.random.randn(num_kernals_4, num_kernals_2, rec_h_4, rec_w_4)
            b_4 = np.random.uniform(-1, 1, (num_kernals_4))
            dp4 = np.random.rand()

            l4 = Conv(l3, weights=w_4, bias=b_4, zero_padding=pad_4, stride=stride_4,
                      activation_fn='ReLU', name='Conv-4', batchnorm=True, dropout=dp4)
            mask_l4 = np.array(np.random.rand(batch_size, *l4.shape[1:]) < dp4, dtype=conf.dtype)
            l4.dropout_mask = mask_l4

            # Layer 5 - MaxPool
            rec_h_5 = 2
            rec_w_5 = 2
            stride_5 = 2
            l5 = Pool(l4, receptive_field=(rec_h_5, rec_w_5), stride=stride_5, pool='AVG',
                      name='MaxPool-5')

            # Layer 6 - Convolution
            num_kernals_6 = 7
            rec_h_6 = 1
            rec_w_6 = 1
            pad_6 = 0
            stride_6 = 1
            w_6 = np.random.randn(num_kernals_6, num_kernals_4, rec_h_6, rec_w_6)
            b_6 = np.random.uniform(-1, 1, (num_kernals_6))
            dp6 = np.random.rand()

            l6 = Conv(l5, weights=w_6, bias=b_6, zero_padding=pad_6, stride=stride_6,
                      activation_fn='ReLU', name='Conv-6', batchnorm=True, dropout=dp6)
            mask_l6 = np.array(np.random.rand(batch_size, *l6.shape[1:]) < dp6, dtype=conf.dtype)
            l6.dropout_mask = mask_l6

            # Layer 7 - MaxPool
            stride_7 = 1
            l7 = Pool(l6, receptive_field=None, stride=stride_7, pool='AVG', name='MaxPool-7')

            # Layer 8 - FC
            w_8 = np.random.randn(np.prod(l7.shape[1:]), 32)
            b_8 = np.random.uniform(-1, 1, (32))
            dp8 = np.random.rand()
            l8 = FC(l7, num_neurons=w_8.shape[-1], weights=w_8, bias=b_8, activation_fn='Tanh',
                    name='FC-8', batchnorm=True, dropout=dp8)
            mask_l8 = np.array(np.random.rand(batch_size, w_8.shape[-1]) < dp8, dtype=conf.dtype)
            l8.dropout_mask = mask_l8

            # Layer 9 - FC
            w_9 = np.random.randn(w_8.shape[-1], 16)
            b_9 = np.random.uniform(-1, 1, (16))
            dp9 = np.random.rand()
            l9 = FC(l8, num_neurons=w_9.shape[-1], weights=w_9, bias=b_9,
                    activation_fn='Linear', name='FC-9', batchnorm=True, dropout=dp9)
            mask_l9 = np.array(np.random.rand(batch_size, w_9.shape[-1]) < dp9, dtype=conf.dtype)
            l9.dropout_mask = mask_l9

            # Layer 10 - SoftMax
            w_10 = np.random.randn(w_9.shape[-1], 10)
            b_10 = np.random.uniform(-1, 1, (10))
            l10 = FC(l9, num_neurons=w_10.shape[-1], weights=w_10, bias=b_10,
                     activation_fn='SoftMax', name='Softmax-10')

            layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]
            nn = NN(X, layers)

            for _ in range(1):
                X = np.random.uniform(-1, 1, (batch_size, depth, num_rows, num_cols))
                test(nn, layers, X)

    def test_backward_resnet(self):
        self.delta = 1e-6

        def test(nn, layers, inp):
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)

            for layer in layers:
                if layer.type == 'Res_Block':
                    res_block_layers = layer.layers
                    for res_layer in res_block_layers:
                        if res_layer.type == 'FC_Layer':
                            self.fc_layer_grads_test(nn, res_layer, inp, inp_grad, self.delta)
                        elif res_layer.type == 'Convolution_Layer':
                            self.conv_layer_grads_test(nn, res_layer, inp, inp_grad, self.delta)
                    skip_conv = layer.skip_convolution
                    if skip_conv is not None:
                        print("skip_conv gradient check")
                        self.conv_layer_grads_test(nn, skip_conv, inp, inp_grad, self.delta)
                else:
                    if layer.type == 'FC_Layer':
                        self.fc_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                    elif layer.type == 'Convolution_Layer':
                        self.conv_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                    elif layer.type == 'Pooling_Layer':
                        continue

            # Inputs finite difference gradients
            self.inputs_3D_grad_test(nn, inp, inp_grad, inputs_grad, self.delta)

        for _ in range(1):
            # Inputs
            batch_size = 8
            depth = 3
            num_rows = 18
            num_cols = 18
            X = np.empty((batch_size, depth, num_rows, num_cols))

            # ResNet Architecture
            # Residual Block 1
            # Layer 1 - Convolution
            num_kernals_1 = 4
            rec_h_1 = 3
            rec_w_1 = 3
            pad_1 = 1
            stride_1 = 1
            w_1 = np.random.randn(num_kernals_1, depth, rec_h_1, rec_w_1)
            b_1 = np.random.uniform(-1, 1, (num_kernals_1))
            dp1 = np.random.rand()

            l1 = Conv(X, weights=w_1, bias=b_1, zero_padding=pad_1, stride=stride_1,
                      activation_fn='ReLU', name='Conv-1', batchnorm=True, dropout=dp1)
            mask_l1 = np.array(np.random.rand(batch_size, *l1.shape[1:]) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1

            # Layer 2 - MaxPool
            rec_h_2 = 2
            rec_w_2 = 2
            stride_2 = 2
            l2 = Pool(l1, receptive_field=(rec_h_2, rec_w_2), stride=stride_2, pool='MAX',
                      name='MaxPool-2')

            # Residual Block 1
            rcp_field_1 = [3, 3, 3]
            num_filters_1 = [4, 4, 4]
            activation_fn_1 = ['Relu', 'Relu', 'Relu']
            stride_1 = 1
            conv_layers_1 = [l2]
            for i, (rcp, n_filters, actv_fn) in enumerate(zip(rcp_field_1, num_filters_1,
                                                              activation_fn_1)):
                pad = int((rcp - 1) / 2)
                conv = Conv(conv_layers_1[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                            zero_padding=pad, stride=(stride_1 if i == 0 else 1), batchnorm=True,
                            activation_fn=actv_fn, name='Conv-1-%d' % (i + 1))
                conv_layers_1.append(conv)

            res_block_1 = ResidualBlock(l2, conv_layers_1[1:], activation_fn='Relu')

            # Residual Block 2
            rcp_field_2 = [1, 3, 1]
            num_filters_2 = [4, 6, 4]
            activation_fn_2 = ['Relu', 'Relu', 'Relu']
            stride_2 = 1
            conv_layers_2 = [res_block_1]
            for i, (rcp, n_filters, actv_fn) in enumerate(zip(rcp_field_2, num_filters_2,
                                                              activation_fn_2)):
                pad = int((rcp - 1) / 2)
                conv = Conv(conv_layers_2[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                            zero_padding=pad, stride=(stride_2 if i == 0 else 1), batchnorm=True,
                            activation_fn=actv_fn, name='Conv-2-%d' % (i + 1))
                conv_layers_2.append(conv)

            res_block_2 = ResidualBlock(res_block_1, conv_layers_2[1:], activation_fn='Relu')

            # Residual Block 3
            rcp_field_3 = [1, 3, 1]
            num_filters_3 = [4, 4, 4]
            activation_fn_3 = ['Relu', 'Relu', 'Relu']
            stride_3 = 2
            conv_layers_3 = [res_block_2]
            for i, (rcp, n_filters, actv_fn) in enumerate(zip(rcp_field_3, num_filters_3,
                                                              activation_fn_3)):
                pad = int((rcp - 1) / 2)
                conv = Conv(conv_layers_3[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                            zero_padding=pad, stride=(stride_3 if i == 0 else 1), batchnorm=True,
                            activation_fn=actv_fn, name='Conv-3-%d' % (i + 1))
                conv_layers_3.append(conv)

            res_block_3 = ResidualBlock(res_block_2, conv_layers_3[1:], activation_fn='Relu')

            # Residual Block 4
            rcp_field_4 = [1, 3, 1]
            num_filters_4 = [4, 4, 6]
            activation_fn_4 = ['Relu', 'Relu', 'Relu']
            stride_4 = 1
            conv_layers_4 = [res_block_3]
            for i, (rcp, n_filters, actv_fn) in enumerate(zip(rcp_field_4, num_filters_4,
                                                              activation_fn_4)):
                pad = int((rcp - 1) / 2)
                conv = Conv(conv_layers_4[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                            zero_padding=pad, stride=(stride_4 if i == 0 else 1), batchnorm=True,
                            activation_fn=actv_fn, name='Conv-4-%d' % (i + 1))
                conv_layers_4.append(conv)

            res_block_4 = ResidualBlock(res_block_3, conv_layers_4[1:], activation_fn='Relu')

            # Residual Block 5
            rcp_field_5 = [1, 3, 1, 3, 1]
            num_filters_5 = [1, 2, 3, 4, 5]
            activation_fn_5 = ['Relu', 'Linear', 'Sigmoid', 'Tanh', 'Relu']
            stride_5 = 2
            conv_layers_5 = [res_block_4]
            for i, (rcp, n_filters, actv_fn) in enumerate(zip(rcp_field_5, num_filters_5,
                                                              activation_fn_5)):
                pad = int((rcp - 1) / 2)
                conv = Conv(conv_layers_5[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                            zero_padding=pad, stride=(stride_5 if i == 0 else 1), batchnorm=True,
                            activation_fn=actv_fn, name='Conv-5-%d' % (i + 1))
                conv_layers_5.append(conv)

            res_block_5 = ResidualBlock(res_block_4, conv_layers_5[1:], activation_fn='Relu')

            # Residual Block 6
            rcp_field_6 = [1]
            num_filters_6 = [10]
            activation_fn_6 = ['ReLU']
            stride_6 = 1
            conv_layers_6 = [res_block_5]
            for i, (rcp, n_filters, actv_fn) in enumerate(zip(rcp_field_6, num_filters_6,
                                                              activation_fn_6)):
                pad = int((rcp - 1) / 2)
                conv = Conv(conv_layers_6[-1], receptive_field=(rcp, rcp), num_filters=n_filters,
                            zero_padding=pad, stride=(stride_6 if i == 0 else 1), batchnorm=True,
                            activation_fn=actv_fn, name='Conv-6-%d' % (i + 1))
                conv_layers_6.append(conv)

            res_block_6 = ResidualBlock(res_block_5, conv_layers_6[1:], activation_fn='Linear')

            # Layer 7 - MaxPool
            stride_7 = 1
            l7 = Pool(res_block_6, receptive_field=(3, 3), stride=stride_7, pool='AVG',
                      name='MaxPool-7')

            # Layer 8 - FC
            w_8 = np.random.randn(np.prod(l7.shape[1:]), 32)
            b_8 = np.random.uniform(-1, 1, (32))
            dp8 = np.random.rand()
            l8 = FC(l7, num_neurons=w_8.shape[-1], weights=w_8, bias=b_8, activation_fn='Tanh',
                    name='FC-8', batchnorm=True, dropout=dp8)
            mask_l8 = np.array(np.random.rand(batch_size, w_8.shape[-1]) < dp8, dtype=conf.dtype)
            l8.dropout_mask = mask_l8

            # Layer 9 - SoftMax
            w_9 = np.random.randn(w_8.shape[-1], 10)
            b_9 = np.random.uniform(-1, 1, (10))
            l9 = FC(l8, num_neurons=w_9.shape[-1], weights=w_9, bias=b_9, activation_fn='SoftMax',
                    name='Softmax-10')

            layers = [l1, l2, res_block_1, res_block_2, res_block_3, res_block_4, res_block_5,
                      res_block_6, l7, l8, l9]
            nn = NN(X, layers)

            for _ in range(1):
                X = np.random.uniform(-1, 1, (batch_size, depth, num_rows, num_cols))
                test(nn, layers, X)

    def test_backward_RNN(self):
        self.delta = 1e-5

        def test(inp, layers):
            nn = NN(inp, layers)
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)

            for layer in layers:
                if layer.type == 'FC_Layer':
                    self.fc_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                elif layer.type == 'RNN_Layer':
                    self.rnn_layer_grads_test(nn, layer, inp, inp_grad, self.delta)

            # Inputs finite difference gradients
            self.inputs_1D_grad_test(nn, inp, inp_grad, inputs_grad, self.delta)

        architecture_type = ['many_to_many', 'many_to_one']
        tune_internal_states = [True, False]
        reduce_size = [0, 3]
        scl = 0.1

        for a_type, tune, r_size in list(itertools.product(architecture_type, tune_internal_states,
                                                           reduce_size)):
            # Case-1 - Continuous Inputs
            # --------------------------
            # Layer 1
            seq_len = 10
            batch_size = seq_len - r_size
            inp_feat_size = 25
            num_neurons_rnn = 15
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1h = np.random.randn(num_neurons_rnn, num_neurons_rnn) * scl
            w_1x = np.random.randn(X.shape[-1], num_neurons_rnn) * scl
            w_1 = {'hidden': w_1h, 'inp': w_1x}
            b_1 = np.random.uniform(-1, 1, (num_neurons_rnn)) * scl
            init_h = np.random.randn(1, num_neurons_rnn) * scl

            # Layer 2
            num_neurons_fc = 20
            w_2 = np.random.randn(b_1.shape[-1], num_neurons_fc) * scl
            b_2 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # RNN Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = RNN(X, num_neurons_rnn, w_1, b_1, seq_len, activation_fn='Tanh',
                     architecture_type=a_type, dropout=dp1, tune_internal_states=tune, name='RNN-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_rnn) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_hidden_state = init_h
                l1.reset_internal_states()

            l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2]
            test(X, layers)

            # Case-2- OneHot Inputs
            # ---------------------
            # Layer 1
            seq_len = 11
            batch_size = seq_len - r_size
            inp_feat_size = 13
            num_neurons_rnn = 24
            X = np.zeros((batch_size, inp_feat_size), dtype=conf.dtype)
            X[range(batch_size), np.random.randint(inp_feat_size, size=batch_size)] = 1
            w_1h = np.random.randn(num_neurons_rnn, num_neurons_rnn) * scl
            w_1x = np.random.randn(X.shape[-1], num_neurons_rnn) * scl
            w_1 = {'hidden': w_1h, 'inp': w_1x}
            b_1 = np.random.uniform(-1, 1, (num_neurons_rnn)) * scl
            init_h = np.random.randn(1, num_neurons_rnn) * scl

            # Layer 2
            num_neurons_fc = 10
            w_2 = np.random.randn(b_1.shape[-1], num_neurons_fc) * scl
            b_2 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # RNN Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = RNN(X, num_neurons_rnn, w_1, b_1, seq_len, activation_fn='Tanh',
                     architecture_type=a_type, dropout=dp1, tune_internal_states=tune, name='RNN-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_rnn) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_hidden_state = init_h
                l1.reset_internal_states()

            l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2]
            test(X, layers)

            # Case-3 - Sandwitched Layers - FC - RNN - FC
            # -------------------------------------------
            # Layer 1
            seq_len = 9
            batch_size = seq_len - r_size
            inp_feat_size = 25
            num_neurons_fc = 30
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1 = np.random.randn(X.shape[-1], num_neurons_fc) * scl
            b_1 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # Layer 2
            num_neurons_rnn = 15
            w_2h = np.random.randn(num_neurons_rnn, num_neurons_rnn) * scl
            w_2x = np.random.randn(w_1.shape[-1], num_neurons_rnn) * scl
            w_2 = {'hidden': w_2h, 'inp': w_2x}
            b_2 = np.random.uniform(-1, 1, (num_neurons_rnn)) * scl
            init_h = np.random.randn(1, num_neurons_rnn) * scl

            # Layer 3
            num_neurons_fc_out = 20
            w_3 = np.random.randn(b_2.shape[-1], num_neurons_fc_out) * scl
            b_3 = np.random.uniform(-1, 1, (num_neurons_fc_out)) * scl

            # RNN Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = FC(X, num_neurons_fc, w_1, b_1, activation_fn='ReLU', dropout=dp1, name='FC-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_fc) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1

            dp2 = np.random.rand()
            l2 = RNN(l1, num_neurons_rnn, w_2, b_2, seq_len, activation_fn='Tanh',
                     architecture_type=a_type, dropout=dp2, tune_internal_states=tune, name='RNN-2')
            mask_l2 = np.array(np.random.rand(batch_size, num_neurons_rnn) < dp2, dtype=conf.dtype)
            l2.dropout_mask = mask_l2
            if tune:
                l2.init_hidden_state = init_h
                l2.reset_internal_states()

            l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3]
            test(X, layers)

            # Case-4 - Sandwitched Layers - RNN - RNN - FC
            # --------------------------------------------
            # Layer 1
            seq_len = 8
            batch_size = seq_len - r_size
            inp_feat_size = 17
            num_neurons_rnn_1 = 11
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1h = np.random.randn(num_neurons_rnn_1, num_neurons_rnn_1) * scl
            w_1x = np.random.randn(inp_feat_size, num_neurons_rnn_1) * scl
            w_1 = {'hidden': w_1h, 'inp': w_1x}
            b_1 = np.random.uniform(-1, 1, (num_neurons_rnn_1)) * scl
            init_h_1 = np.random.randn(1, num_neurons_rnn_1) * scl

            # Layer 2
            num_neurons_rnn_2 = 31
            w_2h = np.random.randn(num_neurons_rnn_2, num_neurons_rnn_2) * scl
            w_2x = np.random.randn(num_neurons_rnn_1, num_neurons_rnn_2) * scl
            w_2 = {'hidden': w_2h, 'inp': w_2x}
            b_2 = np.random.uniform(-1, 1, (num_neurons_rnn_2)) * scl
            init_h_2 = np.random.randn(1, num_neurons_rnn_2) * scl

            # Layer 3
            num_neurons_fc_out = 24
            w_3 = np.random.randn(b_2.shape[-1], num_neurons_fc_out) * scl
            b_3 = np.random.uniform(-1, 1, (num_neurons_fc_out)) * scl

            # RNN Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = RNN(X, num_neurons_rnn_1, w_1, b_1, seq_len, activation_fn='Tanh',
                     architecture_type='many_to_many', dropout=dp1, tune_internal_states=tune,
                     name='RNN-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_rnn_1) < dp1,
                               dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_hidden_state = init_h_1
                l1.reset_internal_states()

            dp2 = np.random.rand()
            l2 = RNN(l1, num_neurons_rnn_2, w_2, b_2, seq_len, activation_fn='Tanh',
                     architecture_type=a_type, dropout=dp2, tune_internal_states=tune, name='RNN-2')
            mask_l2 = np.array(np.random.rand(batch_size, num_neurons_rnn_2) < dp2,
                               dtype=conf.dtype)
            l2.dropout_mask = mask_l2
            if tune:
                l2.init_hidden_state = init_h_2
                l2.reset_internal_states()

            l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3]
            test(X, layers)

    def test_backward_LSTM(self):
        self.delta = 1e-5

        def test(inp, layers):
            nn = NN(inp, layers)
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)

            for layer in layers:
                if layer.type == 'FC_Layer':
                    self.fc_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                elif layer.type == 'RNN_Layer':
                    self.rnn_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                elif layer.type == 'LSTM_Layer':
                    self.lstm_layer_grads_test(nn, layer, inp, inp_grad, self.delta)

            # Inputs finite difference gradients
            self.inputs_1D_grad_test(nn, inp, inp_grad, inputs_grad, self.delta)

        architecture_type = ['many_to_many', 'many_to_one']
        tune_internal_states = [True, False]
        reduce_size = [0, 5]
        scl = 0.1

        for a_type, tune, r_size in list(itertools.product(architecture_type, tune_internal_states,
                                                           reduce_size)):
            # Case-1 - Continuous Inputs
            # --------------------------
            # Layer 1
            seq_len = 10
            batch_size = seq_len - r_size
            inp_feat_size = 25
            num_neurons_lstm = 15
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1 = np.random.randn((num_neurons_lstm + X.shape[-1]), (4 * num_neurons_lstm)) * scl
            b_1 = np.random.uniform(-1, 1, (4 * num_neurons_lstm)) * scl
            init_c = np.random.randn(1, num_neurons_lstm) * scl
            init_h = np.random.randn(1, num_neurons_lstm) * scl

            # Layer 2
            num_neurons_fc = 16
            w_2 = np.random.randn(num_neurons_lstm, num_neurons_fc) * scl
            b_2 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # LSTM Architecture
            # -----------------
            dp1 = np.random.rand()
            l1 = LSTM(X, num_neurons_lstm, w_1, b_1, seq_len, architecture_type=a_type, dropout=dp1,
                      tune_internal_states=tune, name='LSTM-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_lstm) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_cell_state = init_c
                l1.init_hidden_state = init_h
                l1.reset_internal_states()

            l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2]
            test(X, layers)

            # Case-2- OneHot Inputs
            # ---------------------
            # Layer 1
            seq_len = 11
            batch_size = seq_len - r_size
            inp_feat_size = 13
            num_neurons_lstm = 24
            X = np.zeros((batch_size, inp_feat_size), dtype=conf.dtype)
            X[range(batch_size), np.random.randint(inp_feat_size, size=batch_size)] = 1
            w_1 = np.random.randn((num_neurons_lstm + X.shape[-1]), (4 * num_neurons_lstm)) * scl
            b_1 = np.random.uniform(-1, 1, (4 * num_neurons_lstm)) * scl
            init_c = np.random.randn(1, num_neurons_lstm) * scl
            init_h = np.random.randn(1, num_neurons_lstm) * scl

            # Layer 2
            num_neurons_fc = 10
            w_2 = np.random.randn(num_neurons_lstm, num_neurons_fc) * scl
            b_2 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # LSTM Architecture
            # -----------------
            dp1 = np.random.rand()
            l1 = LSTM(X, num_neurons_lstm, w_1, b_1, seq_len, architecture_type=a_type, dropout=dp1,
                      tune_internal_states=tune, name='LSTM-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_lstm) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_cell_state = init_c
                l1.init_hidden_state = init_h
                l1.reset_internal_states()

            l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2]
            test(X, layers)

            # Case-3 - Sandwitched Layers - FC - LSTM - FC
            # --------------------------------------------
            # Layer 1
            seq_len = 9
            batch_size = seq_len - r_size
            inp_feat_size = 25
            num_neurons_fc = 30
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1 = np.random.randn(X.shape[-1], num_neurons_fc) * scl
            b_1 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # Layer 2
            num_neurons_lstm = 15
            w_2 = \
                np.random.randn((num_neurons_lstm + num_neurons_fc), (4 * num_neurons_lstm)) * scl
            b_2 = np.random.uniform(-1, 1, (4 * num_neurons_lstm)) * scl
            init_c = np.random.randn(1, num_neurons_lstm) * scl
            init_h = np.random.randn(1, num_neurons_lstm) * scl

            # Layer 3
            num_neurons_fc_out = 20
            w_3 = np.random.randn(num_neurons_lstm, num_neurons_fc_out) * scl
            b_3 = np.random.uniform(-1, 1, (num_neurons_fc_out)) * scl

            # LSTM Architecture
            # -----------------
            dp1 = np.random.rand()
            l1 = FC(X, num_neurons_fc, w_1, b_1, activation_fn='ReLU', dropout=dp1, name='FC-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_fc) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1

            dp2 = np.random.rand()
            l2 = LSTM(l1, num_neurons_lstm, w_2, b_2, seq_len, architecture_type=a_type,
                      dropout=dp2, tune_internal_states=tune, name='LSTM-2')
            mask_l2 = np.array(np.random.rand(batch_size, num_neurons_lstm) < dp2, dtype=conf.dtype)
            l2.dropout_mask = mask_l2
            if tune:
                l2.init_cell_state = init_c
                l2.init_hidden_state = init_h
                l2.reset_internal_states()

            l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3]
            test(X, layers)

            # Case-4 - Sandwitched Layers - LSTM - LSTM - FC
            # ----------------------------------------------
            # Layer 1
            seq_len = 8
            batch_size = seq_len - r_size
            inp_feat_size = 17
            num_neurons_lstm_1 = 11
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1 = \
                np.random.randn((num_neurons_lstm_1 + X.shape[-1]), (4 * num_neurons_lstm_1)) * scl
            b_1 = np.random.uniform(-1, 1, (4 * num_neurons_lstm_1)) * scl
            init_c_1 = np.random.randn(1, num_neurons_lstm_1) * scl
            init_h_1 = np.random.randn(1, num_neurons_lstm_1) * scl

            # Layer 2
            num_neurons_lstm_2 = 31
            w_2 = np.random.randn((num_neurons_lstm_2 + num_neurons_lstm_1),
                                  (4 * num_neurons_lstm_2)) * scl
            b_2 = np.random.uniform(-1, 1, (4 * num_neurons_lstm_2)) * scl
            init_c_2 = np.random.randn(1, num_neurons_lstm_2) * scl
            init_h_2 = np.random.randn(1, num_neurons_lstm_2) * scl

            # Layer 3
            num_neurons_fc_out = 24
            w_3 = np.random.randn(num_neurons_lstm_2, num_neurons_fc_out) * scl
            b_3 = np.random.uniform(-1, 1, (num_neurons_fc_out)) * scl

            # LSTM Architecture
            # -----------------
            dp1 = np.random.rand()
            l1 = LSTM(X, num_neurons_lstm_1, w_1, b_1, seq_len, architecture_type='many_to_many',
                      dropout=dp1, tune_internal_states=tune, name='LSTM-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_lstm_1) < dp1,
                               dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_cell_state = init_c_1
                l1.init_hidden_state = init_h_1
                l1.reset_internal_states()

            dp2 = np.random.rand()
            l2 = LSTM(l1, num_neurons_lstm_2, w_2, b_2, seq_len, architecture_type=a_type,
                      dropout=dp2, tune_internal_states=tune, name='LSTM-2')
            mask_l2 = np.array(np.random.rand(batch_size, num_neurons_lstm_2) < dp2,
                               dtype=conf.dtype)
            l2.dropout_mask = mask_l2
            if tune:
                l2.init_cell_state = init_c_2
                l2.init_hidden_state = init_h_2
                l2.reset_internal_states()

            l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3]
            test(X, layers)

    def test_backward_GRU(self):
        self.delta = 1e-6

        def test(inp, layers):
            nn = NN(inp, layers)
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)

            for layer in layers:
                if layer.type == 'FC_Layer':
                    self.fc_layer_grads_test(nn, layer, inp, inp_grad, self.delta)
                elif layer.type == 'GRU_Layer':
                    self.gru_layer_grads_test(nn, layer, inp, inp_grad, self.delta)

            # Inputs finite difference gradients
            self.inputs_1D_grad_test(nn, inp, inp_grad, inputs_grad, self.delta)

        architecture_type = ['many_to_many', 'many_to_one']
        tune_internal_states = [True, False]
        reduce_size = [0, 3]
        scl = 0.1

        for a_type, tune, r_size in list(itertools.product(architecture_type, tune_internal_states,
                                                           reduce_size)):
            # Case-1 - Continuous Inputs
            # --------------------------
            # Layer 1
            seq_len = 10
            batch_size = seq_len - r_size
            inp_feat_size = 25
            num_neur_gru = 15
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1 = OrderedDict()
            w_1['gates'] = np.random.randn((num_neur_gru + X.shape[-1]), (2 * num_neur_gru)) * scl
            w_1['candidate'] = np.random.randn((num_neur_gru + X.shape[-1]), num_neur_gru) * scl
            b_1 = OrderedDict()
            b_1['gates'] = np.random.rand(2 * num_neur_gru) * scl
            b_1['candidate'] = np.random.rand(num_neur_gru) * scl
            init_h = np.random.rand(1, num_neur_gru) * scl

            # Layer 2
            num_neurons_fc = 20
            w_2 = np.random.randn(num_neur_gru, num_neurons_fc) * scl
            b_2 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # GRU Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = GRU(X, num_neur_gru, w_1, b_1, seq_len, dropout=dp1, tune_internal_states=tune,
                     architecture_type=a_type, name='GRU-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neur_gru) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_hidden_state = init_h
                l1.reset_internal_states()

            l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2]
            test(X, layers)

            # Case-2- OneHot Inputs
            # ---------------------
            # Layer 1
            seq_len = 11
            batch_size = seq_len - r_size
            inp_feat_size = 13
            num_neur_gru = 24
            X = np.zeros((batch_size, inp_feat_size), dtype=conf.dtype)
            X[range(batch_size), np.random.randint(inp_feat_size, size=batch_size)] = 1
            w_1 = OrderedDict()
            w_1['gates'] = np.random.randn((num_neur_gru + X.shape[-1]), (2 * num_neur_gru)) * scl
            w_1['candidate'] = np.random.randn((num_neur_gru + X.shape[-1]), num_neur_gru) * scl
            b_1 = OrderedDict()
            b_1['gates'] = np.random.rand(2 * num_neur_gru) * scl
            b_1['candidate'] = np.random.rand(num_neur_gru) * scl
            init_h = np.random.rand(1, num_neur_gru) * scl

            # Layer 2
            num_neurons_fc = 10
            w_2 = np.random.randn(num_neur_gru, num_neurons_fc) * scl
            b_2 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # GRU Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = GRU(X, num_neur_gru, w_1, b_1, seq_len, dropout=dp1, tune_internal_states=tune,
                     architecture_type=a_type, name='GRU-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neur_gru) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_hidden_state = init_h
                l1.reset_internal_states()

            l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2]
            test(X, layers)

            # Case-3 - Sandwitched Layers - FC - GRU - FC
            # -------------------------------------------
            # Layer 1
            seq_len = 9
            batch_size = seq_len - r_size
            inp_feat_size = 25
            num_neurons_fc = 30
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1 = np.random.randn(X.shape[-1], num_neurons_fc) * scl
            b_1 = np.random.uniform(-1, 1, (num_neurons_fc)) * scl

            # Layer 2
            num_neur_gru = 15
            w_2 = OrderedDict()
            w_2['gates'] = \
                np.random.randn((num_neur_gru + num_neurons_fc), (2 * num_neur_gru)) * scl
            w_2['candidate'] = np.random.randn((num_neur_gru + num_neurons_fc), num_neur_gru) * scl
            b_2 = OrderedDict()
            b_2['gates'] = np.random.rand(2 * num_neur_gru) * scl
            b_2['candidate'] = np.random.rand(num_neur_gru) * scl
            init_h = np.random.rand(1, num_neur_gru) * scl

            # Layer 3
            num_neurons_fc_out = 20
            w_3 = np.random.randn(num_neur_gru, num_neurons_fc_out) * scl
            b_3 = np.random.uniform(-1, 1, (num_neurons_fc_out)) * scl

            # GRU Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = FC(X, num_neurons_fc, w_1, b_1, activation_fn='ReLU', dropout=dp1, name='FC-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neurons_fc) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1

            dp2 = np.random.rand()
            l2 = GRU(l1, num_neur_gru, w_2, b_2, seq_len, dropout=dp2, tune_internal_states=tune,
                     architecture_type=a_type, name='GRU-2')
            mask_l2 = np.array(np.random.rand(batch_size, num_neur_gru) < dp2, dtype=conf.dtype)
            l2.dropout_mask = mask_l2
            if tune:
                l2.init_hidden_state = init_h
                l2.reset_internal_states()

            l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3]
            test(X, layers)

            # Case-4 - Sandwitched Layers - GRU - GRU - FC
            # --------------------------------------------
            # Layer 1
            seq_len = 8
            batch_size = seq_len - r_size
            inp_feat_size = 17
            num_neur_gru_1 = 11
            X = np.random.uniform(-1, 1, (batch_size, inp_feat_size)) * scl
            w_1 = OrderedDict()
            w_1['gates'] = \
                np.random.randn((num_neur_gru_1 + X.shape[-1]), (2 * num_neur_gru_1)) * scl
            w_1['candidate'] = np.random.randn((num_neur_gru_1 + X.shape[-1]), num_neur_gru_1) * scl
            b_1 = OrderedDict()
            b_1['gates'] = np.random.rand(2 * num_neur_gru_1) * scl
            b_1['candidate'] = np.random.rand(num_neur_gru_1) * scl
            init_h_1 = np.random.rand(1, num_neur_gru_1) * scl

            # Layer 2
            num_neur_gru_2 = 31
            w_2 = OrderedDict()
            w_2['gates'] = \
                np.random.randn((num_neur_gru_2 + num_neur_gru_1), (2 * num_neur_gru_2)) * scl
            w_2['candidate'] = \
                np.random.randn((num_neur_gru_2 + num_neur_gru_1), num_neur_gru_2) * scl
            b_2 = OrderedDict()
            b_2['gates'] = np.random.rand(2 * num_neur_gru_2) * scl
            b_2['candidate'] = np.random.rand(num_neur_gru_2) * scl
            init_h_2 = np.random.rand(1, num_neur_gru_2) * scl

            # Layer 3
            num_neurons_fc_out = 24
            w_3 = np.random.randn(num_neur_gru_2, num_neurons_fc_out) * scl
            b_3 = np.random.uniform(-1, 1, (num_neurons_fc_out)) * scl

            # GRU Architecture
            # ----------------
            dp1 = np.random.rand()
            l1 = GRU(X, num_neur_gru_1, w_1, b_1, seq_len, dropout=dp1, tune_internal_states=tune,
                     architecture_type='many_to_many', name='GRU-1')
            mask_l1 = np.array(np.random.rand(batch_size, num_neur_gru_1) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1
            if tune:
                l1.init_hidden_state = init_h_1
                l1.reset_internal_states()

            dp2 = np.random.rand()
            l2 = GRU(l1, num_neur_gru_2, w_2, b_2, seq_len, dropout=dp2, tune_internal_states=tune,
                     architecture_type=a_type, name='GRU-2')
            mask_l2 = np.array(np.random.rand(batch_size, num_neur_gru_2) < dp2, dtype=conf.dtype)
            l2.dropout_mask = mask_l2
            if tune:
                l2.init_hidden_state = init_h_2
                l2.reset_internal_states()

            l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='SoftMax', name='FC-Out')

            layers = [l1, l2, l3]
            test(X, layers)


if __name__ == '__main__':
    unittest.main()
