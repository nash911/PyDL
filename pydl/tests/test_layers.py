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

from pydl.nn.layers import FC
from pydl.nn.layers import NN
from pydl import conf

class TestLayers(unittest.TestCase):
    def test_score_fn(self):
        def test(inp, w, true_out, bias=False):
            fc = FC(inp, w.shape[-1], w, bias)
            out_fc = fc.score_fn(inp)
            npt.assert_almost_equal(out_fc, true_out, decimal=5)

        # Manually calculated
        # -------------------
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        true_out = np.array([[38, 44, 50, 56],
                             [83, 98, 113, 128]], dtype=conf.dtype)
        test(X, w, true_out)
        test(X, w, true_out+bias, bias)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        num_neurons = [1, 2, 3, 6, 11]
        scale = [1e-6, 1e-3, 1e-1, 1e-0, 2, 3, 10]

        for batch, feat, neur, scl in list(itertools.product(batch_size, feature_size, num_neurons,
                                                             scale)):
            X = np.random.uniform(-scl, scl, (batch, feat))
            w = np.random.randn(feat, neur) * scl
            bias = np.zeros(neur)
            true_out = np.matmul(X, w)
            test(X, w, true_out)
            test(X, w, true_out+bias, bias)


    def test_forward(self):
        def test(inp, w, true_out, bias=False, activation_fn='Sigmoid', batchnorm=False):
            fc = FC(inp, w.shape[-1], w, bias, activation_fn=activation_fn, batchnorm=batchnorm)
            out_fc = fc.forward(inp)
            npt.assert_almost_equal(out_fc, true_out, decimal=5)

        # Manually calculated
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        score_out = np.array([[38, 44, 50, 56],
                              [83, 98, 113, 128]], dtype=conf.dtype)
        true_out = 1.0 / (1.0 + np.exp(-score_out))
        test(X, w, true_out)
        true_out = 1.0 / (1.0 + np.exp(-(score_out+bias)))
        test(X, w, true_out, bias)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        num_neurons = [1, 2, 3, 6, 11]
        scale = [1e-6, 1e-3, 1e-1, 1e-0, 2]
        batchnorm = [True, False]

        for batch, feat, scl, neur, bn in list(itertools.product(batch_size, feature_size, scale,
                                                                 num_neurons, batchnorm)):
            X = np.random.uniform(-scl, scl, (batch, feat))
            w = np.random.randn(feat, neur) * scl
            bias = np.zeros(neur)
            score = np.matmul(X, w) + bias
            if bn:
                score = (score - np.mean(score, axis=0)) / np.sqrt(np.var(score, axis=0) + 1e-32)

            true_out_sig = 1.0 / (1.0 + np.exp(-np.matmul(X, w)))
            test(X, w, true_out_sig, bias=False, activation_fn='Sigmoid', batchnorm=False)

            true_out_sig = 1.0 / (1.0 + np.exp(-score))
            test(X, w, true_out_sig, bias, activation_fn='Sigmoid', batchnorm=bn)

            true_out_tanh = (2.0 / (1.0 + np.exp(-2.0 * score))) - 1.0
            test(X, w, true_out_tanh, bias, activation_fn='Tanh', batchnorm=bn)

            true_out_relu = np.maximum(0, score)
            test(X, w, true_out_relu, bias, activation_fn='ReLU', batchnorm=bn)


    def test_gradients_manually(self):
        def test(inp, w, inp_grad, true_weights_grad, true_inputs_grad, bias=False,
                 true_bias_grad=None):
            fc = FC(inp, w.shape[-1], w, bias)
            weights_grad = fc.weight_gradients(inp_grad, inputs=X)
            bias_grad = fc.bias_gradients(inp_grad)
            inputs_grad = fc.input_gradients(inp_grad)
            npt.assert_almost_equal(weights_grad, true_weights_grad, decimal=5)
            npt.assert_almost_equal(bias_grad, true_bias_grad, decimal=5)
            npt.assert_almost_equal(inputs_grad, true_inputs_grad, decimal=5)

        # Manually calculated - Unit input gradients
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.ones((2, 4), dtype=conf.dtype)
        true_weights_grad = np.sum(X, axis=0, keepdims=True).T * np.ones(w.shape, dtype=conf.dtype)
        true_inputs_grad = np.sum(w, axis=-1, keepdims=True).T * np.ones(X.shape, dtype=conf.dtype)
        true_bias_grad = np.sum(inp_grad, axis=0, keepdims=False)
        test(X, w, inp_grad, true_weights_grad, true_inputs_grad, bias, true_bias_grad)

        # Manually calculated
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.array([[3, 3, 3, 3],
                             [2, 2, 2, 2]], dtype=conf.dtype)
        true_weights_grad = np.array([[11, 11, 11, 11],
                                      [16, 16, 16, 16],
                                      [21, 21, 21, 21]], dtype=conf.dtype)
        true_bias_grad = np.sum(inp_grad, axis=0, keepdims=False)
        true_inputs_grad = np.array([[30, 78, 126],
                                     [20, 52, 84]], dtype=conf.dtype)
        test(X, w, inp_grad, true_weights_grad, true_inputs_grad, bias, true_bias_grad)


    def test_gradients_finite_difference(self):
        self.delta = 1e-5
        def test(inp, w, inp_grad, bias=False):
            fc = FC(inp, w.shape[-1], w, bias)
            weights_grad = fc.weight_gradients(inp_grad, inputs=X)
            bias_grad = fc.bias_gradients(inp_grad)
            inputs_grad = fc.input_gradients(inp_grad)

            # Weights finite difference gradients
            weights_finite_diff = np.empty(weights_grad.shape)
            for i in range(weights_grad.shape[0]):
                w_delta = np.zeros(w.shape, dtype=conf.dtype)
                w_delta[i] = self.delta
                weights_finite_diff[i] = np.sum(((fc.score_fn(inp, w + w_delta) -
                                                  fc.score_fn(inp, w - w_delta)) /
                                                  (2 * self.delta)) * inp_grad, axis=0)

            # Bias finite difference gradients
            fc.bias = bias + self.delta
            lhs = fc.score_fn(inp)
            fc.bias = bias - self.delta
            rhs = fc.score_fn(inp)
            bias_finite_diff = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad, axis=0)
            fc.bias = bias

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[1]):
                i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                i_delta[:,i] = self.delta
                inputs_finite_diff[:,i] = np.sum(((fc.score_fn(inp + i_delta, w) -
                                                   fc.score_fn(inp - i_delta, w)) /
                                                   (2 * self.delta)) * inp_grad, axis=-1,
                                                 keepdims=False)

            # Threshold Gradient Diff Check
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=5)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=5)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=5)

            # # Relative gradient error check
            # max_abs_w_grads = np.maximum(np.abs(weights_grad), np.abs(weights_finite_diff))
            # max_abs_w_grads[max_abs_w_grads==0] = 1
            # w_grads_accuracy = np.abs(weights_grad - weights_finite_diff) / max_abs_w_grads
            # npt.assert_almost_equal(np.zeros_like(w_grads_accuracy), w_grads_accuracy, decimal=5)
            #
            # max_abs_b_grads = np.maximum(np.abs(bias_grad), np.abs(bias_finite_diff))
            # max_abs_b_grads[max_abs_b_grads==0] = 1
            # b_grads_accuracy = np.abs(bias_grad - bias_finite_diff) / max_abs_b_grads
            # npt.assert_almost_equal(np.zeros_like(b_grads_accuracy), b_grads_accuracy, decimal=5)
            #
            # max_abs_inp_grads = np.maximum(np.abs(inputs_grad), np.abs(inputs_finite_diff))
            # max_abs_inp_grads[max_abs_inp_grads==0] = 1
            # inp_grads_accuracy = np.abs(inputs_grad - inputs_finite_diff) / max_abs_inp_grads
            # npt.assert_almost_equal(np.zeros_like(inp_grads_accuracy), inp_grads_accuracy, decimal=5)


        # Manually calculated - Unit input gradients
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.ones((2, 4), dtype=conf.dtype)
        test(X, w, inp_grad, bias)

        # Manually calculated
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.array([[1, 2, 3, 4],
                             [-5, -6, -7, -8]], dtype=conf.dtype)
        test(X, w, inp_grad, bias)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        num_neurons = [1, 2, 3, 6, 11]
        scale = [1e-4, 1e-3, 1e-1, 1e-0, 2, 3, 10]
        unit_inp_grad = [True, False]

        for batch, feat, neur, scl, unit in list(itertools.product(batch_size, feature_size,
                                                                   num_neurons, scale,
                                                                   unit_inp_grad)):
            X = np.random.uniform(-scl, scl, (batch, feat))
            w = np.random.randn(feat, neur) * scl
            bias = np.random.rand(neur) * scl

            inp_grad = np.ones((batch, neur), dtype=conf.dtype) if unit else \
                       np.random.uniform(-10, 10, (batch, neur))
            test(X, w, inp_grad, bias)


    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-8
        def test(inp, w, inp_grad, bias=False, activation_fn='Sigmoid', batchnorm=False):
            fc = FC(inp, w.shape[-1], w, bias, activation_fn=activation_fn, batchnorm=batchnorm)
            y = fc.forward(inp)
            inputs_grad = fc.backward(inp_grad)
            weights_grad = fc.weights_grad
            bias_grad = fc.bias_grad

            # Weights finite difference gradients
            weights_finite_diff = np.empty(weights_grad.shape)
            for i in range(weights_grad.shape[0]):
                for j in range(weights_grad.shape[1]):
                    w_delta = np.zeros(w.shape, dtype=conf.dtype)
                    w_delta[i,j] = self.delta
                    fc.weights = w + w_delta
                    lhs = fc.forward(inp)
                    fc.weights = w - w_delta
                    rhs = fc.forward(inp)
                    weights_finite_diff[i,j] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)
            fc.weights = w

            # Bias finite difference gradients
            bias_finite_diff = np.empty(bias_grad.shape)
            for i in range(bias_grad.shape[0]):
                bias_delta = np.zeros(bias.shape, dtype=conf.dtype)
                bias_delta[i] = self.delta
                fc.bias = bias + bias_delta
                lhs = fc.forward(inp)
                fc.bias = bias - bias_delta
                rhs = fc.forward(inp)
                bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)
            fc.bias = bias

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                    i_delta[i,j] = self.delta
                    inputs_finite_diff[i,j] = np.sum(((fc.forward(inp + i_delta) -
                                                       fc.forward(inp - i_delta)) /
                                                       (2 * self.delta)) * inp_grad, keepdims=False)

            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=2)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

        # Manually calculated - Unit input gradients
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.ones((2, 4), dtype=conf.dtype)
        activation_fn = ['Sigmoid', 'Tanh', 'Softmax']
        batchnorm = [True, False]
        for actv_fn, bn in list(itertools.product(activation_fn, batchnorm)):
            test(X, w, inp_grad, bias, activation_fn=actv_fn, batchnorm=bn)

        # Manually calculated
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.array([[5, 6, 7, 8],
                             [1, 2, 3, 4]], dtype=conf.dtype)
        activation_fn = ['Sigmoid', 'Tanh', 'Softmax']
        batchnorm = [True, False]
        for actv_fn, bn in list(itertools.product(activation_fn, batchnorm)):
            test(X, w, inp_grad, bias, activation_fn=actv_fn, batchnorm=bn)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 8, 11]
        feature_size = [1, 2, 3, 11]
        num_neurons = [1, 2, 3, 11]
        scale = [1e-3, 1e-0, 2]
        unit_inp_grad = [True, False]
        activation_fn = ['Sigmoid', 'Tanh', 'Softmax', 'ReLU']
        batchnorm = [True, False]

        for batch, feat, neur, scl, unit, actv, bn in \
            list(itertools.product(batch_size, feature_size, num_neurons, scale, unit_inp_grad,
                                   activation_fn, batchnorm)):
            X = np.random.uniform(-scl, scl, (batch, feat))
            w = np.random.randn(feat, neur) * scl
            bias = np.zeros(neur)

            inp_grad = np.ones((batch, neur), dtype=conf.dtype) if unit else \
                       np.random.uniform(-1, 1, (batch, neur))
            test(X, w, inp_grad, bias, actv, bn)


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
        b_1 = np.random.uniform(-1, 1, (1, 19))
        l1_score = np.matmul(X, w_1) + b_1
        l1_out = 1.0 / (1.0 + np.exp(-(l1_score)))

        # Layer 2
        w_2 = np.random.randn(l1_out.shape[-1], 15)
        b_2 = np.random.uniform(-1, 1, (1, 15))
        l2_score = np.matmul(l1_out, w_2) + b_2
        l2_out = np.maximum(0, l2_score)

        # Layer 3
        w_3 = np.random.randn(l2_out.shape[-1], 11)
        b_3 = np.random.uniform(-1, 1, (1, 11))
        l3_score = np.matmul(l2_out, w_3) + b_3
        l3_out = (2.0 / (1.0 + np.exp(-2.0*(l3_score)))) - 1.0

        # Layer 4
        w_4 = np.random.randn(l3_out.shape[-1], 9)
        b_4 = np.random.uniform(-1, 1, (1, 9))
        l4_score = np.matmul(l3_out, w_4) + b_4
        l4_out = np.maximum(0, l4_score)

        # Layer 4
        w_5 = np.random.randn(l4_out.shape[-1], 9)
        b_5 = np.random.uniform(-1, 1, (1, 9))
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
            for e in range(s+1, 6):
                layers = layers_list[s:e]
                true_out = out_list[e-1]
                test(inp_list[s], layers, true_out)


    def test_backward(self):
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
                        w_delta[i,j] = self.delta
                        layer.weights = w + w_delta
                        lhs = nn.forward(inp)
                        layer_out_lhs = layer.output
                        layer.weights = w - w_delta
                        rhs = nn.forward(inp)
                        layer_out_rhs = layer.output
                        weights_finite_diff[i,j] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                        if layer.activation.lower() is 'relu':
                            # Replace finite-diff gradients calculated close to 0 with NN calculated
                            # gradients to pass assertion test
                            mask = np.array(np.logical_xor(layer_out_lhs > 0, layer_out_rhs > 0),
                                            dtype=conf.dtype)
                            if np.sum(mask, keepdims=False) > 0.0:
                                weights_finite_diff[i,j] = weights_grad[i,j]

                                # # DEBUGGER - Measure number of finite-diff gradients calculated
                                # # close to 0
                                # ratio_incorrect = np.sum(mask) / mask.size
                                # if ratio_incorrect > 0.0:
                                #     print("Weights Finite-Diff Grad - Incorrect: %f  - Size: %d" %
                                #           (ratio_incorrect * 100.0, lhs.size))
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
                    npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=5)

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
                    npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=5)

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                    i_delta[i,j] = self.delta
                    inputs_finite_diff[i,j] = np.sum(((nn.forward(inp + i_delta) -
                                                       nn.forward(inp - i_delta)) /
                                                       (2 * self.delta)) * inp_grad)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=3)

        for _ in range(1):
            # NN Architecture
            # Layer 1
            X = np.random.uniform(-1, 1, (10, 25))
            w_1 = np.random.randn(X.shape[-1], 19)
            b_1 = np.random.uniform(-1, 1, (1, 19))

            # Layer 2
            w_2 = np.random.randn(w_1.shape[-1], 15)
            b_2 = np.random.uniform(-1, 1, (1, 15))

            # Layer 3
            w_3 = np.random.randn(w_2.shape[-1], 11)
            b_3 = np.random.uniform(-1, 1, (1, 11))

            # Layer 4
            w_4 = np.random.randn(w_3.shape[-1], 9)
            b_4 = np.random.uniform(-1, 1, (1, 9))

            # Layer 5
            w_5 = np.random.randn(w_4.shape[-1], 7)
            b_5 = np.random.uniform(-1, 1, (1, 7))

            # Case-1
            # ------
            l1_a = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh')
            l2_a = FC(l1_a, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid')
            l3_a = FC(l2_a, w_3.shape[-1], w_3, b_3, activation_fn='Tanh')
            l4_a = FC(l3_a, w_4.shape[-1], w_4, b_4, activation_fn='Sigmoid')
            l5_a = FC(l4_a, w_5.shape[-1], w_5, b_5, activation_fn='SoftMax')

            # 5-Layers
            layers = [l1_a, l2_a, l3_a, l4_a, l5_a]
            test(X, layers)

            # Case-2: With BatchNorm
            # ----------------------
            l1_b = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Tanh', batchnorm=False)
            l2_b = FC(l1_b, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid', batchnorm=True)
            l3_b = FC(l2_b, w_3.shape[-1], w_3, b_3, activation_fn='Tanh', batchnorm=True)
            l4_b = FC(l3_b, w_4.shape[-1], w_4, b_4, activation_fn='Sigmoid', batchnorm=False)
            l5_b = FC(l4_b, w_5.shape[-1], w_5, b_5, activation_fn='SoftMax', batchnorm=True)

            # 5-Layers
            layers = [l1_b, l2_b, l3_b, l4_b, l5_b]
            test(X, layers)


if __name__ == '__main__':
    unittest.main()
