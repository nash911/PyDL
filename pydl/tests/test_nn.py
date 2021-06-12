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
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.nn import NN
from pydl import conf

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
                    i_delta[i,j] = self.delta
                    inputs_finite_diff[i,j] = np.sum(((nn.forward(inp + i_delta) -
                                                       nn.forward(inp - i_delta)) /
                                                       (2 * self.delta)) * inp_grad)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

        for _ in range(1):
            # NN Architecture
            # Layer 1
            batch_size = 10
            X = np.random.uniform(-1, 1, (batch_size, 25))
            w_1 = np.random.randn(X.shape[-1], 30)
            b_1 = np.random.uniform(-1, 1, (1, 30))

            # Layer 2
            w_2 = np.random.randn(w_1.shape[-1], 23)
            b_2 = np.random.uniform(-1, 1, (1, 23))

            # Layer 3
            w_3 = np.random.randn(w_2.shape[-1], 16)
            b_3 = np.random.uniform(-1, 1, (1, 16))

            # Layer 4
            w_4 = np.random.randn(w_3.shape[-1], 19)
            b_4 = np.random.uniform(-1, 1, (1, 19))

            # Layer 5
            w_5 = np.random.randn(w_4.shape[-1], 11)
            b_5 = np.random.uniform(-1, 1, (1, 11))

            # Layer 6
            w_6 = np.random.randn(w_5.shape[-1], 9)
            b_6 = np.random.uniform(-1, 1, (1, 9))

            # Layer 7
            w_7 = np.random.randn(w_6.shape[-1], 7)
            b_7 = np.random.uniform(-1, 1, (1, 7))


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
        weights_grad = layer.weights_grad

        # Weights finite difference gradients
        weights_finite_diff = np.empty(weights_grad.shape)
        for i in range(weights_grad.shape[0]):
            for j in range(weights_grad.shape[1]):
                w_delta = np.zeros(w.shape, dtype=conf.dtype)
                w_delta[i,j] = delta
                layer.weights = w + w_delta
                lhs = nn.forward(inp)
                layer_out_lhs = layer.output
                layer.weights = w - w_delta
                rhs = nn.forward(inp)
                layer_out_rhs = layer.output
                weights_finite_diff[i,j] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)

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
        try:
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=3)
        except:
            print("Assertion Error Weights: ", layer.name)
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
                g_delta[i] = delta
                bn.gamma = gamma + g_delta
                lhs = nn.forward(inp)
                bn.gamma = gamma - g_delta
                rhs = nn.forward(inp)
                gamma_finite_diff[i] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            bn.gamma = gamma
            try:
                npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)
            except:
                print("Assertion Error BachNorm Gamma: ", layer.name)
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
            except:
                print("Assertion Error BachNorm Beta: ", layer.name)
                npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)


    def conv_layer_grads_test(self, nn, layer, inp, inp_grad, delta):
        w = layer.weights
        weights_grad = layer.weights_grad

        # Weights finite difference gradients
        weights_finite_diff = np.empty(weights_grad.shape)
        for i in range(weights_grad.shape[0]):
            for j in range(weights_grad.shape[1]):
                for k in range(weights_grad.shape[2]):
                    for l in range(weights_grad.shape[3]):
                        w_delta = np.zeros(w.shape, dtype=conf.dtype)
                        w_delta[i,j,k,l] = delta
                        layer.weights = w + w_delta
                        lhs = nn.forward(inp)
                        layer_out_lhs = layer.output
                        layer.weights = w - w_delta
                        rhs = nn.forward(inp)
                        layer_out_rhs = layer.output
                        weights_finite_diff[i,j,k,l] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)

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
        try:
            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=3)
        except:
            print("Assertion Error Weights: ", layer.name)
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
                for j in range(gamma_grad.shape[1]):
                    for k in range(gamma_grad.shape[2]):
                        g_delta = np.zeros(gamma.shape, dtype=conf.dtype)
                        g_delta[i,j,k] = delta
                        bn.gamma = gamma + g_delta
                        lhs = nn.forward(inp)
                        bn.gamma = gamma - g_delta
                        rhs = nn.forward(inp)
                        gamma_finite_diff[i,j,k] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            bn.gamma = gamma
            try:
                npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)
            except:
                print("Assertion Error BachNorm Gamma: ", layer.name)
                npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=4)

            # Beta finite difference gradients
            beta_finite_diff = np.empty(beta_grad.shape)
            for i in range(beta_grad.shape[0]):
                for j in range(beta_grad.shape[1]):
                    for k in range(beta_grad.shape[2]):
                        b_delta = np.zeros(beta.shape, dtype=conf.dtype)
                        b_delta[i,j,k] = delta
                        bn.beta = beta + b_delta
                        lhs = nn.forward(inp)
                        bn.beta = beta - b_delta
                        rhs = nn.forward(inp)
                        beta_finite_diff[i,j,k] = np.sum(((lhs - rhs) / (2 * delta)) * inp_grad)
            bn.beta = beta
            try:
                npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)
            except:
                print("Assertion Error BachNorm Beta: ", layer.name)
                npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=4)


    def inputs_1D_grad_test(self, nn, inp, inp_grad, inputs_grad, delta):
        inputs_finite_diff = np.empty(inputs_grad.shape)
        for i in range(inputs_grad.shape[0]):
            for j in range(inputs_grad.shape[1]):
                i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                i_delta[i,j] = delta
                inputs_finite_diff[i,j] = np.sum(((nn.forward(inp + i_delta) -
                                                   nn.forward(inp - i_delta)) /
                                                   (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)
        except:
            print("Assertion Error - 1D Inputs")
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)


    def inputs_3D_grad_test(self, nn, inp, inp_grad, inputs_grad, delta):
        inputs_finite_diff = np.empty(inputs_grad.shape)
        for i in range(inputs_grad.shape[0]):
            for j in range(inputs_grad.shape[1]):
                for k in range(inputs_grad.shape[2]):
                    for l in range(inputs_grad.shape[3]):
                        i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                        i_delta[i,j,k,l] = delta
                        inputs_finite_diff[i,j,k,l] = np.sum(((nn.forward(inp + i_delta) -
                                                               nn.forward(inp - i_delta)) /
                                                               (2 * delta)) * inp_grad)
        try:
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)
        except:
            print("Assertion Error - 3D Inputs")
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
            b_1 = np.random.uniform(-1, 1, (1, 30))

            # Layer 2
            w_2 = np.random.randn(w_1.shape[-1], 23)
            b_2 = np.random.uniform(-1, 1, (1, 23))

            # Layer 3
            w_3 = np.random.randn(w_2.shape[-1], 16)
            b_3 = np.random.uniform(-1, 1, (1, 16))

            # Layer 4
            w_4 = np.random.randn(w_3.shape[-1], 19)
            b_4 = np.random.uniform(-1, 1, (1, 19))

            # Layer 5
            w_5 = np.random.randn(w_4.shape[-1], 11)
            b_5 = np.random.uniform(-1, 1, (1, 11))

            # Layer 6
            w_6 = np.random.randn(w_5.shape[-1], 9)
            b_6 = np.random.uniform(-1, 1, (1, 9))

            # Layer 7
            w_7 = np.random.randn(w_6.shape[-1], 7)
            b_7 = np.random.uniform(-1, 1, (1, 7))


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
            b_1 = np.random.uniform(-1, 1, (1, num_kernals_1))
            dp1 = np.random.rand()

            l1 = Conv(X, weights=w_1, bias=b_1, zero_padding=pad_1, stride=stride_1,
                      activation_fn='ReLU', name='Conv-1', batchnorm=True, dropout=dp1)
            mask_l1 = np.array(np.random.rand(batch_size, *l1.shape[1:]) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1

            # Layer 2 - Convolution
            num_kernals_2 = 5
            rec_h_2 = 3
            rec_w_2 = 3
            pad_2 = 0
            stride_2 = 1
            w_2 = np.random.randn(num_kernals_2, num_kernals_1, rec_h_2, rec_w_2)
            b_2 = np.random.uniform(-1, 1, (1, num_kernals_2))
            dp2 = np.random.rand()

            l2 = Conv(l1, weights=w_2, bias=b_2, zero_padding=pad_2, stride=stride_2,
                      activation_fn='ReLU', name='Conv-2', batchnorm=True, dropout=dp2)
            mask_l2 = np.array(np.random.rand(batch_size, *l2.shape[1:]) < dp2, dtype=conf.dtype)
            l2.dropout_mask = mask_l2

            # Layer 3 - MaxPool
            rec_h_3 = 2
            rec_w_3 = 2
            stride_3 = 2
            l3 = Pool(l2, receptive_field=(rec_h_3, rec_w_3), stride=stride_3, name='MaxPool-3')

            # Layer 4 - Convolution
            num_kernals_4 = 3
            rec_h_4 = 1
            rec_w_4 = 1
            pad_4 = 0
            stride_4 = 1
            w_4 = np.random.randn(num_kernals_4, num_kernals_2, rec_h_4, rec_w_4)
            b_4 = np.random.uniform(-1, 1, (1, num_kernals_4))
            dp4 = np.random.rand()

            l4 = Conv(l3, weights=w_4, bias=b_4, zero_padding=pad_4, stride=stride_4,
                      activation_fn='ReLU', name='Conv-4', batchnorm=True, dropout=dp4)
            mask_l4 = np.array(np.random.rand(batch_size, *l4.shape[1:]) < dp4, dtype=conf.dtype)
            l4.dropout_mask = mask_l4

            # Layer 5 - MaxPool
            rec_h_5 = 2
            rec_w_5 = 2
            stride_5 = 2
            l5 = Pool(l4, receptive_field=(rec_h_5, rec_w_5), stride=stride_5, name='MaxPool-5')

            # Layer 6 - FC
            w_6 = np.random.randn(np.prod(l5.shape[1:]), 32)
            b_6 = np.random.uniform(-1, 1, (1, 32))
            dp6 = np.random.rand()
            l6 = FC(l5, num_neurons=w_6.shape[-1], weights=w_6, bias=b_6, activation_fn='Tanh',
                      name='FC-6', batchnorm=True, dropout=dp6)
            mask_l6 = np.array(np.random.rand(batch_size, w_6.shape[-1]) < dp6, dtype=conf.dtype)
            l6.dropout_mask = mask_l6

            # Layer 7 - FC
            w_7 = np.random.randn(w_6.shape[-1], 16)
            b_7 = np.random.uniform(-1, 1, (1, 16))
            dp7 = np.random.rand()
            l7 = FC(l5, num_neurons=w_7.shape[-1], weights=w_7, bias=b_7,
                      activation_fn='Linear', name='FC-7', batchnorm=True, dropout=dp7)
            mask_l7 = np.array(np.random.rand(batch_size, w_7.shape[-1]) < dp7, dtype=conf.dtype)
            l7.dropout_mask = mask_l7

            # Layer 8 - SoftMax
            w_8 = np.random.randn(w_7.shape[-1], 10)
            b_8 = np.random.uniform(-1, 1, (1, 10))
            l8 = FC(l7, num_neurons=w_8.shape[-1], weights=w_8, bias=b_8,
                      activation_fn='SoftMax', name='Softmax-8')

            layers = [l1, l2, l3, l4, l5, l6, l7, l8]
            nn = NN(X, layers)

            for _ in range(1):
                X = np.random.uniform(-1, 1, (batch_size, depth, num_rows, num_cols))
                test(nn, layers, X)


    def performance_test_convolution_pooling(self):
        def test(nn, inp):
            # for _ in range(iterations):
            nn_out = nn.forward(inp)
            inp_grad = np.random.uniform(-1, 1, nn_out.shape)
            inputs_grad = nn.backward(inp_grad)
            nn.update_weights(1e-4)

        for _ in range(1):
            # Inputs
            batch_size = 64
            depth = 8
            num_rows = 32
            num_cols = 32
            X = np.empty((batch_size, depth, num_rows, num_cols))

            # Case-1
            # ------
            # NN Architecture
            # Layer 1 - Convolution
            num_kernals_1 = 16
            rec_h_1 = 5
            rec_w_1 = 5
            pad_1 = 0
            stride_1 = 3
            w_1 = np.random.randn(num_kernals_1, depth, rec_h_1, rec_w_1)
            b_1 = np.random.uniform(-1, 1, (1, num_kernals_1))
            dp1 = np.random.rand()

            l1 = Conv(X, weights=w_1, bias=b_1, zero_padding=pad_1, stride=stride_1,
                      activation_fn='ReLU', name='Conv-1', batchnorm=True, dropout=dp1)
            mask_l1 = np.array(np.random.rand(batch_size, *l1.shape[1:]) < dp1, dtype=conf.dtype)
            l1.dropout_mask = mask_l1

            # Layer 2 - Convolution
            num_kernals_2 = 16
            rec_h_2 = 3
            rec_w_2 = 3
            pad_2 = 0
            stride_2 = 1
            w_2 = np.random.randn(num_kernals_2, num_kernals_1, rec_h_2, rec_w_2)
            b_2 = np.random.uniform(-1, 1, (1, num_kernals_2))
            dp2 = np.random.rand()

            l2 = Conv(l1, weights=w_2, bias=b_2, zero_padding=pad_2, stride=stride_2,
                      activation_fn='ReLU', name='Conv-2', batchnorm=True, dropout=dp2)
            mask_l2 = np.array(np.random.rand(batch_size, *l2.shape[1:]) < dp2, dtype=conf.dtype)
            l2.dropout_mask = mask_l2

            # Layer 3 - MaxPool
            rec_h_3 = 2
            rec_w_3 = 2
            stride_3 = 2
            l3 = Pool(l2, receptive_field=(rec_h_3, rec_w_3), stride=stride_3, name='MaxPool-3')

            # Layer 4 - Convolution
            num_kernals_4 = 8
            rec_h_4 = 1
            rec_w_4 = 1
            pad_4 = 0
            stride_4 = 1
            w_4 = np.random.randn(num_kernals_4, num_kernals_2, rec_h_4, rec_w_4)
            b_4 = np.random.uniform(-1, 1, (1, num_kernals_4))
            dp4 = np.random.rand()

            l4 = Conv(l3, weights=w_4, bias=b_4, zero_padding=pad_4, stride=stride_4,
                      activation_fn='ReLU', name='Conv-4', batchnorm=True, dropout=dp4)
            mask_l4 = np.array(np.random.rand(batch_size, *l4.shape[1:]) < dp4, dtype=conf.dtype)
            l4.dropout_mask = mask_l4

            # Layer 5 - MaxPool
            rec_h_5 = 2
            rec_w_5 = 2
            stride_5 = 2
            l5 = Pool(l4, receptive_field=(rec_h_5, rec_w_5), stride=stride_5, name='MaxPool-5')

            # Layer 6 - FC
            w_6 = np.random.randn(np.prod(l5.shape[1:]), 32)
            b_6 = np.random.uniform(-1, 1, (1, 32))
            dp6 = np.random.rand()
            l6 = FC(l5, num_neurons=w_6.shape[-1], weights=w_6, bias=b_6, activation_fn='Tanh',
                      name='FC-6', batchnorm=True, dropout=dp6)
            mask_l6 = np.array(np.random.rand(batch_size, w_6.shape[-1]) < dp6, dtype=conf.dtype)
            l6.dropout_mask = mask_l6

            # Layer 7 - FC
            w_7 = np.random.randn(w_6.shape[-1], 16)
            b_7 = np.random.uniform(-1, 1, (1, 16))
            dp7 = np.random.rand()
            l7 = FC(l5, num_neurons=w_7.shape[-1], weights=w_7, bias=b_7,
                      activation_fn='Linear', name='FC-7', batchnorm=True, dropout=dp7)
            mask_l7 = np.array(np.random.rand(batch_size, w_7.shape[-1]) < dp7, dtype=conf.dtype)
            l7.dropout_mask = mask_l7

            # Layer 8 - SoftMax
            w_8 = np.random.randn(w_7.shape[-1], 10)
            b_8 = np.random.uniform(-1, 1, (1, 10))
            l8 = FC(l7, num_neurons=w_8.shape[-1], weights=w_8, bias=b_8,
                      activation_fn='SoftMax', name='Softmax-8')

            layers = [l1, l2, l3, l4, l5, l6, l7, l8]
            nn = NN(X, layers)

            for _ in range(100):
                X = np.random.uniform(-1, 1, (batch_size, depth, num_rows, num_cols))
                test(nn, X)

if __name__ == '__main__':
    unittest.main()
