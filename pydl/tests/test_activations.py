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

from pydl.nn.activations import Sigmoid
from pydl.nn.activations import Tanh
from pydl.nn.activations import SoftMax
from pydl.nn.activations import ReLU
from pydl import conf

class TestSigmoid(unittest.TestCase):
    def test_sigmoid_name(self):
        def test(name):
            sig = Sigmoid(name)
            self.assertEqual(sig.name, name)

        name = 'Sigmoid_1'
        test(name)

    def test_sigmoid_forward(self):
        def test(inp, true_out):
            sig = Sigmoid()
            out_sig = sig.forward(inp)
            npt.assert_almost_equal(out_sig, true_out, decimal=5)

        # Manually calculated
        # -------------------
        X = np.zeros((2, 3), dtype=conf.dtype)
        true_out = np.ones((2, 3), dtype=conf.dtype) * 0.5
        test(X, true_out)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        uniform_range = [0.001, 0.01, 0.1, 1, 10]

        for batch, feat, rnge in list(itertools.product(batch_size, feature_size, uniform_range)):
            X = np.random.uniform(-rnge, rnge, (batch, feat))
            true_out = 1 / (1 + np.exp(-X))
            test(X, true_out)


    def test_gradients_finite_difference(self):
        self.delta = 1e-2
        def test(inp, inp_grad):
            sig = Sigmoid()
            sig_grad = sig.backward(inp_grad, inp)

            delta = np.full(sig_grad.shape, self.delta, dtype=conf.dtype)
            sig_finite_diff = ((sig.forward(inp + delta) - sig.forward(inp - delta)) /
                               (2 * self.delta))  * inp_grad

            # Sigmoid function finite difference gradients
            npt.assert_almost_equal(sig_grad, sig_finite_diff, decimal=3)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        inp_range = [0.001, 0.01, 0.1, 1, 10]
        grad_range = [0.001, 0.01, 0.1, 1, 10]
        unit_inp_grad = [True, False]

        for batch, feat, i_rnge, g_rnge, unit in \
            list(itertools.product(batch_size, feature_size, inp_range, grad_range, unit_inp_grad)):
            X = np.random.uniform(-i_rnge, i_rnge, (batch, feat))
            inp_grad = np.ones((batch, feat), dtype=conf.dtype) if unit else \
                       np.random.uniform(-g_rnge, g_rnge, (batch, feat))
            test(X, inp_grad)


class TestTanh(unittest.TestCase):
    def test_tanh_name(self):
        def test(name):
            tanh = Tanh(name)
            self.assertEqual(tanh.name, name)

        name = 'Tanh_1'
        test(name)

    def test_tanh_forward(self):
        def test(inp, true_out):
            tanh = Tanh()
            out_tanh = tanh.forward(inp)
            npt.assert_almost_equal(out_tanh, true_out, decimal=5)

        # Manually calculated
        # -------------------
        X = np.zeros((2, 3), dtype=conf.dtype)
        true_out = np.zeros((2, 3), dtype=conf.dtype)
        test(X, true_out)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        uniform_range = [0.001, 0.01, 0.1, 1, 10]

        for batch, feat, rnge in list(itertools.product(batch_size, feature_size, uniform_range)):
            X = np.random.uniform(-rnge, rnge, (batch, feat))
            true_out = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
            test(X, true_out)


    def test_gradients_finite_difference(self):
        self.delta = 1e-2
        def test(inp, inp_grad):
            tanh = Tanh()
            tanh_grad = tanh.backward(inp_grad, inp)

            delta = np.full(tanh_grad.shape, self.delta, dtype=conf.dtype)
            tanh_finite_diff = ((tanh.forward(inp + delta) - tanh.forward(inp - delta)) /
                                (2 * self.delta))  * inp_grad

            # Tanh function finite difference gradients
            npt.assert_almost_equal(tanh_grad, tanh_finite_diff, decimal=3)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        inp_range = [0.001, 0.01, 0.1, 1, 10]
        grad_range = [0.001, 0.01, 0.1, 1, 10]
        unit_inp_grad = [True, False]

        for batch, feat, i_rnge, g_rnge, unit in \
            list(itertools.product(batch_size, feature_size, inp_range, grad_range, unit_inp_grad)):
            X = np.random.uniform(-i_rnge, i_rnge, (batch, feat))
            inp_grad = np.ones((batch, feat), dtype=conf.dtype) if unit else \
                       np.random.uniform(-g_rnge, g_rnge, (batch, feat))
            test(X, inp_grad)


class TestSoftMax(unittest.TestCase):
    def test_softmax_forward(self):
        def test(inp, true_out):
            softmax = SoftMax()
            out_softmax = softmax.forward(inp)

            # Assert that the probabilities add up to 1.0
            npt.assert_almost_equal(np.sum(out_softmax, axis=-1, keepdims=False),
                                    np.ones(out_softmax.shape[:-1], dtype=conf.dtype), decimal=5)
            npt.assert_almost_equal(out_softmax, true_out, decimal=5)

        # Manually calculated
        # -------------------
        X = np.ones((2, 5), dtype=conf.dtype)
        true_out = np.ones((2, 5), dtype=conf.dtype) * 0.2
        test(X, true_out)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11, 100]
        uniform_range = [0.001, 0.01, 0.1, 1, 3]

        for batch, feat, rnge in list(itertools.product(batch_size, feature_size, uniform_range)):
            X = np.random.uniform(-rnge, rnge, (batch, feat))
            X_normalized = X - np.amax(X, axis=-1, keepdims=True)
            true_out = np.exp(X_normalized) / np.sum(np.exp(X_normalized), axis=-1, keepdims=True)
            test(X, true_out)


    def test_gradients_finite_difference(self):
        self.delta = 1e-2
        def test(inp, inp_grad):
            softmax = SoftMax()
            softmax_grad = softmax.backward(inp_grad, inp)

            # Inputs finite difference gradients
            softmax_finite_diff = np.empty(softmax_grad.shape)
            for i in range(softmax_grad.shape[1]):
                delta = np.zeros(inp.shape, dtype=conf.dtype)
                delta[:,i] = self.delta
                softmax_finite_diff[:,i] = np.sum(((softmax.forward(inp + delta) -
                                                    softmax.forward(inp - delta)) / (2 * self.delta)) *
                                                  inp_grad, axis=-1, keepdims=False)

            # Sigmoid function finite difference gradients
            npt.assert_almost_equal(softmax_grad, softmax_finite_diff, decimal=3)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11, 100]
        inp_range = [0.001, 0.01, 0.1, 1, 10]
        grad_range = [0.001, 0.01, 0.1, 1, 10]
        unit_inp_grad = [True, False]

        for batch, feat, i_rnge, g_rnge, unit in \
            list(itertools.product(batch_size, feature_size, inp_range, grad_range, unit_inp_grad)):
            X = np.random.uniform(-i_rnge, i_rnge, (batch, feat))
            inp_grad = np.ones((batch, feat), dtype=conf.dtype) if unit else \
                       np.random.uniform(-g_rnge, g_rnge, (batch, feat))
            test(X, inp_grad)


class TestReLU(unittest.TestCase):
    def test_relu_forward(self):
        def test(inp, true_out):
            relu = ReLU()
            out_relu = relu.forward(inp)
            npt.assert_almost_equal(out_relu, true_out, decimal=5)

        # Manually calculated
        # -------------------
        X = np.array([[1.0, -3.2, -0.00001, 10, 0.000001],
                      [4, 10, -2.0, -10000, 10000]], dtype=conf.dtype)
        true_out = np.array([[1.0, 0, 0, 10, 0.000001],
                             [4, 10, 0, 0, 10000]], dtype=conf.dtype)
        test(X, true_out)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        uniform_range = [0.001, 0.01, 0.1, 1, 10]

        for batch, feat, rnge in list(itertools.product(batch_size, feature_size, uniform_range)):
            X = np.random.uniform(-rnge, rnge, (batch, feat))
            true_out = np.where(X > 0, X, 0)
            test(X, true_out)


    def test_gradients_finite_difference(self):
        self.delta = 1e-8
        def test(inp, inp_grad):
            relu = ReLU()
            relu_grad = relu.backward(inp_grad, inp)

            delta = np.full(relu_grad.shape, self.delta, dtype=conf.dtype)
            lhs = relu.forward(inp + delta)
            rhs = relu.forward(inp - delta)
            relu_finite_diff = ((lhs - rhs) / (2 * self.delta))  * inp_grad

            # Replace finite-diff gradients calculated close to 0 with NN calculated gradients
            # to pass assertion test
            mask = np.array(np.logical_xor(lhs > 0, rhs > 0), dtype=conf.dtype)
            extraction_mask = np.abs((mask - 1.0), dtype=conf.dtype)
            replace_grads = mask * relu_grad
            relu_finite_diff = (relu_finite_diff * extraction_mask) + replace_grads

            # # DEBUGGER - Measure number of finite-diff gradients calculated close to 0
            # ratio_incorrect = np.sum(mask) / mask.size
            # if ratio_incorrect > 0.0:
            #     print("Incorrect: %f  - Size: %d" % (ratio_incorrect * 100.0, lhs.size) )

            # ReLU function finite difference gradients
            npt.assert_almost_equal(relu_grad, relu_finite_diff, decimal=3)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        inp_range = [0.001, 0.01, 0.1, 1, 10]
        grad_range = [0.001, 0.01, 0.1, 1, 10]
        unit_inp_grad = [True, False]

        for batch, feat, i_rnge, g_rnge, unit in \
            list(itertools.product(batch_size, feature_size, inp_range, grad_range, unit_inp_grad)):
            X = np.random.uniform(-i_rnge, i_rnge, (batch, feat))
            inp_grad = np.ones((batch, feat), dtype=conf.dtype) if unit else \
                       np.random.uniform(-g_rnge, g_rnge, (batch, feat))
            test(X, inp_grad)

if __name__ == '__main__':
    unittest.main()
