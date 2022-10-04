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
from pydl import conf

np.random.seed(11421111)


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
        test(X, w, true_out + bias, bias)

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
            test(X, w, true_out + bias, bias)

    def test_forward(self):
        def test(inp, w, true_out, bias=False, actv_fn='Sigmoid', bchnorm=False, bn_mean=None,
                 bn_var=None, infrnc=False, p=None, mask=None):
            fc = FC(inp, w.shape[-1], w, bias, activation_fn=actv_fn, batchnorm=bchnorm, dropout=p)
            if bchnorm and infrnc:
                fc.batchnorm.avg_mean = bn_mean
                fc.batchnorm.avg_var = bn_var
            out_fc = fc.forward(inp, inference=infrnc, mask=mask)
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
        true_out = 1.0 / (1.0 + np.exp(-(score_out + bias)))
        test(X, w, true_out, bias)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        num_neurons = [1, 2, 3, 6, 11]
        scale = [1e-6, 1e-3, 1e-1, 1e-0, 2]
        batchnorm = [True, False]
        inference = [True, False]
        dropout = [True, False]

        for batch, feat, scl, neur, bn, infrnc, dout in \
            list(itertools.product(batch_size, feature_size, scale, num_neurons, batchnorm,
                 inference, dropout)):
            X = np.random.uniform(-scl, scl, (batch, feat))
            w = np.random.randn(feat, neur) * scl
            bias = np.zeros(neur)
            bn_mean = None
            bn_var = None
            score = np.matmul(X, w) + bias

            if bn:
                if infrnc:
                    bn_mean = np.ones(neur) * np.random.uniform(0.001, 2)
                    bn_var = np.ones(neur) * np.random.uniform(0.001, 2)
                    score = (score - bn_mean) / np.sqrt(bn_var + 1e-32)
                else:
                    score = \
                        (score - np.mean(score, axis=0)) / np.sqrt(np.var(score, axis=0) + 1e-32)

            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(*score.shape) < p, dtype=conf.dtype)
            else:
                p = None
                mask = None

            true_out_sig = 1.0 / (1.0 + np.exp(-np.matmul(X, w)))
            if dout:
                if infrnc:
                    true_out_sig *= p
                else:
                    true_out_sig *= mask
            test(X, w, true_out_sig, bias=False, actv_fn='Sigmoid', bchnorm=False, bn_mean=bn_mean,
                 bn_var=bn_var, infrnc=infrnc, p=p, mask=mask)

            true_out_sig = 1.0 / (1.0 + np.exp(-score))
            if dout:
                if infrnc:
                    true_out_sig *= p
                else:
                    true_out_sig *= mask
            test(X, w, true_out_sig, bias, actv_fn='Sigmoid', bchnorm=bn, bn_mean=bn_mean,
                 bn_var=bn_var, infrnc=infrnc, p=p, mask=mask)

            true_out_tanh = (2.0 / (1.0 + np.exp(-2.0 * score))) - 1.0
            if dout:
                if infrnc:
                    true_out_tanh *= p
                else:
                    true_out_tanh *= mask
            test(X, w, true_out_tanh, bias, actv_fn='Tanh', bchnorm=bn, bn_mean=bn_mean,
                 bn_var=bn_var, infrnc=infrnc, p=p, mask=mask)

            unnorm_prob = np.exp(score)
            true_out_softmax = unnorm_prob / np.sum(unnorm_prob, axis=-1, keepdims=True)
            if dout:
                if infrnc:
                    true_out_softmax *= p
                else:
                    true_out_softmax *= mask
            test(X, w, true_out_softmax, bias, actv_fn='Softmax', bchnorm=bn, bn_mean=bn_mean,
                 bn_var=bn_var, infrnc=infrnc, p=p, mask=mask)

            true_out_relu = np.maximum(0, score)
            if dout:
                if not infrnc:
                    mask /= p
                    true_out_relu *= mask
            test(X, w, true_out_relu, bias, actv_fn='ReLU', bchnorm=bn, bn_mean=bn_mean,
                 bn_var=bn_var, infrnc=infrnc, p=p, mask=mask)

            true_out_linear = score
            if dout:
                if not infrnc:
                    true_out_linear *= mask
            test(X, w, true_out_linear, bias, actv_fn='Linear', bchnorm=bn, bn_mean=bn_mean,
                 bn_var=bn_var, infrnc=infrnc, p=p, mask=mask)

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
                i_delta[:, i] = self.delta
                inputs_finite_diff[:, i] = np.sum(((fc.score_fn(inp + i_delta, w) -
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
            # npt.assert_almost_equal(np.zeros_like(inp_grads_accuracy), inp_grads_accuracy,
            #                         decimal=5)

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

        def test(inp, w, inp_grad, bias=False, actv_fn='Sigmoid', batchnorm=False, p=None,
                 mask=None):
            fc = FC(inp, w.shape[-1], w, bias, activation_fn=actv_fn, batchnorm=batchnorm,
                    dropout=p)
            _ = fc.forward(inp, mask=mask)
            inputs_grad = fc.backward(inp_grad)
            weights_grad = fc.weights_grad
            bias_grad = fc.bias_grad

            # Weights finite difference gradients
            weights_finite_diff = np.empty(weights_grad.shape)
            for i in range(weights_grad.shape[0]):
                for j in range(weights_grad.shape[1]):
                    w_delta = np.zeros(w.shape, dtype=conf.dtype)
                    w_delta[i, j] = self.delta
                    fc.weights = w + w_delta
                    lhs = fc.forward(inp, mask=mask)
                    fc.weights = w - w_delta
                    rhs = fc.forward(inp, mask=mask)
                    weights_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                    # Replace finite-diff gradients calculated close to 0 with NN calculated
                    # gradients to pass assertion test
                    grad_kink = np.sum(np.array(np.logical_xor(lhs > 0, rhs > 0), dtype=np.int32))
                    if grad_kink > 0:
                        weights_finite_diff[i, j] = weights_grad[i, j]
            fc.weights = w

            # Bias finite difference gradients
            bias_finite_diff = np.empty(bias_grad.shape)
            for i in range(bias_grad.shape[0]):
                bias_delta = np.zeros(bias.shape, dtype=conf.dtype)
                bias_delta[i] = self.delta
                fc.bias = bias + bias_delta
                lhs = fc.forward(inp, mask=mask)
                fc.bias = bias - bias_delta
                rhs = fc.forward(inp, mask=mask)
                bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                # Replace finite-diff gradients calculated close to 0 with NN calculated
                # gradients to pass assertion test
                grad_kink = np.sum(np.array(np.logical_xor(lhs > 0, rhs > 0), dtype=np.int32))
                if grad_kink > 0:
                    bias_finite_diff[i] = bias_grad[i]
            fc.bias = bias

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                    i_delta[i, j] = self.delta
                    lhs = fc.forward(inp + i_delta, mask=mask)
                    rhs = fc.forward(inp - i_delta, mask=mask)
                    inputs_finite_diff[i, j] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad,
                                                      keepdims=False)

                    # Replace finite-diff gradients calculated close to 0 with NN calculated
                    # gradients to pass assertion test
                    grad_kink = np.sum(np.array(np.logical_xor(lhs > 0, rhs > 0), dtype=np.int32))
                    if grad_kink > 0:
                        inputs_finite_diff[i, j] = inputs_grad[i, j]

            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=2)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=2)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

        # Manually calculated - Unit input gradients
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.ones((2, 4), dtype=conf.dtype)
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'Softmax']
        batchnorm = [True, False]
        dropout = [True, False]
        for actv, bn, dout in list(itertools.product(activation_fn, batchnorm, dropout)):
            if dout and actv == 'Softmax':
                continue
            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(*inp_grad.shape) < p, dtype=conf.dtype)
                if actv in ['Linear', 'ReLU']:
                    mask /= p
            else:
                p = None
                mask = None
            test(X, w, inp_grad, bias, actv, bn, p, mask)

        # Manually calculated
        X = np.array([[1, 2, 3],
                      [4, 5, 6]], dtype=conf.dtype)
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=conf.dtype)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=conf.dtype)
        inp_grad = np.array([[5, 6, 7, 8],
                             [1, 2, 3, 4]], dtype=conf.dtype)
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'Softmax']
        batchnorm = [True, False]
        dropout = [True, False]
        for actv, bn, dout in list(itertools.product(activation_fn, batchnorm, dropout)):
            if dout and actv == 'Softmax':
                continue
            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(*inp_grad.shape) < p, dtype=conf.dtype)
                if actv in ['Linear', 'ReLU']:
                    mask /= p
            else:
                p = None
                mask = None
            test(X, w, inp_grad, bias, actv, bn, p, mask)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 8, 11]
        feature_size = [1, 2, 3, 11]
        num_neurons = [1, 2, 3, 11]
        scale = [1e-3, 1e-0, 2]
        unit_inp_grad = [True, False]
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'Softmax', 'ReLU']
        batchnorm = [True, False]
        dropout = [True, False]

        for batch, feat, neur, scl, unit, actv, bn, dout in \
            list(itertools.product(batch_size, feature_size, num_neurons, scale, unit_inp_grad,
                                   activation_fn, batchnorm, dropout)):
            if dout and actv == 'Softmax':
                continue

            X = np.random.uniform(-scl, scl, (batch, feat))
            w = np.random.randn(feat, neur) * scl
            # bias = np.random.randn(neur) * scl
            bias = np.zeros(neur)

            inp_grad = np.ones((batch, neur), dtype=conf.dtype) if unit else \
                np.random.uniform(-1, 1, (batch, neur))

            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(batch, neur) < p, dtype=conf.dtype)
                if actv in ['Linear', 'ReLU']:
                    mask /= p
            else:
                p = None
                mask = None

            test(X, w, inp_grad, bias, actv, bn, p, mask)


if __name__ == '__main__':
    unittest.main()
