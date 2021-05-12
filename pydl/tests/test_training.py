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
from pydl.training.training import Training
from pydl.training.training import SGD
from pydl import conf

class TestTraining(unittest.TestCase):
    def test_shuffle_split_data(self):
        self.maxDiff = None
        def test(X, y, y_onehot, shuffle, train_size=70, test_size=30, onehot=True):
            train = Training(nn=None, activatin_type='Dummy_Val')
            train_X, train_y, test_X, test_y = \
                train.shuffle_split_data(X, (y if onehot is True else y_onehot), shuffle=shuffle,
                                         train_size=train_size, test_size=test_size, y_onehot=onehot)
            concat_X = np.vstack((train_X, test_X))
            concat_y = np.vstack((train_y, test_y))

            Xy = np.sum(X * y_onehot, axis=-1, keepdims=False)

            if len(concat_y.shape) == 1:
                concat_Xy = X[range(y.size), y]
            else:
                concat_Xy = np.sum(concat_X * concat_y, axis=-1, keepdims=False)

            self.assertCountEqual(np.array(Xy, dtype=conf.dtype).tolist(),
                                  np.array(concat_Xy, dtype=conf.dtype).tolist())
            self.assertAlmostEqual(np.sum(Xy), np.sum(concat_Xy), places=5)

        # Manually calculated
        # -------------------
        X = np.array([[1, 2, 3, 4],
                      [2, 3, 4, 1],
                      [3, 4, 1, 2],
                      [4, 1, 2, 3]])
        y = np.array([3, 2, 1, 0])
        y_onehot = np.zeros((y.size, y.max()+1))
        y_onehot[np.arange(y.size), y] = 1
        test(X, y, y_onehot, shuffle=True, onehot=False)
        test(X, y, y_onehot, shuffle=False, onehot=True)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [8, 11, 100, 256]
        feature_size = [1, 2, 3, 6, 11]
        train_size = [0, 1, 5, 9.12345, 25.252525, 49.0, 70, 99.99]
        shuffle = [True, False]
        onehot = [True, False]
        for batch, feat, t_sz, shfl, oh in list(itertools.product(batch_size, feature_size,
                                                                  train_size, shuffle, onehot)):
            X = np.random.uniform(-1, 1, (batch, feat))
            y = np.random.randint(feat, size=(batch))
            y[-1] = feat-1
            y_onehot = np.zeros((y.size, y.max()+1))
            y_onehot[np.arange(y.size), y] = 1
            test(X, y, y_onehot, shfl, t_sz, (100.0 - t_sz), oh)


    def test_mse_loss(self):
        def test(X, y, pred, true_out):
            train = Training(nn=None, regression=True)
            loss = train.loss(X, y, prob=pred)
            self.assertAlmostEqual(loss, true_out, places=6)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [100, 256]
        feature_size = [1, 2, 3, 6, 11]
        for batch, feat in list(itertools.product(batch_size, feature_size)):
            X = np.random.uniform(1, 2, (batch, feat))
            pred = np.random.uniform(-1, 1, (batch))
            y = np.random.uniform(-1, 1, (batch))
            true_out = 0.5 * np.mean(np.square(y - pred))
            test(X, y, pred, true_out)


    def test_softmax_loss(self):
        def test(X, y, prob, true_out):
            train = Training(nn=None, activatin_type='Softmax')
            loss = train.loss(X, y, prob=prob)
            self.assertAlmostEqual(loss, true_out, places=6)

        # Manually calculated
        # -------------------
        X = np.array([[1, 2, 3, 4],
                      [2, 3, 4, 1],
                      [3, 4, 1, 2],
                      [4, 1, 2, 3]])
        prob = X / np.sum(X, axis=-1)
        k = X.shape[-1]
        # Case-1
        y = np.array([3, 2, 1, 0])
        y_onehot = np.zeros((y.size, y.max()+1))
        y_onehot[np.arange(y.size), y] = 1
        true_out = -np.log(0.4)
        test(X, y, prob, true_out)
        test(X, y_onehot, prob, true_out)
        # Case-2
        y = np.array([0, 1, 2, 3])
        y_onehot = np.zeros((y.size, y.max()+1))
        y_onehot[np.arange(y.size), y] = 1
        true_out = ((-np.log(0.1) * 2) + (-np.log(0.3) * 2)) / 4
        test(X, y, prob, true_out)
        test(X, y_onehot, prob, true_out)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [100, 256]
        feature_size = [1, 2, 3, 6, 11]
        for batch, feat in list(itertools.product(batch_size, feature_size)):
            X = np.random.uniform(1, 2, (batch, feat))
            prob = X / np.sum(X, axis=-1, keepdims=True)
            k = X.shape[-1]
            y = np.random.randint(k, size=(X.shape[0]))
            y_onehot = np.zeros((y.size, y.max()+1))
            y_onehot[np.arange(y.size), y] = 1
            true_out = np.sum(y_onehot * -np.log(prob)) / batch
            test(X, y, prob, true_out)
            test(X, y_onehot, prob, true_out)


    def test_sigmoid_loss(self):
        def test(X, y, prob, true_out):
            train = Training(nn=None, activatin_type='Sigmoid')
            loss = train.sigmoid_cross_entropy_loss(X, y, prob=prob)
            self.assertAlmostEqual(loss, true_out, places=6)

        # Manually calculated
        # -------------------
        X = np.eye(5, dtype=conf.dtype)
        prob = X
        k = X.shape[-1]
        # Case-1
        y = np.array([0, 1, 2, 3, 4])
        y_onehot = np.zeros((y.size, y.max()+1))
        y_onehot[np.arange(y.size), y] = 1
        true_out = -np.log(1)
        test(X, y, prob, true_out)
        test(X, y_onehot, prob, true_out)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [100, 256]
        feature_size = [1, 2, 3, 6, 11]
        for batch, feat in list(itertools.product(batch_size, feature_size)):
            X = np.random.uniform(1, 2, (batch, feat))
            prob = 1 / (1 + np.exp(-X))
            k = X.shape[-1]
            y = np.random.randint(k, size=(X.shape[0]))
            y_onehot = np.zeros((y.size, y.max()+1))
            y_onehot[np.arange(y.size), y] = 1
            true_out = -np.sum((y_onehot * np.log(prob) + ((1 - y_onehot) * np.log(1 - prob)))) / batch
            test(X, y, prob, true_out)
            test(X, y_onehot, prob, true_out)

            multiclass_onehot = np.zeros((batch, feat))
            for mc in multiclass_onehot:
                y = np.random.randint(k, size=(k))
                mc[y] = 1
            true_out = -np.sum((multiclass_onehot * np.log(prob) + ((1 - multiclass_onehot) * \
                       np.log(1 - prob)))) / batch
            test(X, multiclass_onehot, prob, true_out)


    def test_loss_gradient_finite_diff(self):
        self.delta = 1e-3
        def test(X, y, layers, reg_lambda=0, regression=False):
            nn = NN(X, layers)
            train = Training(nn, reg_lambda=reg_lambda, regression=regression)
            loss = train.loss(X, y)
            loss_grad = train.loss_gradient(X, y)
            inputs_grad = nn.backward(loss_grad, reg_lambda=reg_lambda)

            for n, layer in enumerate(layers):
                w = layer.weights
                weights_grad = layer.weights_grad

                b = layer.bias
                bias_grad = layer.bias_grad

                # Weights finite difference gradients
                weights_finite_diff = np.empty(weights_grad.shape)
                for i in range(weights_grad.shape[0]):
                    for j in range(weights_grad.shape[1]):
                        w_delta = np.zeros(w.shape, dtype=conf.dtype)
                        w_delta[i,j] = self.delta
                        layer.weights = w + w_delta
                        lhs = train.loss(X, y)
                        layer.weights = w - w_delta
                        rhs = train.loss(X, y)
                        weights_finite_diff[i,j] = (lhs - rhs) / (2 * self.delta)
                layer.weights = w
                npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=5)

                # Bias finite difference gradients
                bias_finite_diff = np.empty(bias_grad.shape)
                for i in range(bias_grad.shape[0]):
                    b_delta = np.zeros(b.shape, dtype=conf.dtype)
                    b_delta[i] = self.delta
                    layer.bias = b + b_delta
                    lhs = train.loss(X, y)
                    layer.bias = b - b_delta
                    rhs = train.loss(X, y)
                    bias_finite_diff[i] = (lhs - rhs) / (2 * self.delta)
                layer.bias = b
                npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=5)

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
                        lhs = train.loss(X, y)
                        bn.gamma = gamma - g_delta
                        rhs = train.loss(X, y)
                        gamma_finite_diff[i] = (lhs - rhs) / (2 * self.delta)
                    bn.gamma = gamma
                    npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=5)

                    # Beta finite difference gradients
                    beta_finite_diff = np.empty(beta_grad.shape)
                    for i in range(beta_grad.shape[0]):
                        b_delta = np.zeros(beta.shape, dtype=conf.dtype)
                        b_delta[i] = self.delta
                        bn.beta = beta + b_delta
                        lhs = train.loss(X, y)
                        bn.beta = beta - b_delta
                        rhs = train.loss(X, y)
                        beta_finite_diff[i] = (lhs - rhs) / (2 * self.delta)
                    bn.beta = beta
                    npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=5)

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    i_delta = np.zeros(X.shape, dtype=conf.dtype)
                    i_delta[i,j] = self.delta
                    inputs_finite_diff[i,j] = ((train.loss(X + i_delta, y) -
                                                train.loss(X - i_delta, y)) / (2 * self.delta))
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=5)

        batchnorm = [True, False]
        for bn in batchnorm:
            # NN Architecture
            X = np.random.uniform(-1, 1, (100, 25))

            l1 = FC(X, num_neurons=19, activation_fn='Tanh', batchnorm=bn)
            l2 = FC(l1, num_neurons=15, activation_fn='Sigmoid', batchnorm=bn)
            l3 = FC(l2, num_neurons=11, activation_fn='Sigmoid', batchnorm=bn)
            l4 = FC(l3, num_neurons=9, activation_fn='Tanh', batchnorm=bn)
            l5_a = FC(l4, num_neurons=7, activation_fn='SoftMax', batchnorm=False) # SoftMax Probs
            l5_b = FC(l4, num_neurons=7, activation_fn='Sigmoid', batchnorm=False) # Sigmoid Probs
            l5_c = FC(l4, num_neurons=1, activation_fn='Sigmoid', batchnorm=False) # Binary Classification
            l5_d = FC(l4, num_neurons=1, activation_fn='Linear', batchnorm=False) # Regression

            # SoftMax Probs
            layers = [l1, l2, l3, l4, l5_a]
            k = layers[-1].shape[-1]
            labels = np.random.randint(k, size=(X.shape[0]))
            labels_onehot = np.zeros((labels.size, labels.max()+1))
            labels_onehot[np.arange(labels.size), labels] = 1

            reg_list = [0, 1e-6, 1e-3, 1e-0]
            y_list = [labels, labels_onehot]
            for reg, y in list(itertools.product(reg_list, y_list)):
                test(X, y, layers, reg_lambda=reg)

            # Sigmoid Probs
            layers = [l1, l2, l3, l4, l5_b]
            k = layers[-1].shape[-1]
            labels = np.random.randint(k, size=(X.shape[0]))
            labels_onehot = np.zeros((labels.size, labels.max()+1))
            labels_onehot[np.arange(labels.size), labels] = 1
            multiclass_onehot = np.zeros_like(labels_onehot)
            for mc in multiclass_onehot:
                lab = np.random.randint(k, size=(k))
                mc[lab] = 1

            reg_list = [0, 1e-6, 1e-3, 1e-0]
            y_list = [labels, labels_onehot, multiclass_onehot]
            for reg, y in list(itertools.product(reg_list, y_list)):
                test(X, y, layers, reg_lambda=reg)

            # Binary Classification - Sigmoid Probs
            layers = [l1, l2, l3, l4, l5_c]
            k = 2
            labels = np.random.randint(k, size=(X.shape[0]))
            labels = np.reshape(labels, newshape=(-1,1))

            reg_list = [0, 1e-6, 1e-3, 1e-0]
            y_list = [labels]
            for reg, y in list(itertools.product(reg_list, y_list)):
                test(X, y, layers, reg_lambda=reg)

            # Regression NN
            layers = [l1, l2, l3, l4, l5_d]
            y = np.random.uniform(-1, 1, (100, 1))
            reg_list = [0, 1e-6, 1e-3, 1e-0]
            for reg in reg_list:
                test(X, y, layers, reg_lambda=reg, regression=True)

if __name__ == '__main__':
    unittest.main()
