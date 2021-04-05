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
    def test_split_data(self):
        self.maxDiff = None
        def test(X, y, train_size=70, test_size=30):
            train = Training(nn=None)
            train_X, train_y, test_X, test_y = train.split_data(X, y, train_size, test_size)
            concat_X = np.vstack((train_X, test_X))
            concat_y = np.vstack((train_y, test_y))

            y_onehot = np.zeros((y.size, y.max()+1))
            y_onehot[np.arange(y.size), y] = 1
            Xy = np.sum(X * y_onehot, axis=-1, keepdims=False)
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
        test(X, y)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [8, 11, 100, 256]
        feature_size = [1, 2, 3, 6, 11]
        train_size = [0, 1, 5, 9.12345, 25.252525, 49.0, 70, 99.99]
        for batch, feat, t_sz in list(itertools.product(batch_size, feature_size, train_size)):
            X = np.random.uniform(-1, 1, (batch, feat))
            y = np.random.randint(feat, size=(batch))
            y[-1] = feat-1
            test(X, y, t_sz, (100.0 - t_sz))




    def test_loss(self):
        def test(X, y, prob, true_out):
            train = Training(nn=None)
            loss = train.loss(X, y, prob=prob)
            self.assertEqual(loss, true_out)

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
        test(X, y_onehot, prob, true_out)
        # Case-2
        y = np.array([0, 1, 2, 3])
        y_onehot = np.zeros((y.size, y.max()+1))
        y_onehot[np.arange(y.size), y] = 1
        true_out = ((-np.log(0.1) * 2) + (-np.log(0.3) * 2)) / 4
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
            test(X, y_onehot, prob, true_out)


    def test_loss_gradient_finite_diff(self):
        self.delta = 1e-2
        def test(X, y, layers, reg_lambda=0):
            nn = NN(X, layers)
            train = Training(nn, reg_lambda=reg_lambda)
            loss = train.loss(X, y)
            loss_grad = train.loss_gradient(X, y)
            inputs_grad = nn.backward(loss_grad, reg_lambda=reg_lambda)

            for n, layer in enumerate(layers):
                w = layer.weights
                weights_grad = layer.weights_grad

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

                npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=6)
                layer.weights = w

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    i_delta = np.zeros(X.shape, dtype=conf.dtype)
                    i_delta[i,j] = self.delta
                    inputs_finite_diff[i,j] = ((train.loss(X + i_delta, y) -
                                                train.loss(X - i_delta, y)) / (2 * self.delta))

            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=3)

        for _ in range(5):
            # Manually calculated
            # NN Architecture
            # Layer 1 - Sigmoid
            X = np.random.uniform(-1, 1, (100, 25))
            w_1 = np.random.uniform(-1, 1, (X.shape[-1], 19))
            b_1 = np.random.uniform(-1, 1, (1, 19))

            # Layer 2
            w_2 = np.random.uniform(-1, 1, (w_1.shape[-1], 15))
            b_2 = np.random.uniform(-1, 1, (1, 15))

            # Layer 3
            w_3 = np.random.uniform(-1, 1, (w_2.shape[-1], 11))
            b_3 = np.random.uniform(-1, 1, (1, 11))

            # Layer 4
            w_4 = np.random.uniform(-1, 1, (w_3.shape[-1], 9))
            b_4 = np.random.uniform(-1, 1, (1, 9))

            # Layer 5
            w_5 = np.random.uniform(-1, 1, (w_4.shape[-1], 7))
            b_5 = np.random.uniform(-1, 1, (1, 7))

            l1 = FC(X, w_1.shape[-1], w_1, b_1, activation_fn='Sigmoid')
            l2 = FC(l1, w_2.shape[-1], w_2, b_2, activation_fn='Sigmoid')
            l3 = FC(l2, w_3.shape[-1], w_3, b_3, activation_fn='Sigmoid')
            l4 = FC(l3, w_4.shape[-1], w_4, b_4, activation_fn='Sigmoid')
            l5 = FC(l4, w_5.shape[-1], w_5, b_5, activation_fn='SoftMax')

            # 5-Layers
            layers = [l1, l2, l3, l4, l5]
            k = layers[-1].shape[-1]
            y = np.random.randint(k, size=(X.shape[0]))
            y_onehot = np.zeros((y.size, y.max()+1))
            y_onehot[np.arange(y.size), y] = 1
            test(X, y_onehot, layers, reg_lambda=0) # λ=0
            test(X, y_onehot, layers, reg_lambda=1e-6) # λ=1e-6
            test(X, y_onehot, layers, reg_lambda=1e-4) # λ=1e-4
            test(X, y_onehot, layers, reg_lambda=1e-2) # λ=1e-2
            test(X, y_onehot, layers, reg_lambda=1e-0) # λ=1.0


if __name__ == '__main__':
    unittest.main()
