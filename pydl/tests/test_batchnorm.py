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

from pydl.nn.batchnorm import BatchNorm
from pydl import conf

class TestBatchNorm(unittest.TestCase):
    def test_forward(self):
        def test(inp):
            bn = BatchNorm(feature_size=inp.shape[-1])
            bn_out = bn.forward(inp)
            bn_mu = np.mean(bn_out, axis=0)
            bn_std = np.std(bn_out, axis=0)
            npt.assert_almost_equal(bn_mu, np.zeros_like(bn_mu), decimal=5)
            npt.assert_almost_equal(bn_std, np.ones_like(bn_std), decimal=5)

        # Manually calculated
        # -------------------
        X = np.reshape(np.arange(39483, dtype=conf.dtype), (321, 123))
        test(X)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [2, 3, 6, 16, 64, 256, 512, 1024, 4096]
        feature_size = [1, 2, 3, 6, 11, 25, 100, 1000, 3000]
        scale = [1e-8, 1e-6, 1e-3, 1e-1, 1e-0, 2, 1e+1, 1e+2, 1e+3, 1e+4, 1e+6, 1e+8, 1e+16, 1e+32]

        for batch, feat, scl in list(itertools.product(batch_size, feature_size, scale)):
            X_uniform = np.random.uniform(-scl, scl, (batch, feat))
            X_normal = np.random.randn(batch, feat) * scl
            test(X_uniform)
            test(X_normal)


    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-4
        def test(inp, gamma, beta, inp_grad):
            bn = BatchNorm(gamma=gamma, beta=beta, feature_size=inp.shape[-1])
            bn_fwd = bn.forward(inp)
            x_grad = bn.backward(inp_grad)
            gamma_grad = bn.gamma_grad
            beta_grad = bn.beta_grad

            # Gamma finite difference gradients
            gamma_finite_diff = np.empty(gamma.shape)
            for i in range(gamma_grad.shape[0]):
                gamma_delta = np.zeros(gamma.shape, dtype=conf.dtype)
                gamma_delta[i] = self.delta
                bn.gamma = gamma + gamma_delta
                lhs = bn.forward(inp)
                bn.gamma = gamma - gamma_delta
                rhs = bn.forward(inp)
                gamma_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)
            bn.gamma = gamma

            # Beta finite difference gradients
            beta_finite_diff = np.empty(beta.shape)
            for i in range(beta_grad.shape[0]):
                beta_delta = np.zeros(beta.shape, dtype=conf.dtype)
                beta_delta[i] = self.delta
                bn.beta = beta + beta_delta
                lhs = bn.forward(inp)
                bn.beta = beta - beta_delta
                rhs = bn.forward(inp)
                beta_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)
            bn.beta = beta

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inp.shape)
            for i in range(inp.shape[0]):
                inp_delta = np.zeros(inp.shape, dtype=conf.dtype)
                inp_delta[i] = self.delta
                lhs = bn.forward(inp + inp_delta)
                rhs = bn.forward(inp - inp_delta)
                inputs_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad, axis=0)

            npt.assert_almost_equal(gamma_grad, gamma_finite_diff, decimal=3)
            npt.assert_almost_equal(beta_grad, beta_finite_diff, decimal=3)
            npt.assert_almost_equal(x_grad, inputs_finite_diff, decimal=3)


        # Manually calculated
        # -------------------
        X = np.reshape(np.arange(320, dtype=conf.dtype), (16, 20))
        gamma = np.ones(X.shape[-1], dtype=conf.dtype)
        beta = np.zeros(X.shape[-1], dtype=conf.dtype)
        inp_grad = np.random.uniform(0.0, 0.1, (16, 20))
        test(X, gamma, beta, inp_grad)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [16, 64, 256, 512]
        feature_size = [1, 2, 3, 6, 11, 25, 100]
        inp_scale = [1e-3, 1e-2, 1e-1, 1e-0, 2, 1e+1, 1e+2, 1e+3]
        unit_inp_grad = [True, False]

        for _ in range(1):
            for batch, feat, i_scl, unit in list(itertools.product(batch_size, feature_size,
                                                                   inp_scale, unit_inp_grad)):
                X_uniform = np.random.uniform(-i_scl, i_scl, (batch, feat))
                X_normal = np.random.randn(batch, feat) * i_scl

                gamma = np.random.randn(feat)
                beta = np.random.randn(feat)
                inp_grad = np.ones((batch, feat), dtype=conf.dtype) if unit else \
                           np.random.uniform(-0.0001, 0.0001, (batch, feat))
                test(X_normal, gamma, beta, inp_grad)
                test(X_uniform, gamma, beta, inp_grad)


if __name__ == '__main__':
    unittest.main()
