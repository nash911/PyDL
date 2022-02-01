# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
import sys
from pydl import conf


class BatchNorm(object):
    """The Batch Norm Class."""

    def __init__(self, gamma=None, beta=None, feature_size=None, momentum=0.99, name=None):
        if gamma is not None:
            self._gamma = gamma
        elif feature_size is not None:
            self._gamma = np.ones(feature_size, dtype=conf.dtype)
        else:
            sys.exit("Error: Please provide either initial gamma params of feature_size while " +
                     "initiating Batchorm object")

        if beta is not None:
            self._beta = beta
        elif feature_size is not None:
            self._beta = np.zeros(feature_size, dtype=conf.dtype)
        else:
            sys.exit("Error: Please provide either initial beta params of feature_size while " +
                     "initiating Batchorm object")

        self._momentum = momentum if type(momentum) in ['int', 'float'] else 0.99
        if self._momentum < 0 and self._momentum > 1.0:
            sys.exit("Error: In Batchnorm momentum should be in range [0, 1]")

        self._avg_mean = None
        self._avg_var = None
        self._std_eps = None
        self._X_norm = None
        self._gamma_grad = None
        self._beta_grad = None
        self._out_grad = None

    # Getters
    # -------
    @property
    def avg_mean(self):
        return self._avg_mean

    @property
    def avg_var(self):
        return self._avg_var

    @property
    def gamma(self):
        return self._gamma

    @property
    def beta(self):
        return self._beta

    @property
    def gamma_grad(self):
        return self._gamma_grad

    @property
    def beta_grad(self):
        return self._beta_grad

    # Setters
    # -------
    @avg_mean.setter
    def avg_mean(self, avg_mean):
        assert(avg_mean.shape == self._gamma.shape)
        self._avg_mean = avg_mean

    @avg_var.setter
    def avg_var(self, avg_var):
        assert(avg_var.shape == self._gamma.shape)
        self._avg_var = avg_var

    @gamma.setter
    def gamma(self, g):
        assert(g.shape == self._gamma.shape)
        self._gamma = g

    @beta.setter
    def beta(self, b):
        assert(b.shape == self._beta.shape)
        self._beta = b

    def reinitialize_params(self, feature_size):
        self._avg_mean = None
        self._avg_var = None
        self._gamma = np.ones(feature_size, dtype=conf.dtype)
        self._beta = np.zeros(feature_size, dtype=conf.dtype)

    def forward(self, X, inference=False):
        if inference:
            if self._avg_mean is None or self._avg_var is None:
                return X
            else:
                X_mean = self._avg_mean
                X_var = self._avg_var
        else:
            X_mean = np.mean(X, axis=0)
            X_var = np.var(X, axis=0)

            if self._avg_mean is None or self._avg_var is None:
                self._avg_mean = X_mean
                self._avg_var = X_var
            else:
                self._avg_mean = self._avg_mean * self._momentum + X_mean * (1.0 - self._momentum)
                self._avg_var = self._avg_var * self._momentum + X_var * (1.0 - self._momentum)

        self._std_eps = np.sqrt(X_var + 1e-32)
        self._X_norm = (X - X_mean) / self._std_eps

        #              (X - μ)
        # BNᵧ,ᵦ(X) = ɣ --------- + 𝛃
        #             √(σ² + Ɛ)

        bn = (self._X_norm * self._gamma) + self._beta
        return bn

    def backward(self, inp_grad, inputs=None):
        if self._X_norm is None or self._std_eps is None:
            if inputs is None:
                sys.exit("Error: Please provide inputs for a BatchNorm forward pass, before " +
                         "a backward pass")
            else:
                _ = self.forward(inputs)

        # ∂BNᵧ,ᵦ(X)   m   ∂σ(z)
        # -------- = ∑  -------- x̂ᵢ
        #   ∂ɣ      i=1  ∂BN(xᵢ)
        self._gamma_grad = np.sum(self._X_norm * inp_grad, axis=0)

        # ∂BNᵧ,ᵦ(X)   m   ∂σ(z)
        # -------- = ∑  --------
        #   ∂𝛃      i=1  ∂BN(xᵢ)
        self._beta_grad = np.sum(inp_grad, axis=0)

        #               ∂BN    m  ∂BN           ∂BN
        #            m ----- - ∑ ----- - x̂ᵢ∑  ----- . x̂ⱼ
        # ∂BNᵧ,ᵦ(X)     ∂x̂ᵢ  j=1 ∂x̂ⱼ      j=1 ∂x̂ⱼ
        # -------- = -------------------------------------
        #   ∂xᵢ                   m √(σ² + Ɛ)

        M = inp_grad.shape[0]
        x_norm_grad = self._gamma * inp_grad
        self._out_grad = ((M * x_norm_grad) - (np.sum(x_norm_grad, axis=0, keepdims=True)) -
                          (self._X_norm * np.sum(x_norm_grad * self._X_norm, axis=0,
                                                 keepdims=True))) / (M * self._std_eps)

        self._X_norm = None
        self._std_eps = None

        return self._out_grad

    def update_params(self, alpha):
        self._gamma += self._gamma * alpha
        self._beta += self._beta * alpha
