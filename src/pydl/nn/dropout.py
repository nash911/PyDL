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


class Dropout(object):
    """The Dropout Class
    """

    def __init__(self, p, activation_fn, name=None):
        self._p = p
        self._activation_fn = activation_fn
        self._out_mask = None
        self._name = name

    # Getters
    # -------
    @property
    def p(self):
        return self._p

    @property
    def mask(self):
        return self._out_mask


    # Setters
    # -------
    @p.setter
    def p(self, p):
        self._p = p

    @mask.setter
    def mask(self, mask):
        self._out_mask = mask


    def forward(self, X, mask=None):
        if mask is None:
            if self._activation_fn in ['Linear', 'ReLU']:
                self._out_mask = np.array(np.random.rand(*X.shape) < self._p,
                                          dtype=conf.dtype) / self._p
            elif self._activation_fn in ['Sigmoid', 'Tanh', 'SoftMax']:
                self._out_mask = np.array(np.random.rand(*X.shape) < self._p, dtype=conf.dtype)
            else:
                print("Error: Dropout - Unknown Activation Fn.: ", self._activation_fn)
                sys.exit(" ")
        else:
            self._out_mask = mask
        return self._out_mask * X


    def backward(self, inp_grad, inputs=None):
        if self._out_mask is None:
            if inputs is None:
                sys.exit("Error: Please provide inputs for a Dropout forward pass, before " +
                         "a backward pass")
            else:
                _ = self.forward(inputs)

        out_grad = self._out_mask * inp_grad
        self._out_mask = None

        return out_grad
