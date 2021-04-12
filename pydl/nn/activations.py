# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import abc
from abc import ABC, abstractmethod
import numpy as np
import itertools

from pydl import conf

class Activation(ABC):
    """An abstract class defining the interface of Activation function.
    Args:
        name (str): Name of the activation function.
    """

    def __init__(self, name=None, type=None):
        self._name = name
        self._type = type
        self._inputs = None
        self._outputs = None

    # Getters
    # -------
    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    # Abstract Methods
    # ----------------
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, inp_grad):
        pass


class Sigmoid(Activation):
    """The Sigmoid Class
    """

    def __init__(self, name=None):
        super().__init__(name=name, type='Sigmoid')


    def forward(self, inputs):
        self._inputs = inputs
        self._outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self._outputs

    def backward(self, inp_grad, inputs=None):
        if self._outputs is None:
            self.forward(inputs)
        elif inputs is not None:
            self.forward(inputs)

        # Gradient of the output of the sigmoid fn w.r.t the logits 'z'
        # d/dz σ(z) = σ(z)(1 - σ(z))
        out_grad = self._outputs * (1.0 - self._outputs) * inp_grad
        self._outputs = None
        return out_grad


class SoftMax(Activation):
    """The SoftMax Class
    """

    def __init__(self, name=None):
        super().__init__(name=name, type='SoftMax')

    def forward(self, inputs):
        # Normalization trick to avoid overflow
        self._inputs = inputs - np.amax(inputs, axis=-1, keepdims=True)
        self._outputs = np.exp(self._inputs) / np.sum(np.exp(self._inputs), axis=-1, keepdims=True)
        return self._outputs

    def backward(self, inp_grad, inputs=None):
        if self._outputs is None:
            self.forward(inputs)
        elif inputs is not None:
            self.forward(inputs)

        batch_size = self._outputs.shape[0]
        num_neurons = self._outputs.shape[-1]

        # A syntactic sugar implementation
        delta_ij = np.array([np.eye(num_neurons)]*batch_size) * self._outputs[:,np.newaxis,:]
        out_grad = np.sum((-(self._outputs[:,:,np.newaxis] * self._outputs[:,np.newaxis,:]) +
                           delta_ij) * inp_grad[:,np.newaxis,:], axis=-1)

        # # Syntactic sugar implementation - Has a slight better performance (~5%)
        # out_squared = -(self._outputs[:,:,np.newaxis] * self._outputs[:,np.newaxis,:])
        # diag_indx = list(range(num_neurons)) * batch_size
        # out_squared[list(itertools.chain.from_iterable(itertools.repeat(b, num_neurons) for b in
        #             range(batch_size))), diag_indx, diag_indx] += np.reshape(self._outputs, (-1))
        # out_grad = np.sum(out_squared * inp_grad[:,np.newaxis,:], axis=-1)

        self._outputs = None
        return out_grad


class ReLU(Activation):
    """The ReLU Class
    """

    def __init__(self, name=None):
        super().__init__(name=name, type='ReLU')

    def forward(self, inputs):
        self._inputs = inputs
        self._outputs = np.maximum(0, inputs)
        return self._outputs

    def backward(self, inp_grad, inputs=None):
        if self._outputs is None:
            self.forward(inputs)
        elif inputs is not None:
            self.forward(inputs)

        # Gradient of the output of the ReLU fn w.r.t the logits 'z'
        # d/dz ReLU(z) = 0 if z < 0 else 1
        out_grad = (np.where(self._inputs <= 0, 0, 1) +
                    np.where(self._inputs == 0, np.random.uniform(0.5, 1), 0)) * inp_grad
        self._outputs = None
        return out_grad
