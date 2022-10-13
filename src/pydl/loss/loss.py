# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2022] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    def __init__(self, reg_lambda=0, name=None):
        self._lambda = reg_lambda
        self._name = name

    # Abstract Methods
    # ----------------
    @abstractmethod
    def loss(self, y, y_pred, nn_weights=[]):
        pass

    @abstractmethod
    def gradient(self):
        pass
