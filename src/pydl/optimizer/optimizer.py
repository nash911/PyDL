# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2022] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, step_size=1e-4, name=None):
        self._step_size = step_size
        self._name = name

    # Abstract Methods
    # ----------------
    @abstractmethod
    def update_network(self, t=None):
        pass
