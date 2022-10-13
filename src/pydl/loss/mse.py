# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2022] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np

from pyrl.loss.loss import Loss


class MSE(Loss):
    def __init__(self, name=None):
        super().__init__(name=name)
        self._prediction_delta = None

    def loss(self, y, y_pred, nn_weights=[]):
        #        1  m
        # MSE = --- ∑ (ŷᵢ - yᵢ)²
        #       2m  i
        self._prediction_delta = y - y_pred
        data_loss = 0.5 * np.mean(np.square(self._prediction_delta), axis=0)

        if self._lambda > 0:
            assert len(nn_weights) > 0

            regularization_loss = 0
            for w in nn_weights:
                if w is not None:
                    regularization_loss += np.sum(np.square(w))
            regularization_loss *= (0.5 * self._lambda)
        else:
            regularization_loss = 0

        # MSE Cost Fn.
        return data_loss + regularization_loss

    def gradient(self):
        # ∂MSE      1
        # ---- = - --- (ŷᵢ - yᵢ)
        #  ∂yᵢ      m
        loss_grad = -self._prediction_delta / self._prediction_delta.shape[0]

        self._prediction_delta = None
        return loss_grad
