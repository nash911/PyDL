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

from pydl.nn.dropout import Dropout
from pydl import conf

class TestDropout(unittest.TestCase):
    def test_forward(self):
        def test(inp, p, actvn_fn):
            do = Dropout(p=p, activation_fn=actvn_fn)
            do_out = do.forward(inp)
            p_out = np.mean(do_out)
            if actvn_fn in ['Linear', 'ReLU']:
                p_out *= p
            npt.assert_almost_equal(p_out, p, decimal=3)

        # Test Cases
        # ----------
        X = np.ones((10000, 1000), dtype=conf.dtype)
        dropout_probs = [1e-10, 1e-6, 1e-4, 1e-1, 0.25252525, 0.9999, 1]
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'SoftMax', 'ReLU']
        for p, actv in list(itertools.product(dropout_probs, activation_fn)):
            test(X, p=p, actvn_fn=actv)


if __name__ == '__main__':
    unittest.main()
