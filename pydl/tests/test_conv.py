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

from pydl.nn.conv import Conv
from pydl import conf

class TestConv(unittest.TestCase):
    def test_score_fn(self):
        def test(inp, w, true_out, pad=0, stride=1, bias=False):
            if true_out is None:
                with npt.assert_raises(SystemExit):
                    conv = Conv(inp, zero_padding=pad, stride=stride, weights=w, bias=bias)
                    out_volume = conv.score_fn(inp)
            else:
                conv = Conv(inp, zero_padding=pad, stride=stride, weights=w, bias=bias)
                out_volume = conv.score_fn(inp)
                npt.assert_almost_equal(out_volume, true_out, decimal=8)

        def unroll_input_volume(inp, w, batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad, strd):
            input_padded = np.pad(inp, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant') \
                           if pad > 0 else inp
            out_h = int((inp_h - k_h + (2*pad)) / strd) + 1
            out_w = int((inp_w - k_w + (2*pad)) / strd) + 1
            out_rows = out_h * out_w
            out_cols = w[0].size
            unrolled_inp_shape = tuple((batch, out_rows, out_cols))
            unrolled_inp = np.empty(unrolled_inp_shape)
            kernal_size = k_h * k_w

            for b in range(batch):
                for row in range(out_rows):
                    for col in range(out_cols):
                        inp_dep = int(col/kernal_size)
                        inp_row = int(row/out_w)*strd + int((col%kernal_size)/k_w)
                        inp_col = int(row%out_w)*strd + (col%k_w)
                        unrolled_inp[b, row, col] = input_padded[b, inp_dep, inp_row, inp_col]

            return unrolled_inp


        # Manually calculated
        # -------------------
        inp = np.array([[[2, 0, 0, 1, 2], # Slice-1
                         [0, 1, 2, 0, 0],
                         [1, 1, 2, 0, 2],
                         [1, 1, 1, 1, 0],
                         [2, 0, 2, 2, 1]],
                        [[1, 1, 1, 2, 1], # Slice-2
                         [2, 1, 0, 0, 0],
                         [1, 2, 0, 1, 1],
                         [1, 2, 0, 2, 2],
                         [2, 0, 0, 1, 0]],
                        [[1, 0, 1, 0, 2], # Slice-3
                         [2, 0, 1, 0, 1],
                         [0, 2, 0, 1, 1],
                         [0, 0, 0, 0, 2],
                         [0, 2, 2, 0, 1]]], dtype=conf.dtype)
        X = np.array([inp, inp*2])

        w = np.array([[[[0, 1, 1], # Kernal-1
                        [1, 0, 1],
                        [1, 1, 1]],
                       [[0, 1, 1],
                        [1, -1, 1],
                        [1, 1, 0]],
                       [[1, 1, 1],
                        [0, -1, 0],
                        [1, 1, 1]]],
                      [[[-1, -1, -1], # Kernal-2
                        [1, 0, 0],
                        [0, 1, 0]],
                       [[-1, 0, -1],
                        [0, -1, 0],
                        [-1, 0, 1]],
                       [[-1, 1, -1],
                        [1, -1, 1],
                        [-1, 1, -1]]]], dtype=conf.dtype)

        bias = np.array([1, 0], dtype=conf.dtype)

        out = np.array([[[4, 7, 1], # Slice-1
                         [11, 12, 7],
                         [3, 5, 6]],
                        [[1, 0, -1], # Slice-2
                         [4, 2, 0],
                         [-4, -7, 0]]], dtype=conf.dtype)
        true_out = np.array([out, out*2])

        test(X, w, true_out+bias.reshape(2, 1, 1), pad=1, stride=2, bias=bias)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [1, 2, 3, 5, 7]
        inp_width = [1, 2, 3, 5, 7]
        num_kernals = [1, 2, 3, 8]
        kernal_height = [1, 3, 5]
        kernal_width = [1, 3, 5]
        zero_padding = [0, 1, 2, 3]
        stride = [1, 2, 3]
        # scale = [1e-6, 1e-3, 1e-1, 1e-0, 2, 3, 10]

        for batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad, strd in \
            list(itertools.product(batch_size, inp_depth, inp_height, inp_width, num_kernals,
                                   kernal_height, kernal_width, zero_padding, stride)):
            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))
            w = np.random.randn(num_k, dep, k_h, k_w) * 1.0
            bias = np.zeros(num_k)

            out_height = (inp_h - k_h + (2*pad))/strd + 1
            out_width = (inp_w - k_w + (2*pad))/strd + 1

            if (out_height % 1 != 0 or out_width % 1 != 0):
                true_out = None
            elif (k_h > inp_h or k_w > inp_w):
                true_out = None
            else:
                unrolled_inp = unroll_input_volume(X, w, batch, dep, inp_h, inp_w, num_k, k_h, k_w,
                                                   pad, strd)
                weighted_sum = np.matmul(unrolled_inp, w.reshape(num_k, -1).T)
                true_out = weighted_sum.transpose(0, 2, 1).reshape(-1, num_k, int(out_height),
                                                                   int(out_width))

            test(X, w, true_out, pad=pad, stride=strd, bias=bias)


if __name__ == '__main__':
    unittest.main()
