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

from pydl.nn.pool import Pool
from pydl import conf

class TestPool(unittest.TestCase):
    def unroll_input_volume(self, inp, k_h, k_w, strd_r, strd_c):
        batch, dep, inp_h, inp_w = inp.shape

        strd_r = int(np.min([strd_r, inp_h, k_h]))
        strd_c = int(np.min([strd_c, inp_w, k_w]))

        if inp_h >= k_h:
            pad_h = ((inp_h - k_h) % strd_r)
        else:
            pad_h = k_h - inp_h

        if inp_w >= k_w:
            pad_w = ((inp_w - k_w) % strd_c)
        else:
            pad_w = k_w - inp_w

        if pad_h == 0 and pad_w == 0 and (inp_h-k_h) >= 0 and (inp_w-k_w) >= 0:
            pad_h_tuple = tuple((0,0))
            pad_w_tuple = tuple((0,0))
        else:
            if pad_h == 0 and (inp_h-k_h) >= 0:
                pad_h_tuple = tuple((0,0))
            else:
                pad_h_tuple = tuple((int(np.floor(pad_h/2)), int(np.ceil(pad_h/2))))

            if pad_w == 0 and (inp_w-k_w) >= 0:
                pad_w_tuple = tuple((0,0))
            else:
                pad_w_tuple = tuple((int(np.floor(pad_w/2)), int(np.ceil(pad_w/2))))

        input_padded = np.pad(inp, ((0,0),(0,0),pad_h_tuple,pad_w_tuple), 'constant',
                              constant_values=-np.inf)

        if inp_h >= k_h:
            out_h = int((inp_h - k_h + np.sum(pad_h_tuple)) / strd_r) + 1
        else:
            out_h = 1

        if inp_w >= k_w:
            out_w = int((inp_w - k_w + np.sum(pad_w_tuple)) / strd_c) + 1
        else:
            out_w = 1

        out_rows = out_h * out_w
        out_cols = k_h * k_w
        unrolled_inp_shape = tuple((batch, dep, out_rows, out_cols))
        unrolled_inp = np.empty(unrolled_inp_shape)
        kernal_size = k_h * k_w

        for b in range(batch):
            for d in range(dep):
                for row in range(out_rows):
                    for col in range(out_cols):
                        inp_row = int(row/out_w)*strd_r + int((col%kernal_size)/k_w)
                        inp_col = int(row%out_w)*strd_c + (col%k_w)
                        unrolled_inp[b, d, row, col] = input_padded[b, d, inp_row, inp_col]

        return unrolled_inp, out_h, out_w


    def test_forward(self):
        def test(inp, true_out, stride, rcp_field=None):
            pool = Pool(inp, receptive_field=rcp_field, stride=stride)
            pool_out = pool.forward(inp)
            npt.assert_almost_equal(pool_out, true_out, decimal=8)


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

        out = np.array([[[2, 2, 2], # Slice-1
                         [1, 2, 2],
                         [2, 2, 1]],
                        [[2, 2, 1], # Slice-2
                         [2, 2, 2],
                         [2, 1, 0]],
                        [[2, 1, 2], # Slice-3
                         [2, 1, 2],
                         [2, 2, 1]]], dtype=conf.dtype)
        true_out = np.array([out, out*2])

        test(X, true_out, stride=2, rcp_field=(2, 2))


        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [1, 2, 3, 5, 8]
        inp_width = [1, 2, 3, 5, 8]
        kernal_height = [1, 3, 5, 9]
        kernal_width = [1, 3, 5, 9]
        stride_r = [1, 2, 3, 4]
        stride_c = [1, 2, 3, 4]

        counter = 0

        for batch, dep, inp_h, inp_w, k_h, k_w, strd_r, strd_c in \
            list(itertools.product(batch_size, inp_depth, inp_height, inp_width, kernal_height,
                                   kernal_width, stride_r, stride_c)):

            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))

            unrolled_inp, out_height, out_width = self.unroll_input_volume(X, k_h, k_w, strd_r,
                                                                           strd_c)

            # Performing MaxPooling as true output on unrolled input tensor
            true_out = np.max(unrolled_inp, axis=-1).reshape(batch, dep, out_height, out_width)

            test(X, true_out, stride=(strd_r, strd_c), rcp_field=(k_h, k_w))


    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-8
        def test(inp, unit_inp_grad, stride=None, rcp_field=None):
            pool = Pool(inp, receptive_field=rcp_field, stride=stride)
            y = pool.forward(inp)
            inp_grad = np.ones_like(y, dtype=conf.dtype) if unit_inp_grad \
                       else np.random.uniform(-1, 1, y.shape)

            inputs_grad = pool.backward(inp_grad)

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    for k in range(inputs_grad.shape[2]):
                        for l in range(inputs_grad.shape[3]):
                            i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                            i_delta[i,j,k,l] = self.delta
                            lhs = pool.forward(inp + i_delta)
                            rhs = pool.forward(inp - i_delta)
                            inputs_finite_diff[i,j,k,l] = np.sum(((lhs-rhs) / (2*self.delta)) *
                                                                 inp_grad, keepdims=False)

            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=6)
            assert not np.isinf(inputs_grad).any()


        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [1, 5, 7]
        inp_width = [1, 5, 7]
        kernal_height = [1, 2, 3]
        kernal_width = [1, 2, 3]
        stride_r = [1, 2, 3, 4]
        stride_c = [1, 2, 3, 4]
        unit_inp_grad = [False]
        scale = [1e-0]

        for batch, dep, inp_h, inp_w, k_h, k_w, strd_r, strd_c, unit, scl in \
            list(itertools.product(batch_size, inp_depth, inp_height, inp_width, kernal_height,
                                   kernal_width, stride_r, stride_c, unit_inp_grad, scale)):

            X = np.random.uniform(-scl, scl, (batch, dep, inp_h, inp_w))

            test(X, unit_inp_grad=(True if unit else False), stride=(strd_r,strd_c),
                 rcp_field=(k_h,k_w))


if __name__ == '__main__':
    unittest.main()
