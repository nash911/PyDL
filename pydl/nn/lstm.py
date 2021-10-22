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
from collections import OrderedDict

from pydl.nn.layers import Layer
from pydl.nn.activations import Sigmoid
from pydl.nn.activations import Tanh
from pydl.nn.dropout import Dropout
from pydl import conf


class LSTM(Layer):
    """The LSTM Layer Class."""

    def __init__(self, inputs, num_neurons=None, weights=None, bias=True, seq_len=None, xavier=True,
                 weight_scale=1.0, architecture_type='many_to_many', dropout=None,
                 tune_internal_states=False, name=None):
        super().__init__(name=name)
        self._weights = OrderedDict()
        self._inputs = OrderedDict()
        self._cell_state = OrderedDict()
        self._hidden_state = OrderedDict()
        self._output = OrderedDict()
        self._init_cell_state = None
        self._init_hidden_state = None

        self._weights_grad = OrderedDict()
        self._out_grad = OrderedDict()

        if architecture_type.lower() not in ['many_to_many', 'many_to_one']:
            sys.exit("Error: Unknown model type in LSTM_Layer. Use either 'many_to_many' or " +
                     "'many_to_one'.")
        else:
            self._architecture_type = architecture_type.lower()

        self._type = 'LSTM_Layer'
        self._num_neurons = num_neurons
        self._seq_len = seq_len
        self._inp_size = np.prod(inputs.shape[1:])
        self._weight_scale = weight_scale
        self._xavier = xavier
        self._has_bias = True if type(bias) in [np.ndarray, float, int] else bias
        self._tune_internal_states = tune_internal_states
        self._update_init_internal_states = True

        # Initialize Weights
        if weights is not None:
            if num_neurons is not None:
                # Shape of the concatenated weight matrix should be (m+n, 4n)
                assert(weights.shape == ((num_neurons + self._inp_size), int(4 * num_neurons)))
            else:
                self._num_neurons = int(weights.shape[1] / 4)
            self._weights = weights
        else:
            self._weights = np.random.randn((self._num_neurons + self._inp_size),
                                            int(4 * self._num_neurons)) * weight_scale
            if xavier:
                # Apply Xavier Initialization
                norm_fctr = np.sqrt(self._num_neurons + self._inp_size)
                self._weights /= norm_fctr

        # Initialize Bias
        if type(bias) == np.ndarray:
            self._bias = bias
        elif type(bias) in [float, int]:
            self._bias = np.ones(int(4 * self._num_neurons), dtype=conf.dtype) * bias
        elif bias:
            self._bias = np.zeros(int(4 * self._num_neurons), dtype=conf.dtype)
        else:
            self._bias = None

        # Initialize i.f.o.g gates
        self._i_gate = [Sigmoid() for _ in range(self._seq_len + 1)]
        self._f_gate = [Sigmoid() for _ in range(self._seq_len + 1)]
        self._o_gate = [Sigmoid() for _ in range(self._seq_len + 1)]
        self._g_gate = [Tanh() for _ in range(self._seq_len + 1)]

        # Cell state activation function
        self._cell_state_activation_fn = [Tanh() for _ in range(self._seq_len + 1)]

        # Initialize Cell state
        self._init_cell_state = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        self._cell_state[0] = self._init_cell_state

        # Initialize Hidden state
        self._init_hidden_state = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        if self._tune_internal_states:
            self._hidden_state[0] = self._o_gate[0].forward(self._init_hidden_state) * \
                self._cell_state_activation_fn[0].forward(self._cell_state[0])
        else:
            self._hidden_state[0] = self._init_hidden_state

        self.reset_gradients()

        if dropout is not None and dropout < 1.0:
            if self._architecture_type == 'many_to_many':
                self._dropout = \
                    [Dropout(p=dropout, activation_fn='Linear') for _ in range(self._seq_len + 1)]
            else:  # Many-to-one
                self._dropout = Dropout(p=dropout, activation_fn='Linear')

    # Getters
    # -------
    @property
    def architecture_type(self):
        return self._architecture_type

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def init_cell_state(self):
        return self._init_cell_state

    @property
    def init_hidden_state(self):
        return self._init_hidden_state

    @property
    def shape(self):
        return (None, self._num_neurons)

    @property
    def size(self):
        return self.weights.size

    @property
    def num_neurons(self):
        return self._num_neurons

    @property
    def tune_internal_states(self):
        return self._tune_internal_states

    # Setters
    # -------
    @init_cell_state.setter
    def init_cell_state(self, c):
        assert(c.shape == self._init_cell_state.shape)
        np.copyto(self._init_cell_state, c)

    @init_hidden_state.setter
    def init_hidden_state(self, h):
        assert(h.shape == self._init_hidden_state.shape)
        np.copyto(self._init_hidden_state, h)

    def reinitialize_weights(self, inputs=None, num_neurons=None):
        num_feat = self._inp_size if inputs is None else np.prod(inputs.shape[1:])
        num_neurons = self._num_neurons if num_neurons is None else num_neurons

        # Reinitialize weights
        self._weights = np.random.randn((self._num_neurons + self._inp_size),
                                        int(4 * self._num_neurons)) * self._weight_scale
        if self._xavier:
            # Apply Xavier Initialization
            norm_fctr = np.sqrt(self._num_neurons + self._inp_size)
            self._weights /= norm_fctr

        # Reset layer size
        self._inp_size = num_feat
        self._num_neurons = num_neurons

        if self._has_bias:
            if np.all(self._bias == self._bias[0]):
                self._bias = np.ones(int(4 * num_neurons), dtype=conf.dtype) * self._bias[0]
            else:
                self._bias = np.zeros(int(4 * num_neurons), dtype=conf.dtype)

        self.reset_gradients()

    def reset_gradients(self):
        self._weights_grad = np.zeros_like(self._weights)
        self._bias_grad = np.zeros_like(self._bias)
        self._cell_state_grad = None
        self._hidden_state_grad = None
        self._out_grad = OrderedDict()

    def score_fn(self, inputs, weights=None):
        if weights is None:
            weighted_sum = np.matmul(inputs, self._weights)
        else:
            weighted_sum = np.matmul(inputs, weights)

        if self._has_bias:
            return weighted_sum + self._bias
        else:
            return weighted_sum

    def weight_gradients(self, inp_grad, inputs, reg_lambda=0):
        # d(ifog)/dw: Gradient of the layer's i.f.o.g gates w.r.t the weights 'w'
        grad = inputs.reshape(-1, 1) * inp_grad

        if reg_lambda > 0:
            grad += (reg_lambda * self._weights)
        return grad

    def bias_gradients(self, inp_grad):
        if not self._has_bias:
            return None
        else:
            # d(ifog)/db: Gradient of the layer's i.f.o.g gates w.r.t the bias 'b'
            grad = inp_grad.reshape(-1)
            return grad

    def input_gradients(self, inp_grad, summed=True):
        # d(ifog)/dx: Gradient of the layer's i.f.o.g gates w.r.t the inputs 'X' [h_(t-1), h_(l-1)]
        out_grad = self._weights * inp_grad

        if summed:
            out_grad = np.sum(out_grad, axis=-1, keepdims=False)

        # Return hidden state [h_(t-1)] gradients and input [h_(l-1)] gradients separately
        return out_grad[:self._num_neurons].reshape(1, -1), out_grad[self._num_neurons:]

    def forward(self, inputs, inference=False, mask=None, temperature=1.0):
        if len(inputs.shape) > 2:  # Preceeding layer is a Convolution/Pooling layer or 3D inputs
            try:
                inputs = inputs.squeeze(axis=0)
            except ValueError:
                # Unroll inputs
                batch_size = inputs.shape[0]
                inputs = inputs.reshape(batch_size, -1)

        for t, inp in enumerate(inputs[:, np.newaxis, :], start=1):
            concat_inputs = np.concatenate((self._hidden_state[t - 1], inp), axis=-1)

            # Store concatenated inputs in dict for backprop
            self._inputs[t] = concat_inputs

            # Sum of weighted inputs
            z = self.score_fn(concat_inputs)

            # Calculate i.f.o.g gates
            i_gate = self._i_gate[t].forward(z[:, :self._num_neurons])
            f_gate = self._f_gate[t].forward(z[:, self._num_neurons:int(2 * self._num_neurons)])
            o_gate = \
                self._o_gate[t].forward(z[:, int(2 * self._num_neurons):int(3 * self._num_neurons)])
            g_gate = self._g_gate[t].forward(z[:, int(3 * self._num_neurons):])

            # Update Cell and Hidden states
            # cₜ = f ⊙ cₜ-₁ + i ⊙ g
            self._cell_state[t] = (f_gate * self._cell_state[t - 1]) + (i_gate * g_gate)
            # hₜ = o ⊙ tanh(cₜ)
            self._hidden_state[t] = \
                o_gate * self._cell_state_activation_fn[t].forward(self._cell_state[t])

            # Dropout
            if self._dropout is not None:
                if not inference:  # Training step
                    if self.dropout_mask is None:
                        drop_mask = None if mask is None else mask[t - 1]
                    else:
                        drop_mask = self.dropout_mask[t - 1]

                    # Apply Dropout Mask
                    try:  # Case: Many-to-many
                        self._output[t] = self._dropout[t].forward(self._hidden_state[t], drop_mask)
                    except TypeError:  # Case: Many-to-one
                        if t == inputs.shape[0]:
                            self._output[t] = \
                                self._dropout.forward(self._hidden_state[t], drop_mask)
                else:  # Inference
                    # Inverse Dropout - So the gradients just flow through
                    self._output[t] = self._hidden_state[t]
            else:
                self._output[t] = self._hidden_state[t]

        if self._architecture_type == 'many_to_one':
            # Pass through the output of the final sequence only
            single_out_dict = OrderedDict()
            single_out_dict[list(self._output.keys())[-1]] = list(self._output.values())[-1]
            return single_out_dict
        else:
            return self._output

    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        cell_grad = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        hidden_grad = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        for t in reversed(range(1, self._seq_len + 1)):
            # Add gradients from the next (t+1) hidden sequence and from the layer above (l+1)
            if self._architecture_type == 'many_to_one':
                if t <= list(inp_grad.keys())[-1]:
                    if t == list(inp_grad.keys())[-1]:
                        # Backpropagating through Dropout
                        if self._dropout is not None:
                            drop_grad = self._dropout.backward(inp_grad[t])
                        else:
                            drop_grad = inp_grad[t]
                        grad = hidden_grad + drop_grad
                    else:
                        grad = hidden_grad
                        if len(grad.shape) == 1:
                            grad = np.expand_dims(grad, axis=0)
                else:
                    continue
            else:  # Many-to-one
                try:
                    # Backpropagating through Dropout
                    if self._dropout is not None:
                        drop_grad = self._dropout[t].backward(inp_grad[t])
                    else:
                        drop_grad = inp_grad[t]
                    grad = hidden_grad + drop_grad
                except KeyError:
                    continue

            # Outputs of gate activations
            i_out = self._i_gate[t].output
            f_out = self._f_gate[t].output
            o_out = self._o_gate[t].output
            g_out = self._g_gate[t].output
            cell_state_tanh = self._cell_state_activation_fn[t].output

            # Add gradients from the next cell state sequence (c_t+1) and output gate
            cell_grad += self._cell_state_activation_fn[t].backward(o_out * grad)

            # Output gate gradients
            o_gate_grad = self._o_gate[t].backward(cell_state_tanh * grad)

            # Input gate gradients
            i_gate_grad = self._i_gate[t].backward(g_out * cell_grad)

            # Gate gate gradients
            g_gate_grad = self._g_gate[t].backward(i_out * cell_grad)

            # Forget gate gradients
            f_gate_grad = self._f_gate[t].backward(self._cell_state[t - 1] * cell_grad)

            # Cell state graients
            cell_grad = f_out * cell_grad

            # Concatinate i.f.o.g gradients to backprop through weights and inputs
            concat_grads = np.concatenate((i_gate_grad, f_gate_grad, o_gate_grad, g_gate_grad),
                                          axis=-1)

            # d(ifog)/dw: Gradient of the layer's i.f.o.g gates w.r.t the weights 'w'
            self._weights_grad += \
                self.weight_gradients(concat_grads, self._inputs[t], reg_lambda)

            # d(ifog)/db: Gradient of the layer's i.f.o.g gates w.r.t the bias 'b'
            if self._has_bias:
                self._bias_grad += self.bias_gradients(concat_grads)

            # d(ifog)/dx: Grads of the layer's i.f.o.g gates w.r.t the inputs 'X' [h_(t-1), h_(l-1)]
            hidden_grad, self._out_grad[t] = self.input_gradients(concat_grads)

        if self._tune_internal_states and self._update_init_internal_states:
            o_out = self._o_gate[0].output
            cell_state_tanh = self._cell_state_activation_fn[0].output

            # Gradients of the initial cell state (c_0)
            if len(cell_grad.shape) == 1:
                cell_grad = np.expand_dims(cell_grad, axis=0)
            self._cell_state_grad = \
                self._cell_state_activation_fn[0].backward(o_out * hidden_grad) + cell_grad

            # Gradients of the initial hidden state (h_0)
            if len(hidden_grad.shape) == 1:
                hidden_grad = np.expand_dims(hidden_grad, axis=0)
            self._hidden_state_grad = self._o_gate[0].backward(cell_state_tanh * hidden_grad)

            self._update_init_internal_states = False
        else:
            self._cell_state_grad = None
            self._hidden_state_grad = None

        return self._out_grad

    def update_weights(self, alpha):
        self._weights += -alpha * self._weights_grad

        if self._has_bias:
            self._bias += -alpha * self._bias_grad

        self.reset()

    def reset(self):
        self.reset_gradients()
        self.reset_internal_states(hidden_state='previous_state', cell_state='previous_state')

    def reset_internal_states(self, hidden_state=None, cell_state=None):
        try:
            if hidden_state.lower() == 'previous_state':
                hidden_state = list(self._hidden_state.values())[-1]
                cell_state = list(self._cell_state.values())[-1]
        except AttributeError:
            pass

        try:
            if cell_state.lower() == 'previous_state':
                cell_state = list(self._cell_state.values())[-1]
                hidden_state = list(self._hidden_state.values())[-1]
        except AttributeError:
            pass

        self._inputs = OrderedDict()
        self._cell_state = OrderedDict()
        self._hidden_state = OrderedDict()
        self._output = OrderedDict()

        if cell_state is None:
            self._cell_state[0] = self._init_cell_state
            self._update_init_internal_states = True
        else:
            self._cell_state[0] = cell_state

        if hidden_state is None:
            if self._tune_internal_states:
                self._hidden_state[0] = self._o_gate[0].forward(self._init_hidden_state) * \
                    self._cell_state_activation_fn[0].forward(self._cell_state[0])
                self._update_init_internal_states = True
            else:
                self._hidden_state[0] = self._init_hidden_state
        else:
            self._hidden_state[0] = hidden_state
