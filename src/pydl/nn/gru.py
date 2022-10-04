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


class GRU(Layer):
    """The GRU Layer Class."""

    def __init__(self, inputs, num_neurons=None, weights=None, bias=True, seq_len=None, xavier=True,
                 weight_scale=1.0, architecture_type='many_to_many', dropout=None,
                 reset_pre_transform=True, tune_internal_states=False, name=None):
        super().__init__(name=name)
        self._weights = OrderedDict()
        self._bias = OrderedDict()
        self._inputs = OrderedDict()
        self._candidate_activation = OrderedDict()
        self._hidden_state = OrderedDict()
        self._output = OrderedDict()
        self._init_hidden_state = None

        self._weights_grad = OrderedDict()
        self._bias_grad = OrderedDict()
        self._out_grad = OrderedDict()

        if architecture_type.lower() not in ['many_to_many', 'many_to_one']:
            sys.exit("Error: Unknown model type in GRU_Layer. Use either 'many_to_many' or " +
                     "'many_to_one'.")
        else:
            self._architecture_type = architecture_type.lower()

        self._type = 'GRU_Layer'
        self._num_neurons = num_neurons
        self._seq_len = seq_len
        self._inp_size = np.prod(inputs.shape[1:])
        self._weight_scale = weight_scale
        self._xavier = xavier
        self._has_bias = True if type(bias) in [np.ndarray, float, int, dict, OrderedDict] else bias
        self._reset_pre_transform = reset_pre_transform
        self._tune_internal_states = tune_internal_states
        self._update_init_internal_states = True

        # Initialize Weights
        if weights is not None:
            if num_neurons is not None:
                # Shape of the concatenated gates weight matrix should be (m+n, 2n)
                assert(weights['gates'].shape == ((num_neurons + self._inp_size),
                                                  int(2 * num_neurons)))
                # Shape of the candidate weight matrix should be (m+n, n)
                assert(weights['candidate'].shape == ((num_neurons + self._inp_size), num_neurons))
            else:
                self._num_neurons = weights['candidate'].shape[1]
            self._weights = weights
        else:
            self._weights['gates'] = np.random.randn((self._num_neurons + self._inp_size),
                                                     int(2 * self._num_neurons)) * weight_scale
            self._weights['candidate'] = np.random.randn((self._num_neurons + self._inp_size),
                                                         self._num_neurons) * weight_scale
            if xavier:
                # Apply Xavier Initialization
                norm_fctr = np.sqrt(self._num_neurons + self._inp_size)
                self._weights['gates'] /= norm_fctr
                self._weights['candidate'] /= norm_fctr

        # Initialize Bias
        if type(bias) in [dict, OrderedDict]:
            # Shape of the concatenated gates bias should be (2n)
            assert(bias['gates'].shape == (int(2 * self._num_neurons),))

            # Shape of the candidate bias should be (n)
            assert(bias['candidate'].shape == (self._num_neurons,))

            self._bias = bias
        elif type(bias) in [float, int]:
            self._bias['gates'] = \
                np.hstack((np.zeros(self._num_neurons, dtype=conf.dtype),
                           np.ones(int(self._num_neurons), dtype=conf.dtype) * bias))
            self._bias['candidate'] = np.zeros(self._num_neurons, dtype=conf.dtype)
        elif bias:
            self._bias['gates'] = np.zeros(int(2 * self._num_neurons), dtype=conf.dtype)
            self._bias['candidate'] = np.zeros(self._num_neurons, dtype=conf.dtype)
        else:
            self._bias = None

        # Initialize gates
        self._update_gate = [Sigmoid() for _ in range(self._seq_len + 1)]
        self._reset_gate = [Sigmoid() for _ in range(self._seq_len + 1)]

        # Candidate activation function
        self._candidate_activation_fn = [Tanh() for _ in range(self._seq_len + 1)]

        # Initialize Hidden state
        self._init_hidden_state = np.zeros((1, self.num_neurons), dtype=conf.dtype)
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
    def weights(self):
        return np.hstack((self._weights['gates'], self._weights['candidate']))

    @property
    def gates_weights(self):
        return self._weights['gates']

    @property
    def candidate_weights(self):
        return self._weights['candidate']

    @property
    def bias(self):
        return np.concatenate((self._bias['gates'], self._bias['candidate']), axis=0)

    @property
    def gates_bias(self):
        return self._bias['gates']

    @property
    def candidate_bias(self):
        return self._bias['candidate']

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

    @property
    def weights_grad(self):
        return np.hstack((self._weights_grad['gates'], self._weights_grad['candidate']))

    @property
    def gates_weights_grad(self):
        return self._weights_grad['gates']

    @property
    def candidate_weights_grad(self):
        return self._weights_grad['candidate']

    @property
    def bias_grad(self):
        return np.concatenate((self._bias_grad['gates'], self._bias_grad['candidate']), axis=0)

    @property
    def gates_bias_grad(self):
        return self._bias_grad['gates']

    @property
    def candidate_bias_grad(self):
        return self._bias_grad['candidate']

    # Setters
    # -------
    @weights.setter
    def weights(self, w):
        self.gates_weights = w[:, :-self.num_neurons]
        self.candidate_weights = w[:, -self.num_neurons:]

    @gates_weights.setter
    def gates_weights(self, w):
        assert(w.shape == self._weights['gates'].shape)
        self._weights['gates'] = w

    @candidate_weights.setter
    def candidate_weights(self, w):
        assert(w.shape == self._weights['candidate'].shape)
        self._weights['candidate'] = w

    @bias.setter
    def bias(self, b):
        self.gates_bias = b[:-self.num_neurons]
        self.candidate_bias = b[-self.num_neurons:]

    @gates_bias.setter
    def gates_bias(self, b):
        assert(b.shape == self._bias['gates'].shape)
        self._bias['gates'] = b

    @candidate_bias.setter
    def candidate_bias(self, b):
        assert(b.shape == self._bias['candidate'].shape)
        self._bias['candidate'] = b

    @init_hidden_state.setter
    def init_hidden_state(self, h):
        assert(h.shape == self._init_hidden_state.shape)
        np.copyto(self._init_hidden_state, h)

    def reinitialize_weights(self, inputs=None, num_neurons=None):
        num_feat = self._inp_size if inputs is None else np.prod(inputs.shape[1:])
        num_neurons = self._num_neurons if num_neurons is None else num_neurons

        # Reinitialize weights
        self._weights['gates'] = np.random.randn((num_neurons + num_feat),
                                                 int(2 * num_neurons)) * self._weight_scale
        self._weights['candidate'] = \
            np.random.randn((num_neurons + num_feat), num_neurons) * self._weight_scale
        if self._xavier:
            # Apply Xavier Initialization
            norm_fctr = np.sqrt(num_neurons + num_feat)
            self._weights['gates'] /= norm_fctr
            self._weights['candidate'] /= norm_fctr

        # Reset layer size
        self._inp_size = num_feat
        self._num_neurons = num_neurons

        if self._has_bias:
            if np.all(self._bias['gates'] == self._bias['gates'][0]):
                self._bias['gates'] = \
                    np.ones(int(2 * num_neurons), dtype=conf.dtype) * self._bias['gates'][0]
            else:
                if self._bias['gates'][-1] == self._bias['gates'][-2]:
                    self._bias['gates'] = np.hstack((np.zeros(num_neurons, dtype=conf.dtype),
                                                     np.ones(int(num_neurons), dtype=conf.dtype) *
                                                     self._bias['gates'][-1]))
                else:
                    self._bias['gates'] = np.zeros(int(2 * num_neurons), dtype=conf.dtype)

            if np.all(self._bias['candidate'] == self._bias['candidate'][0]):
                self._bias['candidate'] = \
                    np.ones(num_neurons, dtype=conf.dtype) * self._bias['candidate'][0]
            else:
                self._bias['candidate'] = np.zeros(num_neurons, dtype=conf.dtype)

        self.reset_gradients()

    def reset_gradients(self):
        self._weights_grad['gates'] = np.zeros_like(self._weights['gates'])
        self._weights_grad['candidate'] = np.zeros_like(self._weights['candidate'])
        self._bias_grad['gates'] = np.zeros_like(self._bias['gates'])
        self._bias_grad['candidate'] = np.zeros_like(self._bias['candidate'])
        self._hidden_state_grad = None
        self._out_grad = OrderedDict()

    def score_fn(self, inputs, weights, bias=None):
        weighted_sum = np.matmul(inputs, weights)

        if self._has_bias and bias is not None:
            return weighted_sum + bias
        else:
            return weighted_sum

    def weight_gradients(self, inp_grad, inputs, weights, reg_lambda=0):
        grad = inputs.reshape(-1, 1) * inp_grad

        if reg_lambda > 0:
            grad += (reg_lambda * weights)
        return grad

    def bias_gradients(self, inp_grad):
        if not self._has_bias:
            return None
        else:
            grad = inp_grad.reshape(-1)
            return grad

    def input_gradients(self, weights, inp_grad, summed=True):
        out_grad = weights * inp_grad

        if summed:
            out_grad = np.sum(out_grad, axis=-1, keepdims=False)

        # Return hidden state [h_(t-1)] gradients and input [h_(l-1)] gradients as a single vector
        return out_grad

    def forward(self, inputs, inference=False, mask=None, temperature=1.0):
        if len(inputs.shape) > 2:  # Preceeding layer is a Convolution/Pooling layer or 3D inputs
            try:
                # If there is just an extra dimension added
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
            score = self.score_fn(concat_inputs, self._weights['gates'], self._bias['gates'])

            # Calculate update gate
            # zₜ = σ(U⁽ᶻ⁾hₜ-₁ + W⁽ᶻ⁾Xₜ + b⁽ᶻ⁾)
            update_gate = self._update_gate[t].forward(score[:, :self._num_neurons])

            # Calculate reset gate
            # rₜ = σ(U⁽ʳ⁾hₜ-₁ + W⁽ʳ⁾Xₜ + b⁽ʳ⁾)
            reset_gate = \
                self._reset_gate[t].forward(score[:, self._num_neurons:])

            # Calculate candidate activation (h̃ₜ)
            if self._reset_pre_transform:
                # h̃ₜ = tanh(U(rₜ ⊙ hₜ-₁) + WXₜ + b)
                append_reset_gate = np.concatenate((reset_gate, np.ones_like(inp)), axis=-1)
                reset_inputs = concat_inputs * append_reset_gate
                candidate_score = self.score_fn(reset_inputs, self._weights['candidate'],
                                                self._bias['candidate'])
                candidate_actvn = self._candidate_activation_fn[t].forward(candidate_score)
            else:
                # h̃ₜ = tanh((rₜ ⊙ Uhₜ-₁) + WXₜ + b)
                candidate_score_hidden = \
                    self.score_fn(self._hidden_state[t - 1],
                                  self._weights['candidate'][:self._num_neurons, :]) * reset_gate
                candidate_score_inp = \
                    self.score_fn(inp, self._weights['candidate'][self._num_neurons:, :])
                candidate_score = \
                    candidate_score_hidden + candidate_score_inp + self._bias['candidate']
                candidate_actvn = self._candidate_activation_fn[t].forward(candidate_score)

            # Update Hidden state
            # hₜ = zₜ ⊙ hₜ-₁ + (1 - zₜ) ⊙ h̃ₜ
            self._hidden_state[t] = \
                (update_gate * self._hidden_state[t - 1]) + ((1.0 - update_gate) * candidate_actvn)

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
        hidden_grad = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        for t in reversed(range(1, self._seq_len + 1)):
            # Add gradients from the next (t+1) hidden sequence and from the layer above (l+1)
            if self._architecture_type == 'many_to_one':
                if t <= list(inp_grad.keys())[-1]:
                    if t == list(inp_grad.keys())[-1]:
                        # Process the single incoming gradient from the layer above (l+1)
                        # Backpropagating through Dropout
                        if self._dropout is not None:
                            drop_grad = self._dropout.backward(inp_grad[t])
                        else:
                            drop_grad = inp_grad[t]
                        grad = hidden_grad + drop_grad
                    else:
                        # No gradients coming from the layer above (l+1), so just forward the
                        # gradients from the next (t+1) hidden sequence
                        grad = hidden_grad
                        if len(grad.shape) == 1:
                            grad = np.expand_dims(grad, axis=0)
                else:
                    # The sequence length of the current iteration is shorther than the
                    # layer's seq_len. So skip until the end of the iteration's sequence length.
                    continue
            else:  # Many-to-many
                try:
                    # Backpropagating through Dropout
                    if self._dropout is not None:
                        drop_grad = self._dropout[t].backward(inp_grad[t])
                    else:
                        drop_grad = inp_grad[t]
                    grad = hidden_grad + drop_grad
                except KeyError:
                    # The sequence length of the current iteration is shorther than the
                    # layer's seq_len. So skip until the end of the iteration's sequence length.
                    continue

            # Outputs of gate activations and candidate activation
            z_out = self._update_gate[t].output
            r_out = self._reset_gate[t].output
            cand_out = self._candidate_activation_fn[t].output

            # Update gate gradients (z'ₜ):
            # z'ₜ = σ'((∂hₜ/∂zₜ) * gₜ) = σ'((hₜ-₁ - h̃ₜ) * gₜ)
            update_gate_grad = \
                self._update_gate[t].backward((self._hidden_state[t - 1] - cand_out) * grad)

            # Candidate activation gradients (h̃'ₜ):
            # h̃'ₜ = tanh'((∂hₜ/∂h̃ₜ) * gₜ) = tanh'((1-zₜ) * gₜ)
            cand_grad = self._candidate_activation_fn[t].backward((1.0 - z_out) * grad)

            # Reset gate gradients (r'ₜ):
            if self._reset_pre_transform:
                # r'ₜ = σ'((∂h̃ₜ/∂rₜ) * gₜ) = σ'((hᵀ₍ₜ-₁₎∙Uᵀ) * h̃'ₜ)
                r_input_grad = np.sum((self._weights['candidate'][:self._num_neurons, :] *
                                       self._hidden_state[t - 1].reshape(-1, 1) * cand_grad),
                                      axis=-1, keepdims=False)
                reset_gate_grad = self._reset_gate[t].backward(r_input_grad)
            else:
                # r'ₜ = σ'((∂h̃ₜ/∂rₜ) * gₜ) = σ'(Uhₜ-₁ * h̃'ₜ)
                r_input_grad = self.score_fn(self._hidden_state[t - 1],
                                             self._weights['candidate'][:self._num_neurons, :])
                reset_gate_grad = self._reset_gate[t].backward(r_input_grad * cand_grad)

            # Concatinate update and reset gate gradients to backprop through weights and inputs
            concat_grads = np.concatenate((update_gate_grad, reset_gate_grad), axis=-1)

            # ∂(zₜ, rₜ)/∂w: Gradient of the layer's update and reset gates w.r.t the weights 'w'
            self._weights_grad['gates'] += \
                self.weight_gradients(concat_grads, self._inputs[t], self._weights['gates'],
                                      reg_lambda)

            # ∂(h̃ₜ)/∂w: Gradient of the layer's candidate activation w.r.t the weights 'w'
            if self._reset_pre_transform:
                append_reset_gate = \
                    np.concatenate((r_out, np.ones((1, self._inp_size), dtype=conf.dtype)), axis=-1)
                reset_inputs = self._inputs[t] * append_reset_gate
                self._weights_grad['candidate'] += \
                    self.weight_gradients(cand_grad, reset_inputs, self._weights['candidate'],
                                          reg_lambda)
            else:
                append_reset_gate = np.concatenate((np.tile(r_out, (self._num_neurons, 1)),
                                                    np.ones((self._inp_size, self._num_neurons),
                                                            dtype=conf.dtype)), axis=0)
                self._weights_grad['candidate'] += \
                    self.weight_gradients(cand_grad, self._inputs[t], self._weights['candidate'],
                                          reg_lambda) * append_reset_gate

            if self._has_bias:
                # ∂(zₜ, rₜ)/∂b: Gradient of the layer's update and reset gates w.r.t the bias 'b'
                self._bias_grad['gates'] += self.bias_gradients(concat_grads)

                # ∂(h̃ₜ)/∂b: Gradient of the layer's candidate activation w.r.t the bias 'b'
                self._bias_grad['candidate'] += self.bias_gradients(cand_grad)

            # ∂(zₜ, rₜ)/∂X: Gradient of the layer's update and reset gates w.r.t the
            # inputs: 'X' [h_(t-1), h_(l-1)]
            input_grads = self.input_gradients(self._weights['gates'], concat_grads)

            # ∂(h̃ₜ)/∂xₜ: Gradient of the layer's candidate activation w.r.t the inputs (xₜ)
            if self._reset_pre_transform:
                cand_weights_reset = self._weights['candidate'] * append_reset_gate.reshape(-1, 1)
                input_grads += self.input_gradients(cand_weights_reset, cand_grad)
            else:
                cand_weights_reset = self._weights['candidate'] * append_reset_gate
                input_grads += self.input_gradients(cand_weights_reset, cand_grad)

            # ∂(h̃ₜ)/∂hₜ-₁: Gradient of the layer's candidate activation w.r.t the
            # previous hidden state (hₜ-₁)
            hidden_grad = input_grads[:self._num_neurons].reshape(1, -1) + (z_out * grad)
            self._out_grad[t] = input_grads[self._num_neurons:]

        if self._tune_internal_states and self._update_init_internal_states:
            # Gradients of the initial hidden state (h_0)
            if len(hidden_grad.shape) == 1:
                hidden_grad = np.expand_dims(hidden_grad, axis=0)
            self._hidden_state_grad = hidden_grad
            self._update_init_internal_states = False
        else:
            self._hidden_state_grad = None

        return self._out_grad

    def update_weights(self, alpha):
        pass

    def reset(self):
        self.reset_gradients()
        self.reset_internal_states(hidden_state='previous_state')

    def reset_internal_states(self, hidden_state=None):
        try:
            if hidden_state.lower() == 'previous_state':
                hidden_state = list(self._hidden_state.values())[-1]
        except AttributeError:
            pass

        self._inputs = OrderedDict()
        self._hidden_state = OrderedDict()
        self._output = OrderedDict()

        if hidden_state is None:
            self._hidden_state[0] = self._init_hidden_state
            if self._tune_internal_states:
                self._update_init_internal_states = True
        else:
            self._hidden_state[0] = hidden_state
