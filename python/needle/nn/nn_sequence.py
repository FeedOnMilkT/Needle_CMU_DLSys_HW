"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module

import math


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar((1 + ops.exp(-x)), -1)        
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        bound = 1 / math.sqrt(hidden_size)

        self.W_ih = Parameter(
          init.rand(input_size, hidden_size, low = -bound, high = bound,
                    device = device, dtype = dtype)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=-bound, high=bound,
                      device=device, dtype=dtype)
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(hidden_size, low=-bound, high=bound,
                          device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.rand(hidden_size, low=-bound, high=bound,
                          device=device, dtype=dtype)
            )


        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
          h = init.zeros(bs, self.hidden_size, device = self.device, dtype = self.dtype)

        out = X @ self.W_ih + h @ self.W_hh

        if self.bias:
            out = out + self.bias_ih.broadcast_to(out.shape)
            out = out + self.bias_hh.broadcast_to(out.shape)

        if self.nonlinearity == "relu":
          return ops.relu(out)
        return ops.tanh(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        self.rnn_cells = []
        for l in range(num_layers):
          in_dim = input_size if l == 0 else hidden_size
          cell = RNNCell(in_dim, hidden_size, bias=bias,
                           nonlinearity=nonlinearity,
                           device=device, dtype=dtype)
          self.rnn_cells.append(cell)
        
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 is None:
          h0 = init.zeros(self.num_layers, bs, self.hidden_size,
                           device=self.device, dtype=self.dtype)
        
        inputs = list(ops.split(X, axis = 0))
        hidden_states = list(ops.split(h0, axis = 0))
        
        for i in range(len(inputs)):
            if len(inputs[i].shape) == 3:
                inputs[i] = inputs[i].reshape((bs, inputs[i].shape[-1]))

        for i in range(len(hidden_states)):
            if len(hidden_states[i].shape) == 3:
                hidden_states[i] = hidden_states[i].reshape((bs, self.hidden_size))

        outputs = []

        for t in range(seq_len):
          x_t = inputs[t]
          for l in range(self.num_layers):
            h_next = self.rnn_cells[l](x_t, hidden_states[l])
            hidden_states[l] = h_next
            x_t = h_next
          outputs.append(x_t)

        output = ops.stack(outputs, axis = 0)

        h_n = ops.stack(hidden_states, axis = 0)
        
        return output, h_n

        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        k = 1 / math.sqrt(hidden_size)   

        self.W_ih = Parameter(
          init.rand(input_size, 4 * hidden_size, low = -k, high = k, 
                      device = device, dtype = dtype)
        )

        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size, low=-k, high=k,
                      device=device, dtype=dtype)
        )   

        if bias:
            self.bias_ih = Parameter(
                init.rand(4 * hidden_size, low=-k, high=k,
                          device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.rand(4 * hidden_size, low=-k, high=k,
                          device=device, dtype=dtype)
            )

        self.sigmoid = Sigmoid()

        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        gates = X @ self.W_ih + h0 @ self.W_hh

        if self.bias:
            gates = gates + self.bias_ih.broadcast_to(gates.shape)
            gates = gates + self.bias_hh.broadcast_to(gates.shape)

        cols = list(ops.split(gates, axis=1))
        cols = [c if len(c.shape) == 2 else c.reshape((bs, 1)) for c in cols]

        i_raw = ops.stack(cols[0 : self.hidden_size], axis = 1).reshape((bs, self.hidden_size))
        f_raw = ops.stack(cols[self.hidden_size:2*self.hidden_size], axis=1).reshape((bs, self.hidden_size))
        g_raw = ops.stack(cols[2*self.hidden_size:3*self.hidden_size], axis=1).reshape((bs, self.hidden_size))
        o_raw = ops.stack(cols[3*self.hidden_size:4*self.hidden_size], axis=1).reshape((bs, self.hidden_size))

        i = self.sigmoid(i_raw)
        f = self.sigmoid(f_raw)
        g = ops.tanh(g_raw)
        o = self.sigmoid(o_raw)

        c_next = f * c0 + i * g
        h_next = o * ops.tanh(c_next)

        return h_next, c_next        
                
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = []
        for l in range(num_layers):
            in_dim = input_size if l == 0 else hidden_size
            cell = LSTMCell(
                in_dim, hidden_size,
                bias=bias, device=device, dtype=dtype
            )
            self.lstm_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size,
                            device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size,
                            device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        inputs = list(ops.split(X, axis = 0))
        hidden_states = list(ops.split(h0, axis = 0))
        cell_states = list(ops.split(c0, axis = 0))

        processed_inputs = []
        for x in inputs:
            if len(x.shape) == 3:
                processed_inputs.append(x.reshape((bs, x.shape[-1])))
            else:
                processed_inputs.append(x)
        inputs = processed_inputs

        processed_hidden = []
        for hh in hidden_states:
            if len(hh.shape) == 3:
                processed_hidden.append(hh.reshape((bs, self.hidden_size)))
            else:
                processed_hidden.append(hh)
        hidden_states = processed_hidden

        processed_cell = []
        for cc in cell_states:
            if len(cc.shape) == 3:
                processed_cell.append(cc.reshape((bs, self.hidden_size)))
            else:
                processed_cell.append(cc)
        cell_states = processed_cell

        outputs = []

        for t in range(seq_len):
            x_t = inputs[t]

            for l in range(self.num_layers):
                h_next, c_next = self.lstm_cells[l](x_t, (hidden_states[l], cell_states[l]))
                hidden_states[l] = h_next
                cell_states[l] = c_next
                x_t = h_next

            outputs.append(x_t)

        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(hidden_states, axis=0)
        c_n = ops.stack(cell_states, axis=0)

        return output, (h_n, c_n)        

        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

  
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        """
        x: shape (seq_len, bs)
        output: shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape

        # init.one_hot expects i to be a Needle Tensor, because inside it calls i.numpy()
        if isinstance(x, np.ndarray):
            x_idx = Tensor(x.astype("int32").reshape(-1), device=self.device, requires_grad=False)
        else:
            x_idx = Tensor(x.numpy().astype("int32").reshape(-1), device=self.device, requires_grad=False)

        x_one_hot = init.one_hot(
            self.num_embeddings, x_idx, device=self.device, dtype=self.dtype
        )
        out = x_one_hot @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION