from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)

import math


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)
    """
    def matmul(self, a, b_transpose):
        
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)
    """
    def matmul(self, a, b):
        # a: (B, H, M, K), b: (B, H, K, N)
        B, H, M, K = a.shape
        B2, H2, K2, N = b.shape
        assert B == B2 and H == H2 and K == K2

        # Use backend matmul kernel per (batch, head) block instead of
        # materializing huge broadcasted tensors.
        a_blocks = ops.split(a.reshape((B * H, M, K)), axis=0)
        b_blocks = ops.split(b.reshape((B * H, K, N)), axis=0)
        out_blocks = [ops.matmul(a_blocks[i], b_blocks[i]) for i in range(B * H)]

        return ops.stack(out_blocks, axis=0).reshape((B, H, M, N))

    def softmax(self, logit):
        """
        The softmax function; 
        """
        lse = ops.logsumexp(logit, axes=(3,))
        lse = lse.reshape((*logit.shape[:-1], 1)).broadcast_to(logit.shape)
        return ops.exp(logit - lse)

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        
        # k: (B, H, T_k, D) -> (B, H, D, T_k)
        k_t = ops.transpose(k, axes = (2, 3))

        # scores: (B, H, T_q, D) @ (B, H, D, T_k) -> (B, H, T_q, T_k)
        scores = self.matmul(q, k_t) / math.sqrt(q_dim)

        if self.causal:
          mask = self.create_causal_mask(queries_len, keys_values_len, q.device)
          mask_tensor = Tensor(
            mask,
            device = q.device,
            dtype = q.dtype,
            requires_grad = False
          )
          mask_tensor = mask_tensor.broadcast_to(scores.shape)
          scores = scores + mask_tensor

        probs = self.softmax(scores)
        probs = self.dropout(probs)

        # result: (B, H, T_q, T_k) @ (B, H, T_k, D) -> (B, H, T_q, D)
        # v_t = ops.transpose(v, axes = (2, 3))
        result = self.matmul(probs, v)
        # result = ops.transpose(result, axes = (2, 3))

        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        
        q = q.reshape((batch_size * queries_len, q_dim))
        q = self.prenorm_q(q)
        q = q.reshape((batch_size, queries_len, q_dim))

        k = k.reshape((batch_size * keys_values_len, k_dim))
        k = self.prenorm_k(k)
        k = k.reshape((batch_size, keys_values_len, k_dim))

        v = v.reshape((batch_size * keys_values_len, v_dim))
        v = self.prenorm_v(v)
        v = v.reshape((batch_size, keys_values_len, v_dim))

        q = q.reshape((batch_size * queries_len, q_dim))
        q = self.q_projection(q)
        q = q.reshape((batch_size, queries_len, self.num_head * self.dim_head))

        k = k.reshape((batch_size * keys_values_len, k_dim))
        k = self.k_projection(k)
        k = k.reshape((batch_size, keys_values_len, self.num_head * self.dim_head))

        v = v.reshape((batch_size * keys_values_len, v_dim))
        v = self.v_projection(v)
        v = v.reshape((batch_size, keys_values_len, self.num_head * self.dim_head))

        q = q.reshape((batch_size, queries_len, self.num_head, self.dim_head))
        k = k.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))
        v = v.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))

        q = ops.transpose(q, axes=(1, 2))
        k = ops.transpose(k, axes=(1, 2))
        v = ops.transpose(v, axes=(1, 2))

        result, probs = self.attn(q, k, v)
        self.probs = probs
        result = ops.transpose(result, axes=(1, 2))
        result = result.reshape((batch_size, queries_len, self.num_head * self.dim_head))

        result = result.reshape((batch_size * queries_len, self.num_head * self.dim_head))
        result = self.out_projection(result)
        result = result.reshape((batch_size, queries_len, self.out_features))


        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.attn = AttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )

        self.dropout_attn = Dropout(dropout)

        self.norm = LayerNorm1d(
            q_features,
            device=device,
            dtype=dtype,
        )

        self.linear1 = Linear(
            q_features,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        self.relu = ReLU()

        self.dropout_ff1 = Dropout(dropout)

        self.linear2 = Linear(
            hidden_size,
            q_features,
            device=device,
            dtype=dtype,
        )

        self.dropout_ff2 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        # attention residual
        attn_out = self.attn(x)                     # (B, T, D)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out

        # FFN prenorm residual
        y = x.reshape((batch_size * seq_len, x_dim))
        y = self.norm(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.dropout_ff1(y)
        y = self.linear2(y)
        y = self.dropout_ff2(y)
        y = y.reshape((batch_size, seq_len, x_dim))

        x = x + y

        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.pos_embedding = Embedding(
            sequence_len,
            embedding_size,
            device=device,
            dtype=dtype,
        )

        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerLayer(
                    q_features=embedding_size,
                    num_head=num_head,
                    dim_head=dim_head,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    causal=causal,
                    device=device,
                    dtype=dtype,
                )
            )

        self.layers = Sequential(*layers)
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, emb_dim = x.shape

        pos_ids = Tensor(
            np.arange(seq_len).reshape((1, seq_len)),
            device=self.device,
            requires_grad=False
        )

        pos_emb = self.pos_embedding(pos_ids)          # (1, T, D)
        pos_emb = ops.broadcast_to(pos_emb, x.shape)   # (B, T, D)

        x = x + pos_emb

        x = self.layers(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
