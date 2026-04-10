"""Operator implementations."""

from numbers import Number
from re import T
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # ND backend does not support elementwise tensor^tensor power directly.
        # Use numpy fallback here.
        return NDArray(a.numpy() ** b.numpy(), device=a.device)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad * b * power(a, b - 1)
        grad_b = out_grad * power(a, b) * log(a)
        return grad_a, grad_b
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        grad_lhs = out_grad / rhs
        grad_rhs = -(out_grad * lhs / power_scalar(rhs, 2))
        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        if self.axes is None:
            i, j = len(a.shape) - 2, len(a.shape) - 1
        else:
            i, j = self.axes
        axes[i], axes[j] = axes[j], axes[i]
        return a.permute(tuple(axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = self.shape

        in_len = len(input_shape)
        out_len = len(output_shape)

        padded_input_shape = (1,) * (out_len - in_len) + input_shape

        axes = []
        for i, (in_dim, out_dim) in enumerate(zip(padded_input_shape, output_shape)):
            if in_dim == 1 and out_dim != 1:
                axes.append(i)

        grad = out_grad
        if len(axes) > 0:
            grad = summation(grad, axes=tuple(axes))
        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum()
        if isinstance(self.axes, int):
            return a.sum(axis=self.axes)
        res = a
        for ax in sorted(self.axes, reverse=True):
            res = res.sum(axis=ax)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape

        if self.axes is None:
            reshape_shape = tuple(1 for _ in input_shape)
            return broadcast_to(reshape(out_grad, reshape_shape), input_shape)

        axes = self.axes
        if isinstance(axes, int):
            axes = (axes,)

        reshape_shape = list(input_shape)
        for ax in axes:
            reshape_shape[ax] = 1

        return broadcast_to(reshape(out_grad, tuple(reshape_shape)), input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        grad_lhs = matmul(out_grad, transpose(rhs))
        grad_rhs = matmul(transpose(lhs), out_grad)

        if len(grad_lhs.shape) > len(lhs.shape):
            axes = tuple(range(len(grad_lhs.shape) - len(lhs.shape)))
            grad_lhs = summation(grad_lhs, axes=axes)

        if len(grad_rhs.shape) > len(rhs.shape):
            axes = tuple(range(len(grad_rhs.shape) - len(rhs.shape)))
            grad_rhs = summation(grad_rhs, axes=axes)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):        
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = Tensor(
          node.realize_cached_data() > 0,
          device=out_grad.device,
          dtype=out_grad.dtype,
          requires_grad=False,
        )
        return mask * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = tanh(node.inputs[0])
        return out_grad * add_scalar(negate(power_scalar(y, 2)), 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        n = len(args)
        out_shape = shape[:self.axis] + (n, ) + shape[self.axis:]

        out = array_api.empty(out_shape, device = args[0].device)

        for i, arr in enumerate(args):
          idx = [slice(None)] * len(out_shape)
          idx[self.axis] = i
          out[tuple(idx)] = arr
        
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        res = []
        out_shape = A.shape[:self.axis] + A.shape[self.axis+1:]
        for i in range(A.shape[self.axis]):
          idx = [slice(None)] * len(A.shape)
          idx[self.axis] = slice(i, i + 1)
          piece = A[tuple(idx)].compact().reshape(out_shape)
          res.append(piece)
        return tuple(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)

        for ax in self.axes:
          new_shape[ax] = a.shape[ax] * (self.dilation + 1)

        out = array_api.full(tuple(new_shape), 0, device = a.device)

        idx = []

        for i in range(len(new_shape)):
          if i in self.axes:
            idx.append(slice(0, new_shape[i], self.dilation + 1))
          else:
            idx.append(slice(0, new_shape[i], 1))
          
        out[tuple(idx)] = a
        return out

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        idx = []
        for i in range(len(a.shape)):
          if i in self.axes:
            idx.append(slice(0, a.shape[i], self.dilation + 1))
          else:
            idx.append(slice(0, a.shape[i], 1))
          
        return a[tuple(idx)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if self.padding > 0:
          A = A.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        s = self.stride

        H_out = (H - K) // s + 1
        W_out = (W - K) // s + 1

        A_strided = A.as_strided(
          shape = (N, H_out, W_out, K, K, C_in),
          strides = (
                A.strides[0],
                A.strides[1] * s,
                A.strides[2] * s,
                A.strides[1],
                A.strides[2],
                A.strides[3],            
          )
        ).compact()

        A_col = A_strided.reshape((N * H_out * W_out, K * K * C_in))
        B_col = B.compact().reshape(((K * K * C_in), C_out))

        out = A_col @ B_col
        return out.reshape((N,H_out, W_out, C_out))


        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K = W.shape[0]
        p = self.padding
        s = self.stride

        if s > 1:
          out_grad = dilate(out_grad, axes=(1, 2), dilation=s - 1)

        # dX
        W_flip = flip(W, axes = (0, 1))
        W_flip = transpose(W_flip, axes=(2, 3)) # (K,K,Cout,Cin)
        X_grad = conv(out_grad, W_flip, stride=1, padding=K - 1 - p)

        # dW
        X_perm = transpose(X, axes = (0, 3)) # (Cin,H,W,N)
        # X_perm = transpose(X_perm, axes= (1, 2)) 

        OG_perm = transpose(out_grad, axes=(0, 1))
        OG_perm = transpose(OG_perm, axes=(1, 2))

        W_grad = conv(X_perm, OG_perm, stride=1, padding=p)
        W_grad = transpose(W_grad, axes=(0, 1))
        W_grad = transpose(W_grad, axes=(1, 2))

        return X_grad, W_grad


        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


