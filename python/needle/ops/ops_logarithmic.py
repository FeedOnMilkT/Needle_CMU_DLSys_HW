from typing import Optional

from ..autograd import NDArray, Tensor, TensorOp
from .ops_mathematic import *
from ..backend_selection import array_api


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=1, keepdims=True)
        max_z = array_api.broadcast_to(max_z, Z.shape)
        shifted = Z - max_z

        sum_exp = array_api.sum(array_api.exp(shifted), axis=1, keepdims=True)
        log_sum_exp = array_api.log(sum_exp)
        log_sum_exp = array_api.broadcast_to(log_sum_exp, Z.shape)

        return shifted - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        y = node

        sum_out_grad = summation(out_grad, axes=(1,))
        sum_out_grad = reshape(sum_out_grad, (z.shape[0], 1))
        sum_out_grad = broadcast_to(sum_out_grad, z.shape)

        return out_grad - exp(y) * sum_out_grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # axes=None: reduce over all entries
        if self.axes is None:
            Z_flat = Z.compact().reshape((1, Z.size))
            max_z = Z_flat.max(axis=1, keepdims=True)                     # (1,1)
            max_z_b = array_api.broadcast_to(max_z, Z_flat.shape)        # (1,N)
            shifted = Z_flat - max_z_b
            sum_exp = array_api.sum(array_api.exp(shifted), axis=1)      # (1,)
            return array_api.log(sum_exp) + array_api.reshape(max_z, sum_exp.shape)

        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        assert len(axes) == 1, "Only support a single reduction axis"

        ax = axes[0]
        max_z = Z.max(axis=ax, keepdims=True)
        max_z_b = array_api.broadcast_to(max_z, Z.shape)
        shifted = Z - max_z_b
        sum_exp = array_api.sum(array_api.exp(shifted), axis=ax)

        return array_api.log(sum_exp) + array_api.reshape(max_z, sum_exp.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        input_shape = z.shape

        # axes=None: flatten first, everything becomes shape-safe
        if self.axes is None:
            flat_dim = 1
            for s in input_shape:
                flat_dim *= s

            z_flat = reshape(z, (1, flat_dim))

            max_z_data = z.realize_cached_data().compact().reshape((1, flat_dim)).max(axis=1, keepdims=True)
            max_z = Tensor(
                max_z_data,
                device=z.device,
                dtype=z.dtype,
                requires_grad=False,
            )

            max_z_b = broadcast_to(max_z, z_flat.shape)          # (1, flat_dim)
            shifted = z_flat - max_z_b
            shift_exp = exp(shifted)                             # (1, flat_dim)

            sum_exp = summation(shift_exp, axes=(1,))            # (1,)
            sum_exp = reshape(sum_exp, (1, 1))
            sum_exp = broadcast_to(sum_exp, z_flat.shape)        # (1, flat_dim)

            out_grad = reshape(out_grad, (1, 1))
            out_grad = broadcast_to(out_grad, z_flat.shape)      # (1, flat_dim)

            grad = out_grad * shift_exp / sum_exp
            return reshape(grad, input_shape)

        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        assert len(axes) == 1, "Only support a single reduction axis"
        ax = axes[0]

        max_z_data = z.realize_cached_data().max(axis=ax, keepdims=True)
        max_z = Tensor(
            max_z_data,
            device=z.device,
            dtype=z.dtype,
            requires_grad=False,
        )

        max_z_b = broadcast_to(max_z, input_shape)
        shifted = z - max_z_b
        shift_exp = exp(shifted)

        sum_exp = summation(shift_exp, axes=(ax,))
        reshape_shape = list(input_shape)
        reshape_shape[ax] = 1
        reshape_shape = tuple(reshape_shape)

        sum_exp = reshape(sum_exp, reshape_shape)
        sum_exp = broadcast_to(sum_exp, input_shape)

        out_grad = reshape(out_grad, reshape_shape)
        out_grad = broadcast_to(out_grad, input_shape)

        return out_grad * shift_exp / sum_exp
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)