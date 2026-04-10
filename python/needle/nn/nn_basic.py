"""
The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    @staticmethod
    def _move_value(value: object, device: Any | None = None, dtype: str | None = None):
        if isinstance(value, Parameter):
            target_device = device if device is not None else value.device
            target_dtype = dtype if dtype is not None else value.dtype
            if value.device == target_device and value.dtype == target_dtype:
                return value
            moved = Tensor(
                value,
                device=target_device,
                dtype=target_dtype,
                requires_grad=value.requires_grad,
            )
            # In-place update keeps object identity so optimizer param refs stay valid.
            value.cached_data = moved.realize_cached_data()
            value.requires_grad = moved.requires_grad
            return value
        if isinstance(value, Tensor):
            target_device = device if device is not None else value.device
            target_dtype = dtype if dtype is not None else value.dtype
            if value.device == target_device and value.dtype == target_dtype:
                return value
            moved = Tensor(
                value,
                device=target_device,
                dtype=target_dtype,
                requires_grad=value.requires_grad,
            )
            value.cached_data = moved.realize_cached_data()
            value.requires_grad = moved.requires_grad
            return value
        if isinstance(value, Module):
            value.to(device=device, dtype=dtype)
            return value
        if isinstance(value, dict):
            return {k: Module._move_value(v, device=device, dtype=dtype) for k, v in value.items()}
        if isinstance(value, list):
            return [Module._move_value(v, device=device, dtype=dtype) for v in value]
        if isinstance(value, tuple):
            return tuple(Module._move_value(v, device=device, dtype=dtype) for v in value)
        return value

    def to(self, device: Any | None = None, dtype: str | None = None):
        """Recursively move module parameters/buffers to the target device/dtype."""
        for k, v in list(self.__dict__.items()):
            if k == "training":
                continue
            self.__dict__[k] = self._move_value(v, device=device, dtype=dtype)
        if hasattr(self, "device") and device is not None:
            self.device = device
        if hasattr(self, "dtype") and dtype is not None:
            self.dtype = dtype
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.kaiming_uniform(
            in_features,
            out_features,
            device = device,
            dtype = dtype
          )
        )

        if bias:
          self.bias = Parameter(
            init.kaiming_uniform(
              out_features,
              1,
              device = device,
              dtype = dtype
            ).reshape((1, out_features))
          )
        else:
          self.bias = None

        
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias is not None:
          out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        flatten_dim = 1
        for dim in X.shape[1:]:
          flatten_dim *= dim
        return ops.reshape(X, (batch_size, flatten_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
          x = m(x)

        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_classes = logits.shape[1]
        y_one_hot = init.one_hot(
            num_classes,
            y,
            device=logits.device,
            dtype=logits.dtype,
            requires_grad=False,
        )
        logits_y = ops.summation(logits * y_one_hot, axes=(1,))
        logsum = ops.logsumexp(logits, axes=(1,))
        loss = logsum - logits_y
        return ops.summation(loss) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device = device, dtype = dtype))
        self.bias = Parameter(init.zeros(dim, device = device, dtype = dtype))

        self.running_var = init.ones(dim, device = device, dtype = dtype)
        self.running_mean = init.zeros(dim, device = device, dtype = dtype)      
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, dim = x.shape

        if self.training:
          #pass
          o_mean = ops.summation(x, axes = (0, )) / batch_size # (D, )
          mean = ops.reshape(o_mean, (1, dim))
          mean = ops.broadcast_to(mean, x.shape)

          x_centered = x - mean

          o_var = ops.summation(ops.power_scalar(x_centered, 2), axes = (0, )) / batch_size
          var = ops.reshape(o_var, (1, dim))
          var = ops.broadcast_to(var, x.shape)

          x_hat = x_centered / ops.power_scalar(var + self.eps, 0.5)

          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * o_mean

          if batch_size > 1:
            un_biased_var = o_var * batch_size / (batch_size - 1)
          else:
            un_biased_var = o_var

          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * o_var
        else:
          mean = ops.reshape(self.running_mean, (1, dim))
          mean = ops.broadcast_to(mean, x.shape)

          var = ops.reshape(self.running_var, (1, dim))
          var = ops.broadcast_to(var, x.shape)

          x_hat = (x - mean) / ops.power_scalar(var + self.eps, 0.5)

        w = ops.reshape(self.weight, (1, dim))
        w = ops.broadcast_to(w, x.shape)

        b = ops.reshape(self.bias, (1, dim))
        b = ops.broadcast_to(b, x.shape)

        return x_hat * w + b
                 
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.ones(dim, device = device, dtype = dtype)
        )
        self.bias = Parameter(
          init.zeros(dim, device = device, dtype = dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_dim = x.shape[1]

        mean = ops.summation(x, axes = (1, )) / feature_dim # (B, )
        mean = ops.reshape(mean, (batch_size, 1)) #(B, 1)
        mean = ops.broadcast_to(mean, x.shape) #(B, F)

        x_centered = x - mean

        var = ops.summation(ops.power_scalar(x_centered, 2), axes = (1, )) / feature_dim
        var = ops.reshape(var, (batch_size, 1))
        var = ops.broadcast_to(var, x.shape)

        x_hat = x_centered / (ops.power_scalar(var + self.eps, 0.5))

        w = ops.reshape(self.weight, (1, feature_dim))
        w= ops.broadcast_to(w, x.shape)

        b = ops.reshape(self.bias, (1, feature_dim))
        b = ops.broadcast_to(b, x.shape)

        return x_hat * w + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          mask = init.randb(*x.shape, p = 1 - self.p, device = x.device, dtype = x.dtype)
          return x * mask / (1 - self.p)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))
