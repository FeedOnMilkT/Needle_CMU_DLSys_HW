import math
from .init_basic import *
from typing import Any


import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    shape = kwargs.pop("shape", None)
    if shape is not None:
        return rand(*shape, low=-a, high=a, **kwargs)
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    shape = kwargs.pop("shape", None)
    if shape is not None:
        return randn(*shape, mean=0.0, std=std, **kwargs)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    shape = kwargs.pop("shape", None)
    if shape is not None:
        return rand(*shape, low=-bound, high=bound, **kwargs)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain * math.sqrt(1 / fan_in)
    shape = kwargs.pop("shape", None)
    if shape is not None:
        return randn(*shape, mean=0.0, std=std, **kwargs)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION