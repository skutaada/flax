# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax.nn import (
  celu,
  elu,
  gelu,
  glu,
  hard_sigmoid,
  hard_silu,
  hard_swish,
  hard_tanh,
  leaky_relu,
  log_sigmoid,
  log_softmax,
  logsumexp,
  one_hot,
  relu,
  identity,
  relu6,
  selu,
  sigmoid,
  silu,
  soft_sign,
  softmax,
  softplus,
  standardize,
  swish,
)
from jax.numpy import tanh
from jax import numpy as jnp

from flax import nnx
from flax.typing import Array, Dtype
from flax.nnx.module import Module

__all__ = [
  'celu',
  'elu',
  'gelu',
  'glu',
  'hard_sigmoid',
  'hard_silu',
  'hard_swish',
  'hard_tanh',
  'leaky_relu',
  'log_sigmoid',
  'log_softmax',
  'logsumexp',
  'one_hot',
  'relu',
  'identity',
  'relu6',
  'selu',
  'sigmoid',
  'silu',
  'soft_sign',
  'softmax',
  'softplus',
  'standardize',
  'swish',
  'tanh',
]


class PReLU(Module):
  """Parametric Rectified Linear Unit (PReLU) activation function.

  Note that PReLU is a Flax layer and not a simple activation function, so
  it needs to be initialized before being called.

  Example usage::
    >>> import flax.nnx as nnx

    >>> class MLP(nnx.Module):
    ...   def __init__(self, in_features, out_features, *, rngs):
    ...     self.l = nnx.Linear(in_features, out_features, rngs=rngs)
    ...     self.act = nnx.PReLU(out_features)
    ...
    ...   def __call__(self, x):
    ...     x = self.l(x)
    ...     x = self.act(x)
    ...     return x

  Attributes:
    num_parameters: the number of trainable parameters
    param_dtype: the dtype passed to parameter initializers (default: float32).
    negative_slope_init: the value to initialize the negative slope (default 0.01).
  """

  def __init__(self, num_parameters: int, param_dtype: Dtype = jnp.float32, negative_slope_init: float = 0.01):
    self.negative_slope = nnx.Param(
      jnp.full((num_parameters, ), negative_slope_init, dtype=param_dtype)
    )

  def __call__(self, inputs: Array) -> Array:
    """Applies an activation to the inputs.

    Args:
      inputs: the nd-array to apply the activation function to.

    Returns:
      The transformed input.
    """
    return jnp.where(inputs >= 0, inputs, jnp.asarray(self.negative_slope, dtype=inputs.dtype) * inputs)
