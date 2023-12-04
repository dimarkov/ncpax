# Copyright 2023 Dimitrije Markovic
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import jax.random as jr
from jax import lax
from equinox import Module, field
from typing import Optional, Callable
from jaxtyping import PRNGKeyArray

class LTCCell(Module):
    params: Module
    sensory_activation: Callable
    hidden_activation: Callable
    ode_unfolds: int = field(static=True)
    epsilon: float = field(static=True)

    def __init__(
        self,
        params: Module,
        sensory_activation: Callable,
        hidden_activation: Callable,
        ode_unfolds: int = 4,
        epsilon: float = 1e-8,
        **kwargs
    ):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.torch.LTC`.


        :param wiring:
        :param in_features:
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        """
        self.params = params
        self.sensory_activation = sensory_activation
        self.hidden_activation = hidden_activation
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon

    def _ode_solver(self, inputs, state, elapsed_time):
        sensory_response = self.sensory_activation(
            inputs, self.params.sensory
        )

        δt = elapsed_time / self.ode_unfolds
        w_τ = self.params.w_tau
        A = self.params.A

        # Unfold the multiply ODE multiple times into one RNN step
        def step_fn(carry, t):
            state = carry
            response = self.hidden_activation(
                sensory_response, state, self.params.hidden
            ) 

            numerator = state + δt * A * (response)
            denominator = 1 + δt * w_τ + δt * response

            # Avoid dividing by 0
            new_state = numerator / (denominator + self.epsilon)

            return new_state, None
        
        last_state, _ = lax.scan(step_fn, state, jnp.arange(self.ode_unfolds))

        return last_state

    def __call__(self, input, hidden, elapsed_time: Optional[float] = 1.0, **kwargs):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(sensory_size,)`.
        - `hidden`: The hidden state, which should be a 2-tuple of JAX arrays, each of
            shape `(hidden_size,)`.
        - `elapsed_time`: A float denoting the time elapsed between the consequent steps

        **Returns:**

        The updated hidden state, shape `(hidden_size,)`.
        """
        hidden = self._ode_solver(input, hidden, elapsed_time)

        return hidden
