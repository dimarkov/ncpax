# Copyright 2022 Dimitrije Markovic
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

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp

import jax.numpy as jnp
from equinox import Module, field
from typing import Optional, Callable


class CfCCell(eqx.Module):
    params: Module
    sensory_activation: Callable
    hidden_activation: Callable

    def __init__(
        self,
        params: Module,
        sensory_activation: Callable,
        hidden_activation: Callable,
        **kwargs
    ):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps.
        """

        self.params = params
        self.sensory_activation = sensory_activation
        self.hidden_activation = hidden_activation
   
    def __call__(self, input, hidden, elapsed_time: Optional[float] = 1.0, **kwargs):
        sensory_response = self.sensory_activation(
            input, self.params.sensory
        )

        neg_sensory_response = self.sensory_activation(
            - input, self.params.sensory
        )

        response = self.hidden_activation(
                sensory_response, hidden, self.params.hidden
            )
        
        neg_response = self.hidden_activation(
                neg_sensory_response, -hidden, self.params.hidden
            )

        w_tau = self.params.w_tau
        A = self.params.A

        B = (hidden - A)

        last_state = B * jnp.exp(- elapsed_time * (w_tau + response)) * neg_response + A
        
        return last_state
    

class CfCNNCell(eqx.Module):
    params: Module
    mode: str

    def __init__(
        self,
        params: Module,
        mode: str = "with_gate",
        **kwargs
    ):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell
        that uses feedforward neural networks to approximate the solution.

        .. Note::
            This is an RNNCell that process single time-steps.
        """
        
        self.params = params
        self.mode = mode
   
    def __call__(self, input, hidden, elapsed_time: Optional[float] = 1.0, **kwargs):

        x = jnp.concatenate([input, hidden], -1)
        x = self.params.backbone(x)
        ff1 = jnp.tanh(self.params.ff1(x))
        ff2 = jnp.tanh(self.params.ff2(x))

        t_a = self.params.time_a(x)
        t_b = self.params.time_b(x)
        t_interpolate = jnn.sigmoid(t_a * elapsed_time + t_b)

        if self.mode == "no_gate":
            last_state = ff1 + t_interpolate * ff2
        else:
            last_state = ff1 * (1 - t_interpolate) + t_interpolate * ff2

        return last_state