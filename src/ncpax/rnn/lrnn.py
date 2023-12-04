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
import equinox as eqx
from jax import random as jr
from jax import lax, vmap

class LiquidRNN(eqx.Module):
    input_mapping: eqx.nn.Linear
    output_mapping: eqx.nn.Linear
    cell: eqx.Module

    def __init__(
            self, 
            rnn_cell,
            input_size,
            output_size, 
            use_input_bias=True,
            use_output_bias=True, 
            *, 
            key
        ):
        ckey, lkey = jr.split(key)
        self.cell = rnn_cell
        self.input_mapping = eqx.nn.Linear(
            input_size, 
            self.cell.params.sensory_size,
            use_bias=use_input_bias, 
            key=ckey
        )

        self.output_mapping = eqx.nn.Linear(
            self.cell.params.motor_size, 
            output_size, 
            use_bias=use_output_bias, 
            key=lkey
        )

    def __call__(self, inputs, init_hidden=None, timespans=None):

        init_hidden = jnp.zeros(self.cell.params.hidden_size) if init_hidden is None else init_hidden

        embedding = vmap(self.input_mapping)(inputs)

        def step_fn(carry, xs):
            h_state = carry
            ts, input = xs
            h_state = self.cell(input, h_state, ts)
            return h_state, h_state


        timespans = jnp.ones(len(inputs)) if timespans is None else timespans
        last_state, hidden_sequence = lax.scan(step_fn, init_hidden, (timespans, embedding))

        motor_sequence = hidden_sequence[..., :self.cell.params.motor_size]
        output_sequence = vmap(self.output_mapping)(motor_sequence)

        return output_sequence, last_state