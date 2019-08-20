import numpy as np
import torch
import torch.nn as nn

from .torch_model import TorchModel
from rltime.general.type_registry import get_registered_type
from .utils import repeat_interleave, make_tensor
import logging


class SequentialModel(TorchModel):
    """Implements a 'sequential' model which just runs the input through a
    given configuration of modules serially"""

    def __init__(self, observation_space, layer_configs,
                 extra_input_layer=None):
        """Initializes a sequential model

        Args:
            observation_space: The observation space of the input
            layer_configs: A list of layer configurations. Each layer config is
                a dictionary containing the 'type' (Either a python class or
                registered type string, and 'args' to pass to the module
                initialization)
            extra_input_layer: Layer (index) at which to append the extra input
                vector (If exists). Must be a layer which accepts flattened
                inputs (i.e. can't be before a CNN layer). Can be negative
                (For example -1 means before the last layer). If 'None' the
                layer is auto-selected to be the first recurrent layer in the
                model, and if there is none then before the last layer (i.e.
                extra_input_layer=-1)
        """
        super().__init__(observation_space)
        self.layers = nn.ModuleList()
        self.layer_input_shapes = []

        # Configure which layer receives the extra input vector, if exists
        if self.extra_input_shape is not None:
            self.extra_input_layer = extra_input_layer \
                if extra_input_layer is not None \
                else self._auto_detect_extra_input_layer(layer_configs)
            if self.extra_input_layer < 0:
                self.extra_input_layer += len(layer_configs)
            assert(self.extra_input_layer < len(layer_configs))
        else:
            self.extra_input_layer = None

        # Initialize all the modules
        inp_shape = self.main_input_shape
        for i, layer_config in enumerate(layer_configs):
            module_cls = get_registered_type("modules", layer_config["type"])
            if i == self.extra_input_layer:
                # If this the layer to receive the extra-input vector flatten
                # the input shape and add the extra-input vector size
                inp_shape = (
                    np.prod(inp_shape) + np.prod(self.extra_input_shape),)
            # Create the module for this layer with the requested args
            layer = module_cls(
                inp_shape=inp_shape, **layer_config.get("args", {}))
            self.layers.append(layer)
            self.layer_input_shapes.append(inp_shape)
            inp_shape = layer.out_shape
        assert(len(inp_shape) == 1)
        self.out_size = inp_shape[0]

        # Track layer preprocessors which can optionally be configured with
        # set_layer_preprocessor()
        self.layer_pre_processors = {}

    def _auto_detect_extra_input_layer(self, layer_configs):
        """Auto detect what layer to use for extra inputs, if not specified.

        This chooses the first recurrent layer, if exists, otherwise the last
        layer
        """
        for i, layer_config in enumerate(layer_configs):
            # Note that is_recurrent should be a static class method
            layer_cls = get_registered_type("modules", layer_config['type'])
            if layer_cls.is_recurrent():
                return i
        # No recurrent layers, default to the last layer
        return len(layer_configs) - 1

    def _get_actual_layer_index(self, layer_index):
        """Converts a layer index to the actual layer index, i.e. if it's
        negative calculate the index from the end"""
        if layer_index < 0:
            layer_index = len(self.layers) + layer_index
        assert(layer_index >= 0 and layer_index < len(self.layers))
        return layer_index

    def set_layer_preprocessor(self, layer_index, preprocessor):
        """Adds a preprocessor to a layers input.

        This preprocessor is called before the layer on every forward pass.
        Returns the input shape of the requested layer
        """
        layer_index = self._get_actual_layer_index(layer_index)

        assert(layer_index not in self.layer_pre_processors), \
            "SequentialModel supports at most 1 pre-processor per layer ATM"
        self.layer_pre_processors[layer_index] = preprocessor
        return self.get_layer_in_shape(layer_index)

    def get_layer_in_shape(self, layer_index):
        """Returns the input shape of the given layer"""
        layer_index = self._get_actual_layer_index(layer_index)
        return self.layer_input_shapes[layer_index]

    def get_layer_out_shape(self, layer_index):
        """Returns the output shape of the given layer"""
        layer_index = self._get_actual_layer_index(layer_index)
        return self.layers[layer_index].out_shape

    def is_cuda(self):
        """Returns whether this model is on the GPU

        TODO: There's probably a better way to know this as we are a pytorch
        module, also might be better to return the actual device we are on
        """
        return self.layers[0].is_cuda()

    def is_recurrent(self):
        """Returns whether this model is a recurrent model, i.e. if any of the
        layers are recurrent"""
        return np.any([layer.is_recurrent() for layer in self.layers])

    def make_input_state(self, x, initials):
        """Makes the input state for the model

        The input state combines the input observation (x) with any
        module-specific input states such as RNN hidden states

        Args:
            x: The (batched) input itself, i.e. the batched observations
            initials: For each batch item whether it is an initial state
                (i.e. start of an episode)
        Returns: The combined model input state, which is a dictionary
            containing 'x' and the input state for each module (If applicable)
        """
        state = {"x": x}
        for i, layer in enumerate(self.layers):
            state[f'layer{i}_state'] = layer.get_state(initials)
        return state

    def _combine_extra_inputs(self, x, extra_inputs):
        """Combines the extra inputs to the current input vector

        Special handling for multi-sample batches
        """
        # If 'x' became a multi-sample batch, we need to repeat the extra
        # inputs accordingly
        if x.shape[0] != extra_inputs.shape[0]:
            assert(x.shape[0] % extra_inputs.shape[0] == 0)
            extra_inputs = repeat_interleave(
                extra_inputs, x.shape[0] // extra_inputs.shape[0], dim=0)

        # It's assumed the extra-input receiving layer supports 1D inputs,
        # so we flatten and concat all the inputs
        return torch.cat(
            [
                x.view(x.shape[0], -1),
                extra_inputs.view(extra_inputs.shape[0], -1)
            ],
            dim=-1)

    def forward(self, inp, timesteps):
        """ Performs the forward pass of the model

        Args:
            inp: The input dictionary (Result of calling make_input_state())
            timesteps: The amount of timesteps in the batch layout (For RNN
                usage). timesteps should be on the first dimension,
                i.e. a (batch, ...) input would be viewable as:
                (timesteps, batch_size/timesteps, ...) where the second
                dimension are the indepent multi-batch trajectories and the
                first dimension are consecutive timesteps from the same
                episode. This value defines the RNN sequence length to use if
                there are any RNN layers.
        Returns: Dictionary containing:
            output: The result
            layer_inputs: The inputs to each of the layers for optional
                additional processing/extensions
        """
        inp = make_tensor(inp, device="cuda" if self.is_cuda() else "cpu")
        x, extra_inputs = self._get_inputs(inp["x"])
        assert((extra_inputs is None) == (self.extra_input_layer is None))

        # The result includes also internal layer inputs in case the policy
        # wants to use them (For example dueling DQN)
        result = {"layer_inputs": []}
        for i, layer in enumerate(self.layers):
            # Concatenate extra model inputs if defined to do so in this layer
            if i == self.extra_input_layer:
                x = self._combine_extra_inputs(x, extra_inputs)

            # If there is a preprocessor defined for this layer, call it
            if i in self.layer_pre_processors:
                x, extra_outputs = self.layer_pre_processors[i](x)
                result.update(extra_outputs)

            result['layer_inputs'].append(x)

            # Call the layer, passing it additional state variables from the
            # input state if applicable (e.g. RNN hidden states)
            layer_state = inp.get(f'layer{i}_state', {})
            x = layer(x, timesteps=timesteps, **layer_state)

        result["output"] = x
        return result
