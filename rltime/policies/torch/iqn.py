import torch
import numpy as np
import logging

from .dqn import DQNPolicy
from rltime.models.torch.utils import linear, repeat_interleave


class IQNPolicy(DQNPolicy):
    """An IQN Policy (https://arxiv.org/abs/1806.06923) """

    def __init__(self, *args, embedding_dim=64, num_sampling_quantiles=32,
                 injection_layer=-1, **kwargs):
        """Initializes an IQN policy

        Args:
            embedding_dim: The embedding size to use in the quantile layer
                (Defaults to 64 as in the paper)
            num_sampling_quantiles: How many quantile samples to use on each
                forward pass (This value is used for all forward-passes,
                including actor-action-selection, and target+training values,
                i.e. N and N' from the paper. There is no option ATM to define
                them separately).
                Default value of 32 is used as this seems good enough, also
                according to the paper (Which uses 64 for N and N' and 32 for
                actor-action-selection)
                In case the quantile layer is injected before an LSTM layer
                then 32 causes a big slowdown in training time, and it is
                recommended to use 8 in this case which should also be good
                enough according to the paper, though this wasn't verified
            injection_layer: Before which model layer to inject the quantile
                layer and convert the batch to a multi-sample one. The
                default (-1) means to inject it before the last layer. For
                example for a CNN->FC model it will behave like in the paper
                (After the CNN layer). For a CNN->LSTM->FC model it will
                inject it between the LSTM and FC layers (Alternatively,
                setting injection_layer=1 will always inject it immediately
                after the CNN layer)
        """
        super().__init__(*args, **kwargs)

        self.num_sampling_quantiles = num_sampling_quantiles
        self.embedding_dim = embedding_dim

        # For IQN we need to 'inject' ourselves in the middle of the model,
        # typically after layer0 (The CNN layer) / before the last layer
        inner_layer_shape = self.model.set_layer_preprocessor(
            injection_layer, self._apply_quantile_layer)
        quantile_layer_size = int(np.prod(inner_layer_shape))
        logging.getLogger().info(f"IQN Layer size: {quantile_layer_size}")

        # The quantile layer uses an embedding and random samples and merges
        # to the 'state reprsentation' output of the model, usually after
        # the CNN layer or LSTM layer, depending on the model and
        # 'injection_layer' option (See _apply_quantile_layer).
        self.quantile_layer = linear(embedding_dim, quantile_layer_size)

        # The embedding range used in _apply_quantile_layer, create it only
        # once (This should have requires_grad=False by default so shouldn't
        # change during backprop)
        # register_buffer ensures it's moved to GPU/CPU etc together with the
        # whole policy/model
        self.register_buffer(
            "embedding_range",
            torch.arange(1, self.embedding_dim+1, dtype=torch.float32))

    def _apply_quantile_layer(self, x):
        """Applies the quantile layer to the intermediate model ouptut 'x'"""

        # We are essentially multiplying the batch size from this point
        # by <num_sampling_quantiles>
        batch_size = x.shape[0]
        quantiled_batch_size = batch_size * self.num_sampling_quantiles

        # Flatten, it's assumed all model layers accept flattened inputs from
        # this point onward
        x = x.view((batch_size, -1))

        # Repeat the inputs according to the amount of quantile samples
        # Note we use interleaved repeating (And not tiling), so the final
        # grouping is (batch_size, num_sampling_quantiles,...). This ensures we
        # don't mess up timestep grouping of the original batch (Which is on
        # dim=0), and also makes the batch stay on dim=0
        xt = repeat_interleave(x, self.num_sampling_quantiles, dim=0)  # (quantiled_batch_size, state_size)

        # Generate the uniform random quantile samples (Directly on the current
        # device), we also return these to the caller for use during training
        quantiles = torch.rand(
            quantiled_batch_size, device=self.embedding_range.device)  # (quantiled_batch_size,)

        # apply the quantile network as defined in the paper
        quantiles_net = quantiles.unsqueeze(1).repeat([1, self.embedding_dim])  # (quantiled_batch_size, embedding_dim)
        quantiles_net = self.embedding_range*np.pi*quantiles_net  # (quantiled_batch_size, embedding_dim)
        quantiles_net = torch.cos(quantiles_net)  # (quantiled_batch_size, embedding_dim)
        quantiles_net = self.quantile_layer(quantiles_net)  # (quantiled_batch_size, state_size)
        quantiles_net = torch.nn.functional.relu(quantiles_net)

        # Element-wise multiply the repeated model output with the quantile
        # layer output
        out = xt*quantiles_net  # (quantiled_batch_size, state_size)

        # The modified output will continue flowing through the model
        # (typically through FC and/or LSTM layers),
        # We add the quantiles themselves to the output dictionary so we get
        # them back and return them from predict() (Needed for training)
        return out, {"quantiles": quantiles}

    def _shape_action_outputs(self, output):
        """Reshape output with sampling quantiles on the correct axis.
        actions are on the last dim=2
        """
        output = output.view(
            (-1, self.num_sampling_quantiles, output.shape[-1]))  # (batch_size, num_sampling_quantiles, num_actions)
        return output, 2

    def _predict_postprocess(self, output, model_output):
        # This will handle dueling layer, if configured
        output = super()._predict_postprocess(output, model_output)

        # Return also the quantiles used during this forward pass
        # (Needed during training)
        return output, model_output['quantiles']

    def _actor_predict_postprocess(self, pred):
        # The actor prediction/action-selection for IQN just reduces the
        # quantile-samples axis (1) using mean, then it's same as DQN
        # (argmax on actions, see super().actor_predict())
        # Note we take index0 from the pred, as it also includes the quantiles
        # (See _predict_postprocess())
        assert(pred[0].shape[1] == self.num_sampling_quantiles)
        return pred[0].mean(1)
