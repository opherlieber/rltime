import torch

from .dqn import DQNPolicy


class DistDQNPolicy(DQNPolicy):
    """Distributional DQN policy (NOT VERIFIED/TESTED)"""
    def __init__(self, *args, num_atoms=51, vmin=-10, vmax=10, **kwargs):
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        super().__init__(*args, **kwargs)

        # Create the persistent support vector (To avoid transferring to GPU
        # each time), need to create it as a buffer so that it is persistent
        # and moved between CPU/CUDA together with the rest of the modules
        self.register_buffer(
            "support", torch.linspace(self.vmin, self.vmax, self.num_atoms))

    def _outputs_per_action(self):
        """Override DQN to configure num_atoms outputs for each action"""
        return self.num_atoms

    def _shape_action_outputs(self, output):
        """Reshape output with atoms on the correct axis"""
        output = output.view((output.shape[0], -1, self.num_atoms))  # (batch_size, num_actions, num_atoms)
        return output, 1

    def _actor_predict_postprocess(self, pred):
        """DistDQN postprocessing for action selection"""
        assert(pred.shape[2] == self.num_atoms)
        res = torch.nn.functional.softmax(pred, dim=-1) * self.support
        return res.sum(2)
