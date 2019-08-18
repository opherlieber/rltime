import torch

from .a2c import A2C
from rltime.general.utils import anneal_value


class PPO(A2C):
    """A PPO training algorithm for an actor-critic policy

    https://arxiv.org/abs/1707.06347

    The only difference from the base A2C training is the action-gain
    calculation of the gradient computation (Also this defaults to
    adv_norm = True unlike A2C which defaults to False)

    This should be equivalent to the OpenAI baselines implementation when
    using a single local vectorized actor
    """
    def _train(self, clip_value, clip_anneal=None, adv_norm=True, **kwargs):
        """Entry point for PPO training

        Args:
            clip_value: The clip value to use in the PPO clipped surrogate loss
            clip_anneal: Whether to anneal the clip_calue to 0 thoughout the
                training (Can also be a specific value to anneal it to)
        """
        self._clip_value = clip_value
        self._clip_anneal = clip_anneal

        super()._train(adv_norm=adv_norm, **kwargs)

    def _calc_clip_value(self):
        """Calculates the current clip value to use, either fixed or annealed
        to 0 throughout the training
        """
        return anneal_value(
            self._clip_value, self.get_train_progress(), self._clip_anneal)

    def _calc_action_gain(self, action_log_probs, advantages,
                          org_policy_outputs):
        """Override the A2C action-gain calculation for PPO clipped surrogate
        loss
        """
        # The original log_probs from acting-time
        old_log_probs = self.policy.make_tensor(
            org_policy_outputs['action_log_probs'])

        # Action gain with clipped surrogate loss
        assert(action_log_probs.shape == old_log_probs.shape)
        assert(action_log_probs.shape == advantages.shape)
        ratio = torch.exp(action_log_probs - old_log_probs)
        ploss1 = ratio * advantages
        clip_value = self._calc_clip_value()
        ploss2 = torch.clamp(
            ratio, 1.0 - clip_value, 1.0 + clip_value) * advantages
        return torch.min(ploss1, ploss2).mean()
