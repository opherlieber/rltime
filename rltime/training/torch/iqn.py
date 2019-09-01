import torch
import numpy as np

from .dqn import DQN
from rltime.policies.torch.iqn import IQNPolicy


class IQN(DQN):
    """IQN Training Algorithm (https://arxiv.org/pdf/1806.06923.pdf)"""

    @staticmethod
    def create_policy(**kwargs):
        return IQNPolicy.create(**kwargs)

    def _get_bootstrap_target_value(self, target_states, timesteps):
        # Do a forward pass and get the quantile target-values from the target
        # network
        target_quantile_values = self.target_policy.predict(
            target_states,
            timesteps=timesteps)[0]  # (batch,quantile_samples,actions)

        # For double-Q we choose the greedy-action for the target values using
        # the online network, otherwise from the target network
        if not self.double_q:
            action_selection_network = self.target_policy
        else:
            action_selection_network = self.policy
        # Note that, unlike regular DQN, here we need to do another
        # forward-pass also if double_q=False so that the action-selection
        # random quantiles are independent from the ones used to get
        # 'target_quantile_values'
        action_selection_quantile_values = action_selection_network.predict(
            target_states,
            timesteps=timesteps)[0]  # (batch,quantile_samples,actions)

        # Second dim is the amount of quantile samples per training sample
        num_quantile_samples = target_quantile_values.shape[1]

        # Choose the argmax 'best action' per batch item based on mean of all
        # quantile samples
        target_actions = action_selection_quantile_values.mean(1).argmax(
            dim=-1, keepdim=True)  # (batch, 1)

        # Now we take for each batch item it's quantile-target-values based
        # on the greedily selected 'best next action'
        target_actions = target_actions.unsqueeze(1).repeat(
            [1, num_quantile_samples, 1])  # (batch, quantile_samples, 1)
        target_values = torch.gather(
            target_quantile_values, dim=-1,
            index=target_actions).squeeze(-1)  # (batch, quantile_samples)

        return target_values  # (batch, quantile_samples)

    def _compute_grads(self, states, targets, policy_outputs, extra_data,
                       timesteps):
        batch_size = targets.shape[0]

        # The actions selected at acting time for this transition, these are
        # the action-qvalues we will be training now
        actions = self.policy.make_tensor(
            policy_outputs['actions']).long()  # (batch_size,)

        # Note: In current implementation num_target_quantile_samples and
        # num_online_quantile_samples are always the same, but the
        # loss-calculation here supports them being different. These are the
        # values N' and N from the paper respectively.
        num_target_quantile_samples = targets.shape[1]

        # Do a forward pass on the states we are training
        online_quantile_values, online_quantiles = self.policy.predict(
            states, timesteps)
        num_online_quantile_samples = online_quantile_values.shape[1]

        # Gather the outputs only for the specific actions chosen when these
        # transitions were taken, so to train only those specific
        # action-qvalues
        actions = actions.unsqueeze(-1).unsqueeze(-1)  # (batch_size,1,1)
        actions = actions.repeat([1, num_online_quantile_samples, 1])  # (batch_size,num_online_quantile_samples,1)
        chosen_action_quantile_values = torch.gather(
            online_quantile_values, dim=-1, index=actions).squeeze(-1)  # (batch_size, num_online_quantile_samples)

        assert(self.loss_mode == "huber"), "IQN supports only huber loss"

        # The pair-wise sampled td-errors
        sampled_td_errors = targets.unsqueeze(2) - \
            chosen_action_quantile_values.unsqueeze(1)  # (batch_size, num_target_quantile_samples, num_online_quantile_samples)
        # Element-wise huber loss on the sampled td errors
        quantile_value_loss = self._calc_loss(sampled_td_errors)

        # The quantile random values drawn during the forward training pass
        online_quantiles = online_quantiles.view(
            batch_size, num_online_quantile_samples)
        online_quantiles = online_quantiles.unsqueeze(1).repeat(
            [1, num_target_quantile_samples, 1])  # (batch_size, num_target_quantile_samples, num_online_quantile_samples)

        # For each online quantile 'r' the quantile loss is penalized by 'r' if
        # it's an overestimated target and by '1-r' if it's underestimated
        is_underestimated = (sampled_td_errors < 0).float().detach()  # TODO: Should this really be detached? Some implementations do some don't
        penalty = torch.abs(online_quantiles - is_underestimated)  # r or (1-r)
        loss = penalty * quantile_value_loss / self.huber_kappa  # (batch_size, num_target_quantile_samples, num_online_quantile_samples)

        loss = loss.sum(2)  # (batch_size,num_target_quantile_samples)
        # Note the mean here adds the 1/N' term from the paper
        loss = loss.mean(1)  # (batch_size,)

        # Report these per-batch-item losses back if needed (e.g. prioritized
        # replay), we delay this to after backward() to avoid stalling the cuda
        # pipe.
        # TODO: What are the correct errors to report???
        # Option1: report the mean absolute sampled td-erorrs (Before applying
        # huber and asymetric loss penalty)
        losses_to_report = sampled_td_errors.abs().mean(1).mean(1)
        # Option2: report the final loss
        # losses_to_report = loss

        # If we got importance weights (From replay for example), multiply the
        # losses element-wise by them
        loss = self._apply_importance_weights_if_needed(loss, extra_data)

        loss = self._aggregate_losses(loss, timesteps)
        loss.backward()

        # If someone wants to get the td-error back, send them (Prioritized
        # replay for example).
        self._report_losses_if_needed(losses_to_report, extra_data)

        self.value_log.log("qloss", loss.item(), group="train")
        self.value_log.log(
            "td_mean", sampled_td_errors.abs().mean().item(), group="train")
