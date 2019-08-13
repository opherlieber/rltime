import torch
import numpy as np

from .torch_trainer import TorchTrainer
from rltime.policies.torch.dqn import DQNPolicy


class DQN(TorchTrainer):
    """The basic (non-distributional) DQN training algorithm

    By default this used a replay buffer but this can be changed to use
    priortized replay, or 'online' mode for online nstep q-learning
    """

    def _train(self, double_q=False, loss_mode="huber", huber_kappa=1.0,
               loss_aggregation="mean", loss_timestep_aggregation=None,
               history_mode={"type": "replay"}, **kwargs):
        """ Entry point for DQN training

        Args:
            double_q: Whether to use 'doubleq' method for target value
                calculation, i.e. to use the online policy for selecting the
                greedy action to use to take the value estimations from the
                target policy (Instead of using the target-policy for both)
            loss_mode: How to calculate the loss on the td-errors: Either 'mse'
                or 'huber' is supported. huber loss restricts the gradient of
                the loss to specific value (e.g. '1.0' if huber_kappa=1.0)
                rather then exploding as the td-errors grow, while still
                maintaining squared-error behavior for lower losses
            huber_kappa: Kappa value to use in the huber-loss formula if
                loss_mode=huber.
            loss_aggregation: How to aggregate the losses across the batch.
                Supported values are 'mean' or 'sum'
            loss_timestep_aggregation: Optionally use a different loss
                aggregation across timesteps. For example 'mean' across
                timesteps and 'sum' the mbatch (Relevant only if multi-step
                training, i.e. nstep_train>1)
        """
        self.double_q = double_q
        self.loss_mode = loss_mode
        self.huber_kappa = huber_kappa
        self.loss_aggregation = self._get_aggregator(loss_aggregation)
        self.loss_timestep_aggregation = None \
            if not loss_timestep_aggregation \
            else self._get_aggregator(loss_timestep_aggregation)
        super()._train(history_mode=history_mode, **kwargs)

    @staticmethod
    def create_policy(**kwargs):
        return DQNPolicy.create(**kwargs)

    def _get_bootstrap_target_value(self, target_states, timesteps):
        # Target values are taken from the target policy
        target_values = self.target_policy.predict(
            target_states, timesteps=timesteps)
        if not self.double_q:
            # Without double-Q, actions are chosen using the same target policy
            # values
            action_selection_values = target_values
        else:
            # For double Q, we use the online policy to select the best target
            # action, and take the value of that action from the target policy
            # This is supposed to improve training results but costs an
            # additional forward pass
            action_selection_values = self.policy.predict(
                target_states, timesteps=timesteps)

        target_actions = action_selection_values.argmax(dim=-1, keepdim=True)
        target_values = target_values.gather(
            dim=-1, index=target_actions).squeeze(-1)
        return target_values

    def _report_losses_if_needed(self, losses, extra_train_data):
        """Notifies the history buffer about the losses for each index, if
        requested"""
        if "loss_indices" not in extra_train_data:
            return
        notify_errors = losses.data.cpu().numpy()
        loss_indices = extra_train_data["loss_indices"]
        assert(notify_errors.shape == loss_indices.shape[:1])
        self.history_buffer.update_losses(loss_indices, notify_errors)

    def _apply_importance_weights_if_needed(self, losses, extra_train_data):
        """Applies importance weights to the losses, if supplied"""
        if "importance_weights" not in extra_train_data:
            return losses

        importance_weights = self.policy.make_tensor(
            extra_train_data["importance_weights"])
        assert(importance_weights.shape == losses.shape)
        losses = losses * importance_weights
        # Add the average importance weights to the result log
        self.value_log.log(
            "importance_weights",
            np.mean(extra_train_data["importance_weights"]), group="train")
        return losses

    def _calc_loss(self, errors):
        """Calculates the losses given the batch-wise 'td-errors'

        This is either squared-error or huber loss
        """
        if self.loss_mode == "mse":
            return errors.pow(2)
        elif self.loss_mode == "huber":
            # Huber loss element-wise
            abs_errors = torch.abs(errors)
            return torch.where(
                abs_errors <= self.huber_kappa,
                0.5 * errors.pow(2),
                self.huber_kappa * (abs_errors - (0.5 * self.huber_kappa)))
        else:
            assert(False), \
                f"{self.loss_mode} is not a valid q-learning loss mode"

    def _get_aggregator(self, name):
        assert (name in ['mean', 'sum'])
        return torch.mean if name == 'mean' else torch.sum

    def _aggregate_losses(self, losses, timesteps):
        """Aggregates the losses to a single value using the requested
        aggregation modes
        """
        assert(len(losses.shape) == 1)
        if self.loss_timestep_aggregation:
            # Optionally use a different aggregation across timesteps
            # Timesteps are on the first dimension
            losses = self.loss_timestep_aggregation(
                losses.view(timesteps, -1), dim=0)
        return self.loss_aggregation(losses)

    def _compute_grads(self, states, targets, policy_outputs, extra_data,
                       timesteps):
        # Q-Learning loss function is basically huber/mean-squared error
        # between the outputs (qvalues) of the acted actions
        # and the bootstrapped nstep-discounted reward return starting from
        # that state/action
        # Forward training pass to return all the qvalues
        all_qvalues = self.policy.predict(states, timesteps)

        # Take the actions used when this transition happened at acting time
        actions = self.policy.make_tensor(
            policy_outputs['actions']).long().unsqueeze(-1)
        assert(actions.shape[:-1] == all_qvalues.shape[:-1])
        # Gather only the qvalues for the acted actions, to train only those
        # action-q-values
        chosen_qvalues = torch.gather(
            all_qvalues, dim=-1, index=actions).squeeze(-1)

        assert(targets.shape == chosen_qvalues.shape)
        td_errors = (chosen_qvalues - targets)

        loss = self._calc_loss(td_errors)

        # If we got importance weights (From prioritized replay for example),
        # multiply the losses element-wise by them
        loss = self._apply_importance_weights_if_needed(loss, extra_data)

        # Aggregate the losses and do the backprop pass
        loss = self._aggregate_losses(loss, timesteps)
        loss.backward()

        # If someone wants to get the td-error back, send them (Prioritized
        # replay for example)
        # NOTE: This needs to be after the backward() to avoid stalling the
        # cuda pipe since it copies the losses back to CPU
        self._report_losses_if_needed(td_errors, extra_data)

        # Log some stats
        self.value_log.log("qloss", loss.item(), group="train")
        self.value_log.log(
            "qvalue", chosen_qvalues.mean().item(), group="train")
