"""WARNING: This training algo has not been fully verified/tested and may
still have bugs and not reach any results at all."""
import torch
import torch.nn.functional as F

from .dqn import DQN
from rltime.policies.torch.dist_dqn import DistDQNPolicy


class DistDQN(DQN):
    """Distributional RL algorithm (AKA C51)

    https://arxiv.org/pdf/1707.06887.pdf

    Implementation heavily based on https://github.com/higgsfield/RL-Adventure
    (Rainbow notebook)

    WARNING: This has not been fully verified/tested and may still have bugs
    and not reach any results at all.
    """
    def _train(self, *args, loss_mode="crossentropy", **kwargs):
        """DistDQN uses crossentropy loss so we make it the default, though
        it's possible to override this for mse/huber if wanted"""
        super()._train(*args, loss_mode=loss_mode, **kwargs)

    @staticmethod
    def create_policy(**kwargs):
        return DistDQNPolicy.create(**kwargs)

    def calc_target_values(self, returns, target_states, target_masks, nsteps,
                           timesteps):
        with torch.no_grad():
            assert(not self.vf_scale_epsilon), \
                "DistDQN does not support value function rescaling"

            # Make tensors of everything
            target_states, returns, target_masks, nsteps = \
                self.target_policy.make_tensor(
                    (target_states, returns, target_masks, nsteps))
            returns, target_masks, nsteps = (
                returns.unsqueeze(-1), target_masks.unsqueeze(-1),
                nsteps.unsqueeze(-1))

            batch_size = returns.shape[0]
            num_atoms = self.policy.num_atoms
            # Make these tensors first to avoid cuda stall
            offset = torch.linspace(0, (batch_size - 1)*num_atoms, batch_size)
            offset = offset.long().unsqueeze(1).expand(batch_size, num_atoms)
            offset = self.policy.make_tensor(offset)
            proj_dist = self.policy.make_tensor(
                torch.zeros((batch_size, num_atoms)))

            # Calculate the target disribution
            target_dist_logits = self.target_policy.predict(
                target_states, timesteps=timesteps)  # (batch,num_actions,num_atoms)
            # For double-Q we choose the action for the target value using the
            # online network, otherwise from the target network
            if not self.double_q:
                action_selection_target_dist_logits = target_dist_logits
            else:
                action_selection_target_dist_logits = self.policy.predict(
                    target_states, timesteps=timesteps)  # (batch,num_actions,num_atoms)

            target_dist_actions = F.softmax(
                action_selection_target_dist_logits, dim=-1)
            target_dist_actions = target_dist_actions * self.policy.support  # (batch,num_actions,num_atoms)
            target_dist_actions = target_dist_actions.sum(2).argmax(
                dim=-1, keepdim=True)  # (batch, 1)
            target_dist_actions = target_dist_actions.unsqueeze(1).repeat(
                [1, 1, num_atoms])  # (batch, 1, num_atoms)
            # Now we select for each batch item it's target-dist based on the
            # 'best next action' we just calculated
            target_dist = torch.gather(
                target_dist_logits, dim=1,
                index=target_dist_actions).squeeze(1)  # (batch, num_atoms)
            target_dist = F.softmax(target_dist, dim=-1)

            # Project distribution
            # TODO: How does this work when target_mask=0 (i.e. end of episode)
            # in this case target_dist is not relevant as target_state is
            # undefined but it's still used
            delta_z = float(self.policy.vmax-self.policy.vmin) / (num_atoms-1)
            Tz = returns + target_masks * (self.gamma**nsteps)
            Tz = Tz * self.policy.support
            Tz = Tz.clamp(min=self.policy.vmin, max=self.policy.vmax)
            b = (Tz - self.policy.vmin) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1),
                (target_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1),
                (target_dist * (b - l.float())).view(-1))

            return proj_dist  # (batch, atoms)

    def _compute_grads(self, states, targets, policy_outputs, extra_data,
                       timesteps):
        actions = self.policy.make_tensor(
            policy_outputs['actions']).long().unsqueeze(-1).unsqueeze(-1)  # (batch_size,1,1)

        # Do a forward pass on the states we are training
        logits = self.policy.predict(states, timesteps)  # (batch_size,num_actions,num_atoms)
        num_atoms = logits.shape[-1]
        assert(len(logits.shape) == 3 and (num_atoms == self.policy.num_atoms))

        # Gather the outputs only for the specific actions chosen when these
        # transitions were taken, so to train only those specific action-values
        actions = actions.repeat([1, 1, num_atoms])  # (batch_size,1,num_atoms)
        chosen_action_logits = torch.gather(
            logits, dim=1, index=actions).squeeze(1)  # (batch_size, num_atoms)
        assert(chosen_action_logits.shape == targets.shape)

        if self.loss_mode == "crossentropy":
            # TODO: Find a 'cross-entropy-with-logits' version that works in
            # pytorch, instead of softmax+clamping
            log_probs = F.softmax(
                chosen_action_logits, dim=-1).clamp(1e-5, 1-1e-5).log()  # (batch_size, num_atoms)
            loss = -(targets * log_probs).sum(1)  # (batch_size)
        else:
            loss = self._calc_loss(
                F.softmax(chosen_action_logits, dim=-1) - targets).sum(1)

        # Report these per-batch-item losses back if needed (e.g. prioritized
        # replay), we delay this to after backward() to avoid stalling the cuda
        # pipe
        losses_to_report = loss

        # If we got importance weights (From replay for example), multiply the
        # losses element-wise by them
        loss = self._apply_importance_weights_if_needed(loss, extra_data)

        loss = self._aggregate_losses(loss, timesteps)
        loss.backward()

        # If someone wants to get the errors back, send them
        # (Prioritized replay for example).
        self._report_losses_if_needed(losses_to_report, extra_data)

        self.value_log.log("qloss", loss.item(), group="train")
