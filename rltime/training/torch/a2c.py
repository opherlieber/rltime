from .torch_trainer import TorchTrainer
from rltime.policies.torch.actor_critic import ActorCriticPolicy
from rltime.general.utils import anneal_value


class A2C(TorchTrainer):
    """A2C Training Algorithm

    This is like A3C but with centralized training. If using local/vectorized
    acting it should be equivalent to the OpenAI baselines A2C implementation.
    This is also the base class for other Advantage-Actor-Critic
    implementations such as PPO.

    By default this uses 'online' history mode which is common for all
    actor-critic implementations, however it's possible to change this to one
    of the replay modes if it makes sense.

    GAE returns are implemented here, but the default advlam=1.0 should result
    in the default discount function as in other A2C implementations, though
    it's still possible to use GAE also for A2C if it makes sense
    """
    def _train(self, entropy_factor, entropy_anneal=None, vf_coef=1.0,
               advlam=1.0, adv_norm=False, history_mode={"type": "online"},
               **kwargs):
        """Entry point for A2C and general actor-critic training

        Args:
            entropy_factor: Entropy coefficient to use for the policy entropy
                term in the loss
            entropy_anneal: Whether to anneal the entropy_factor to 0
                throughout the training
            vf_coef: Value-function loss coefficient to control the value-loss
                term contribution in the total loss
            advlam: 'Lambda' value to use for 'Generalized Advantage
                Estimation' returns. The default (1.0) is equivalent to
                disabling GAE and using the basic (gamma**nstep * reward)
                discounting
        """
        self.entropy_factor = entropy_factor
        self.entropy_anneal = entropy_anneal
        self.vf_coef = vf_coef
        self.advlam = advlam
        self.adv_norm = adv_norm
        super()._train(history_mode=history_mode, **kwargs)

    @staticmethod
    def create_policy(**kwargs):
        return ActorCriticPolicy.create(**kwargs)

    def _get_discount_function(self, gamma):
        """Override return discounting for 'truncated generalized advantage
        returns' (GAE), controlled by 'advlam', from the PPO paper.

        Note the final return will not include the T+1 target bootstrap and the
        initial -V(St). The advantage would be the final return minus V(St)
        (After bootstrapping)
        Note that when advlam=1.0 this code is equivalent to the regular
        discount ((self.gamma ** nstep) * reward)
        """
        def discount(nstep, reward, policy_output):
            state_value = policy_output['values']

            return (gamma**nstep) * (self.advlam**(nstep - 1)) * \
                (state_value + self.advlam*(reward - state_value))

        return discount

    def _discount_bootstrap_target_value(self, target_values, nsteps):
        """Discount the bootstrap value using GAE advlam (Equivalent to the
        regular/base method if advlam=1.0)"""
        return (self.gamma**nsteps) * (self.advlam**(nsteps-1)) * target_values

    def _get_bootstrap_target_value(self, target_states, timesteps):
        """For actor-critic the bootstrap is just the value-estimation for the
        target-state

        Note we use target_policy here, though usually for actor-critic this is
        just the main/online policy, though it's still possible to use a
        delayed one
        """
        return self.target_policy.get_state_value(
            target_states, timesteps=timesteps)

    def _calc_entropy_factor(self):
        """Returns the current entropy factor to use"""
        return anneal_value(
            self.entropy_factor, self.get_train_progress(),
            self.entropy_anneal)

    def _calc_action_gain(self, action_log_probs, advantages,
                          org_policy_outputs):
        """Calculates the action-gain term for the loss

        The default 'vanilla' advantage-actor-critic just does
        log_prob*advantage for the action gain. this can be overwridden for
        example for PPO clipped action gain.
        """
        assert(action_log_probs.shape == advantages.shape)
        return (action_log_probs * advantages).mean()

    def _compute_grads(self, states, targets, policy_outputs, extra_data,
                       timesteps):
        # Get the current log-probs/values/entropy for the acted actions
        action_log_probs, state_values, entropy = self.policy.evaluate_actions(
            states, timesteps, policy_outputs['actions'])
        entropy = entropy.mean()
        assert(len(action_log_probs.shape) == 1)

        # Calc advantage
        assert(len(targets.shape) == 1)
        # ?? Is it more correct to use the baseline value from the original
        # policy output at acting time, or the current one which changed each
        # mini-epoch?
        # Currently using the original value at acting time
        adv_values = self.policy.make_tensor(policy_outputs['values'])
        assert(adv_values.shape == targets.shape)
        # This is the missing -V(St) term from our GAE calculation:
        advantages = (targets - adv_values)
        if self.adv_norm:
            # Normalize advantages if requested
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-5)
        action_gain = self._calc_action_gain(
            action_log_probs, advantages, policy_outputs)

        # Value loss is just MSE on the target value and current value
        value_target = targets
        assert(value_target.shape == state_values.shape)
        value_loss = (value_target - state_values).pow(2).mean()

        total_loss = value_loss * self.vf_coef - action_gain - \
            self._calc_entropy_factor() * entropy
        total_loss.backward()

        # Log the various metrics
        self.value_log.log(
            'state_value_mean', state_values.mean().item(), group="train")
        self.value_log.log('value_loss', value_loss.item(), group="train")
        self.value_log.log('policy_loss', -action_gain.item(), group="train")
        self.value_log.log('policy_entropy', entropy.item(), group="train")
