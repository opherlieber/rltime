from rltime.training.multi_step_trainer import MultiStepTrainer
import torch
from rltime.models.torch.utils import set_lr


class TorchTrainer(MultiStepTrainer):
    """ Base class for all multi-step pytorch training. """

    def _train(self, clip_grad=None, clip_grad_dynamic_alpha=None,
               adam_epsilon=1e-8, vf_scale_epsilon=None, **kwargs):
        """Triggers the training

        Args:
            clip_grad: Whether to clip gradient-norms before the optimizer
                step. If not None this is the value to use for clipping.
            clip_grad_dynamic_alpha: If not None and clip_grad is not None then
                use dynamic gradient-norm clipping instead of fixed. In this
                case an exponential moving-average of the gradient-norms are
                tracked using this value as the 'alpha' value. The
                gradient-norm will be clipped to the moving-average value times
                the 'clip_grad' parameter value (For example set clip_grad=1.5
                to clip the gradient norm to 1.5x of the moving-average
                gradient-norm).
                This may be usefull to avoid 'destructive' gradient updates
                without losing the general magnitude of the gradients, however
                it may not help with 'exploding gradient' issues.
            adam_epsilon: Epsilon value to use for the adam optimizer
            vf_scale_epsilon: If not None then value-functon rescaling will be
                used for the target value, as defined in the R2D2 paper, using
                this value as the epsilon value. This is used as an alternative
                to reward clipping so should be enabled together with
                'clip_rewards' set to False in the base class
        """
        self.clip_grad = float(clip_grad) if clip_grad is not None else None
        self.clip_grad_dynamic_alpha = clip_grad_dynamic_alpha
        if clip_grad_dynamic_alpha is not None:
            self._grad_norm_moving_average = None
        self.adam_epsilon = adam_epsilon
        assert(vf_scale_epsilon is None or vf_scale_epsilon > 0)
        assert((not vf_scale_epsilon) or (not self.clip_rewards)), \
            "Value function rescaling only makes sense with clip_rewards=False"
        self.vf_scale_epsilon = vf_scale_epsilon

        super()._train(**kwargs)

    def _vf_scale(self, x):
        """Performs value-function rescaling of the given value, if enabled"""
        if not self.vf_scale_epsilon:
            return x
        # Value function scaling as in the R2D2 paper
        return torch.sign(x) * \
            (torch.sqrt(torch.abs(x) + 1) - 1) + (self.vf_scale_epsilon * x)

    def _vf_unscale(self, scaled_x):
        """Computes the inverse of _vf_scale(x), if vf-rescaling is enabled"""
        if not self.vf_scale_epsilon:
            return scaled_x

        # We need double() otherwise we lose too much precision for low eps
        # values such as 1e-3, due to the eps**2 terms
        scaled_x = scaled_x.double()
        abs_scaled_x = torch.abs(scaled_x)
        eps = self.vf_scale_epsilon
        # TODO: Can this be simplified somehow?
        x = abs_scaled_x / eps - (
                (1 / (2. * (eps**2))) *
                torch.sqrt(
                    4 * self.vf_scale_epsilon*abs_scaled_x +
                    (2. * eps + 1)**2)
            ) + \
            (2. * eps + 1) / (2. * (eps ** 2))
        x *= torch.sign(scaled_x)

        # SANITY CHECK to make sure the inverse is working, enable only to
        # test this function
        # assert(torch.all(torch.abs(scaled_x - self._vf_scale(x))<1e-5)), ("_vf_unscale() sanity failed:",(scaled_x, self._vf_scale(x)),(scaled_x == self._vf_scale(x)))

        return x.float()

    def train_init(self, lr):
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), eps=self.adam_epsilon)

    def _compute_grads(self, states, targets, policy_outputs, extra_data,
                       timesteps):
        """Should be implemented by the sub-class to compute the gradients on
        the given states and targets"""
        raise NotImplementedError

    def _get_bootstrap_target_value(self, target_states, timesteps):
        """Should be implemented by the sub-class to compute the bootstrap
        target value on the given target states"""
        raise NotImplementedError

    def _discount_bootstrap_target_value(self, target_values, nsteps):
        """Discounts the calculated target-bootstrap value using the default
        discount method and given target nsteps"""
        return (self.gamma ** nsteps) * target_values

    def calc_target_values(self, returns, target_states, target_masks, nsteps,
                           timesteps):
        """Calculates target values for the given target-states

        Optionally using value-function-rescaling if configured, supports 1D
        and 2D targets
        """
        with torch.no_grad():  # Target values should not generate gradients
            # Make tensors of everything (non_blocking can potentially help but
            # doesn't really since memory isn't pinned)
            target_states, returns, target_masks, nsteps = \
                self.target_policy.make_tensor(
                    (target_states, returns, target_masks, nsteps),
                    non_blocking=True)
            # Calculate the bootstrap target value using the given target
            # states
            bootstrap_target_value = self._get_bootstrap_target_value(
                target_states, timesteps)

            # value function rescaling as in the R2D2 paper, here we
            # unscale/inverse the bootstrapped target value, and later rescale
            # the finalized target at the end. Note this has no effect in the
            # default case of vf_scale_epsilon=None
            bootstrap_target_value = self._vf_unscale(bootstrap_target_value)

            # Target value can be either 1D or 2D (2D for example in case of
            # distributional or IQN target), in any case the batch should be on
            # dim=0
            assert(returns.shape == (bootstrap_target_value.shape[0],))
            assert(target_masks.shape == (bootstrap_target_value.shape[0],))
            assert(nsteps.shape == (bootstrap_target_value.shape[0],))
            assert(len(bootstrap_target_value.shape) in [1, 2])
            if len(bootstrap_target_value.shape) == 2:
                returns = returns.unsqueeze(-1)
                target_masks = target_masks.unsqueeze(-1)
                nsteps = nsteps.unsqueeze(-1)

            # Calculate the final discounted nstep-bootstrapped (optionally
            # rescaled) target value.
            # In any case where the episode terminated within the target nstep
            # target_mask will be 0.
            # In case of a second/distributional dimension then
            # returns/nsteps/target_masks will be broadcasted to that dimension
            return self._vf_scale(
                returns +
                self._discount_bootstrap_target_value(
                    bootstrap_target_value, nsteps)*target_masks)

    def set_lr(self, lr):
        """Sets a new LR value for the optimizer"""
        set_lr(self.optimizer, lr)

    def _get_grad_norm_clip_value(self, cur_grad_norm):
        """Returns the value to clip the gradient norm to, if any, given the
        current gradient norm"""
        if not self.clip_grad:
            # No grad clipping
            return None
        elif self.clip_grad_dynamic_alpha is None:
            # Standard fixed clipping to the specified value
            return self.clip_grad
        else:
            # Dynamic clipping, calculate moving-average of the grad norm and
            # clip to that value with 'clip_grad' factor
            if self._grad_norm_moving_average is None:
                self._grad_norm_moving_average = cur_grad_norm
            else:
                self._grad_norm_moving_average = \
                    (self._grad_norm_moving_average *
                     self.clip_grad_dynamic_alpha) + cur_grad_norm * (
                         1 - self.clip_grad_dynamic_alpha)
            # Log the current grad norm moving-average value
            self.value_log.log(
                "grad_norm_ma", self._grad_norm_moving_average, group="train")
            return self._grad_norm_moving_average * self.clip_grad

    def train_batch(self, *args, **kwargs):
        # Reset gradients
        self.policy.zero_grad()

        # Compute the gradients on the given training arugments
        self._compute_grads(*args, **kwargs)

        # Log and optionally clip the global gradient norm before updating the
        # weights
        grad_norm = self.policy.get_grad_norm()
        self.value_log.log("grad_norm", grad_norm, group="train")
        self.value_log.log(
            "grad_norm_max", grad_norm, group="train", agg="max")
        clip_value = self._get_grad_norm_clip_value(grad_norm)
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), clip_value)
            self.value_log.log(
                "grad_norm_clipped", self.policy.get_grad_norm(),
                group="train")

        # Perform the optimizer weight update
        self.optimizer.step()
