from gym import spaces

from .torch_policy import TorchPolicy
from rltime.models.torch.utils import linear

from .distributions import get_dist_layer_class


class ActorCriticPolicy(TorchPolicy):
    def __init__(self, model_config, observation_space, action_space,
                 critic_separate_model=False):
        """Initialize an actor-critic policy

        Args:
            model_config: The configuration for creating the model
            observation_space: The observation space defining the model input
            action_space: The action space for outputting actons as
            critic_separate_model: Whether to use a duplicate/separate model
                for the critic (Otherwise, same model is used for both actor
                and critic)
        """
        super().__init__(model_config, observation_space)

        # Optionally use a separate identical model for the critic
        if not critic_separate_model:
            self.value_model = None
        else:
            self.value_model = self._create_model_from_config(
                model_config, observation_space)

        # Create the actor distribution and critic layers
        self.actor = get_dist_layer_class(action_space)(
            action_space, self.model.out_size)
        self.critic = linear(self.model.out_size, 1)

    def actor_predict(self, inp, timesteps, force_best=False):
        """Performs action-selection on the input for acting

        force_best chooses the argmax best action instead of sampling from the
        the distribution, this might make sense in certain use-cases (Only for
        evaluation/inference, not training), and only for discrete spaces
        """
        dist, values = self.get_dist_and_state_value(inp, timesteps)

        # Choose the batch actions
        if not force_best:
            # Standard case: Sample the actions from the distribution
            actions = dist.sample().data
        else:
            # Deterministic argmax action-selection (For eval/inference only)
            # This will fail for non-discrete spaces
            actions = dist.logits.argmax(dim=-1)

        # Return also the log-probs for the actions to be used during training
        action_log_probs = dist.log_prob(actions).data.cpu().numpy()

        # For multi-action distributions aggregate log_probs using sum()
        if len(action_log_probs.shape) == 2:
            action_log_probs = action_log_probs.sum(-1)

        actions = actions.cpu().numpy()
        values = values.cpu().data.numpy()

        # In addition to the actions, we also return the log_probs and value
        # estimations which are typically needed during training
        return {
            "actions": actions,
            "action_log_probs": action_log_probs,
            "values": values
        }

    def get_state_value(self, inp, timesteps):
        """Returns only the state-value estimation for the input"""
        use_model = self.value_model \
            if self.value_model is not None \
            else self.model
        res = use_model(inp, timesteps)['output']
        res = self.critic(res).squeeze(-1)
        return res

    def evaluate_actions(self, inp, timesteps, actions):
        """Evaluates the given actions for the given input for training,
        returning the log-probs/values/entropy
        """
        dist, values = self.get_dist_and_state_value(inp, timesteps)
        action_log_probs = dist.log_prob(self.make_tensor(actions))

        # For multi-action distribution aggregate log_prob using sum()
        if len(action_log_probs.shape) == 2:
            action_log_probs = action_log_probs.sum(-1)
        entropy = dist.entropy()
        return action_log_probs, values, entropy

    def get_dist_and_state_value(self, x, timesteps):
        """Does a forward pass and returns the output distribution and
        state-value estimation"""
        model_output = self.model(x, timesteps)['output']
        dist = self.actor(model_output)

        # If separate value-function model, use that one as the critic
        if self.value_model is not None:
            model_output = self.value_model(x, timesteps)['output']
        values = self.critic(model_output).squeeze(-1)

        return dist, values
