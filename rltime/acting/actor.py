import numpy as np

from rltime.general.type_registry import get_registered_type
from rltime.env_wrappers.vec_env.simple import make_simple_vec_env
from rltime.env_wrappers.vec_env.sub_proc import make_sub_proc_vec_env
from rltime.general.utils import deep_apply, deep_dictionary_update
from .acting_interface import ActingInterface


class Actor(ActingInterface):
    """Implements the basic (vectorized) actor logic

    This will be just a local vectorized actor if used directly (Using the
    training policy directly and synchronously to training), or can be part of
    a more complex acting configuration if wrapped by one of the acting pool
    classes
    """

    def __init__(self, env_creator, num_envs=1, exploration_config=None,
                 use_proc="auto", base_env_id=0, total_env_ids=None):
        """Initializes the actor

        Args:
            env_creator: The env creation function to create a new instance
                of the environment we want to act on
            num_envs: The amount of ENVs to predict/act on in parallel
                (Vectorized action-selection and environment steps)
            exploration_config: Exploration configuration for overriding
                policy actions, if applicable.
                Dictionary in the form: {"type":type,"args":{}}
            use_proc: Whether the vectorized ENV should run each ENV in a
                sub-process or not. The default value of 'auto' sets this
                to False for num_envs=1 and True for num_envs>1. In case of
                very fast ENVs it is sometimes preferable to set this to False
                also for num_envs>1
            base_env_id: The base ENV ID for this actor, if running it along
                other actors (Used for seeding the ENVs and for exploration)
            total_env_ids: The total amount of ENVs running, if running this
                actor alongside other actors (Relevant only for certain
                exploration configurations)
        """

        self._num_envs = num_envs
        self._base_env_id = base_env_id
        total_env_ids = total_env_ids or num_envs
        self._env_ids = list(range(base_env_id, base_env_id + num_envs))

        # Init exploration manager if configured
        if exploration_config:
            exploration_cls = get_registered_type(
                "exploration", exploration_config['type'])
            self._exploration_manager = exploration_cls(
                **exploration_config.get("args"), total_actors=total_env_ids)
        else:
            self._exploration_manager = None

        self._actor_policy = None
        self._progress = 0

        # Create the vec-env
        if use_proc == "auto":
            use_proc = self._num_envs > 1
        self._vec_env = self._create_envs(env_creator, use_proc)

        super().__init__(
            self._vec_env.observation_space, self._vec_env.action_space)

    def get_env_count(self):
        return self._num_envs

    def update_state(self, progress, policy_state=None):
        """Updates the training progress and policy state"""
        self._progress = progress
        if policy_state is not None:
            # Load the updated policy state to the acting policy
            self._actor_policy.load_state(policy_state)

    def set_actor_policy(self, actor_policy):
        """Configures the acting policy

        This is assumed to be called only once. We reset the vec-env
        for the first time and create the initial 'state' to act on
        """
        self._actor_policy = actor_policy

        # Make the initial policy state
        self.last_state = self._actor_policy.make_state(
            self._vec_env.reset(), np.array([True]*self._num_envs))

    def close(self):
        self._vec_env.close()

    def _create_envs(self, env_creator, use_proc):
        maker = make_sub_proc_vec_env if use_proc else make_simple_vec_env
        return maker(env_creator, self._num_envs, base_seed=self._base_env_id)

    def get_samples(self, min_samples):
        """Performs the next vectorized actings steps"""

        # We always get at least 1 sample here, even if requested 0
        min_samples = max(1, min_samples)

        # Each vectorized iteration gets us 'num_envs' samples, so make sure
        # we get at least the minimum requested amount
        iters = (min_samples + self._num_envs - 1) // self._num_envs

        samples = []
        for _ in range(iters):
            # Predict/Select the next action using the acting policy
            pred = self._actor_policy.actor_predict(
                self.last_state, timesteps=1)

            # If we have an exploration manager, use it to (possibly) remap the
            # chosen actions
            if self._exploration_manager is not None:
                actions, exp_info = self._exploration_manager.remap_actions(
                        pred['actions'], self._env_ids,
                        self._action_space, self._progress
                )
                pred['actions'] = actions

            # Perform the actions on the ENVs
            obs, rewards, dones, infos = self._vec_env.step(pred['actions'])

            # The new state is a combination of the new observation as well as
            # any additional state information from the policy (For example
            # RNN hidden states for each RNN layer)
            states = self._actor_policy.make_state(obs, np.array(dones))

            # Although we did vectorized work above, the final samples need
            # to be singular so we split them
            for i in range(self._num_envs):
                sample_pred = deep_apply(pred, lambda x: x[i])
                sample_state = deep_apply(states, lambda x: x[i])
                info = infos[i]
                if self._exploration_manager is not None:
                    # Add exploration stats to the info for logging
                    info["exploration"] = deep_apply(exp_info, lambda x: x[i])

                samples.append(
                    self._create_sample(
                        sample_pred, sample_state, rewards[i], dones[i], info,
                        self._env_ids[i]
                    )
                )

            self.last_state = states

        return samples





