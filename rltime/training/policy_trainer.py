import time
import numpy as np
import logging
import pickle

from rltime.general.value_log import ValueLog
from rltime.general.utils import deep_dictionary_update


class PolicyTrainer():
    """ Base class for all policy trainers.

    In particular handles:
    - Policy/Model creation
    - Target policy creation and updating, if defined
    - Sampling actors for new data upon request
    - Tracking/logging rewards and stats from the actors
    - Saving policy checkpoints

    Actual training class should override _train() which will get all training
    kwargs not relevant to this class
    """
    def __init__(self, logger, actors, model_config, policy_args={}):
        self.logger = logger
        self.actors = actors
        self.model_config = model_config
        self.policy_args = policy_args
        self.episode_rewards = {}
        self.episode_lens = {}
        self.action_hist = {}
        self.action_hist_count = 0
        self.target_update_freq = 0
        self.value_log = ValueLog()

    @staticmethod
    def create_policy(**kwargs):
        raise NotImplementedError

    def init_policies(self):
        """Initializes the policies to train, as well as the separate
        target-policy if requested, and configures the actors with the
        policy"""

        observation_space, action_space = self.actors.get_spaces()
        logging.getLogger().info(f"Creating Policies")
        logging.getLogger().info(f"Observation Space: {observation_space}")
        logging.getLogger().info(f"Action Space: {action_space}")
        policy_creation_args = dict(
            model_config=self.model_config,
            observation_space=observation_space,
            action_space=action_space,
            **self.policy_args)

        self.policy = self.create_policy(**policy_creation_args)

        if not self.target_update_freq:
            # No separate target policy, use the main one for target values
            self.target_policy = self.policy
        else:
            # Separate policy for target values
            self.target_policy = self.create_policy(**policy_creation_args)

        self.actors.set_actor_policy(self.policy)
        logging.getLogger().info(f"Training Policy:\n{self.policy}")
        policy_size = len(pickle.dumps(self.policy.get_state()))/1024./1024.
        logging.getLogger().info("Policy Size: %.2fMB" % policy_size)

    def sync_target(self):
        """Syncs the target policy with the online policy"""
        self.target_policy.copy_from(self.policy)

    def _format_action_hist(self):
        """Formats the action histogram for logging (How much each action was
        performed during acting) and resets it"""
        if not self.action_hist:
            return []
        ret = [0]*(max(self.action_hist.keys())+1)
        for key, value in self.action_hist.items():
            ret[key] = round(value / self.action_hist_count, 3)
        self.action_hist = {}
        self.action_hist_count = 0
        return ret

    def update_actors(self):
        """Updates the actors with current training state"""
        progress = self.get_train_progress()
        # Note we don't pass the policy state to the actor, since it already
        # has the main policy, so it's up to it to decide what to do (For
        # example, if it has remote actors it may trigger an update_state()
        # call to them using the policy state, otherwise it doesn't need to do
        # anything special)
        self.actors.update_state(progress=progress)

    def _track_rewards(self, samples):
        """Tracks training rewards and stats from newly received actor samples
        """
        for sample in samples:
            info = sample['info']
            done = sample['done']
            reward = sample['reward']
            env_id = sample['env_id']
            ep_info = info.get("episode_info", None)
            if ep_info is None:
                self.episode_rewards[env_id] = \
                    self.episode_rewards.get(env_id, 0) + reward
                self.episode_lens[env_id] = self.episode_lens.get(env_id, 0)+1
                ep_done = done
            else:
                # If we have a monitor ENV passing back the actual
                # reward/length/done use it (This should usually be the case)
                self.episode_rewards[env_id] = ep_info['reward']
                self.episode_lens[env_id] = ep_info['length']
                ep_done = ep_info['done']

            if ep_done:
                # Log total and this-interval episode counts
                self.value_log.log(
                    "episodes", 1, agg="sum", group="this_interval")
                self.value_log.log(
                    "episodes", 1, agg="sum", group="total", scope=None)

                # Log the episode reward and length values (Average/max, using
                # the requested window sizes)
                for key, value in [
                        ("reward", self.episode_rewards[env_id]),
                        ("episode_length", self.episode_lens[env_id])]:
                    for last in self.episode_history_windows:
                        self.value_log.log(
                            key, value, scope=last, group=f"last{last}",
                            precision=2)
                        self.value_log.log(
                            key+"_max", value, agg="max", scope=last,
                            group=f"last{last}", precision=2)

                self.episode_rewards[env_id] = 0
                self.episode_lens[env_id] = 0

            # Log exploration info, if available
            if "exploration" in info:
                self.value_log.log_dict(
                    info['exploration'], group="acting->exploration")

            # Log any stats provided by the ENV, if exists
            if "env_stats" in info:
                self.value_log.log_dict(
                    info['env_stats'], group="acting->env_stats")

            # Track action histogram for logging
            action = sample['policy_output'].get('actions', None)
            if action is not None:
                if isinstance(action, np.ndarray):
                    # Box actions, we will log the mean value of each action
                    for i, val in enumerate(action.reshape(-1)):
                        self.action_hist[i] = self.action_hist.get(i, 0) + val
                else:
                    # Discrete actions, we will log how many times we chose
                    # each action
                    self.action_hist[action] = \
                        self.action_hist.get(action, 0) + 1
                self.action_hist_count += 1

    def _update_steps_trained(self, steps):
        """Called by the sub-class with how many steps it trained (for log)"""
        self.value_log.log(
            "steps_trained", steps, agg="sum", group="this_interval")
        self.value_log.log(
            "steps_trained", steps, agg="sum", group="total", scope=None)
        self.ts_steps_trained += steps

    def _get_train_state(self):
        # Optional training state to save with the checkpoint (For example
        # optimizer state such as momentums etc)
        return {}

    def _save_checkpoint(self):
        """Saves a training checkpoint using the logger

        Checkpoint includes the policy state (i.e. the weights), as well as
        any optional training state (e.g. optimizer state)
        """
        data = {
            "policy_state": self.policy.get_state(),
            "train_state": self._get_train_state()
            }
        self.logger.save_checkpoint(data, self.steps)

    def _log_checkpoint(self):
        """Called at each log interval to log the training progress"""
        end_time = time.time()

        # Log stats for the interval
        self.value_log.log(
            'steps_acted_per_second',
            int(self.ts_steps/(end_time-self.ts_start+1e-5)),
            group="this_interval")
        self.value_log.log(
            'steps_trained_per_second',
            int(self.ts_steps_trained/(end_time-self.ts_start+1e-5)),
            group="this_interval")
        self.value_log.log(
            'train_ratio', self.ts_steps_trained / self.ts_steps,
            group="this_interval")
        self.value_log.log(
            'seconds', (end_time-self.ts_start), group="this_interval",
            precision=2)
        self.value_log.log('steps_acted', self.ts_steps, group="this_interval")
        self.value_log.log(
            'steps_acted', self.ts_steps, agg="sum", group="total", scope=None)

        # Get the aggregated value-log dictionary
        log_info = self.value_log.get()

        # Add the 'action histogram' to the log
        action_hist = self._format_action_hist()
        deep_dictionary_update(log_info, {"acting": {"actions": action_hist}})

        self.logger.log_result("train", log_info, self.steps)

        # Save the policy/training checkpoint (TODO: make this configurable
        # to difference frequency than the log interval)
        self._save_checkpoint()

        self.ts_start = end_time
        self.ts_steps = 0
        self.ts_steps_trained = 0

    def _start_timer(self, name):
        """Starts timing an operation with the given name

        Only one at a time supported
        """
        self._timer_name = name
        self._timer_time = time.time()

    def _end_timer(self):
        """Ends timing an operation and logs the time"""
        duration = ((time.time()-self._timer_time) * 1000.)
        self.value_log.log(
            self._timer_name, duration, agg="mean", group="timings_mean_ms",
            precision=2)
        self.value_log.log(
            self._timer_name, duration, agg="sum", group="timings_total_ms",
            precision=2)

    def _process_new_samples(self, new_samples):
        """Process new samples from actors"""
        self._track_rewards(new_samples)

        if self.clip_rewards:
            for sample in new_samples:
                sample['reward'] = np.sign(sample['reward'])

    def sample_actors(self, min_samples):
        """Retrieve new samples from the actors"""
        self._start_timer("sample_actors")
        samples = self.actors.get_samples(min_samples)

        if not samples:
            return None

        self._process_new_samples(samples)

        total_steps = len(samples)
        self.steps += total_steps
        self.ts_steps += total_steps

        # Update/Sync the target policy if relevant
        if self.target_update_freq > 0:
            if (self.steps // self.target_update_freq) != \
                    ((self.steps - total_steps) // self.target_update_freq):
                self.sync_target()

        # Log if crossed the frequency
        if (self.steps // self.log_freq) != \
                ((self.steps-total_steps) // self.log_freq):
            self._log_checkpoint()

        self._end_timer()
        return samples

    def train(self, total_steps, log_freq=10000, target_update_freq=0,
              clip_rewards=False, early_stop_steps=None,
              episode_history_windows=[10, 100], **kwargs):
        """ Trains the policy

        Args:
            total_steps: The total (acting) steps to run
            log_freq: Every how many (acted) steps to perform logging
            target_update_freq: If not zero, a second 'target policy' is
                created and updated/synced to the main one every this many
                acted steps
            clip_reward: Whether to clip the rewards received from acting
                (to [-1,0,1])
            early_stop_steps: If specified, stop training earlier at this many
                steps
            episode_history_windows: Log episode stats (reward/length) using
                these window sizes
            kwargs: Additional training arguments for the training
                sub-class, passed to the call to _train()

        """
        self.target_update_freq = target_update_freq
        self.episode_history_windows = episode_history_windows
        self.log_freq = log_freq
        self.clip_rewards = clip_rewards
        self.ts_start = time.time()
        self.steps = 0
        self.ts_steps = 0
        self.ts_steps_trained = 0
        self.total_steps = total_steps
        self.early_stop_steps = early_stop_steps

        # Initialize the policies and update the actors with the latest policy
        # state
        self.init_policies()
        self.update_actors()

        logging.getLogger().info(
            "Training start with total acting ENVs: "
            f"{self.actors.get_env_count()}")
        self._train(**kwargs)
        logging.getLogger().info(f"Training finished")

    def _train(self, **kwargs):
        raise NotImplementedError

    def train_is_done(self):
        """Checks if training is done"""
        return (
            (self.get_train_progress() >= 1.0) or
            (self.early_stop_steps is not None and
             self.steps >= self.early_stop_steps)
        )

    def get_train_progress(self):
        return (self.steps / self.total_steps)
