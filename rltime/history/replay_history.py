from .history import History
import numpy as np
from collections import deque


class ReplayHistoryBuffer(History):
    """A replay history buffer. Uniform sampling by default.

    Saves the last X samples, and samples uniformly from all of them (e.g.
    DQN replay buffer).
    Supports multi-step/multi-env sample retrieval for training multi-step
    transitions across multiple environments in parallel.
    """
    def __init__(self, size, train_frequency, avoid_episode_crossing=False,
                 **kwargs):
        """Init a replay history buffer.

        Args:
            size (int): Size of the replay buffer, in single-step/single-env
                transitions/samples
            train_frequency (int): How often to train (on average) on each
                newly added sample, for example: a value of 4 means on each
                sample we generate/get, we train 4 samples (On average, not
                necessarily evenly per sample)
                This is the inverse of the 'replay period' commonly seen in DQN
                papers. For example common DQN does a replay-period of 4 with a
                batch size 32, this would be equivalent to train_frequency=8
                here (32/4), i.e. every ENV transition is trained 8 times on
                average.
                Setting to 0/None means unconstrained training, which behaves
                like distributed DQN implementations (APEX/R2D2). This might be
                faster in some cases but makes the training much less
                deterministic and dependent on the amount of CPUs/Actors (For
                example you can get a very low or very high train frequency
                depending on how many actors/CPUs are running and what GPU is
                training and how much the training machine is overloaded)
                WARNING: Set to 'None' only if using the asynchronous acting
                options. Setting to 0 when using synced/vectorized acting will
                cause unexpected results
            avoid_episode_crossing: Trys to avoid choosing sequences which
                cross episode boundaries. If enabled and such a sequence is
                chosen, the sequence is shifted to either the end of the
                previous episode or start of the next one. This works as long
                as episodes are longer than nstep_train which should usually
                be the case except very early in the training.
                This is only relevent for multi-step (nstep_train>1) training.
        """
        super().__init__(**kwargs)
        self.avoid_episode_crossing = avoid_episode_crossing
        self.size = size
        # A 'linear' history of all samples, in the order they were added.
        # Note these are just references to the same sample in the main
        # 'buffer' in the parent class, there is no RAM duplication here.
        # This defines the order samples are removed from the replay buffer
        self.linear_history = deque()
        self.train_frequency = train_frequency
        # This signifies how much train quota we have, for each sample
        # received this increases by 'train_frequency' and decreases on every
        # sample trained
        self.train_quota = 0

    def needed_feed_count(self, mbatch_size, num_envs):
        # If train_frequency is not defined, we want to train/act
        # asynchronously, so always request only available samples
        if not self.train_frequency:
            return 0
        else:
            if self.train_quota > 0:
                return None
            else:
                # Request the amount we need to reach positive training quota
                # but at least the amount of ENVs so not to get to little at
                # a time
                return max(
                    int(-self.train_quota / self.train_frequency), num_envs)

    def _sample_added(self, sample):
        # If history is full, remove the oldest sample
        if len(self.linear_history) >= self.size:
            assert(len(self.linear_history) == self.size)
            sample_to_remove = self.linear_history.popleft()
            # sanity check, this must be the 1st sample in the buffer for this
            # ENV
            env_id = sample_to_remove["env_id"]
            assert(self.buffer[env_id] and
                   self.buffer[env_id][0] is sample_to_remove)
            self._remove_samples(env_id, 1)

        self.linear_history.append(sample)
        if self.train_frequency:
            self.train_quota += self.train_frequency

    def _get_train_data(self, mbatch_size, train_progress):
        available_per_env = {}
        # The total sequence length we want for each batch item includes
        # the nstep_train and additional amount of requested prefix steps
        total_seq_len = self.nstep_train + self.prefix_steps
        # Calculate how many samples we actually have per ENV.
        total_available = 0
        for env_id, env_samples in self.buffer.items():
            # How many samples this env has available, also accounting for
            # target nstep (To guarantee the full nstep_target also near the
            # end)
            available = len(env_samples) - (total_seq_len+self.nstep_target-1)
            if available > 0:
                available_per_env[env_id] = available
                total_available += available

        # If we don't have enough choices return
        if total_available < mbatch_size:
            # If buffer is full we can't ever train so fail, this would mean
            # you should increase the buffer size or decrease nstep_train
            assert(len(self.linear_history) < self.size)
            return None

        # Choose mbatch_size unique multi-step trajectories from all available
        # ones
        choices = np.random.choice(total_available, mbatch_size)
        sample_ranges = []
        for choice in choices:
            sample_range = None
            # We need to find where the chosen index falls from all our
            # available envs/samples
            for env_id, amount in available_per_env.items():
                if choice < amount:
                    # Adjust the start offset if applicable
                    start = self._refine_sample_range(
                        env_id, choice, total_seq_len)
                    # Generate the trajectory
                    sample_range = self._make_sample_range(
                        env_id, start, total_seq_len)
                    break
                else:
                    choice -= amount
            assert(sample_range), \
                "ReplayHistoryBuffer chosen index somehow fell out of range " \
                "of available samples"
            sample_ranges.append(sample_range)

        return self._make_train_batch(sample_ranges)

    def _refine_sample_range(self, env_id, start_index, amount):
        """If requested at init(), avoid crossing episodes within a multi-step
        sample when possible
        """
        if not self.avoid_episode_crossing:
            return start_index

        env_buffer = self.buffer[env_id]

        # Ignore the last sample since that one is fine to end at 'done'
        for index in range(amount - 1):
            if env_buffer[start_index + index]['done']:
                # We encountered a done, so we shift this sequence either to
                # end of prev episode or start of next episode, depending on
                # where we fell relative to middle.
                # Although there's typically much more to learn at the end than
                # the start, it may be preferable to avoid over-sampling
                # end-of-episode sequences more than we already are with this
                # method
                # Note that we don't handle the case of 2 'dones' within the
                # same nstep_train since that should be rare especially after
                # learning takes off
                if index < amount / 2:
                    start_index = max(start_index - (amount - index - 1), 0)
                else:
                    start_index = min(
                        start_index + index + 1, len(env_buffer)-amount)
                break

        return start_index

    def get_train_data(self, mbatch_size, train_progress=None):
        # Reduce quota even if we don't end up finding enough samples (To avoid
        # quota explosion initially when filling up the buffer)
        if self.train_frequency:
            self.train_quota -= mbatch_size * self.nstep_train
            # Sanity check, make sure we aren't training too slowly
            assert(self.train_quota < 100 * mbatch_size * self.nstep_train)
            # Or vice-versa
            assert(self.train_quota > -100 * mbatch_size * self.nstep_train)

        # Use this internal function which can be overridden by sub-sclasses
        return self._get_train_data(mbatch_size, train_progress)
