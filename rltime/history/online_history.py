from .history import History


class OnlineHistoryBuffer(History):
    """A history buffer for 'online' multi-step training updates

    Might still end up 'slightly' off-policy depending on the actor
    configuration used.

    Samples are removed once they are retrieved. Typical usage is: Feed samples
    until reaching the requested nstep_train trajectory (per ENV) and then get
    the multi-step samples for training and repeat.

    When doing local/synced acting this will be deterministic, each env will
    get exactly nstep_train samples and then train on them and discard them.
    When doing async/distributed acting certain envs can fill up faster than
    others, and in particular we can sometimes get a training batch with 2
    or more consecutive nstep_train sequences from the same ENV (Only if there
    is no other choice)
    """
    def __init__(self, max_delayed_steps=5000, fixed_target=True, **kwargs):
        """Initializes an online history buffer

        Args:
            max_delayed_steps: The maximal amount of steps/samples to
                accumulate (per ENV), before discarding old ones. Discarded
                steps are 'lost' and never trained, usually this means acting
                is faster than training which is problematic. The amount of
                discarded steps, if any, are returned in the call to update().
                This is only relevant when using asynchronous or remote acting.
            fixed_target: When creating the nstep_train sequence for training,
                whether we fix/force the target to not go beyond the sequence.
                Typically in online training all transitions in the sequence
                will have the same target state (the 'next-state' of the last
                transition in the sequence). Setting this value to True ensures
                this is the case, also if the ENV has already received
                more samples. This is mainly relevant for non-local training
                where the history buffer fills up asynchronously.
                This is mainly to ensure target nsteps within training
                sequences are fixed/detetministic regardless of whether
                sync/async acting is done.
        """
        super().__init__(**kwargs)

        self.last_env = None
        self.max_delayed_steps = max_delayed_steps
        self.fixed_target = fixed_target

    def _check_sample_availability(self, mbatch_size):
        """Checks we have a certain amount of unique multi-step samples.

        An ENV can have more than 1 multi-sample for example if
        nstep_train is 16 and an ENV has 32+ samples, it has 2 multi-step
        samples available for training
        """
        valid_count = 0
        for key in self.buffer:
            valid_count += int(len(self.buffer[key]) / self.nstep_train)
        return valid_count >= mbatch_size

    def update(self, samples):
        ret = super().update(samples)
        # After adding new samples, discard samples from any ENV which has too
        # many, this means acting is too fast for training which should be
        # avoided
        discarded = 0
        for env_id in self.buffer:
            if len(self.buffer[env_id]) > self.max_delayed_steps:
                remove = len(self.buffer[env_id]) - self.max_delayed_steps
                self._remove_samples(env_id, remove)
                discarded += remove
        ret['discarded_steps'] = discarded
        return ret

    def needed_feed_count(self, mbatch_size, num_envs):
        # For online history request new samples only if we don't have enough
        # to train on
        can_train = self._check_sample_availability(mbatch_size)
        return None if can_train else num_envs

    def get_train_data(self, mbatch_size, train_progress=None):
        """Returns a sample batch for training.

        Each 'trajectory' contains 'nstep_train' consecutive timesteps and
        there are a total of 'mbatch_size' such sequences. It's possible to
        get more than 1 trajectory for the same environment if there are not
        enough ones from unique environments.
        If there are not enough samples to satisfy the requirement, returns
        None. Otherwise returns the training data.
        """
        assert(self.prefix_steps == 0), \
            "Online history does not support prefix/burnin steps"
        steps_per_sequence = self.nstep_train

        if not self._check_sample_availability(mbatch_size):
            return None

        all_env_ids = sorted(self.buffer.keys())
        # Evenly get from all ENVs, starting from the last one plus 1
        index = 0 \
            if not self.last_env \
            else (all_env_ids.index(self.last_env)+1) % len(all_env_ids)
        sample_ranges = []

        # Note that this loop is guaranteed to complete since
        # _check_sample_availability() succeeded
        while len(sample_ranges) < mbatch_size:
            env_id = all_env_ids[index]
            if len(self.buffer[env_id]) >= steps_per_sequence:
                sample_ranges.append(
                    self._make_sample_range(
                        env_id, 0, steps_per_sequence, self.fixed_target))
                # In online history each nstep_train sequence is removed
                # as soon as it is trained
                self._remove_samples(env_id, steps_per_sequence)
                self.last_env = env_id
            index = (index + 1) % len(all_env_ids)

        # Create the final training batch from all the chosen ranges
        return self._make_train_batch(sample_ranges)
