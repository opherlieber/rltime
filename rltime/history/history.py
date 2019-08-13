import numpy as np
from rltime.general.utils import deep_stack, deep_apply
from .data_structures.cyclic_array import CyclicArray
from rltime.general.object_wrapper import ObjectWrapper
from rltime.general.backend import SharedSampleList


class History():
    """Base class for a multi-step/multi-env history buffer.

    All actor samples go to a history buffer first, and are then retrieved for
    training on nstep trajectories per environment

    Can be 'online' samples trained on immediately and discarded, or a replay
    buffer, depending on the subclass used.
    """
    def __init__(self, nstep_target, nstep_train, prefix_steps=0,
                 discount_function=None, state_store=None):
        """ Initialize a history buffer

        Args:
            nstep_target: The nstep to use for the target-states and discounted
                rewards. This is not always guaranteed, depending on the
                subclass used
            nstep_train: The nstep to train. When getting training data for
                a given 'mbatch' size, the history buffer will return mbatch
                trajectories each with consecutive timesteps of this size,
                i.e. a total of exactly mbatch*nstep_train transitions for
                training
            prefix_steps: Additional 'prefix' steps to return for each sequence
                for example for state burnin
            discount_function: Function to use for discounting nstep_target
                returns. Required if nstep_target>1.
            state_store: Optional class to store and load state data. This can
                be used to override the default (numpy) storage of state data,
                and to supply a custom method for stacking state data
                (For example directly to the GPU)
        """

        self.nstep_target = nstep_target
        self.nstep_train = nstep_train
        self.prefix_steps = prefix_steps
        assert(nstep_target == 1 or discount_function is not None), \
            "History buffer must get a 'discount_function' for nstep_target>1"
        self.discount_function = discount_function
        self.state_store = state_store
        if state_store:
            state_store.init_for_use()

        # The sample 'buffer'. Each key is for a unique 'ENV ID', containing
        # the timestep-ordered samples of that ENV
        self.buffer = {}
        # Contains the last sample received for each ENV (In order to take its
        # 'next state' as the 'state' of the next sample)
        self.last_sample_buffer = {}
        # Preparation for stacking support in the buffer, to save history
        # buffer RAM and actor transfer time, when using frame stacking greater
        # than 1. This is currently not supported
        self.stack = 1

    def _sample_added(self, sample):
        """Optional additional things to do in the sub-class for every new
        sample added"""
        pass

    def _sample_removed(self, sample):
        """Optional additional things to do in the sub-class for every sample
        removed"""
        pass

    def _update_nstep(self, env_id, index, nstep_target):
        """Updates the nstep/discounted-return for the given sample index

        Updating is based on available subsequent samples and the configured
        nstep_target
        """
        env_buffer = self.buffer[env_id]
        assert(index < len(env_buffer))
        sample = env_buffer[index]

        # Update up to the requested nstep_target or the end of the buffer,
        # whatever is available
        end = min(index + nstep_target, len(env_buffer))
        for target_index in range(index + sample['nstep'], end):
            target_sample = env_buffer[target_index]
            # Discounting stops if we saw a 'done' sample
            if sample['target_mask']:
                sample['return'] += self.discount_function(
                    sample['nstep'], target_sample['reward'],
                    target_sample['policy_output'])

            # Note that even if saw a 'done' already, we still update the nstep
            # and the target_state (But not the return), this is to ensure
            # consecutive timesteps for the target states (For recurrent
            # network bootstrapping). Since target_mask is 0 it shouldn't
            # affect the eventual target value calculations (Except maybe for
            # dist_dqn???)
            sample['nstep'] += 1

            # The previous targets' next-state is now the new n-step target
            # state for the base sample
            sample['target_state'] = target_sample['next_state']

            if target_sample['done']:
                # This ensures the 'target state' is not used for boostrapping
                # since it is not relevant when the episode terminated within
                # the nstep
                sample['target_mask'] = 0.

    def _remove_samples(self, env_id, amount):
        """Called by the sub-class to remove samples from a specific ENV

        Currently assumes all implementations remove in-order within the ENV,
        starting from the oldest
        """
        env_buffer = self.buffer[env_id]
        assert(len(env_buffer) >= amount)

        for _ in range(amount):
            self._sample_removed(env_buffer[0])
            env_buffer.pop(0)

    def update(self, new_samples):
        """ Updates the buffer with new samples

        Returns: optional stats/info (Override this for additional
            functionality and stats)
        """
        if isinstance(new_samples, SharedSampleList):
            new_samples.unpack()

        for sample in new_samples:
            # If the state data is wrapped, unwrap it
            if isinstance(sample['next_state'], ObjectWrapper):
                sample['next_state'] = sample['next_state'].get_object()

            if self.state_store is not None:
                # If we got a state store we use it to store the state which
                # should be more efficient later when retrieving/stacking train
                # data
                sample['next_state'] = self.state_store.store(
                    sample['next_state'])

            # Initialize the discounted return and nstep of the sample (Will be
            # adjusted if/when needed)
            sample['return'] = float(sample['reward'])
            sample['target_mask'] = 1 - sample['done']

            # The nstep target state by default is the next state, but can be
            # moved forward according to the configured nstep target (see
            # _update_nstep)
            # Note we aren't actually duplicating any memory it is referencing
            # the same state object
            sample['target_state'] = sample['next_state']
            sample['nstep'] = 1
            # Each sample is chained together with previous samples from
            # same ENV
            env_id = sample['env_id']
            if env_id not in self.last_sample_buffer:
                # Corner case for first ever sample for an ENV, we don't have
                # the next_state of the previous step (Should be negligible/no
                # impact)
                sample['state'] = sample['next_state']
            else:
                # New samples state is next_state of previous sample (This
                # avoids duplicate memory for next_state/state)
                sample['state'] = self.last_sample_buffer[env_id]['next_state']

            if env_id not in self.buffer:
                self.buffer[env_id] = CyclicArray()  # Was ' = []', now faster
            self.buffer[env_id].append(sample)
            # Remember the last sample per ENV (it's 'next state' is the state
            # of the next sample which will arrive)
            self.last_sample_buffer[env_id] = sample
            self._sample_added(sample)
        return {}

    def _make_sample_range(self, env_id, index, steps, fixed_target=False):
        """Prepares a timestep range from an ENV ID for train-batching"""
        assert (index >= 0 and (index + steps <= len(self.buffer[env_id])))
        ret = []
        nstep_target = self.nstep_target

        for i in range(index, index+steps):
            # Update nstep discounts/targets (It may already be updated,
            # depending on sub-class, in which case _update_nstep does nothing)
            if fixed_target:
                # In fixed_target mode we never give a target beyond the
                # actual sequence being trained (Common in online training)
                nstep_target = min(nstep_target, index + steps - i)
            self._update_nstep(env_id, i, nstep_target)
            sample = self.buffer[env_id][i]
            ret.append({
                    "target_states": sample["target_state"],
                    "states": sample["state"],
                    "returns": sample['return'],
                    "nsteps": sample['nstep'],
                    "target_masks": sample["target_mask"],
                    "policy_outputs": sample["policy_output"]
                })
        return ret

    def _make_train_batch(self, sample_ranges, extra_data=None):
        """Makes the training data dictionary given the chosen sample ranges

        See _make_sample_range. Each sample range defines the env_id and
        nstep_train range of samples within that ENV ID to combine for training

        The result is a dictionary containing the relevant stacked
        training fields:
        - states
        - target_states (n-step target states, or just the 'next state' if
          nstep_target=1)
        - returns (The nstep discounted return sum for this state, not
          including the final bootstrap target-state)
        - nsteps (The actual target nsteps, might be lower than requested nstep
          target in certain cases)
        - target_masks (Whether the target states are relevant or the episode
          terminated before them)
        - policy_outputs (The policy output at acting time for each state)
        All fields are shaped (nstep_train, mbatch)
        """
        mbatch = len(sample_ranges)
        nstep = len(sample_ranges[0])

        # Reorder/transpose the samples so that grouping will be by ENVs and
        # not timesteps (So that timestep will be axis=0 after stacking)
        samples = []
        for ranges in zip(*sample_ranges):
            samples.extend(ranges)

        # Stack all the data on axis=0
        train_data = {}
        # First stack all non-heavy (non-state) data
        for key in samples[0].keys():
            if key in ['target_states', 'states']:
                # These are handled after
                continue
            else:
                vals = [sample[key] for sample in samples]
                train_data[key] = deep_stack(vals)

        # Generate the stacked states and target states
        seqlen = self.nstep_train + self.prefix_steps
        if self.nstep_target < seqlen and \
                np.all(train_data['nsteps'] == self.nstep_target):

            # Stack optimization for the case of overalapping states and target
            # states. In particular for nstep_target<<nstep_train this improves
            # history buffer overheads by ~30-40% as most of the time is spent
            # stacking add loading state data to the GPU
            # (TODO: Add also optimization for case where all states have the
            # same single target state, i.e. online history)
            all_states = [sample['states'] for sample in samples]
            assert(len(all_states) == mbatch*seqlen)
            target_offset = mbatch*self.nstep_target
            all_states += [sample['target_states']
                           for sample in samples[-target_offset:]]
            all_stacked = self.state_store.stack(all_states)
            # Cut out states and target states from the overlapping stacked
            # data
            train_data['states'] = deep_apply(
                all_stacked, lambda x: x[:mbatch*seqlen])
            train_data['target_states'] = deep_apply(
                all_stacked, lambda x: x[target_offset:])
        else:
            # Defualt case, stack the target states and states separately
            for key in ['target_states', 'states']:
                train_data[key] = self.state_store.stack(
                    [sample[key] for sample in samples])

        # Now the train_data contains each field in one long sequence of size
        # (nstep_train*mbatch), reshape it to the expected output shape
        train_data = deep_apply(
            train_data, lambda x: x.reshape((nstep, mbatch)+x.shape[1:]))

        # Add extra train data if reuested (Stacking along axis=1 so that the
        # dimensions are (nstep_train,mbatch) like rest of data)
        if extra_data is not None:
            assert(len(extra_data) == mbatch)
            train_data['extra_data'] = deep_stack(
                extra_data, args={"axis": 1})
        else:
            train_data['extra_data'] = {}

        return train_data

    def needed_feed_count(self, mbatch_size, num_envs):
        """Returns how many new samples, if any, should be feeded now to the
        history buffer, in order to train the given mbatch size

        None means none, 0 means 'any available', otherwise the actual amount
        """
        raise NotImplementedError

    def get_train_data(self, mbatch_size, train_progress=None):
        """Gets multi-step samples for training from the history buffer.

        This is buffer-type dependent. The total amount of samples/transitions
        returned is <mbatch_size * (self.prefix_steps+self.nstep_train)>
        The sub-class should generate the result by calling
        _make_sample_range() <mbatch_size> times and finally
        _make_train_batch() on the sample ranges

        Args:
            mbatch_size: The amount of independent nstep_train sequences to
                get total
            train_progress: The current progress of the training in range
                [0,1], which may be needed for certain implementations

        Returns:
            A dictionary containing the train data, or None if no training
            needed/possible (In which case feed more samples).
            The dictionary contains:
            - "states": The policy input states
            - "target_states": The nstep-target states
            - "target_masks": 0/1 mask specifying if the target_state is
              relevant or should be masked (Due to 'done' during the nstep)
            - "returns": The discounted nstep returns for the state (Not
              including final target bootstrap)
            - "nsteps" : The actual target nstep of each sample (Might be less
              than the configured nstep_target in some cases)
            - "policy_outputs": The original policy outputs at acting time
              (Including the actions and possibly more policy-dependent info
              like logprob, state-values etc..)

            All values are shaped:
            ((prefix_steps+nstep_train), mbatch_size, ...)
        """
        raise NotImplementedError

    def update_losses(self, indices, losses):
        """Optional update of losses from training, for example for prioritized
        replay"""
        pass
