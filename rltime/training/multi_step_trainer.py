
import numpy as np
import math

from .policy_trainer import PolicyTrainer
from rltime.general.type_registry import get_registered_type
from rltime.history.parallel_history import ParallelHistoryWrapper
from rltime.general.utils import deep_apply


class MultiStepTrainer(PolicyTrainer):
    """Base class for multi-step history-based training.

    Handles:
    - Mult-step/Multi-env history buffer management using the requested
      history mode (Online, replay, prioritized-replay...)
    - Balancing acting/training as configured
    - Recurrent state burn-in, if configured
    - Getting the training data, triggering the target-value calculation
    - Training, including optional mini-batch/epoch support and shuffling
    - LR Scheduling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_target_values(self, returns, target_states, target_masks, nsteps,
                           timesteps):
        """Calculates the bootstrapped nstep target values for the given target
        states"""
        raise NotImplementedError

    def train_init(self, lr):
        """Initializes training with the specified learning rate, called at
        start of training (After policies are created)"""
        raise NotImplementedError

    def set_lr(self, lr):
        """Sets/changes the learning rate during training"""
        raise NotImplementedError

    def train_batch(self, states, targets, policy_outputs, extra_data,
                    timesteps):
        """Trains the given states with the calculated target values"""
        raise NotImplementedError

    def get_train_indexes(self, batch_size, mini_batch_size, nstep):
        """ Generates shuffled training indices for the biven batch and
        mini-batch size, only relevant when using mini-batches"""
        inds = np.arange(batch_size)
        if not self.policy.is_recurrent():
            # If it's non-recurrent we just shuffle everything with everything
            np.random.shuffle(inds)
        else:
            # For recurrent we need to leave the timestep grouping,
            # and can only randomly choose different multi-step sequences
            assert((mini_batch_size % nstep) == 0)
            envs_per_mini_batch = mini_batch_size // nstep
            inds = inds.reshape((nstep, -1))
            env_inds = np.arange(inds.shape[-1])
            np.random.shuffle(env_inds)

            inds = np.concatenate([
                inds[..., env_inds[i:i+envs_per_mini_batch]].ravel()
                for i in np.arange(0, inds.shape[-1], envs_per_mini_batch)
            ])

        return inds

    def _get_discount_function(self, gamma):
        """Returns the nstep discounting function used by the history buffer"""
        def discount(nstep, reward, policy_output):
            return (gamma ** nstep) * reward
        return discount

    def _sample_and_update_history(self, min_samples):
        """Gets samples fron the actors (At least min_samples, or more if
        available) and updates the history buffer"""
        new_samples = self.sample_actors(min_samples)

        if new_samples:
            # Store the new samples in the history buffer
            self._start_timer("history_update")
            stats = self.history_buffer.update(new_samples)
            self._end_timer()

            # Log history stats
            # self.value_log.log_dict(stats, group="history")

    def _burn_in(self, train_data, burn_in_timesteps, do_target_states):
        # SUPER HACKY VERSION JUST TO SEE IF IT HELPS, IF YES NEED TO COME UP
        # WITH AN ELEGANT SOLUTION WITH MODEL SUPPORT
        self._start_timer("burn_in")

        # Burn-in both the training states and target states if requested
        # using the approriate current online and target policies
        burn_in_targets = [(self.policy, train_data['states'])]
        if do_target_states:
            burn_in_targets += [
                (self.target_policy, train_data['target_states'])]
        for policy, states in burn_in_targets:
            # Extract only the burnin timesteps from the states, and do a
            # forward pass on them, so that the current model RNN states
            # will be the 'burned in' initial state for the first timestep of
            # the actual training timesteps
            states_burnin = deep_apply(
                states, lambda x: x[:burn_in_timesteps])
            # Flatten the states for the model
            states_burnin = deep_apply(
                states_burnin,
                lambda x: x.reshape((x.shape[0]*x.shape[1],)+x.shape[2:]))
            policy.actor_predict(
                states_burnin, timesteps=burn_in_timesteps)

            # Replace the layer states with the updated ones
            # TODO: Add support for model to do this and not here
            for i, layer in enumerate(policy.model.layers):
                key = f"layer{i}_state"
                if not states[key]:
                    continue
                new_state = layer.get_state(
                    states[key]['initials'][burn_in_timesteps].cpu().numpy())
                for vkey in new_state.keys():
                    states[key][vkey][burn_in_timesteps] = \
                        policy.make_tensor(new_state[vkey])
        # Now we remove all the burnin timesteps from all the train data,
        # leaving only the actual timesteps we want to train
        new_train_data = deep_apply(
            train_data, lambda x: x[burn_in_timesteps:])
        self._end_timer()
        return new_train_data

    def _init_history_buffer(self, mode, async_history, nstep_target,
                             nstep_train, prefix_steps):
        """Initialize the history buffer for the training session"""
        history_cls = get_registered_type("history", mode.get("type"))
        history_args = dict(
            **mode.get("args", {}), nstep_target=nstep_target,
            nstep_train=nstep_train,
            prefix_steps=prefix_steps,
            discount_function=self._get_discount_function(self.gamma),
            state_store=self.policy.get_state_store(async_history))

        if async_history:
            # If requested, run the history buffer on a separate process
            self.history_buffer = ParallelHistoryWrapper(
                history_cls, history_args)
        else:
            # Local history buffer
            self.history_buffer = history_cls(**history_args)

    def _train(self, gamma, nstep_train, lr, history_mode, mbatch_size=None,
               nstep_target=None, lr_anneal=False, epochs=1, minibatches=1,
               warmup_steps=0, actor_update_frequency_steps=1000,
               burn_in_timesteps=0, rnn_steps_train=None, rnn_bootstrap=False,
               async_history=False):
        """ Entry point for multi-step training

        Args:
            gamma: The gamma value to use for discounting nstep-target returns
            nstep_train: The multi-step to train (i.e. consecutive timesteps
                from an ENV to retrieve for training from the history buffer)
            lr: The base learning rate to use for training
            history_mode: The history buffer configuration, in the form
                {"type":type,"args":args}. <type> can be for example "online",
                "replay", "prioritized_replay" or a python class.
            mbatch_size: The 'multi-batch-size' to use, i.e. the amount of
                independent nstep_train sequences to train at once. The
                full batch size trained on will be <nstep_train*mbatch_size>.
                If not specified the total ENV count across all actors is used
            nstep_target: The nstep to use for the target-value. By default
                this is set to nstep_train, though it's commonly different
                for replay training (e.g. typically 1/3/5 for DQN variants)
            lr_anneal: Whether to anneal the LR across the training period.
                can be 'True' to anneal to zero or the value to anneal to
            epochs: Amount of 'mini-epochs'/iterations to perform on each
                training batch (Relevant mainly for PPO)
            minibatches: Split the training batch to this many parts and
                train them one after the other (Relevant mainly for PPO)
            warmup_steps: Initial amount of steps to not train on, to allow
                for example replay buffers to fill up a bit first
            actor_update_frequency_steps: How often to update the actors with
                the latest policy state (Only relevant for non-local actors
                which need to continuously get updated policy weights). In any
                case this will never be less than the training batch size
                even if the value requested is lower.
            burn_in_timesteps: Perform a 'burn_in' of the training states using
                this many timesteps, to improve stale RNN states which were
                acted a long time ago (See R2D2 paper). Only relevant for
                recurrent policies. Burn-in also applies to the target states
                if rnn_bootstrap=True
            rnn_steps_train: If using a recurrent policy, this is the amount
                of timesteps to serially pass through the RNN. The default case
                (None) sets this to nstep_train, but in some cases where
                nstep_train is very large it may be desirable to use shorter
                LSTM sequences. nstep_train must be divisible by
                rnn_steps_train. Note this does not change the training batch,
                only how the RNN in the model 'sees' it. For example if
                nstep_train is 128 and rnn_steps_train is 32, we still train
                all 128 steps, just any RNN in the model will treat it as 4x32
                independent sequences
            rnn_bootstrap: Whether to use recurrent sequencing also for the
                bootstrap target value calculation. If False the RNN does not
                do any sequencing, and each timestep is run through the RNN
                with the state it had at acting time as a single timestep.
                This is only supported in history buffers which have a fixed
                nstep for each transition (e.g. replay buffers), and not for
                online history (which has a fixed target state, though for
                online history this option is less relevant since the state
                is not stale)
                Note this is relevant ONLY for target values. Training uses
                RNN sequencing in any case (According to rnn_steps_train)
            async_history: Still experimental, wraps the history buffer to run
                it asynchronously/parallel to the training/acting, to help
                reduce overheads of generating the training batches.
        """
        self.train_init(lr)
        self.gamma = gamma
        base_lr = lr

        # If not specified, set mbatch_size to be the (total) ENV count across
        # all actors
        mbatch_size = mbatch_size or self.actors.get_env_count()
        # If not specified, set nstep_target to nstep_train
        nstep_target = nstep_target or nstep_train

        # Init the history buffer, new acting samples are fed to this buffer,
        # and multi-step train samples are retrieved from it
        self._init_history_buffer(
            history_mode, async_history, nstep_target, nstep_train,
            prefix_steps=burn_in_timesteps)

        # Track the last step-count when the actors were updated with the
        # policy weights
        actors_last_update_steps = 0

        # By default if not specified use the full nstep_train for RNN
        # sequencing
        rnn_steps_train = rnn_steps_train or nstep_train

        assert((not burn_in_timesteps) or self.policy.is_recurrent()), \
            "burn_in_timesteps only makes sense for recurrent policies"

        env_count = self.actors.get_env_count()
        while not self.train_is_done():
            train_progress = self.get_train_progress()
            warming_up = self.steps < warmup_steps

            # Check how many new actor samples, if any, the history buffer
            # wants from the actors
            num_samples = self.history_buffer.needed_feed_count(
                mbatch_size, env_count)
            if num_samples is not None:
                if warming_up:
                    # While warming up make sure we always get new data to
                    # avoid hogging the CPU
                    num_samples = max(num_samples, env_count)
                self._sample_and_update_history(num_samples)

            # Get data to train on (This may or may not be the latest samples,
            # depends if we are using replay for example, but we shouldn't care
            # where they came from)
            # If 'None' is returned, it means the history buffer wants to be
            # fed more samples first (Depends on the specific history mode we
            # are running)
            self._start_timer("get_train_data")
            train_data = self.history_buffer.get_train_data(
                mbatch_size, train_progress=train_progress)
            if train_data is None:
                # This means it still needs more samples
                continue
            self._end_timer()

            # Skip training during warmup
            if warming_up:
                continue

            # The train data is currently shaped (nstep_train, mbatch_size)

            # Perform burnin of the recurrent states, if requested. Burn-in
            # also the target-states only if using RNN bootstrappng
            if burn_in_timesteps:
                train_data = self._burn_in(
                    train_data, burn_in_timesteps,
                    do_target_states=rnn_bootstrap)

            # Target value calculation
            self._start_timer("calc_target_values")
            # Reshape/flatten everything to a 1 dimension batch for training
            train_data = deep_apply(
                train_data,
                lambda x: x.reshape((x.shape[0]*x.shape[1],)+x.shape[2:]))

            if rnn_bootstrap:
                assert(np.all(train_data['nsteps'] == nstep_target)), \
                    "rnn_bootstrap is only supported with history buffers " \
                    "which guarantee the full/fixed target nstep, e.g. " \
                    "replay history"
            # The returns should already be nstep_target discounted, all that's
            # missing for the final 'target value' is the discounted bootstrap
            # target
            train_data['targets'] = self.calc_target_values(
                train_data['returns'], train_data['target_states'],
                train_data['target_masks'], nsteps=train_data['nsteps'],
                timesteps=1 if not rnn_bootstrap else rnn_steps_train)

            self._end_timer()

            # Train
            self._start_timer("train")

            # Now all data is organized in 1 big batch, for example for
            # nstep_train=20 and mbatch_size=32 we will have 640 for the 1st
            # axis of all the train_data fields (The timestep information is
            # still used though in case of recurrent network to re-organize the
            # data to timesteps when needed in the model)

            # Perform mini-batch updates if requested
            batch_size = train_data['returns'].shape[0]
            assert((batch_size % minibatches) == 0)
            mini_batch_size = batch_size // minibatches

            for _ in range(epochs):
                # Shuffle the train indices (Only meaningful in case of
                # minibatches>1)
                inds = self.get_train_indexes(
                    batch_size, mini_batch_size, nstep_train)

                for i in range(0, batch_size, mini_batch_size):
                    if minibatches > 1:
                        tinds = inds[i:i+mini_batch_size]
                        slicer = lambda y: deep_apply(y, lambda x: x[tinds])
                    else:
                        # Avoid re-indexing if it has no meaning (If there is
                        # no mini-batching then shuffling makes no difference
                        # to training)
                        slicer = lambda x: x

                    # Train on the batch, in case of an LSTM in the model the
                    # rnn_steps_train value will be used by the LSTM module to
                    # reshape the batch to timesteps of that length for the
                    # LSTM sequencing
                    self.train_batch(slicer(train_data['states']),
                                     slicer(train_data['targets']),
                                     slicer(train_data['policy_outputs']),
                                     slicer(train_data['extra_data']),
                                     rnn_steps_train)

                    # Log the actual/total batch-size used for training
                    self.value_log.log(
                        "batch_size", mini_batch_size, group="train")

                    self._update_steps_trained(mini_batch_size)

            # If requested, anneal the LR
            if lr_anneal not in [False, None]:
                anneal_to = 0.0 if lr_anneal is True else float(lr_anneal)
                lr = base_lr - train_progress*(base_lr - anneal_to)
                self.set_lr(lr)

            self.value_log.log("lr", lr, group="train")

            # Update the actors with new policy at the requested frequency
            # (This will always be at least the batch size, even if
            # actor_update_frequency_steps is less). Note that in any case,
            # this will only actually do something inside if we have remote or
            # sub-process actors, and should do nothing for vectorized local
            # actors.
            if not actor_update_frequency_steps or (
                    self.steps - actors_last_update_steps >=
                    actor_update_frequency_steps):
                self.update_actors()
                actors_last_update_steps = self.steps

            self._end_timer()

        # Close the parallel history buffer if configured
        if async_history:
            self.history_buffer.close()
