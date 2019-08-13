from .acting_interface import ActingInterface
import multiprocessing as mp
import cloudpickle
import logging
import queue
from rltime.general.backend import SharedSampleList, SharedSampleBatch
from rltime.general.preceiver import ParallelReceiver


class ActorPool(ActingInterface):
    """Base class for actor pools

    An actor pool is multiple Actor() instances running asynchronously and in
    prallel, for example each one in a separate sub-process and/or in a remote
    machine, and samples are retrieved from all of them continuously.

    This implements the same interface as Actor, so it is transparent to the
    training code if it's working with a single actor or a pool of actors.

    Optionally the handling/receiving can be processed on separate 'receiver'
    sub-processes which handle pre-processing (such as decompression, and
    moving state data to shared memory) before sending it back to the training
    process. This can help minimize the training process oveheads when doing
    high throughput training.

    As a future improvement the receivers should also feed samples directly
    to the history buffer to minimize the training process overheads even more
    """
    def __init__(self, instances, actor_cls, actor_args, num_receivers=1,
                 min_receiver_batch=1000, max_backlog=10):
        """Initialize the actor pool

        Args:
            instances: Total amount of actor instances to run in the pool.
                Each one runs 1 or more envs depending on the actor_args
            actor_cls: The actor class to use (Usually 'Actor')
            actor_args: Args for the Actor() object running in each
                instance.
            num_receivers: If >0 then sample receiving is done asynchronously
                in sub-processes, using this many receiver processes. Each
                receiver manages <instances/num_receivers> actors. The
                receivers move samples to shared memory before sending them
                back, allowing near-zero overhead transferring of the samples
                back to the training process (And then to the history buffer if
                it's also in a sub-process). This is recommended for
                high-throughput training/acting (Around 1 receiver per 10K
                samples-per-second is usually needed for high throughput atari
                training)
            min_receiver_batch: The minimum amount of samples to send back to
                the main process at once for a receiver process (Only relevant
                if num_receivers>0)
            max_backlog: The max amount of receiver batches to have pending
                before pausing (Each such batch is at least min_receiver_batch
                in size), total across all receivers together (Only relevant
                if num_receivers>0)
        """
        logging.getLogger().info(
            f"Initialiazing actor pool with {instances} instances")
        # The total amount of ENVs we have across all actor instances
        self._total_envs = actor_args['num_envs'] * instances
        actor_args = {**actor_args, "total_env_ids": self._total_envs}

        if not num_receivers:
            # No receivers, create all the actor instances directly from here
            spaces = self._create_actors(0, instances, actor_cls, actor_args)
            self._receivers = None
        else:
            # Create sub-process receivers
            assert(instances % num_receivers == 0)
            # Queue per receiver, to send the receiver the policy/state
            self._state_queues = []
            # The receiver processes
            self._receivers = []
            self._min_receiver_batch = min_receiver_batch
            instances_per_receiver = instances // num_receivers

            ctx = mp.get_context('fork')
            # Shared queue for all receivers to send the sample batches back
            self._sample_queue = ctx.Queue(max_backlog)
            # Signal to receivers to close
            self._close_event = ctx.Event()
            for i in range(num_receivers):
                state_queue = ctx.Queue()
                p = ctx.Process(
                    target=self._receiver_main_loop,
                    args=(
                        i, instances_per_receiver, actor_cls,
                        cloudpickle.dumps(actor_args), state_queue)
                    )
                p.start()
                self._receivers.append(p)
                self._state_queues.append(state_queue)

            # Retrieve the spaces from the receivers
            for i in range(num_receivers):
                spaces = self._sample_queue.get()
            # Recv from the sample_queue continuously on a separate thread
            self._sample_receiver = ParallelReceiver(
                lambda timeout: self._sample_queue.get(timeout=timeout),
                max_size=max_backlog)

        super().__init__(*spaces)

    def _create_actors(self, my_index, instances, actor_cls, actor_args):
        """Implemented by the sub-class to create the actors"""
        raise NotImplementedError

    def _start_get_samples(self, actor_policy_creator):
        """Implemented by the sub-class to kick off acting with the given
        policy creator"""
        raise NotImplementedError

    def _get_ready_samples(self, amount, timeout):
        """Implemented by the sub-class to return ready samples"""
        raise NotImplementedError

    def get_env_count(self):
        return self._total_envs

    def _close_actors(self):
        """Implemented by the sub-class to close the actors gracefully"""
        raise NotImplementedError

    def _receiver_main_loop(self, my_index, instances, actor_cls, actor_args,
                            state_queue):
        """Entry point for each receiver sub-process (If using receivers)"""
        # Create the actors for this receiver, and send back the spaces
        actor_args = cloudpickle.loads(actor_args)
        spaces = self._create_actors(
            my_index, instances, actor_cls, actor_args)
        self._sample_queue.put(spaces)

        # First thing in the state queue will always be the policy creator
        policy_creator = cloudpickle.loads(state_queue.get())
        # Kick off acting
        self._start_get_samples(policy_creator)

        samples = []
        # Continuously get ready samples until close is requested
        while not self._close_event.is_set():
            get_count = max(self._min_receiver_batch - len(samples), 1)
            samples.extend(self._get_ready_samples(get_count, 0.01))
            if len(samples) > self._min_receiver_batch:
                # Wrap the samples in a SharedSampleBatch. In particular this
                # should move them to shared memory allowing us to pass this
                # batch between processes quickly
                batch = SharedSampleBatch(samples)

                # Send the batch of samples back to the main process
                self._sample_queue.put(batch)
                samples = []

            # Check if there is an updated policy state to load
            try:
                state_obj = state_queue.get_nowait()
                # Send the state update to the actors
                self._update_state_actual(
                    state_obj['progress'], state_obj['policy_state'])
            except queue.Empty:
                pass

        # Close the actors gracefully before exiting the process
        self._close_actors()

    def get_samples(self, min_samples):
        if not self._receivers:
            # No receivers, just get the ready samples directly
            # First get anything that's ready without waiting
            res = self._get_ready_samples(0, timeout=0)
            # Now wait for at least the min requested
            while len(res) < min_samples:
                res.extend(self._get_ready_samples(
                    min_samples - len(res), timeout=0.01))

        else:
            # Check how many are alreay available
            available = self._sample_receiver.available()
            res = SharedSampleList()
            # Get all available, and at least the requested amount
            while len(res) < min_samples or available > 0:
                batch = self._sample_receiver.get()
                res.append(batch)
                available -= 1
        return res

    def _update_state_actual(self, progress, policy_state):
        """Implemented by the sub-class to perform the actor state update"""
        raise NotImplementedError

    def update_state(self, progress, policy_state=None):
        if policy_state is None:
            policy_state = self._source_actor_policy.get_state()
        if not self._receivers:
            # No receivers, just update directly
            self._update_state_actual(progress, policy_state)
        else:
            # Send the update request to each receiver sub-provess
            obj = {"progress": progress, "policy_state": policy_state}
            for state_queue in self._state_queues:
                state_queue.put(obj)

    def set_actor_policy(self, actor_policy):
        # Save the source policy and make the policy creator for the actors
        self._source_actor_policy = actor_policy
        policy_creator = actor_policy.get_creator()
        if not self._receivers:
            # No receivers, just kick-off acting directly
            self._start_get_samples(policy_creator)
        else:
            # Send the policy creator to each receiver so it can start acting
            obj = cloudpickle.dumps(policy_creator)
            for state_queue in self._state_queues:
                state_queue.put(obj)

    def close(self):
        if not self._receivers:
            self._close_actors()
        else:
            # Signal and wait for all receiver sub-processes to exit
            self._close_event.set()
            for receivers in self._receivers:
                receivers.join()
            self._sample_receiver.close()
