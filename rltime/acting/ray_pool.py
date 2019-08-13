import numpy as np
import pickle
import math
from threading import Lock
from collections import OrderedDict
import logging

# Don't assume/require ray or LZ4 to be installed unless using them
try:
    import ray
except ImportError:
    ray = None
try:
    import lz4.frame as lz4_frame
except ImportError:
    lz4_frame = None

from .actor_pool import ActorPool
from .actor_wrapper import ActorWrapper
from rltime.general.utils import deep_apply
from rltime.general.object_wrapper import ObjectWrapper
from rltime.general.compression import CompressedStateWrapper
import itertools


class RayActor(ActorWrapper):
    """Actor wrapper when using RAY to override/add some methods"""

    def __init__(self, compress_samples, actor_cls, actor_args):
        actor = actor_cls(**actor_args)
        super().__init__(actor)
        self._compress_samples = compress_samples

    def create_actor_policy(self, creator):
        policy = creator()
        super().set_actor_policy(policy)

    def get_samples(self, min_samples):
        samples = super().get_samples(min_samples)
        # If requested, compress the states before sending back
        if self._compress_samples:
            for sample in samples:
                sample['next_state'] = \
                    CompressedStateWrapper(sample['next_state'])

        return samples


class RayPool(ActorPool):
    """Implements an actor pool using the RAY engine

    Note that you need to setup and start your ray node/cluster ('ray start...'
    or 'ray up...') ahead of time (See the ray docs), and pass
    the ray redis address to the initialization here (If it's not the default
    "localhost:6379")
    """

    def __init__(self, min_samples_per_request=1, requests_per_actor=2,
                 cpus_per_worker=1, custom_resources_per_worker={},
                 compress_samples=True, ray_address="localhost:6379",
                 **kwargs):
        """Initialize a ray pool

        Args:
            min_samples_per_request: The min samples to receive for each remote
                actor call. Small values can slow down overall throughput due
                to overheads while large values may result in a larger delay
                between the training and acting policy, and larger batches of
                updates to the training process
            requests_per_actor: Amount of sampling requests to trigger per
                actor at a time, to ensure they are always busy (2 should
                usually be enough). These are executed serially by ray, this
                just minimizes the overheads between subsequent requests
            cpus_per_worker: Amount of ray CPU resources to allocate per ray
                worker
            custom_resources_per_worker: Additional custom ray resources to
                allocate per worker
            compress_samples: Whether to compress samples from the workers.
                This should usually be True if actors are on different
                machines, though it should be fast enough not to hurt in any
                case
        """
        self._requests_per_actor = requests_per_actor
        self._min_samples_per_request = min_samples_per_request
        self._cpus_per_worker = cpus_per_worker
        self._custom_resources_per_worker = custom_resources_per_worker
        self._ray_address = ray_address
        if compress_samples and lz4_frame is None:
            raise RuntimeError(
                "Please install LZ4 for using compression in a ray actor pool")
        self._compress_samples = compress_samples
        super().__init__(**kwargs)

    def _create_actors(self, my_index, instances, actor_cls, actor_args):
        """Creates the RAY workers with the requested configuration"""
        if ray is None:
            raise RuntimeError("Ray actor pool requires RAY to be installed")
        # Init the ray driver
        ray.init(redis_address=self._ray_address)

        # Make sure we have enough resources in the ray cluster
        self._ensure_resources(instances)

        # Create each actor in a ray remote actor
        self.ray_workers = []
        worker_template = ray.remote(
            num_cpus=self._cpus_per_worker,
            resources=self._custom_resources_per_worker)(RayActor)

        # Calculate the unique ENV id range start for this pool
        envs_per_actor = actor_args['num_envs']
        base_env_id = my_index * instances * envs_per_actor
        for i in range(instances):
            # Add the unique env IDs info to the actor args
            actor_args_instance = dict(
                **actor_args, base_env_id=base_env_id + i*envs_per_actor)

            # Create the ray remote actor
            worker = worker_template.remote(
                compress_samples=self._compress_samples,
                actor_cls=actor_cls,
                actor_args=actor_args_instance)
            self.ray_workers.append(worker)

        # Get the spaces and env count from the 1st worker (Should be same for
        # all)
        spaces = ray.get(self.ray_workers[0].get_spaces.remote())

        # TODO: This isn't always updated at this point yet, maybe get spaces
        # from all workers in previous line to ensure they have started
        logging.getLogger().info(
            "Remaining RAY resources after actor creation: "
            f"{ray.available_resources()}")

        return spaces

    def _ensure_resources(self, instances):
        """Checks we have enough ray resources to create the request

        TODO: This doesn't really work with more than 1 receiver as they create
        and check in parallel. In any case ray will not error if we create an
        actor without resources it will just wait and not be used until it can
        be run
        """
        available = ray.available_resources()
        required = {
            "CPU": self._cpus_per_worker,
            **self._custom_resources_per_worker
        }
        required = {key: val * instances for key, val in required.items()}
        if not np.all(
                [available.get(key, 0) >= required[key] for key in required]):
            raise RuntimeError(
                "Not enough RAY resources to start the acting pool. "
                f"Need: {required} Available: {available}")

    def _update_state_actual(self, progress, policy_state):
        assert(policy_state is not None)
        # Send the updated state to all workers, using the same ray object
        policy_state_obj = ray.put(policy_state)
        for worker in self.ray_workers:
            worker.update_state.remote(
                progress=progress, policy_state=policy_state_obj)

    def _trigger_new_request(self, worker):
        """Trigger a new get_samples request on the worker"""
        req_id = worker.get_samples.remote(
            min_samples=self._min_samples_per_request)
        self._pending_requests[req_id] = worker

    def _start_get_samples(self, actor_policy_creator):
        """Initiates sample retrieval and loads the policy to the actors"""

        # Send the policy creation object to the workers so they can
        # create their own policy copies
        policy_obj = ray.put(actor_policy_creator)

        # Use this to track all sampling requests we have pending per worker
        self._pending_requests = OrderedDict()
        for worker in self.ray_workers:
            worker.create_actor_policy.remote(policy_obj)
            # Initiate acting requests per worker
            for _ in range(self._requests_per_actor):
                self._trigger_new_request(worker)

    def _get_ready_samples(self, amount, timeout):
        """Gets (at least) the requested amount of ready samples from the
        workers, or up to the requested timeout
        """
        all_reqs = list(self._pending_requests.keys())
        if not amount:
            # Return all
            num_returns = len(all_reqs)
        else:
            # 'amount' is not a hard-limit, but we try to return around that
            # amount if possible
            num_returns = math.ceil(amount / self._min_samples_per_request)
            num_returns = min(num_returns, len(all_reqs))

        ready_reqs, _ = ray.wait(
            all_reqs, num_returns=num_returns, timeout=timeout)

        res = []
        for req in ready_reqs:
            res.extend(ray.get(req))

            # Initiate a new request for the actor that gave this one
            self._trigger_new_request(self._pending_requests[req])
            # Clear the old request
            del self._pending_requests[req]

        # Decompress all samples if compression is enabled
        # TODO: Add option for keeping it compressed also in replay buffer
        if self._compress_samples:
            for sample in res:
                sample['next_state'] = sample['next_state'].get_object()

        return res

    def _close_actors(self):
        # Trigger close() on all actors and wait for them to finish
        close_reqs = [worker.close.remote() for worker in self.ray_workers]
        ray.get(close_reqs)
