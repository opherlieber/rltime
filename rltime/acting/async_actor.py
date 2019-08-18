import threading
from .actor import Actor


class AsyncActor(Actor):
    """An actor extension that runs the acting itself on a separate thread

    This is useful when working with real-time environments which require
    no delay between acting steps, and mininal delay when loading a new policy
    state.
    """

    def __init__(self, max_pending=50, **kwargs):
        """Inits the async actor

        Args:
            max_pending: Max amount of samples to accumulate until stopping.
                For slow real-time ENVs this should be large enough to not
                stall the acting between calls to get_samples(). For very fast
                ENVs this avoids an exploding amount of samples
            kwargs: Args for the actual Actor class
        """
        super().__init__(**kwargs)
        self._thread = None
        self._max_pending = 20

    def set_actor_policy(self, actor_policy):
        super().set_actor_policy(actor_policy)
        # Can now start the acting thread
        self._queue = []
        self._queue_fill_event = threading.Event()
        self._queue_empty_event = threading.Event()
        self._close_event = threading.Event()
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def run(self):
        # TODO Fix this dependency. The policy itself sets the thread limit
        # to 1, but this configuration seems to be per-thread in pytorch
        # so need to set it here too :(
        import torch
        torch.set_num_threads(1)

        while not self._close_event.is_set():
            # If queue is full, wait for it not to be
            while len(self._queue) >= self._max_pending:
                self._queue_empty_event.wait()
                self._queue_empty_event.clear()
            # Get the next sample(s)
            samples = super().get_samples(1)
            self._queue.extend(samples)
            self._queue_fill_event.set()

    def get_samples(self, min_samples):
        assert(min_samples <= self._max_pending)
        # Wait for queue to have requested amount
        while len(self._queue) < min_samples:
            self._queue_fill_event.wait()
            self._queue_fill_event.clear()

        # Return ALL available samples
        amount = len(self._queue)
        res = self._queue[:amount]
        del self._queue[:amount]
        self._queue_empty_event.set()
        return res

    def close(self):
        # Signal and wait for thread to close
        self._close_event.set()
        self._thread.join()
        # Close the actual actor
        super().close()
