import multiprocessing as mp
import queue
import cloudpickle
import numpy as np
from collections import deque


class ParallelHistoryWrapper():
    """Wraps a history buffer and executes it's requests in parallel on a
    separate process.

    This is particularly helpfull for hiding prioritized replay overheads and
    state data stacking time

    This is not recommended when using 'Online' history mode as it can cause
    off-policy training data (Unless the training is ok with that)

    This can also allow loading training data to the GPU while training the
    previous batch on the main process, using shared CUDA tensors on supported
    platforms
    """
    def __init__(self, history_cls, history_args):
        super().__init__()

        self._last_needed_feed_count = 0
        self.results = {}
        self.pending_counts = {}

        # Make sure to use 'spawn' and not 'fork' to allow shared CUDA tensors
        # on linux
        ctx = mp.get_context('spawn')
        self.close_event = ctx.Event()
        self.qevent = ctx.Event()
        # Queue for requests, such as getting training data
        self.request_queue = ctx.Queue(10)
        # Queue for updates like new acting samples and priority updates
        self.update_queue = ctx.Queue(10)
        # Queue for sending back request results
        self.result_queue = ctx.Queue()

        self._process = ctx.Process(
            target=self.run,
            args=(history_cls, cloudpickle.dumps(history_args)))

        self._process.start()

    def run(self, history_cls, history_args):
        try:
            # Create the history buffer
            history_buffer = history_cls(**cloudpickle.loads(history_args))

            while not self.close_event.is_set():
                # Prioritize updates over requests
                if self.update_queue.qsize() > 0:
                    req = self.update_queue.get()
                elif self.request_queue.qsize() > 0:
                    req = self.request_queue.get()
                else:
                    self.qevent.wait()
                    self.qevent.clear()
                    continue
                func = getattr(history_buffer, req['op'])
                result = func(*req['args'], **req['kwargs'])

                if req['need_result']:
                    self.result_queue.put(
                        {"op": req['op'], "result": result})

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print("ParallelHistory Failed With Exception:", exc)

        self.result_queue.close()
        self.result_queue.join_thread()

    def _put_op(self, op, args, kwargs, need_result):
        q = self.update_queue if not need_result else self.request_queue
        q.put({
            "op": op, "args": args,
            "kwargs": kwargs, "need_result": need_result})
        self.qevent.set()
        if need_result:
            self.pending_counts[op] = self._pending(op)+1

    def update(self, *args, **kwargs):
        self._put_op("update", args, kwargs, need_result=False)

    def update_losses(self, *args, **kwargs):
        self._put_op("update_losses", args, kwargs, need_result=False)

    def _pending(self, op):
        return self.pending_counts.get(op, 0)

    def _ready(self, op):
        return len(self.results.get(op, []))

    def _get_ready_results(self, block=True):
        block_actual = block
        try:
            # We get all available items from the queue, blocking only on
            # the first one (if requested)
            while True:
                result = self.result_queue.get(block=block_actual, timeout=60)
                op = result['op']
                if op not in self.results:
                    self.results[op] = deque()

                self.results[op].append(result['result'])
                assert(self.pending_counts[op] > 0)
                self.pending_counts[op] -= 1
                block_actual = False
        except queue.Empty:
            if block_actual:
                raise RuntimeError(
                    "Parallel history operation seems stuck")

    def get_train_data(self, *args, **kwargs):
        op = "get_train_data"
        # Always make sure we have at least 2 future requests running (after
        # this one)
        while self._pending(op) + self._ready(op) < 3:
            self._put_op("get_train_data", args, kwargs, need_result=True)

        # Wait for at least 1 ready result
        while not self._ready(op):
            self._get_ready_results()

        res = self.results[op].popleft()
        return res

    def needed_feed_count(self, *args, **kwargs):
        op = "needed_feed_count"
        self._put_op(op, args, kwargs, need_result=True)
        self._get_ready_results(block=False)
        while self.results.get(op, None):
            self._last_needed_feed_count = \
                self.results["needed_feed_count"].popleft()
        return self._last_needed_feed_count

    def close(self):
        self.close_event.set()
        # Wait for any pending results to arrive
        while np.any([amount > 0 for amount in self.pending_counts.values()]):
            self._get_ready_results()
        self._process.join()
        self.result_queue.close()
        self.result_queue.join_thread()
