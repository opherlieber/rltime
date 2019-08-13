import threading
import queue


class ParallelReceiver(threading.Thread):
    """Helper class to perform parallel 'receiving' of objects from a sender

    For example from a Multiprocessing queue or TCP connection.
    This class is NOT thread-safe and should be called/used by a single entity
    """
    def __init__(self, recv_func, max_size=None):
        """Initialize the receiver

        Args:
            recv_func: The function for receiving the object from the sender,
                with 1 optional parameter: timeout in seconds.
                The recv_func can either return None or raise queue.Empty if
                timed out
            max_size: Tf specified, is the max amount of entries to hold (In
                addition to any entries the sender might hold, for example if
                it's a multiprocessing queue)
        """
        super().__init__()
        self.recv_func = recv_func
        self.exit_event = threading.Event()
        self.add_item_event = threading.Event()
        self.remove_item_event = threading.Event()
        self.objects = []
        self.max_size = max_size
        self.start()

    def run(self):
        while not self.exit_event.is_set():
            try:
                # If full, wait for remove event
                if self.max_size and len(self.objects) >= self.max_size:
                    self.remove_item_event.wait(0.1)
                    self.remove_item_event.clear()
                    continue

                # Timout is to allow user to close us gracefully when the
                # sender is not sending anymore
                obj = self.recv_func(0.1)
                if obj is not None:
                    self.objects.append(obj)
                    self.add_item_event.set()
            except queue.Empty:
                pass

    def available(self):
        """Returns how many objects are available"""
        return len(self.objects)

    def get(self):
        """Gets the next object, waiting if not available"""
        while len(self.objects) == 0:
            self.add_item_event.wait()
            self.add_item_event.clear()

        res = self.objects.pop()
        self.remove_item_event.set()
        return res

    def get_nowait(self):
        """Gets an item without blocking

        Raises queue.Empty if not available
        """
        if self.available() > 0:
            return self.get()
        else:
            raise queue.Empty

    def close(self):
        self.exit_event.set()
        self.join()
