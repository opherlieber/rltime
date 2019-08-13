class CyclicArray():
    """Implements a cyclic array

    Supports only indexing/slicing, 'append to end' and 'pop from front'
    This is used to overcome slowness of builtin python list.pop() which
    slows down big replay buffer updating by ~10x once it becomes full

    This is NOT thread safe
    """

    def __init__(self, start_capacity=10000):
        self._capacity = start_capacity
        self._array = [None] * self._capacity
        self._first = 0
        self._amount = 0

    def _increase_capacity(self):
        new_array = [None] * (self._capacity*2)
        new_array[:self._amount] = self[:]
        del self._array
        self._array = new_array
        self._first = 0
        self._capacity *= 2

    def append(self, item):
        if self._amount == self._capacity:
            self._increase_capacity()
        assert(self._amount < self._capacity)
        self._array[(self._first + self._amount) % self._capacity] = item
        self._amount += 1

    def pop(self, index):
        assert(index == 0)
        assert(self._amount > 0)
        res = self._array[self._first]
        self._amount -= 1
        self._first = (self._first + 1) % self._capacity
        return res

    def _get_offset(self, offset):
        if self._amount == 0:
            return self._first
        if offset < 0:
            offset += self._amount
        if offset < 0:
            offset = 0
        elif offset > self._amount:
            offset = self._amount
        return offset

    def __len__(self):
        return self._amount

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = 0 if key.start is None else self._get_offset(key.start)
            stop = self._amount if key.stop is None else \
                self._get_offset(key.stop)

            assert(start <= stop)
            amount = stop - start
            array_start = (start + self._first) % self._capacity
            tail = self._capacity - array_start
            if tail >= amount:
                # Typical case for small slices
                return self._array[array_start:array_start+amount]
            else:
                return self._array[array_start:] + self._array[:amount - tail]
        else:
            offset = self._get_offset(int(key))
            assert(offset >= 0 and offset < self._amount)
            return self._array[(self._first + offset) % self._capacity]
