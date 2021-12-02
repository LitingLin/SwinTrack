import threading

# Notices: 1. The correctness is relied on GIL
#          2. Iterator is not thread-safe, so don't access by different threads in the same time
class _SimplePrefetcherIterator:
    def __init__(self, iterable, low_limit: int, high_limit: int):
        super(_SimplePrefetcherIterator, self).__init__()
        self.iterable = iterable
        self.queue = []
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.low_limit_condition = threading.Condition(threading.Lock())
        self.produced_condition = threading.Condition(threading.Lock())
        self.end_flag = False
        self.thread_exit_flag = False
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        self.exp = None

    def __del__(self):
        if self.thread.is_alive():
            self.thread_exit_flag = True
            self.low_limit_condition.notify()
            self.thread.join()

    def __next__(self):
        if len(self.queue) == 0:
            if self.end_flag:
                raise StopIteration
            else:
                with self.produced_condition:
                    while True:
                        # Release GIL
                        if self.produced_condition.wait(0.5):
                            break
                        else:
                            if len(self.queue) != 0:
                                break
                            elif self.end_flag:
                                if self.exp is not None:
                                    raise self.exp
                                raise StopIteration
                            # Release GIL
                            if not self.thread.is_alive():
                                if self.exp is not None:
                                    raise self.exp
                                else:
                                    raise Exception('Worker exited unexpected')

        item = self.queue.pop(0)

        if len(self.queue) <= self.low_limit:
            with self.low_limit_condition:
                self.low_limit_condition.notify()
        return item

    def worker(self):
        try:
            iterator = iter(self.iterable)
            while True:
                if self.thread_exit_flag:
                    return
                if len(self.queue) >= self.high_limit:
                    with self.low_limit_condition:
                        self.low_limit_condition.wait()
                        continue
                try:
                    item = next(iterator)
                    self.queue.append(item)
                    if len(self.queue) == 1:
                        with self.produced_condition:
                            self.produced_condition.notify()
                except (StopIteration, IndexError):
                    break
        except Exception as e:
            self.exp = e
        finally:
            self.end_flag = True


class SimplePrefetcher:
    def __init__(self, iterable, buffer_low_limit: int = 1, buffer_high_limit: int = 3):
        assert buffer_low_limit < buffer_high_limit
        assert buffer_low_limit >= 0
        self.iterable = iterable
        self.low_limit = buffer_low_limit
        self.high_limit = buffer_high_limit

    def __iter__(self):
        return _SimplePrefetcherIterator(self.iterable, self.low_limit, self.high_limit)

    def __len__(self):
        return len(self.iterable)

    def __getattr__(self, item):
        return getattr(self.iterable, item)
