import itertools
import threading
import time

import surreal.utils as U
from surreal.distributed import RedisClient
from surreal.distributed.obs_fetch_queue import ObsFetchQueue
from surreal.distributed.exp_queue import ExpQueue
from surreal.utils.common import StoppableThread


class _EvictThread(StoppableThread):
    def __init__(self,
                 evict_func,
                 evict_args,
                 evict_kwargs,
                 sleep_interval=1.):
        """
        Args:
            evict_func: call evict from Replay object
            evict_args: passed to evict_func
            evict_kwargs: passed to evict_func
            sleep_interval:
        """
        self._evict_func = evict_func
        self._evict_args = evict_args
        self._evict_kwargs = evict_kwargs
        self._sleep_interval = sleep_interval
        super().__init__()

    def run(self):
        while True:
            if self.is_stopped():
                break
            self._evict_func(*self._evict_args, **self._evict_kwargs)
            time.sleep(self._sleep_interval)


class Replay(object):
    def __init__(self, *,
                 redis_client,
                 batch_size,
                 name='replay',
                 fetch_queue_size=5):
        U.assert_type(redis_client, RedisClient)
        self.batch_size = batch_size
        self._client = redis_client
        self._exp_queue = ExpQueue(
            redis_client=redis_client,
            queue_name=name,
        )
        self._obs_fetch_queue = ObsFetchQueue(
            redis_client=redis_client,
            maxsize=fetch_queue_size,
        )
        # keeps the integrity of the replay memory data structure
        self._replay_lock = threading.Lock()
        # avoid evicting and fetching at the same time
        self._evict_lock = threading.Lock()
        self._evict_thread = None

    def _insert(self, exp_dict):
        """
        Add a new experience to the replay.
        Includes evict logic if exceeds memory capacity

        Args:
            exp_dict: experience dictionary with
                {"obs_pointers", "reward", "action", "info"} keys
        """
        raise NotImplementedError

    def _sample(self, batch_size, batch_i):
        """
        This function is called in the `exp_download_queue` thread, its operation
        is async, i.e. overlaps with the insertion operations.

        Args:
            batch_size: passed from self.batch_size, defined in the
                constructor upfront.
            batch_i: the i-th batch it is sampling.
            Note that `batch_size` is

        Returns:
            a list of exp_dicts
        """
        raise NotImplementedError

    def _evict(self, *args, **kwargs):
        """
        Evict old experiences.

        Returns:
            list of exp dicts that contain `exp_pointer` or `obs_pointers`.
            if the exp is not stored on Redis, they will be ignored.
        """
        return []

    def start_sample_condition(self):
        """
        Tells the thread to start sampling only when this condition is met.
        For example, only when the replay memory has > 10K experiences.

        Returns:
            bool: whether to start sampling or not
        """
        raise NotImplementedError

    def aggregate_batch(self, exp_list):
        """
        Will be called in `next_batch()` method to produce the actual inputs
        to the neural network training loop.

        Args:
            exp_list: list of experience dictionaries with actual observations
                {"obses", "reward", "action", "info"} keys

        Returns:
            batched Tensors, batched action/reward vectors, etc.
        """
        raise NotImplementedError

    def insert(self, exp_dict):
        """
        Must not sample and insert at the same time
        """
        with self._replay_lock:
            return self._insert(exp_dict)

    def sample(self, batch_i):
        with self._replay_lock:
            return self._sample(self.batch_size, batch_i)

    def evict(self, *args, **kwargs):
        with self._evict_lock:
            with self._replay_lock:
                evicted_exp_list = self._evict(*args, **kwargs)
            evict_exp_pointers = []
            evict_obs_pointers = []
            for exp in evicted_exp_list:
                U.assert_type(exp, dict)
                if 'exp_pointer' in exp:
                    evict_exp_pointers.append(exp['exp_pointer'])
                if 'obs_pointers' in exp:
                    obs_pointers = exp['obs_pointers']
                    U.assert_type(obs_pointers, list)
                    evict_obs_pointers.extend(obs_pointers)

            print('DEBUG delete start', len(evict_obs_pointers))
            ref_pointers = ['ref-' + _p for _p in evict_obs_pointers]
            ref_counts = self._client.mdecr(ref_pointers)
            # only evict when ref count drops to 0
            print('DEBUG ref_counts', ref_counts)
            evict_obs_pointers = [evict_obs_pointers[i]
                                  for i in range(len(evict_obs_pointers))
                                  if ref_counts[i] <= 0]
            # mass delete exp and obs (only when ref drop to 0) on Redis
            _ret = self._client.mdel(evict_obs_pointers + evict_exp_pointers)
            print('DEBUG delete done', _ret)
            return evicted_exp_list

    def start_queue_threads(self):
        """
        Call this method to launch all background threads that talk to Redis.
        """
        self._exp_queue.start_enqueue_thread()
        self._exp_queue.start_dequeue_thread(self.insert)
        self._obs_fetch_queue.start_enqueue_thread(
            sampler=self.sample,
            start_sample_condition=self.start_sample_condition,
            evict_lock=self._evict_lock
        )

    def stop_queue_threads(self):
        self._exp_queue.stop_enqueue_thread()
        self._exp_queue.stop_dequeue_thread()
        self._obs_fetch_queue.stop_enqueue_thread()

    def start_evict_thread(self, *args, sleep_interval=1., **kwargs):
        if self._evict_thread is not None:
            raise RuntimeError('evict_thread already running')
        self._evict_thread = _EvictThread(
            evict_func=self.evict,
            evict_args=args,
            evict_kwargs=kwargs,
            sleep_interval=sleep_interval
        )
        self._evict_thread.start()
        return self._evict_thread

    def stop_evict_thread(self):
        t = self._evict_thread
        t.stop()
        self._evict_thread = None
        return t

    def next_batch(self):
        exp_list = self._obs_fetch_queue.dequeue()
        return self.aggregate_batch(exp_list)

    def batch_iterator(self):
        """
        Yields:
            batch_i, (batched inputs to neural network)
        """
        for batch_i in itertools.count():
            yield batch_i, self.next_batch()
