"""
Sample pointers from replay buffer and pull the actual observations
"""
from .redis_client import RedisClient
from .packs import ObsPack, ExpPointerPack, ExpFullPack


class ExpSender(object):
    def __init__(self,
                 redis_client,
                 queue_name,
                 *,
                 pointers_only=True,
                 save_exp_on_redis=True,
                 obs_cache_size=10000000):
        """
        Args:
            redis_client
            queue_name: name of the ExpQueue
            pointers_only (default True): send only pointers instead of full obs
              if True, the exp_dict will contain only 'obs_pointers' field
              otherwise it will have 'obses' field.
            save_exp_on_redis: whether to save Exp dict on Redis replay server
              or not. if False, only push exp to the Redis queue without saving.
            obs_cache_size: max size of the cache of new_obs hashes so that we
              don't send duplicate new_obs to Redis. Will be ignored if
              `pointers_only` is False.
        """
        assert isinstance(redis_client, RedisClient)
        self._client = redis_client
        self.queue_name = queue_name
        self._visited_obs = set() # avoid resending new_obs
        self._pointers_only = pointers_only
        self._save_exp_on_redis = save_exp_on_redis
        self._obs_cache_size = obs_cache_size

    def _add_to_visited(self, obs_pointer):
        self._visited_obs.add(obs_pointer)
        if len(self._visited_obs) > self._obs_cache_size:
            self._visited_obs.pop()

    def send(self, obses, action, reward, done, info):
        # TODO can buffer multiple send with redis_mset dict and flush to
        # network all at once to save bandwidth
        """
        Args:
            exp_dict: {obses: [np_image0, np_image1], action, reward, info}

        - Send the observations with their hash as key
        - Send the experience tuple with its hash as key
        - Send the PointerPack to Redis queue
        """
        redis_mset = {}
        if self._pointers_only:
            # observation pack
            obs_pointers = []
            ref_pointers = []
            for obs in obses:
                pack = ObsPack(obs)
                obs_pointer, binary = pack.serialize()
                if obs_pointer not in self._visited_obs:
                    self._add_to_visited(obs_pointer)
                    redis_mset[obs_pointer] = binary
                obs_pointers.append(obs_pointer)
                # reference counter, evict when dropped to zero
                ref_pointers.append('ref-' + obs_pointer)
            self._client.mincr(ref_pointers)
            # experience pack
            exp_pack = ExpPointerPack(obs_pointers, action, reward, done, info)
        else:
            exp_pack = ExpFullPack(obses, action, reward, done, info)
        exp_pointer, binary = exp_pack.serialize()
        if self._save_exp_on_redis:
            redis_mset[exp_pointer] = binary
        self._client.mset(redis_mset)
        # send to queue
        self._client.enqueue(self.queue_name, binary)

