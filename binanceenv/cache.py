from dbbinance.fetcher.cachemanager import CacheManager

__version__ = 0.002

cache_manager_obj = CacheManager(max_memory_gb=2, mutex='mlp')
eval_cache_manager_obj = CacheManager(max_memory_gb=2, mutex='mlp')
