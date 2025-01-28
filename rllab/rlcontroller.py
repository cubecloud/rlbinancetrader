from abc import ABC, abstractmethod
from threading import RLock

import objsize
from dbbinance.fetcher.slocks import SThLock, SMpLock
from dbbinance.fetcher.singleton import Singleton
from typing import Union
import multiprocessing as mp
from multiprocessing.managers import SyncManager

__version__ = 0.001

logger = mp.get_logger()

mp_timesteps_counter = mp.Value('i', 1)
mp_episodes_counter = mp.Value('i', -1)
mp_count = mp.Value('i', 0)


class SyncContainer(SyncManager):
    pass


class Controller:
    def __init__(self,
                 max_memory_gb: Union[float, int] = 3,
                 start_host: bool = True,
                 host: str = "127.0.0.1",
                 port: int = 5003,
                 authkey: bytes = b"password",
                 th_rlock=None,
                 unique_name='train'
                 ):
        """
        Initialize the Cache class with an optional maximum memory limit in gigabytes.

        Args:
            max_memory_gb (float or int):   The maximum memory limit in gigabytes.
            start_host (bool):              Start host or just connect to it
            host (str):                     Host IP
            port (int):                     Host port
            authkey (bytes):                Authorization password (bytes)
        """
        self.start_host = start_host
        self.host = host
        self.port = port
        self.unique_name = f'{unique_name}'

        self.manager = None
        self.__cache = {}
        self.__hits = {}
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)  # Convert max_memory_gb to bytes
        self.host_instance = False
        self.manager = SyncContainer((self.host, self.port), authkey=authkey)
        self.thrlock_obj = SThLock(th_rlock if th_rlock is not None else RLock(),
                                   unique_name=f'{self.unique_name}_rlock')
        self.lock = self.thrlock_obj.lock
        if self.start_host:
            """
            Start host instance 
            using Threading lock (SThLock) to avoid race conditions with multiple clients 
            in one process
            SyncManager is not thread-safe, but for multiprocessing it is  
            """
            self.manager.register('get_cache', callable=lambda: self.__cache)
            self.manager.register('get_hits', callable=lambda: self.__hits)
            self.manager.start()
            self.host_instance = True
        else:
            """
            Connect to host instance
            using Threading lock (SThLock) to avoid race conditions with multiple clients 
            in one process
            SyncManager is not thread-safe, but for multiprocessing it is
            """
            self.manager.register('get_cache')
            self.manager.register('get_hits')
            self.manager.connect()
            self.host_instance = False


class MainController:
    def __init__(self,
                 start_host: bool = True,
                 host: str = "127.0.0.1",
                 port: int = 5003,
                 authkey: bytes = b"password",
                 unique_name='train'
                 ):
        self.start_host = start_host
        self.host = host
        self.port = port
        self.unique_name = f'{unique_name}'
        self.manager = SyncContainer((self.host, self.port), authkey=authkey)
        self.train_controller = None
        self.test_controller = None
        self.check_controller = None

    @classmethod
    def init_controller(cls, host, port, authkey):
        return SyncContainer((host, port), authkey=authkey)

    def get_controller(self, host, port, authkey):
        pass
