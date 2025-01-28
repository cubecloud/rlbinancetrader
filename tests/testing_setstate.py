from multiprocessing import Process, Value, Lock
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
import threading
import time

# Создаем общую переменную
mp_counter = Value('i', 0)


class MyClass:
    id_counter = mp_counter

    def __init__(self):
        self.data = None
        with MyClass.id_counter.get_lock():
            MyClass.id_counter.value += 1
        self.idnum = MyClass.id_counter.value
        print(self.idnum)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Используем блокировку для изменения счетчика
        with MyClass.id_counter.get_lock():
            MyClass.id_counter.value += 1
            self.idnum = MyClass.id_counter.value
        print(self.idnum, flush=True)


def worker_thread(env):
    env.__setstate__({})


def worker(envs_var):
    # env_v = CloudpickleWrapper(envs_var)
    envs = [env for env in envs_var.var]
    for env in envs:
        # print(env.__dict__.items())
        env.__setstate__({})  # имитируем вызов __setstate__

    # threads = []
    # for env in envs:
    #     t = threading.Thread(target=worker_thread, args=(env,))
    #     threads.append(t)
    #     t.start()
    #
    # for t in threads:
    #     t.join()


if __name__ == '__main__':
    start_time = time.time()

    envs_lst = [MyClass() for _ in range(1000)]
    processes = []
    for _ in range(2):
        p = Process(target=worker, args=(CloudpickleWrapper(envs_lst),))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    print(f'Final counter value: {mp_counter.value}')
    print(f'Total execution time: {total_time:.4f} seconds')
