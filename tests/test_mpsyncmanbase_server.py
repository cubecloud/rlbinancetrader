from rllab import MpSyncManBase
import time

if __name__ == "__main__":
    if not MpSyncManBase.is_server_running(port=5500):
        man_obj = MpSyncManBase(n_envs=2, port=5500)
        print(man_obj.command)
        man_obj.command.update({1: (1, 1)})
        print(man_obj.command)
        print(man_obj.command.get(1))
        print(man_obj.command_done)
        # print("Shared dictionary created:", man_obj.items())
        # man_obj.update({800: 800})
        # cache = list(man_obj.items())
        # # print(type(man_obj.keys()))
        # # print(type(man_obj.keys()[0]))

        # # while cache == man_obj.items():
        # #     print(f'\r{man_obj.items()}', end='')
        # #     time.sleep(5)
        # while True:
        #     print(f'\r{man_obj.items()} -> length = {len(man_obj)}', end='')
        #     time.sleep(5)
        # # print(f'\n{man_obj.items()}')
        # # man_obj.shutdown()
    else:
        print('Server is running')
