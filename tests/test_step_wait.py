import timeit
import numpy as np

class EnvWrapper:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.env_indices_per_process = np.arange(n_envs)
        self.remotes = [None] * n_envs  # simulate remote connections

    def step_async(self):
        data = {}
        for pr_idx, remote in enumerate(self.remotes):
            data.update({pr_idx: remote()})  # simulate receiving data from remote connection
        return data

    def test_original_code(self):
        data = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            data.update({pr_idx: self.remotes[pr_idx]()})
        results = [data[env_ix] for env_ix in range(self.n_envs)]
        obs, rews, dones, infos, reset_infos = zip(*results)
        return obs, rews, dones, infos, reset_infos

    def test_improved_code(self):
        data = {}
        for pr_idx, remote in enumerate(self.remotes):
            data.update({pr_idx: remote()})
        obs, rews, dones, infos, reset_infos = zip(*(data[env_ix] for env_ix in range(self.n_envs)))
        return obs, rews, dones, infos, reset_infos

# Create an instance of EnvWrapper
env_wrapper = EnvWrapper(int(2520//14))

# Simulate remote connections
env_wrapper.remotes = [lambda: (np.ones((48,21)), np.zeros((1)), np.ones((1)).astype(bool), dict(), dict()) for _ in range(env_wrapper.n_envs)]

# Run the tests
original_time = timeit.timeit(lambda: env_wrapper.test_original_code(), number=100000)
improved_time = timeit.timeit(lambda: env_wrapper.test_improved_code(), number=100000)

print(f"Original code: {original_time:.6f} seconds")
print(f"Improved code: {improved_time:.6f} seconds")

# original_obs, original_rews, original_dones, original_infos, original_reset_infos = env_wrapper.test_original_code()
# improved_obs, improved_rews, improved_dones, improved_infos, improved_reset_infos = env_wrapper.test_improved_code()

# print("Original code return values:")
# print(original_obs)
# print(original_rews)
# print(original_dones)
# print(original_infos)
# print(original_reset_infos)

# print("Improved code return values:")
# print(improved_obs)
# print(improved_rews)
# print(improved_dones)
# print(improved_infos)
# print(improved_reset_infos)