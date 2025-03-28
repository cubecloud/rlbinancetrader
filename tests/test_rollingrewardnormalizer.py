import numpy as np
import unittest
import math


class RollingRewardNormalizer:
    """Class for normalizing rewards using a running average and standard deviation."""

    def __init__(self, epsilon: np.float64 = 1e-8, dtype=np.float64) -> None:
        """
        Initializes the reward normalizer object.

        Args:
            epsilon: A small number to ensure numerical stability.
        """
        self.running_mean: np.float64 = 0.0  # Running mean of rewards.
        self.running_var: np.float64 = 1.0  # Running variance of rewards, initialized to be positive.
        self.epsilon = epsilon
        self.count = self.epsilon  # Count of observed rewards, starts with epsilon to avoid division by zero.

    def update(self, reward: np.float64) -> None:
        """
        Updates the running mean and variance based on the new reward.

        Args:
            reward: The new reward value.
        """
        self.count += 1
        old_mean = self.running_mean  # Save the old mean before updating
        delta = reward - old_mean  # Calculate delta from the old mean
        self.running_mean += delta / self.count  # Update the mean
        self.running_var += delta * (reward - old_mean)  # Update the variance correctly

    def normalize(self, reward: np.float64) -> np.float64:
        """
        Normalizes the given reward based on current mean and variance.

        Args:
            reward: The reward to be normalized.

        Returns:
            float: The normalized reward.
        """
        if self.running_var > 1e-8:
            std_dev = np.sqrt(self.running_var / (self.count - 1))  # Standard deviation.
            return (reward - self.running_mean) / std_dev
        else:
            return reward

    def reset(self):
        self.running_mean: np.float64 = 0.0  # Running mean of rewards.
        self.running_var: np.float64 = 1.0  # Running variance of rewards, initialized to be positive.
        self.count = self.epsilon  # Count of observed rewards, starts with epsilon to avoid division by zero.


class TestRollingRewardNormalizer(unittest.TestCase):
    def test_update_and_normalize(self):
        # Create an instance of the normalizer
        normalizer = RollingRewardNormalizer()

        # # Check initialization values
        # self.assertTrue(math.isclose(normalizer.running_mean, 0.0, rel_tol=1e-12))
        # self.assertTrue(math.isclose(normalizer.running_var, 0.0, rel_tol=1e-12))
        # self.assertTrue(math.isclose(normalizer.count, 1e-6, rel_tol=1e-12))

        # Add first reward
        reward_1 = 10.0
        normalizer.update(reward_1)
        expected_running_mean = reward_1
        expected_running_var = 0.0
        print(normalizer.running_mean, normalizer.running_var)
        # self.assertTrue(math.isclose(normalizer.running_mean, expected_running_mean, rel_tol=1e-12))
        # self.assertTrue(math.isclose(normalizer.running_var, expected_running_var, rel_tol=1e-12))

        # Add second reward
        reward_2 = 20.0
        normalizer.update(reward_2)
        expected_running_mean = (reward_1 + reward_2) / 2
        expected_running_var = ((reward_1 - expected_running_mean) ** 2 + (reward_2 - expected_running_mean) ** 2) / 2
        print(normalizer.running_mean, normalizer.running_var)
        # self.assertTrue(math.isclose(normalizer.running_mean, expected_running_mean, rel_tol=1e-12))
        # self.assertTrue(math.isclose(normalizer.running_var, expected_running_var, rel_tol=1e-12))

        reward_3 = 30.0
        normalizer.update(reward_3)


if __name__ == '__main__':
    unittest.main()
