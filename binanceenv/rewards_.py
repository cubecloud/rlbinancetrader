from abc import ABC, abstractmethod

__version__ = 0.0005


class RewardsABC(ABC):

    def __init__(self):
        self.current_timestep: int = 0
        self.episode_rewards: list = []
        self.action_reward_dict: dict = {0: self.action_zero_reward,
                                         1: self.action_one_reward,
                                         2: self.action_two_reward,
                                         3: self.action_three_reward,
                                         'step': self.step_reward,
                                         'last': self.last_reward,
                                         }

    def get_action_reward(self, action, action_data):
        return self.action_reward_dict[action](action_data)

    @abstractmethod
    def step_reward(self, action_data):
        reward = None
        return reward

    @abstractmethod
    def action_zero_reward(self, action_data):
        reward = None
        return reward

    @abstractmethod
    def action_one_reward(self, action_data):
        reward = None
        return reward

    @abstractmethod
    def action_two_reward(self, action_data):
        reward = None
        return reward

    @abstractmethod
    def action_three_reward(self, action_data):
        reward = None
        return reward

    @abstractmethod
    def last_reward(self, action_data):
        reward = None
        return reward

    def reset(self):
        self.current_timestep: int = 0
        self.episode_rewards: list = []

