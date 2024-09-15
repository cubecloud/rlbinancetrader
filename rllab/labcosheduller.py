from typing import Callable
from math import cos, pi

__version__ = 0.008


class CoSheduller:
    def __init__(self, warmup: int = 15, learning_rate: float = 1e-4, min_learning_rate: float = 1e-6,
                 total_epochs: int = 300, epsilon: int = 100):

        self.warmup = warmup
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.total_epochs = total_epochs
        self.last_lr = self.learning_rate
        self.epsilon = epsilon
        self.__call__()

    def __call__(self) -> Callable[[float], float]:
        return self.scheduler

    def scheduler(self, progress_remaining: float) -> float:
        # progress_remaining = 1.0 - (num_timesteps / total_timesteps)
        epoch = int((1 - progress_remaining) * self.total_epochs)
        if epoch % self.epsilon == 0:
            """ Warm up from zero to learning_rate """
            if epoch <= self.warmup - 1:
                lr = self.min_learning_rate + (((self.learning_rate - self.min_learning_rate) / self.warmup) *
                                               epoch)
            else:
                """ using cos learning rate """
                formula = self.min_learning_rate + 0.5 * (self.learning_rate - self.min_learning_rate) * (
                        1 + cos(max(epoch + 1 - self.warmup, 0) * pi / max(self.total_epochs - self.warmup, 1)))
                # min calc min and max with zero centered logic - close to zero = less
                lr = max(formula, self.min_learning_rate)
            self.last_lr = lr
        return self.last_lr



