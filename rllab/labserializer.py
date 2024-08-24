import copy
from typing import Union, Callable, ClassVar
import numpy as np
from rllab.labtools import CoSheduller
from torch.nn import ReLU, LeakyReLU
from customnn.mlpextractor import MlpExtractorNN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import A2C, PPO, DQN, TD3, DDPG, SAC
from binanceenv import BinanceEnvBase
from binanceenv import BinanceEnvCash

__version__ = 0.009

#   underscore at the end of class name -> call object itself to get method

lab_serializer: dict = {'learning_rate': {'CoSheduller_': CoSheduller},
                        'MlpExtractorNN': MlpExtractorNN,
                        'ReLU': ReLU,
                        'LeakyReLU': LeakyReLU,
                        'action_noise': {'OrnsteinUhlenbeckActionNoise': OrnsteinUhlenbeckActionNoise},
                        'SAC': SAC,
                        'A2C': A2C,
                        'PPO': PPO,
                        'DQN': DQN,
                        'TD3': TD3,
                        'DDPG': DDPG,
                        'BinanceEnvBase': BinanceEnvBase,
                        'BinanceEnvCash': BinanceEnvCash
                        }


def deserialize_kwargs(_agent_kwargs: Union[dict, str]) -> Union[dict, Callable]:
    agent_kwargs = copy.deepcopy(_agent_kwargs)
    data_update: dict = {}
    if isinstance(agent_kwargs, dict):
        for _key, _value in agent_kwargs.items():
            if isinstance(_value, str):
                for serializer_key, serializer_value in lab_serializer.items():
                    if _value.lower() == serializer_key.lower():
                        deserialized_obj = lab_serializer.get(_value, None)
                        data_update.update({_key: deserialized_obj})
            elif isinstance(_value, dict):
                for serializer_key, serializer_value in lab_serializer.items():
                    if _key.lower() == serializer_key.lower():
                        for _k, _v in _value.items():
                            deserialized_obj = lab_serializer.get(_key, None).get(_k, None)
                            if deserialized_obj is not None:
                                data_update.update({_key: deserialized_obj(**_v)})
                            else:
                                deserialized_obj = lab_serializer.get(_key, None).get(f'{_k}_', None)
                                if deserialized_obj is not None:
                                    data_update.update({_key: deserialized_obj(**_v)()})
                if not data_update:
                    data_update.update({_key: deserialize_kwargs(_value)})
        agent_kwargs.update(data_update)
    elif isinstance(agent_kwargs, str):
        deserialized_obj = lab_serializer.get(agent_kwargs, None)
        if deserialized_obj is not None:
            agent_kwargs = deserialized_obj
    return agent_kwargs


if __name__ == '__main__':
    total_timesteps = 16_000_000
    buffer_size = 1_000_000
    learning_start = 750_000
    batch_size = 1024

    action_noise_box = OrnsteinUhlenbeckActionNoise(mean=5e-1 * np.ones(3), sigma=4.99e-1 * np.ones(3), dt=1e-2)

    sac_policy_kwargs = dict(
        features_extractor_class='MlpExtractorNN',
        features_extractor_kwargs=dict(features_dim=256),
        share_features_extractor=True,
        activation_fn='ReLU',
        # net_arch=net_arch,
    )
    sac_kwargs = dict(policy="MlpPolicy",
                      buffer_size=buffer_size,
                      learning_starts=learning_start,
                      policy_kwargs=sac_policy_kwargs,
                      batch_size=batch_size,
                      stats_window_size=100,
                      ent_coef='auto_0.0001',
                      learning_rate={'CoSheduller': dict(warmup=learning_start,
                                                         learning_rate=2e-4,
                                                         min_learning_rate=1e-5,
                                                         total_epochs=total_timesteps,
                                                         epsilon=100)},
                      action_noise={'OrnsteinUhlenbeckActionNoise': dict(mean=5e-1 * np.ones(3),
                                                                         sigma=4.99e-1 * np.ones(3),
                                                                         dt=1e-2)},
                      train_freq=(2, 'step'),
                      target_update_interval=10,  # update target network every 10 _gradient_ steps
                      device="auto",
                      verbose=1)

    print(deserialize_kwargs(sac_kwargs))
