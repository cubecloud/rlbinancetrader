from typing import Callable, Union, Dict
import copy

__version__ = 0.0010


def deserialize_kwargs(_agent_kwargs: Union[dict, str],
                       lab_serializer=None) -> Union[dict, Callable]:
    """

    Args:
        lab_serializer (dict):
    """
    if lab_serializer is None:
        lab_serializer = {}

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
                    data_update.update({_key: deserialize_kwargs(_value, lab_serializer)})
        agent_kwargs.update(data_update)
    elif isinstance(agent_kwargs, str):
        deserialized_obj = lab_serializer.get(agent_kwargs, None)
        if deserialized_obj is not None:
            agent_kwargs = deserialized_obj
    return agent_kwargs
