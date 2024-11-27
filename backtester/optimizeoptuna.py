import optuna
from typing import Dict, Any, Tuple, Callable


def prepare_strategy_params(trial: optuna.Trial,
                            kwargs: Dict[str, Any],
                            constraint: Callable[[dict], bool] = None,
                            suffix: str = '') -> Dict[str, Any] or None:
    """
    Prepare strategy parameters from kwargs for use in an Optuna study.

    Args:
        suffix:
        constraint:
        trial:
        kwargs: A dictionary of strategy parameters.

    Returns:
        A dictionary of strategy parameters prepared for use in an Optuna study.
    """

    class AttrDict(dict):
        def __getattr__(self, item):
            return self[item]

    strategy_params: dict = {}

    for ix, (param_name, value) in enumerate(kwargs.items()):
        if param_name == 'ind_params':
            strategy_params[param_name] = prepare_strategy_params(trial, value, constraint=constraint)
        elif (param_name.startswith('model_') or param_name in ('minus', 'plus')) and isinstance(value, dict):
            result_value = prepare_strategy_params(trial, value, constraint=constraint, suffix=f'_{ix:02}')
            if result_value is not None:
                if constraint(AttrDict(result_value)):
                    strategy_params[param_name] = result_value
                else:
                    strategy_params[param_name] = None
            else:
                strategy_params[param_name] = None
        elif isinstance(value, (int, float, str)):
            trial.set_user_attr(f'{param_name}{suffix}', value)
            strategy_params[param_name] = value
        elif isinstance(value, (tuple, list)):
            if all(isinstance(item, int) for item in value):
                strategy_params[param_name] = suggest_int_tuple(trial, f'{param_name}{suffix}', value)
            elif all(isinstance(item, float) for item in value):
                strategy_params[param_name] = suggest_float_tuple(trial, f'{param_name}{suffix}', value)
            else:
                strategy_params[param_name] = trial.suggest_categorical(f'{param_name}{suffix}', value)
        elif isinstance(value, range):
            strategy_params[param_name] = trial.suggest_int(f'{param_name}{suffix}', value.start, value.stop, value.step)
        else:
            raise ValueError(f"Unsupported parameter type for f'{param_name}{suffix}': {type(value)}")

    if any(value is None for value in strategy_params.values()):
        strategy_params: dict = None
    return strategy_params


def suggest_int_tuple(trial: optuna.Trial, param_name: str,
                      value: Tuple[int, int, int] or Tuple[int, int]) -> int or None:
    """
    Generate a suggestion for an integer tuple value based on the length of the value.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        param_name (str): The name of the parameter.
        value (Tuple[int, int, int] or Tuple[int, int]): The tuple value.

    Returns:
        Any: The suggested value based on the length of the tuple value.
    """
    if len(value) == 2:
        return trial.suggest_int(param_name, value[0], value[1])
    elif len(value) == 3:
        return trial.suggest_int(param_name, value[0], value[1], value[2])
    else:
        return None


def suggest_float_tuple(trial: optuna.Trial, param_name: str,
                        value: Tuple[float, float, float] or Tuple[float, float]) -> float or None:
    """
    Generate a suggestion for a floating-point tuple value.

    Parameters:
        - trial (optuna.Trial): The optuna Trial object.
        - param_name (str): The name of the parameter.
        - value  Tuple[float, float, float] or Tuple [float, float]: The tuple of values.

    Returns:
        - Any: The suggested value for the parameter.

    """
    if len(value) == 2:
        return trial.suggest_uniform(param_name, value[0], value[1])
    elif len(value) == 3:
        temp = trial.suggest_float(param_name, value[0], value[1], step=value[2])
        return temp
    else:
        return None
