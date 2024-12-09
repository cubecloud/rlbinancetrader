import numpy as np

__version__ = 0.002


def minmax_normalization_1_1(val_arr):
    val_arr_min = np.min(val_arr)
    return (val_arr - val_arr_min) / (np.max(val_arr) - val_arr_min) * 2 + -1


def minmax_normalization_custom(val_arr, features_range=(-1, 1)):
    val_arr_min = np.min(val_arr)
    return (val_arr - val_arr_min) / (np.max(val_arr) - val_arr_min) * (
            features_range[1] - features_range[0]) + features_range[0]


def minmax_normalization(val_arr):
    val_arr_min = np.min(val_arr)
    return (val_arr - val_arr_min) / (np.max(val_arr) - val_arr_min)
