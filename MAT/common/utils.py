import numpy as np
import torch as tr
from typing import Union
from MAT import logger


def sparse_2_onehot(label: np.ndarray, n_class: int, dtype=np.float32) -> np.ndarray:
    """
    Convert categorical label to one-hot encoded label
    :param label: array shape [num label]
    :param n_class: number of class
    :param dtype: data type of output one-hot label
    :return: one-hot encoded label, array shape [num label, num class]
    """
    onehot = np.zeros([len(label), n_class], dtype=dtype)
    onehot[np.arange(len(label)), label] = 1
    return onehot


def sliding_window(time_series_data: np.ndarray, window_size: int, step_size: int,
                   count_last_lines: bool = False) -> np.ndarray:
    """
    Slide window from a time series and yield window by window.
    :param time_series_data: shape (num time steps, *)
    :param window_size: window size
    :param step_size: step size
    :param count_last_lines: still yield the last lines of sequence when it's shorter than step size
    :return: yield window by window, each one has shape [window size, *]
    """
    first_index = 0
    while True:
        last_index = first_index + window_size

        if last_index <= time_series_data.shape[0]:
            yield time_series_data[first_index: last_index]
            if last_index == time_series_data.shape[0]:
                break
        elif count_last_lines:
            yield time_series_data[-window_size:]
            break
        else:
            break

        first_index += step_size


def class_weight(label, return_type, rescale_by='mean') -> Union[dict,]:
    """
    Calculate class weight based on quantity
    :param label: sparse label array shape [num instances, ]
    :param return_type: "array" or "dict"
    :param rescale_by: "mean", "max", or "min"
    :return: array or dict, class weight
    """

    # count number of label of each class
    values, counts = np.unique(label, return_counts=True)

    # calculate class weight
    cweight = len(label) / counts

    # rescale class weight
    if rescale_by == 'min':
        cweight = cweight / cweight.min()
    elif rescale_by == 'max':
        cweight = cweight / cweight.max()
    elif rescale_by == 'mean':
        cweight = cweight / cweight.mean()

    if return_type == 'array':
        return cweight
    elif return_type == 'dict':
        return dict(zip(values, cweight))
    else:
        raise ValueError('return_type is "array" or "dict"')


def interp_1d(observed_data: np.ndarray,
              new_len: int = None,
              observed_timestamp: np.ndarray = None):
    """
    1D linear interpolation.
    :param observed_data: shape [len, ]
    :param new_len: integer
    :param observed_timestamp: shape [len, ], a timestamp for each input observed sample;
        if not specified, timestamps are assumed to be constant
    :return: array shape [new_len, ]
    """
    if new_len is None and observed_timestamp is None:
        logger.warning('Both new_len and timestamp are None, interpolation will not take effect.')
        return observed_data

    if observed_timestamp is not None:
        if new_len is None:
            new_len = len(observed_data)
        return np.interp(np.linspace(observed_timestamp[0], observed_timestamp[-1], new_len),
                         observed_timestamp,
                         observed_data)
    else:
        return np.interp(np.linspace(0, len(observed_data) - 1, new_len),
                         np.arange(len(observed_data)),
                         observed_data)


def interp_1ds(observed_data: np.ndarray,
               new_len: int = None,
               observed_timestamp: np.ndarray = None) -> np.ndarray:
    """
    Apply interp_1d on all input feature.
    :param observed_data: shape [len, feature], raw data
    :param new_len: length that we want after interpolation
    :param observed_timestamp: shape [len, ], a timestamp for each input observed sample;
        if not specified, timestamps are assumed to be constant
    :return: array shape [new len, feature]
    """
    if len(observed_data.shape) != 2:
        raise ValueError(
            f'observation must have 2 dimensions (time and feature), '
            f'found {observed_data.shape}.')

    if (observed_timestamp is not None) and (len(observed_timestamp.shape) != 1):
        raise ValueError(
            f'timestamp must have 1 dimensions (time), '
            f'found {observed_timestamp.shape}.')

    if new_len is None:
        new_len = len(observed_data)

    interp_data = []
    for i in range(observed_data.shape[-1]):
        interp_data.append(interp_1d(observed_data[:, i], new_len, observed_timestamp))
    interp_data = np.stack(interp_data, axis=1)
    return interp_data
