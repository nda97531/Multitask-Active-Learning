import numpy as np
import torch as tr
from MAT import logger


def sparse_2_onehot(label, n_class):
    """
    label shape (n,)
    """
    onehot = np.zeros([len(label), n_class], dtype=np.float32)
    onehot[np.arange(len(label)), label] = 1.
    return onehot


def load_dataset_single_file(data_file, p_lb_file, list_valid_subject_id, is_train_set,
                             data_channels=(3, 3),
                             normalize=True,
                             min_vals=[], max_vals=[],
                             lb_col_in_p_lb=None):
    p_lb = np.load(p_lb_file)
    select_index = np.isin(p_lb[:, 0], list_valid_subject_id)
    if is_train_set:
        select_index = ~select_index
    label = p_lb[select_index]

    if lb_col_in_p_lb is not None:
        label = label[:, lb_col_in_p_lb]

    data_array = np.load(data_file)
    if np.sum(data_channels) != data_array.shape[-1]:
        raise ValueError('number of channels not match!')

    return_array = []
    for index, n_channel in enumerate(data_channels):
        first_col = np.sum(data_channels[:index], dtype=int)
        this_data = data_array[:, :, first_col:first_col + n_channel]

        if normalize:
            if len(min_vals) < len(data_channels) or len(max_vals) < len(data_channels):
                min_vals.append(this_data.min())
                max_vals.append(this_data.max())
            this_data = (this_data - min_vals[index]) / (max_vals[index] - min_vals[index])
        return_array.append(this_data[select_index])

    return_array.append(label)

    if is_train_set and normalize:
        return return_array, min_vals, max_vals
    else:
        return return_array


def sliding_window(time_series_data, window_size, step_size, count_last_lines=False):
    """
    :param time_series_data: shape (timestep, channel)
    :param window_size: window size in lines
    :param step_size: step size in lines
    :param count_last_lines: still yield the last lines of sequence when it's shorter than step size
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


def class_weight(label, return_type, rescale_by='mean'):
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
    cweight = 1. * len(label) / counts

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


def check_std(window, min_std_allowed):
    """
    :param window: shape (timestep, channel)
    :return: boolean
    """
    return window.std(axis=0).mean() >= min_std_allowed


def closest_greater_index_in_array(array, value):
    diff = array - value
    greater_index = np.where(diff >= 0)[0][0]
    return greater_index


def closest_less_index_in_array(array, value):
    diff = array - value
    less_index = np.where(diff <= 0)[0][-1]
    return less_index


def closest_index_in_array(array, value):
    diff = np.abs(array - value)
    return np.argmin(diff)


def interp_1d(observed_data: np.ndarray,
              new_len: int = None,
              observed_timestamp: np.ndarray = None):
    """
    1D linear interpolation
    Args:
        observed_data: shape [len, ]
        new_len: integer
        observed_timestamp: shape [len, ]
    Returns:
        shape [new_len, ]
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
               observed_timestamp: np.ndarray = None):
    """
    Apply interp_1d on all input feature.
    Args:
        observed_data: shape [len, feature]
        new_len: integer
        observed_timestamp: shape [len, ]
    Returns:
        shape [new len, feature]
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


def torch_num_params(model):
    keys = list(model.state_dict().keys())
    total_params = 0
    for key in keys:
        total_params += tr.tensor(model.state_dict()[key].shape).prod().item()
    return total_params
