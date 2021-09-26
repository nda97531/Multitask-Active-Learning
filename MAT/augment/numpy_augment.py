from typing import Union
import numpy as np
from overrides import overrides
from MAT import logger


class Augmenter:
    def apply(self, org_data: np.ndarray) -> np.ndarray:
        data = np.copy(org_data)
        return data


class Rotate(Augmenter):
    def __init__(self,
                 input_shape: Union[list, tuple],
                 rotate_x_range: Union[list, tuple],
                 rotate_y_range: Union[list, tuple],
                 rotate_z_range: Union[list, tuple]
                 ) -> None:
        """

        :param input_shape: input window shape,
            e.g. [2500,6]
        :param rotate_x_range, rotate_y_range, rotate_z_range:
            list of 2 positive floats, range of degree to rotate signals,
            e.g. [10., 20.]
                signal will be rotated by 10~20 degs or -20~-10 degs
        """
        self.input_len = input_shape[0]
        self.input_channel = input_shape[-1]
        self.num_modals = self.input_channel // 3

        if self.input_channel % 3 != 0:
            logger.error('input data is not triaxial')

        # convert angles from angle to radian
        self.rotate_x_range = np.array(rotate_x_range) / 180. * np.pi
        self.rotate_y_range = np.array(rotate_y_range) / 180. * np.pi
        self.rotate_z_range = np.array(rotate_z_range) / 180. * np.pi

    @overrides
    def apply(self, org_data: np.ndarray) -> np.ndarray:
        """
        Apply augmentation methods in self.list_aug_func
        :param org_data:
            shape (time step, channel) channel must be divisible by 3,
            otherwise bugs may occur
        :return: array shape (time step, channel)
        """
        data = super().apply(org_data)

        # check input shape
        if data.shape[0] != self.input_len \
                or data.shape[1] != self.input_channel:
            raise ValueError(
                f'wrong input shape, expect ({self.input_len}, '
                f'{self.input_channel}) but get {data.shape}')

        rotate_angles = np.array([
            np.random.uniform(*self.rotate_x_range),
            np.random.uniform(*self.rotate_y_range),
            np.random.uniform(*self.rotate_z_range)
        ]) * np.random.choice([-1, 1], size=3)

        for i in range(0, self.input_channel, 3):
            data[:, i:i + 3] = Rotate.rotate(data[:, i:i + 3], *rotate_angles)

        return data

    @staticmethod
    def rotate(data, rotate_x, rotate_y, rotate_z):
        """
        Rotate an array
        :param data: shape (time step, 3)
        :param rotate_x, rotate_y, rotate_z: angle in RADIAN
        :return: array shape (time step, 3)
        """
        cos_x = np.cos(rotate_x)
        sin_x = np.sin(rotate_x)
        cos_y = np.cos(rotate_y)
        sin_y = np.sin(rotate_y)
        cos_z = np.cos(rotate_z)
        sin_z = np.sin(rotate_z)

        # create rotation filters
        rotate_x = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        rotate_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        rotate_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])

        # rotate original data by multiply it with rotation filters
        rotate_filters = np.matmul(np.matmul(rotate_x, rotate_y), rotate_z)
        data = np.matmul(rotate_filters, data.T).T
        return data
