from typing import Union, List
import torch as tr
import numpy as np
from torch.utils.data import Dataset
from MAT.augment.numpy_augment import Augmenter
from MAT import DEVICE


class DatasetWindowsFusion(Dataset):
    def __init__(self,
                 data: List[np.ndarray],
                 label: np.ndarray,
                 augment_rate: Union[float, List[float]] = 0.,
                 augmenter: Union[Augmenter, List[Augmenter]] = None):
        """
        :param data: a list or tuple, each element is a numpy array (timestep, channel) containing data of one modality
        :param label: a numpy array
        :param augment_rate: float or list/tuple of floats that has the same length as the number of modals.
        If float, then apply for all modals.
        :param augmenter: an Augmenter object or list/tuple of Augmenter objects.
        If Augmenter, then apply for all modals.
        """
        if type(augment_rate) is list or type(augment_rate) is tuple:
            if type(augmenter) is not list and type(augmenter) is not tuple:
                raise ValueError('augment_rate is list/tuple, augmenter must also be list/tuple')
            self.separated_augmentation = True
        else:
            if type(augmenter) is list or type(augmenter) is tuple:
                raise ValueError('augment_rate is not list/tuple, augmenter must not be list/tuple')
            self.separated_augmentation = False

        self.data = data
        for i in range(len(self.data)):
            self.data[i].flags.writeable = False
        self.label = tr.from_numpy(label).to(DEVICE)
        self.augment_rate = augment_rate
        self.augmenter = augmenter

    def __getitem__(self, index):
        data = []
        # same augmenter for all modals
        if not self.separated_augmentation and np.random.rand() < self.augment_rate:
            for i in range(len(self.data)):
                data.append(tr.from_numpy(
                    self.augmenter.apply(self.data[i][index])
                ).float().permute([1, 0]).to(DEVICE))

        # unique augmenter for each modal
        elif self.separated_augmentation:
            for i in range(len(self.augment_rate)):
                if np.random.rand() < self.augment_rate[i]:
                    data.append(tr.from_numpy(
                        self.augmenter[i].apply(self.data[i][index])
                    ).float().permute([1, 0]).to(DEVICE))
                else:
                    data.append(tr.from_numpy(
                        self.data[i][index]
                    ).float().permute([1, 0]).to(DEVICE))

        # no augmentation
        else:
            for i in range(len(self.data)):
                data.append(tr.from_numpy(
                    self.data[i][index]
                ).float().permute([1, 0]).to(DEVICE))

        label = self.label[index]
        return data, label

    def __len__(self):
        return self.label.shape[0]


class DatasetWindows(Dataset):
    def __init__(self, data, label, augment_rate=0., augmenter: Augmenter = None):
        """
        :param data: a numpy array (timestep, channel)
        :param label: a numpy array
        :param augment_rate: float from 0 to 1
        :param augmenter: Augmenter
        """
        self.data = data
        self.data.flags.writeable = False
        self.label = tr.from_numpy(label).to(DEVICE)
        self.augment_rate = augment_rate
        if augment_rate > 0.:
            self.augmenter = augmenter

    def __getitem__(self, index):
        data = self.data[index]
        if np.random.rand() < self.augment_rate:
            data = self.augmenter.apply(data)

        data = tr.from_numpy(data).float().permute([1, 0]).to(DEVICE)
        label = self.label[index]

        return data, label

    def __len__(self) -> int:
        return self.label.shape[0]
