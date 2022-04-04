#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from MAT.models.model_soft_fusion import SoftFusionOneSigmoid_lua

os.chdir('../')
import numpy as np
from MAT import DEVICE
from MAT.augment import numpy_augment
from MAT.data_generator import DatasetWindowsFusion
from MAT.models.model_tcn_classifier import TCN
from MAT.common.train import mat_train
from MAT import logger

# ## hyper-parameters

# In[2]:


gamma = 1.0

# ## load target data

# In[3]:


target_data_folder = f'../Dataset/MotionSense'

# data: array shape [num windows, window length, num channels]
# label: array shape [num windows, 1(categorical label)]
target_train_data = np.load(os.path.join(target_data_folder, "data_train.npy"))
target_train_label = np.load(os.path.join(target_data_folder, "p_lb_train.npy"))
target_valid_data = np.load(os.path.join(target_data_folder, "data_valid.npy"))
target_valid_label = np.load(os.path.join(target_data_folder, "p_lb_valid.npy"))

logger.info(f"Target train data: {target_train_data.shape}")
logger.info(f"Target train label: {target_train_label.shape}")
logger.info(f"Target valid data: {target_valid_data.shape}")
logger.info(f"Target valid label: {target_valid_label.shape}")

# ## load source data

# In[4]:


source_data_folder = '../Dataset/MobiActV2'

# data: array shape [num windows, window length, num channels]
source_data = np.load(os.path.join(source_data_folder, "data_new_picked.npy"))
# label: array shape [num windows, 2(subject id, categorical label, session ID)]
source_label = np.load(os.path.join(source_data_folder, "p_lb_ss_new_picked.npy"))

# only keep label column => shape [num windows, 1(categorical label)]
source_label = source_label[:, 1:2].astype(int)

logger.info(f"Source data: {source_data.shape}")
logger.info(f"Source label: {source_label.shape}")

# ## set weight and mask for each instance of source/target set

# In[5]:


# set weight and mask placeholder for instances from target domain
# array shape [num windows, 3(label, weight, mask)]
target_train_label = np.concatenate([
    target_train_label,
    np.empty([len(target_train_label), 2])
], axis=1)

# set weight and mask placeholder for instances from source domain
# array shape [num windows, 3(label, weight, mask)]
source_label = np.concatenate([
    source_label,
    np.empty([len(source_label), 2])
], axis=1)

# calculate source and target weights
total_train_set_len = len(target_train_label) + len(source_label)
source_weight = total_train_set_len / len(source_label)
target_weight = (total_train_set_len / len(target_train_label)) * gamma
# set sample weight for source and target sets
source_label[:, 1] = source_weight
target_train_label[:, 1] = target_weight

# set multi-task mask
source_label[:, 2] = 0  # source mask
target_train_label[:, 2] = 1  # target mask

# ## init dataset loader with augmentation

# In[6]:


# combine target and source train sets into 1
# data array shape [num windows, window length, num channels]
train_data = np.concatenate([target_train_data, source_data])
# label array shape [num windows, 3(label, weight, mask)]
train_label = np.concatenate([target_train_label, source_label])

# create Dataset object (split data into list of 2 modalities)
train_set = DatasetWindowsFusion(
    [train_data[:, :, :3], train_data[:, :, 3:]],
    train_label,
    augment_rate=0.5,
    augmenter=numpy_augment.Rotate(input_shape=[300, 3],
                                   rotate_x_range=[0., 20.],
                                   rotate_y_range=[0., 20.],
                                   rotate_z_range=[0., 20.])
)
valid_set = DatasetWindowsFusion(
    [target_valid_data[:, :, :3], target_valid_data[:, :, 3:]],
    target_valid_label
)

logger.info(f"Total train data: {train_data.shape}")
logger.info(f"Total train label: {train_label.shape}")
logger.info(f"Train set size: {len(train_set)}")
logger.info(f"Valid set size: {len(valid_set)}")

# ## init model

# In[7]:


# get number of classes in source/target dataset
n_class_source = len(np.unique(source_label[:, 0]))
n_class_target = len(np.unique(target_train_label[:, 0]))

# init model object
model1 = TCN(n_classes=None,  # type:int
             how_flatten="spatial attention gap",
             n_tcn_channels=(64,) * 6 + (128,) * 2,  # type: list, tuple
             tcn_kernel_size=2,  # type:int
             dilation_base=2,  # type:int
             tcn_droprate=0.2,  # type: float
             use_spatial_dropout=False,
             n_fc_layers=1,
             fc_droprate=0.5,  # type: float
             use_init_batchnorm=True,
             use_last_fc=False,
             )
model1 = model1.to(DEVICE)

model2 = TCN(n_classes=None,  # type:int
how_flatten="spatial attention gap",
             n_tcn_channels=(64,) * 6 + (128,) * 2,  # type: list, tuple
             tcn_kernel_size=2,  # type:int
             dilation_base=2,  # type:int
             tcn_droprate=0.2,  # type: float
             use_spatial_dropout=False,
             n_fc_layers=1,
             fc_droprate=0.5,  # type: float
             use_init_batchnorm=True,
             use_last_fc=False,
             )
model2 = model2.to(DEVICE)

model = SoftFusionOneSigmoid_lua(
    n_classes=[n_class_source, n_class_target],
    model_i1=model1,
    model_i2=model2,
    freeze_encoder=False,
    input_size=128,
    feature_size=128,
    last_fc_drop_rate=0.5,
    freeze_softmask=False
)
model = model.to(DEVICE)

logger.info(f"Number of source classes: {n_class_source}")
logger.info(f"Number of target classes: {n_class_target}")

# ## train

# In[8]:


mat_train(
    model,
    train_set=train_set,
    valid_set=valid_set,
    weights_save_name="results/mat_motionsense",
    only_save_best_of_best=True,
    save_before_early_stop=False,
    curve_save_name=None,
    learning_rate=1e-3,
    weight_decay=0.,
    batch_size=32,
    max_epoch=100,
    class_weight=None,
    patience=10
)
