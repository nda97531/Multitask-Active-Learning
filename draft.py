import os
import numpy as np
from MAT import DEVICE
from MAT.augment import numpy_augment
from MAT.dataset.data_generator import DatasetWindowsFusion
from MAT.models.model_tcn_classifier import TCN
from MAT.common.train import mat_train

# load target data
target_data_folder = f'../Dataset/MotionSense'
# data: array shape [num windows, window length, num channels]
target_data = np.load(os.path.join(target_data_folder, "data_new_map.npy"))
# label: array shape [num windows, 2(subject id, categorical label)]
target_label = np.load(os.path.join(target_data_folder, "p_lb_new_map.npy"))

# get training window indices by subject IDs
train_subjects = np.unique(np.linspace(1, 24, 4, False).astype(int))
train_window_idx = np.isin(target_label[:, 0], train_subjects)
# split dataset into train set and test set, remove subject IDs from label array
# data: array shape [num windows, window length, num channels]
# label: array shape [num windows, 1(categorical label)]
target_train_data = target_data[train_window_idx]
target_train_label = target_label[train_window_idx, 1:]
target_valid_data = target_data[~train_window_idx]
target_valid_label = target_label[~train_window_idx, 1:]

# set weight and mask placeholder for instances from target domain
# array shape [num windows, 3(label, weight, mask)]
target_train_label = np.concatenate([
    target_train_label,
    np.empty([len(target_train_label), 2])
], axis=1)

# load source dataset
source_data_folder = '../Dataset/MobiActV2'

pick_index = np.load(os.path.join(source_data_folder, 'mobiactv2_r2_pick_index_one-tenth.npy'))
# data: array shape [num windows, window length, num channels]
source_data = np.load(os.path.join(source_data_folder, "data_new.npy"))[pick_index]
# label: array shape [num windows, 1(categorical label)]
source_label = np.load(os.path.join(source_data_folder, "p_lb_ss_new.npy"))[pick_index, 2:3]

# set weight and mask placeholder for instances from source domain
# array shape [num windows, 3(label, weight, mask)]
source_label = np.concatenate([
    source_label,
    np.empty([len(source_label), 2])
], axis=1)

# calculate source and target weights
total_train_set_len = len(target_train_label) + len(source_label)
source_weight = total_train_set_len / len(source_label)
gamma = 1
target_weight = (total_train_set_len / len(target_train_label)) * gamma
# set sample weight for source and target sets
source_label[:, 1] = source_weight
target_train_label[:, 1] = target_weight

# set multi-task mask
source_label[:, 2] = 0  # source mask
target_train_label[:, 2] = 1  # target mask

# combine target and source train sets into 1
train_data = np.concatenate([target_train_data, source_data])
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

# create model object
n_class_source = len(np.unique(source_label[:, 0]))
n_class_target = len(np.unique(target_train_label[:, 0]))
model = TCN(
    n_classes=[n_class_source, n_class_target],
    how_flatten="spatial attention gap",
    n_tcn_channels=(64,) * 6 + (128,) * 2,
    tcn_kernel_size=2,
    dilation_base=2,
    tcn_droprate=0.2,
    use_spatial_dropout=False,
    n_fc_layers=1,
    fc_droprate=0.5,
    use_init_batchnorm=True
).to(DEVICE)

mat_train(
    model,
    train_set=train_set,
    valid_set=valid_set,
    weights_save_name="param/demo/mat_motionsense",
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
