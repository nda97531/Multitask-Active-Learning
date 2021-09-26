from typing import Union
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import math
from MAT.models.conv_with_associated import Conv1dWithAssociated


class ResTempBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size, dilation, droprate=0.2, use_spatial_dropout=False,
                 n_conv_layers=2):
        super(ResTempBlock, self).__init__()

        org_channel_in = n_channels_in

        block = []
        for i in range(n_conv_layers):
            if i == n_conv_layers - 1:
                droprate = 0.

            block += [Conv1dWithAssociated(in_filters=n_channels_in,
                                           out_filters=n_channels_out,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           padding=((kernel_size - 1) * dilation, 0),
                                           dilation=dilation,
                                           droprate=droprate,
                                           use_spatial_dropout=use_spatial_dropout), ]
            if i == 0:
                n_channels_in = n_channels_out

        self.block = nn.Sequential(*block)

        self.downsample = nn.Conv1d(org_channel_in, n_channels_out, 1) if org_channel_in != n_channels_out else None

    def forward(self, x):
        out = self.block(x)
        res = x if self.downsample is None else self.downsample(x)
        return tr.relu(out + res)


class TCN(nn.Module):
    def __init__(self,
                 n_classes: Union[int, tuple, list],
                 how_flatten,
                 n_tcn_channels=(64,) * 5 + (128,) * 2,  # type: list, tuple
                 tcn_kernel_size=2,  # type:int
                 dilation_base=2,  # type:int
                 tcn_droprate=0.2,  # type: float
                 use_spatial_dropout=False,
                 n_fc_layers=1,
                 fc_droprate=0.7,  # type: float
                 use_init_batchnorm=False,
                 use_last_fc=True,
                 freeze_feature_extractor=False
                 ):
        super(TCN, self).__init__()

        if type(n_classes) in {tuple, list} and len(n_classes) != 2:
            raise ValueError("only support 2 outputs for multitask case")

        self.B = len(n_tcn_channels)
        self.N = 2
        self.k = tcn_kernel_size

        self.use_init_batchnorm = use_init_batchnorm
        self.use_last_fc = use_last_fc

        if use_init_batchnorm:
            self.batchnorm_raw = nn.BatchNorm1d(3)

        # create TCN layers
        layers = []
        for i in range(len(n_tcn_channels)):
            dilation_rate = dilation_base ** i
            in_channels = 3 if i == 0 else n_tcn_channels[i - 1]
            out_channels = n_tcn_channels[i]

            layers.append(ResTempBlock(
                n_channels_in=in_channels,
                n_channels_out=out_channels,
                kernel_size=tcn_kernel_size,
                dilation=dilation_rate,
                droprate=tcn_droprate,
                use_spatial_dropout=use_spatial_dropout,
                n_conv_layers=2
            ))
        self.feature_extractor = nn.Sequential(*layers)

        # create flatten layer
        if how_flatten == "last time step":
            self.flatten = lambda x: x[:, :, -1]
        elif how_flatten == "gap":
            self.flatten = lambda x: F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        elif "attention gap" in how_flatten:
            if how_flatten == "channel attention gap":
                self.cbam = CBAM(n_tcn_channels[-1],
                                 apply_channel_att=True, apply_spatial_att=False,
                                 reduction_ratio=math.sqrt(n_tcn_channels[-1])
                                 )
            elif how_flatten == "spatial attention gap":
                self.cbam = CBAM(n_tcn_channels[-1],
                                 apply_channel_att=False, apply_spatial_att=True,
                                 reduction_ratio=math.sqrt(n_tcn_channels[-1])
                                 )
            else:
                self.cbam = CBAM(n_tcn_channels[-1],
                                 apply_channel_att=True, apply_spatial_att=True,
                                 reduction_ratio=math.sqrt(n_tcn_channels[-1])
                                 )
            self.flatten = lambda x: F.adaptive_avg_pool1d(self.cbam(x), 1).squeeze(-1)
        else:
            raise ValueError("how_flatten must be 'last time step'/'gap'/'attention gap'/'channel attention gap'/"
                             "'spatial attention gap'")

        # add FC layers
        fc = []
        for i in range(n_fc_layers - 1):
            fc += [
                nn.Dropout(fc_droprate),
                nn.Linear(n_tcn_channels[-1], n_tcn_channels[-1]),
                nn.ReLU()
            ]
        self.FCs = nn.Sequential(*fc)

        # add the last FC layer (2 layers if multi-task)
        if type(n_classes) is int:
            self.classifier = nn.Sequential(
                nn.Dropout(fc_droprate),
                nn.Linear(n_tcn_channels[-1], n_classes)
            )
        elif type(n_classes) in {tuple, list}:
            self.classifier = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(fc_droprate),
                    nn.Linear(n_tcn_channels[-1], n_classes[0])
                ),
                nn.Sequential(
                    nn.Dropout(fc_droprate),
                    nn.Linear(n_tcn_channels[-1], n_classes[1])
                )
            ])

        if freeze_feature_extractor:
            for param in self.batchnorm_raw.parameters():
                param.requires_grad = False
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x, classifier_idx: int = None, multitask_mask=None):
        """
        :param x: [batch, channel, ...]
        :param classifier_idx: if this index is specified, only use the corresponding classifier layer
        :param multitask_mask: an 1d torch tensor of booleans with len = batch size,
            'True' for samples with num class being the first element in n_classes list, 'False' for the second one.
            only used if classifier_idx is None
        :return:

        """
        if self.use_init_batchnorm:
            x = self.batchnorm_raw(x)

        x = self.feature_extractor(x)
        # batch size, feature, time step

        x = self.flatten(x)
        # batch size, feature

        if not self.use_last_fc:
            return x

        x = self.FCs(x)

        if type(self.classifier) is nn.ModuleList:
            if classifier_idx is not None:
                x = self.classifier[classifier_idx](x)
            else:
                if multitask_mask.dtype is not tr.bool:
                    raise TypeError("multitask_mask must be a tensor of boolean values")
                x0 = self.classifier[0](x[~multitask_mask])
                x1 = self.classifier[1](x[multitask_mask])
                x = (x0, x1)
        else:
            x = self.classifier(x)

        return x

    def receptive_field(self):
        """
        :return: receptive field
        """
        return 1 + self.N * (self.k - 1) * (2 ** self.B - 1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio: float = 16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, int(gate_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(gate_channels // reduction_ratio), gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)
                channel_att_raw = self.mlp(max_pool)
            else:
                raise ValueError("pool_types must be avg/max")

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = tr.sigmoid(channel_att_sum).unsqueeze(-1)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()

        self.spatial = Conv1dWithAssociated(2, 1, kernel_size, padding=(kernel_size - 1) // 2,
                                            activation=nn.Identity())

    def forward(self, x):
        # input shape: [batch, channel, time]

        x_compress = tr.cat((tr.max(x, 1, keepdim=True)[0], tr.mean(x, 1, keepdim=True)), dim=1)  # channel pooling
        # shape: [batch, 2, time]

        x_out = self.spatial(x_compress)
        scale = tr.sigmoid(x_out)  # broadcasting
        # shape: [batch, 1, time]

        return x * scale  # shape: [batch, channel, time]


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio: float = 16, pool_types=('avg', 'max'),
                 apply_channel_att=True, apply_spatial_att=True):
        super(CBAM, self).__init__()

        att = []
        if apply_channel_att:
            att.append(ChannelGate(gate_channels, reduction_ratio, pool_types))
        if apply_spatial_att:
            att.append(SpatialGate())

        self.att = nn.Sequential(*att)

    def forward(self, x):
        x_out = self.att(x)
        return x_out


if __name__ == '__main__':
    # model = TCN(n_classes=13,
    #             n_tcn_channels=(128,) * 8,
    #             tcn_kernel_size=2,
    #             )
    # print(model.receptive_field())

    with tr.no_grad():
        layer = CBAM(16, 2)

        ten = tr.ones([1, 16, 10])

        ten = layer(ten)

        print(ten.shape)
