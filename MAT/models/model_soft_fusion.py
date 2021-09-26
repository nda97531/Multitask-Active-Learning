import torch as tr
import torch.nn as nn


class SoftFusionOneSigmoid_lua(nn.Module):
    def __init__(self,
                 n_classes,
                 model_i1, model_i2, freeze_encoder=False,
                 input_size=128,
                 feature_size=128,
                 freeze_softmask=False,
                 last_fc_drop_rate=0.):
        super(SoftFusionOneSigmoid_lua, self).__init__()

        self.model_i1 = model_i1
        self.model_i2 = model_i2

        self.softmask = nn.Sequential(
            nn.Linear(input_size * 2, feature_size * 2),
            nn.Sigmoid()
        )

        if type(n_classes) is int:
            self.classifier = nn.Sequential(*[
                nn.Dropout(last_fc_drop_rate),
                nn.Linear(feature_size*2, n_classes),
                nn.Softmax(dim=-1)
            ])
        elif type(n_classes) in {tuple, list}:
            self.classifier = nn.ModuleList([
                nn.Sequential(*[
                    nn.Dropout(last_fc_drop_rate),
                    nn.Linear(feature_size*2, n_classes[0]),
                    nn.Softmax(dim=-1)
                ]),
                nn.Sequential(*[
                    nn.Dropout(last_fc_drop_rate),
                    nn.Linear(feature_size*2, n_classes[1]),
                    nn.Softmax(dim=-1)
                ])
            ])

        if freeze_encoder:
            for param in self.model_i1.parameters():
                param.requires_grad = False
            for param in self.model_i2.parameters():
                param.requires_grad = False

        if freeze_softmask:
            for param in self.softmask.parameters():
                param.requires_grad = False

    def forward(self, x, classifier_idx: int = None, multitask_mask=None):
        x1, x2 = x

        x1 = self.model_i1(x1)
        x2 = self.model_i2(x2)

        x = tr.cat([x1, x2], dim=-1)

        soft_mask = self.softmask(x)

        x = x * soft_mask

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
