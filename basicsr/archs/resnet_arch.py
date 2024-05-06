import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import resnet as resnet

from basicsr.utils.registry import ARCH_REGISTRY

NAMES = {
    'resnet101': [
        'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'
    ]
}

@ARCH_REGISTRY.register()
class ResnetFeatureExtractor(nn.Module):
    def __init__(self,
                 layer_name_list,
                 resnet_type='resnet101',
                 weights=None,
                 requires_grad=False):
        super(ResnetFeatureExtractor, self).__init__()
        self.layer_name_list = layer_name_list

        self.names = NAMES[resnet_type]

        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if weights is not None:
            assert os.path.exists(weights), f'weights {weights} does not exist.'
            state_dict = torch.load(weights)
            self.resnet = getattr(resnet, resnet_type)(weights=None)
            resnet.load_state_dict(state_dict)
        else:
            # Load pretrained model
            state_dict = None
            self.resnet = getattr(resnet, resnet_type)(pretrained=True)

        # Remove the last fc layers
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # TODO: Remove non-used layers


        # Freeze the model
        if not requires_grad:
            self.resnet.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.resnet.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)



