import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.resnet_arch import ResnetFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class TextPerceptualLoss(nn.Module):
    """ Text Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        model (nn.Module): The model used as feature extractor.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 model : nn.Module,
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(TextPerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights

        self.model = model

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def foward(self, x, gt):
        # TODO: implement with layer weights

        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract text features
        x_features = self.resnet(x)
        gt_features = self.resnet(gt.detach())

        if self.perceptual_weight > 0:
            if self.criterion_type == 'fro':
                percep_loss = torch.norm(x_features - gt_features, p='fro')
            else:
                percep_loss = self.criterion(x_features, gt_features)
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        if self.style_weight > 0:
            if self.criterion_type == 'fro':
                style_loss = torch.norm(self._gram_mat(x_features) - self._gram_mat(gt_features), p='fro')
            else:
                style_loss = self.criterion(self._gram_mat(x_features), self._gram_mat(gt_features))
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

