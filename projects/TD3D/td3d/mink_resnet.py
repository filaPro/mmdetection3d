# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

from mmdet3d.models.backbones import MinkResNet
from mmdet3d.registry import MODELS


@MODELS.register_module()
class TD3DMinkResNet(MinkResNet):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details. The only difference
    with MinkResNet is the `norm`, `return_stem` and `stem_stride` parameters.
    These classes should be merged in the future.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input channels, 3 for RGB.
        num_stages (int): Resnet stages. Defaults to 4.
        pool (bool): Whether to add max pooling after first conv.
            Defaults to True.
        norm (str): Norm type ('instance' or 'batch') for stem layer.
            Usually ResNet implies BatchNorm but for some reason
            original MinkResNet implies InstanceNorm. Defaults to 'instance'.
        return_stem (bool): Whether to return stem conv output.
            Defaults to True.
        stem_stride (int): Stride of the stem conv layer. Defaults to 2.
    """

    def __init__(self,
                 depth: int,
                 in_channels: int,
                 num_stages: int = 4,
                 pool: bool = True,
                 norm: str = 'instance',
                 return_stem: bool = False,
                 stem_stride: int = 2):
        super(TD3DMinkResNet, self).__init__(depth, in_channels, num_stages,
                                             pool)
        self.return_stem = return_stem
        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=stem_stride,
            dimension=3)
        norm_layer = ME.MinkowskiInstanceNorm if norm == 'instance' else \
            ME.MinkowskiBatchNorm
        self.norm1 = norm_layer(self.inplanes)
    
    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        if self.return_stem:
            outs.append(x)
        if self.pool:
            x = self.maxpool(x)
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs
