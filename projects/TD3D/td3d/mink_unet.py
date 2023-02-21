# Copyright (c) OpenMMLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/minkunet.py # noqa
# and mmdet3d/models/backbones/mink_resnet.py
from typing import List, Tuple, Union

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    # blocks are used in the static part of MinkResNet
    ME = BasicBlock = Bottleneck = SparseTensor = None

import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TD3DMinkUNet(BaseModule):
    r"""Minkowski UNet. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.
    MinkUNet class in mmdet3d may have different interface in
    the future so we call it TD3DMinkUNet for now.

    Args:
        depth (int): Depth of resnet, from {14, 18, 34, 50, 101}.
        in_channels (int): Number of input channels, 3 for RGB.
        num_stages (int): Resnet stages. Defaults to 4.
        pool (bool): Whether to add max pooling after first conv.
            Defaults to True.
        num_planes (tuple[int]): Number of planes per block before
            block.expansion. First half is for down blocks and
            last is for up blocks. Defaults to 
            (32, 64, 128, 256, 128, 128, 96, 96).
    """
    arch_settings = {
        14: (BasicBlock, (1, 1, 1, 1, 1, 1, 1, 1)), 
        18: (BasicBlock, (2, 2, 2, 2, 2, 2, 2, 2)),
        34: (BasicBlock, (2, 3, 4, 6, 2, 2, 2, 2)),
        50: (Bottleneck, (2, 3, 4, 6, 2, 2, 2, 2)),
        101: (Bottleneck, (2, 3, 4, 23, 2, 2, 2, 2))
    }

    def __init__(self,
                 depth: str,
                 in_channels: int,
                 out_channels: int,
                 num_planes: Tuple[int] =
                    (32, 64, 128, 256, 128, 128, 96, 96)):
        super(TD3DMinkUNet, self).__init__()
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )
        assert num_planes % 2 == 0
        height = num_planes // 2
        self.height = height
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        block, stage_blocks = self.arch_settings[depth]

        init_channels = 32
        self.inplanes = init_channels
        self.in_conv = self._make_conv(
            in_channels, self.inplanes, kernel_size=5)
        
        for i in range(height):
            self.__setattr__(
                f'down_conv_{i}',
                self._make_conv(self.inplanes, self.inplanes, kernel_size=2,
                    stride=2))
            self.__setattr__(
                f'down_block_{i}',
                self._make_layer(block, num_planes[i], stage_blocks[i]))
        for i in range(height):
            self.__setattr__(
                f'up_conv_{height - 1 - i}',
                self._make_conv(self.inplanes, self.inplanes, kernel_size=2,
                    stride=2, transpose=True))
            down_channels = num_planes[height - 2 - i] * block.expansion \
                if i != 0 else init_channels
            self.inplanes = num_planes[height + i] + down_channels
            self.__setattr__(
                f'up_block_{height - 1 - i}',
                self._make_layer(block, num_planes[num_planes // 2 + i],
                    stage_blocks[height + i]))
        self.out_conv = ME.MinkowskiConvolution(
            num_planes[-1] * block.expansion, out_channels, kernel_size=1,
            bias=True, dimension=3)
        

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int,
                    blocks: int, stride: int) -> nn.Module:
        """Make single level of residual blocks.

        Args:
            block (BasicBlock | Bottleneck): Residual block class.
            planes (int): Number of convolution filters.
            blocks (int): Number of blocks in the layers.
            stride (int): Stride of the first convolutional layer.

        Returns:
            nn.Module: With residual blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_conv(in_channels: int,
                   out_channels: int,
                   transpose: bool = False,
                   kernel_size: int = 1,
                   stride: int = 1) -> nn.Module:
        """Construct Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            transpose (bool): Use transposed convolution if True.
                Defaults to False.
            kernel_size (int): Kernel size. Defaults to 3.
            stride (int): Stride of the convolution. Defaults to 1.
        
        Returns:
            torch.nn.Module: With corresponding layers.
        """
        conv = ME.MinkowskiConvolutionTranspose if transpose \
            else ME.MinkowskiConvolution
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=3), ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    def forward(self, x: SparseTensor) -> SparseTensor:
        """Forward pass of UNet.

        Args:
            x (SparseTensor): Input sparse tensor.

        Returns:
            SparseTensor: Output sparse tensor.
        """
        x = self.in_conv(x)
        outs = [x]
        for i in range(self.height):
            x = self.__getattr__(f'down_conv_{i}')(x)
            x = self.__getattr__(f'down_block_{i}')(x)
            if i != self.height - 1:
                outs.append(x)
        for i in range(self.height - 1, -1, -1):
            x = self.__getattr__(f'up_conv_{i}')(x)
            x = ME.cat(x, outs[i])
            x = self.__getattr__(f'up_block_{i}')(x)
        x = self.out_conv(x)
        return x
