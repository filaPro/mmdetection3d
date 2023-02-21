# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/td3d/blob/master/mmdet3d/models/necks/ngfc_neck.py # noqa
from typing import List, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TD3DNeck(BaseModule):
    r"""Neck of `TD3D <https://arxiv.org/abs/2302.02871>`_.

    Args:
        in_channels (tuple[int]): Number of channels in input tensors.
        out_channels (int): Number of channels in all output tensors
            except first.
        seg_out_channels (int): Number of channels in the first output tensor.
    """

    def __init__(self, in_channels: Tuple[int], out_channels: int,
                 seg_out_channels: int):
        super(TD3DNeck, self).__init__()
        self._init_layers(in_channels, out_channels, seg_out_channels)

    def _init_layers(self, in_channels: Tuple[int], out_channels: int,
                     seg_out_channels: int):
        """Initialize layers.

        Args:
            in_channels (tuple[int]): Number of channels in input tensors.
            out_channels (int): Number of channels in all output tensors
            except first.
            seg_out_channels (int): Number of channels in the first output tensor.
        """
        for i in range(len(in_channels)):
            if i > 0:
                stride = 4 if i == 1 else 2
                self.add_module(
                    f'up_block_{i}',
                    self._make_block(in_channels[i], in_channels[i - 1], True,
                                     stride))
            if len(in_channels) - 1 > i > 0:
                self.add_module(
                    f'lateral_block_{i}',
                    self._make_block(in_channels[i], in_channels[i]))
            n_channels = out_channels if i != 0 else seg_out_channels
            self.add_module(f'out_block_{i}',
                            self._make_block(in_channels[i], n_channels))

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: List[SparseTensor]) -> List[SparseTensor]:
        """Forward pass.

        Args:
            x (list[SparseTensor]): Features from the backbone.

        Returns:
            List[Tensor]: Output features from the neck.
        """
        outs = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                if i > 0:
                    x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)
            outs.append(out)
        return outs[::-1]

    @staticmethod
    def _make_block(in_channels: int,
                    out_channels: int,
                    transpose: bool = False,
                    stride: int = 1) -> nn.Module:
        """Construct Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            transpose (bool): Use transposed convolution if True.
                Defaults to False.
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
                kernel_size=3,
                stride=stride,
                dimension=3), ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))
