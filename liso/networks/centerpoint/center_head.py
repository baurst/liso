import copy
from typing import Dict

import torch
from liso.networks.centerpoint.weight_init import kaiming_init
from torch import nn


class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        norm_cfg,
        head_conv=64,
        final_kernel=1,
        bn=False,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = []
            for _ in range(num_conv - 1):
                fc.append(
                    nn.Conv2d(
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    )
                )
                if bn:
                    fc.append(nn.BatchNorm2d(head_conv, **norm_cfg))
                fc.append(nn.ReLU())

            fc.append(
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )
            fc = torch.nn.Sequential(*fc)

            for m in fc.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = {}
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(
        self,
        common_heads: Dict,
        norm_cfg,
        in_channels=(128,),
        stride=1,
        share_conv_channel=64,
    ):
        super(CenterHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = 1

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                share_conv_channel,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(share_conv_channel, **norm_cfg),
            nn.ReLU(inplace=True),
        )

        self.tasks = nn.ModuleList()

        heads = copy.deepcopy(common_heads)
        self.tasks.append(
            SepHead(
                share_conv_channel,
                heads,
                norm_cfg=norm_cfg,
                bn=True,
                final_kernel=3,
            )
        )

    def forward(self, x, *kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))
        assert len(ret_dicts) == 1, len(ret_dicts)
        return ret_dicts[0]
