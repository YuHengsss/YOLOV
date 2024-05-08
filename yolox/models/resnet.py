from torch import nn
from .network_blocks import BaseConv, ResNetBottleneck

ResNet_cfg = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ResNet(nn.Module):

    def __init__(self, block=ResNetBottleneck, depth=50, groups=1, width_per_group=64, act='relu', out_features=''):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.act = act
        self.stem = BaseConv(3, self.inplanes, 7, 2, groups, act=act)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.out_features = out_features

        layers = ResNet_cfg[depth]
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, )
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, )
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, )

    def make_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.act
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = {}
        conv1 = self.stem(inputs)
        x = self.maxpool(conv1)
        outputs["stem"] = x
        x = self.layer1(x)
        outputs["stage2"] = x
        x = self.layer2(x)
        outputs["stage3"] = x
        x = self.layer3(x)
        outputs["stage4"] = x
        x = self.layer4(x)
        outputs["stage5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
