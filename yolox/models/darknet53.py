from torch import nn
from .network_blocks import Res_unit,ConvBnLeaky

class Darknet53(nn.Module):

    def __init__(
            self,
            in_channels = 3,
            stem_out_channel = 32,
            out_features = ('stage3','stage4','stage5')
    ):
        super().__init__()
        self.out_features = out_features
        self.blocknum = [1,2,8,8,4]
        self.stem = nn.Sequential(
            ConvBnLeaky(in_channels,out_channels=stem_out_channel,ksize=3,stride=1)
        )
        self.stage1 = nn.Sequential(
            *self.make_group_layer(in_channels=stem_out_channel,block_num=self.blocknum[0],stride=2)
        )#in 32 out 64
        self.stage2 = nn.Sequential(
            *self.make_group_layer(in_channels=2*stem_out_channel,block_num=self.blocknum[1],stride=2)
        )#in 64 out 128
        self.stage3 = nn.Sequential(
            *self.make_group_layer(in_channels=4 * stem_out_channel, block_num=self.blocknum[2], stride=2)
        )  # in 128 out 256
        self.stage4 = nn.Sequential(
            *self.make_group_layer(in_channels=8 * stem_out_channel, block_num=self.blocknum[3], stride=2)
        )  # in 256 out 512
        self.stage5 = nn.Sequential(
            *self.make_group_layer(in_channels=16 * stem_out_channel, block_num=self.blocknum[4], stride=2)
        )  # in 512 out 1024


    def make_group_layer(self,in_channels,block_num,stride=1):
        return [
            ConvBnLeaky(in_channels=in_channels,out_channels=in_channels*2,ksize=3,stride=stride),
            *[(Res_unit(in_channels*2)) for _ in range(block_num)]
        ]

    def forward(self,x):
        output = {}
        x = self.stem(x)
        output['stem'] = x
        x = self.stage1(x)
        output['stage1'] = x
        x = self.stage2(x)
        output['stage2'] = x
        x = self.stage3(x)
        output['stage3'] = x
        x = self.stage4(x)
        output['stage4'] = x
        x = self.stage5(x)
        output['stage5'] = x

        return {k:v for k,v in output.items() if k in self.out_features}

