"""
U-net with different filter sizes: 42, 84, 168, 336
"""
import torch
from torchvision.transforms.functional import resize
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()

        self.in_c = in_c 
        self.out_c = out_c
        self.batchNorm = nn.BatchNorm2d(self.out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_c, self.out_c, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(self.out_c, self.out_c, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, feature_maps=[42, 84, 168, 336]):
        super(UNet, self).__init__()

        self.up_part = nn.ModuleList()
        self.down_part = nn.ModuleList()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)   

        self.name = "u_net_one"

        for feature in feature_maps:
            self.down_part.append(DoubleConv(in_c, feature))
            in_c = feature

        for feature in reversed(feature_maps):
            self.up_part.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.up_part.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(feature_maps[-1], feature_maps[-1] * 2)
        self.output = nn.Conv2d(feature_maps[0], out_c, kernel_size=1)

    def forward(self, x):

        skips = []
        for down in self.down_part:
            x = down(x)
            skips.append(x)
            x = self.maxPool(x)

        x = self.bottleneck(x)

        skips = list(reversed(skips))
        for i in range(0, len(self.up_part), 2):
            x = self.up_part[i](x)
            skip_connection = skips[i // 2]

            if x.shape != skip_connection.shape:
                x = resize(x, size=skip_connection.shape[2:])

            concat = torch.cat((skip_connection, x), dim=1)
            x = self.up_part[i+1](concat)

        output = self.output(x)
        return output

