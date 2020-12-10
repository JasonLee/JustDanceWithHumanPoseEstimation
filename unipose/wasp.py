import torch
import torch.nn as nn
import torch.nn.functional as F

class _AtrousModule(nn.Module):
    def __init__(self,in_channels , dilation):
        super(_AtrousModule, self).__init__()

        self.atrous_conv = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, dilation=dilation)
        # TODO: Research why batch Norms are used
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.atrous_conv(x)
        return self.relu(x)


class WASP(nn.Module):

    def __init__(self):
        super(WASP, self).__init__()
        self.rate = [6, 12, 18, 24]
        
        # 2048 is from first block of ResNet
        self.atrous1 = _AtrousModule(2048, self.rate[0])
        self.atrous2 = _AtrousModule(256, self.rate[1])
        self.atrous3 = _AtrousModule(256, self.rate[2])
        self.atrous4 = _AtrousModule(256, self.rate[3])

        # TODO: This might share weights which is bad
        self.conv1and1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.ReLU()
        )
        
        # kernel size?
        self.avg_pool = nn.AvgPool2d(kernel_size=3)

    def forward(self, x):
        x1 = self.atrous1(x)
        x2 = self.atrous2(x1)
        x3 = self.atrous3(x2)
        x4 = self.atrous4(x3)

        x1 = self.conv1and1(x1)
        x2 = self.conv1and1(x2)
        x3 = self.conv1and1(x3)
        x4 = self.conv1and1(x4)

        avg_pool = self.avg_pool(x)

        # Check channels after pooling, may have shrink

        return torch.cat(x1, x2, x3, x4, avg_pool)

if __name__ == "__main__":
    net = WASP()
    print(net)