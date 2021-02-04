import torch
import torch.nn as nn
import torch.nn.functional as F

class _AtrousModule(nn.Module):
    def __init__(self,in_channels , dilation, padding):
        super(_AtrousModule, self).__init__()

        self.atrous_conv = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class WASP(nn.Module):

    def __init__(self):
        super(WASP, self).__init__()
        self.rate = [6, 12, 18, 24]
        
        # 2048 is from first block of ResNet
        self.atrous1 = _AtrousModule(in_channels=2048, dilation=self.rate[0], padding=self.rate[0])
        self.atrous2 = _AtrousModule(in_channels=256, dilation=self.rate[1], padding=self.rate[1])
        self.atrous3 = _AtrousModule(in_channels=256, dilation=self.rate[2], padding=self.rate[2])
        self.atrous4 = _AtrousModule(in_channels=256, dilation=self.rate[3], padding=self.rate[3])

        # TODO: This might share weights which is bad
        self.conv1and1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        
        # Paper is misleading? Just says average pooling
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

    def forward(self, x):
        x1 = self.atrous1(x)
        x2 = self.atrous2(x1)
        x3 = self.atrous3(x2)
        x4 = self.atrous4(x3)

        x1 = self.conv1and1(x1)
        x2 = self.conv1and1(x2)
        x3 = self.conv1and1(x3)
        x4 = self.conv1and1(x4)

        avg_pool = self.global_avg_pool(x)
        # Get right size
        avg_pool = F.interpolate(avg_pool, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # Check channels after pooling, may have shrink
        # print("size", x1.size(), x2.size(), x3.size(), x4.size(), avg_pool.size())

        x = torch.cat((x1, x2, x3, x4, avg_pool), dim=1)

        
        x = self.conv1(x)
        x = self.relu(x)
        return x

if __name__ == "__main__":
    net = WASP()
    print(net)