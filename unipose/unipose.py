import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder
from resnet import ResNet
from wasp import WASP


class UniPose(nn.Module):

    def __init__(self, num_heatmap):
        super(UniPose, self).__init__()
        self.wasp = WASP()
        self.resnet = ResNet()
        self.decoder = Decoder(num_heatmap)


    def forward(self, x):
        resnet_first, resnet_last = self.resnet(x)
        wasp_x = self.wasp(resnet_last)
        x = decoder(wasp_x, resnet_first, x.size()[2:])

