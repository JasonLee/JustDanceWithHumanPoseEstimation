import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

# Get feature maps from ResNet
class Decoder(nn.Module):
    def __init__(self, num_heatmaps):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1, stride=1),
            nn.ReLU()
        )
            
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=304, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.Dropout(),
            nn.Conv2d(in_channels=256, out_channels=num_heatmaps, kernel_size=1, stride=1) 
        )
        
    def forward(self, wasp_feat, lowlvl_feat, input_size):
        lowlvl_feat = self.conv1(lowlvl_feat)
        lowlvl_feat = self.maxpool(lowlvl_feat)

        # Keep same size
        w = F.interpolate(wasp_feat, size=wasp_feat.shape[2:],mode='bilinear')

        x = torch.cat(lowlvl_feat, w)

        x = self.conv2(x)

        x = F.interpolate(x, size=input_size, mode='bilinear')