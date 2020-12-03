import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


# Get feature maps from ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        
        modules=list(resnet101.children())[:-1]

        resnet101 = nn.Sequential(*modules)

        for p in resnet101.parameters():
            p.requires_grad = False

        self.resnet101 = resnet101
        
    def forward(self, x):
        return self.resnet101(x)

resnet = ResNet()

print(resnet)
