import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


# Get feature maps from ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self._resnet101 = models.resnet101(pretrained=True)

        self.resnet101a = self._get_subset_resnet(-5)
        self.resnet101b = self._get_subset_resnet(-2)

    # Remove layers to get feature map to get channel = 256
    def _get_subset_resnet(self, from_last):
        modules = list(self._resnet101.children())[:from_last]
        resnet101 = nn.Sequential(*modules)
                
        for p in resnet101.parameters():
            p.requires_grad = False
        
        return resnet101

    # first block, last block from resnet
    def forward(self, x):
        return self.resnet101a(x), self.resnet101b(x)

if __name__ == "__main__":
    model = ResNet()
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    
    # Expecting 256 and 2048 channels respectively
    print(output.size())
    print(low_level_feat.size())
    