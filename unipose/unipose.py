import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder
from resnet import ResNet101
from wasp import WASP


class UniPose(nn.Module):
    def __init__(self, num_heatmap=16):
        super(UniPose, self).__init__()
        self.wasp = WASP()
        self.resnet = ResNet101(pretrained=True, output_stride=8)
        self.decoder = Decoder(num_heatmap)


    def forward(self, x):
        x, low_level_feat = self.resnet(x)
        wasp_x = self.wasp(x)
        x = self.decoder(wasp_x, low_level_feat)

        # Keypoints only
        return x

if __name__ == "__main__":
    model = UniPose()
    model.eval()
    input = torch.rand(1, 3, 384, 384)
    output = model(input)
    print(output.size())
    print(output.type())
    print(type(output))