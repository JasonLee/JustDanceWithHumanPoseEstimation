import torch.nn as nn
import torch.optim
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageDraw

from unipose import UniPose
from utils import local_maxima

class Tester():
    def __init__(self):
        self.model = UniPose()
        self.model.cuda()

        checkpoint = torch.load('models/model_5.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        img = Image.open('image2.jpg')
        img = img.convert('RGB')
        points = self.test(img)
        # points = [[10, 10]]

        self.draw_on_image(img, points)


    def test(self, image):
        tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
        image = tran(image)
        image = torch.unsqueeze(image, 0)

        self.model.eval()
        val_loss = 0.0
        mPCKH = 0

        image = image.cuda()
        out_heatmaps = self.model(image)

        points = local_maxima(out_heatmaps.detach().cpu().numpy())

        return points


    def draw_on_image(self, image, points):
        draw = ImageDraw.Draw(image)
        size = 5

        for i,(col, row) in enumerate(points[0]):
            draw.text((row, col), str(i), fill=(255, 0, 0, 128))
            draw.text((col, row), str(i), fill=(0, 255, 0, 128))
            # draw.ellipse((row-size, col-size, row+size, col+size), fill = 'blue', outline ='blue')

        image.save('test.png')
        
if __name__ == '__main__':
    start = Tester()
