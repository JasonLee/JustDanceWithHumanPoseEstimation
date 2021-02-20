import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageDraw

from unipose import UniPose
from utils import local_maxima
from dataloader import MPIIDataset

class Tester():
    def __init__(self):
        # self.start()
        self.test_from_dataset()

    def start(self):
        self.model = UniPose()
        self.model.cuda()

        checkpoint = torch.load('models/best.pth')
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

    def test_from_dataset(self):
        model = UniPose()
        model.cuda()
        model.eval()

        checkpoint = torch.load('models/best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        image_path = "D:\Downloads - SSD\mpii_human_pose_v1\images"
        anno_path = "D:\Downloads - SSD\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat"

        dataset = MPIIDataset(image_path, anno_path)

        image, heatmaps, coords = dataset[0]
 
        tran = transforms.ToPILImage()
        pil_image = tran(image)

        image = torch.unsqueeze(image, 0)
        image = image.cuda()
        out_heatmaps = model(image)

        heatmaps = np.expand_dims(heatmaps, 0)
        heatmap_points = local_maxima(heatmaps)
        outheatmap_points = local_maxima(out_heatmaps.detach().cpu().numpy())

        self.draw_heatmaps(out_heatmaps.detach().cpu().numpy())
        self.draw_on_image(pil_image, coords, heatmap_points, outheatmap_points)

    def draw_heatmaps(self, points):
        import matplotlib.pyplot as plt
        print(points[0].shape)
        plt.imshow(points[0,1], cmap='hot', interpolation='nearest')

        plt.show()


    def draw_on_image(self, image, points, heatmap_points, outheatmap_points):
        draw = ImageDraw.Draw(image)

        print(points.shape)
        print(heatmap_points.shape)
        print(outheatmap_points.shape)


        # Overlaps
        for i,(row, col) in enumerate(points):
            draw.text((row, col), str(i), fill=(255, 255, 255, 128))

        for i,(row, col) in enumerate(heatmap_points[0]):
            draw.text((row, col), str(i), fill=(255, 0, 0, 128))

        for i,(row, col) in enumerate(outheatmap_points[0]):
            draw.text((row, col), str(i), fill=(0, 0, 255, 128))

        image.save('test.png')
        
if __name__ == '__main__':
    start = Tester()
