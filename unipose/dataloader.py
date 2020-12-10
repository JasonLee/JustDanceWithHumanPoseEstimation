import glob
from torch.utils.data import Dataset
from PIL import Image
from utils import loadmat
import os

import cv2
import numpy as np
from math import exp

class MPIIDataset(Dataset):
    # TODO: change size
    IMAGE_SIZE = 300

    def __init__(self, image_path, annotation_path):
        # self.images = sorted(glob.glob(image_path + '/*.*'))
        annotations = loadmat(annotation_path)
        annotations = annotations['RELEASE']
        annolist = annotations['annolist']

        self.annotated_images = []
        self.joint_arr = []
        self.root = image_path

        for i in range(len(annolist)):
            anno_list_i = annolist[i]
            annorect = anno_list_i['annorect']

            # One person in image 
            if isinstance(annorect, dict):
                if "annopoints" in annorect:
                    image_name = anno_list_i['image']['name']
                    joints = annorect['annopoints']['point']
                    
                    # sort based on joint id
                    self.joint_arr.append(sorted(joints, key=lambda k: k['id']))
                    self.annotated_images.append({'name': image_name})


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.annotated_images[index]['name']))
        old_width, old_height = img.size
        new_image = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        
        joint_heatmaps = []

        for i in range(len(self.joint_arr[index])):
            x = self.joint_arr[index][i]['x']
            y = self.joint_arr[index][i]['y']

            # Adjust joint locations
            x = round(x * self.IMAGE_SIZE / old_width)
            y = round(y * self.IMAGE_SIZE / old_height)

            self.joint_arr[index][i]['x'] = x
            self.joint_arr[index][i]['y'] = y


            joint_heatmaps.append(self.gaussian_heatmap((x, y), self.IMAGE_SIZE, 3))

        return {'image': new_image, 'joints': self.joint_arr[index], 'heatmaps': joint_heatmaps}

    def __len__(self):
        """ Returns length of the dataset """
        return len(self.annotated_images)
    
    def gaussian_heatmap(self, center=(2, 2), image_size=256, sig=1):
        """
        It produces single gaussian at expected center
        :param center:  the mean position (X, Y) - where high value expected
        :param image_size: The total image size (width, height)
        :param sig: The sigma value
        :return:
        """
        x_axis = np.linspace(0, image_size-1, image_size) - center[0]
        y_axis = np.linspace(0, image_size-1, image_size) - center[1]
        xx, yy = np.meshgrid(x_axis, y_axis)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

        return kernel



image_path = "D:\Downloads - SSD\mpii_human_pose_v1\images"
anno_path = "D:\Downloads - SSD\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat"

dataset = MPIIDataset(image_path, anno_path)
print(len(dataset))
dataset[0]
# print(dataset[0])

# annotations = loadmat("D:\Downloads - SSD\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat")
# annotations = annotations['RELEASE']
# annolist = annotations['img_train']
# print(sum(annolist))
