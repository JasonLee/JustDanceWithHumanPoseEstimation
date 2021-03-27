import glob
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import PIL
from utils import loadmat
import os
from torchvision import transforms
from utils import gaussian_heatmap

import cv2
import numpy as np
from math import exp

class MPIIDataset(Dataset):
    # TODO: change size
    IMAGE_SIZE = 368
    JOINT_LEN = 16

    def __init__(self, image_path, annotation_path):
        # self.images = sorted(glob.glob(image_path + '/*.*'))
        annotations = loadmat(annotation_path)
        annotations = annotations['RELEASE']
        annolist = annotations['annolist']

        self.annotated_images = []
        self.joint_arr = []
        self.root = image_path
        self.transform = transforms.Compose([transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE), PIL.Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[128.0, 128.0, 128.0],
                                                                 std=[256.0, 256.0, 256.0])
                                            ])    

        for i in range(len(annolist)):
            anno_list_i = annolist[i]
            annorect = anno_list_i['annorect']

            # One person in image 
            if isinstance(annorect, dict):
                if "annopoints" in annorect:
                    image_name = anno_list_i['image']['name']
                    joints = annorect['annopoints']['point']

                    # 16 Joints
                    # Joints recognized
                    joint = sorted(joints, key=lambda k: k['id'])
                    
                    if len(joint) <= 8:
                        continue
                    
                    # sort based on joint id
                    self.joint_arr.append(sorted(joints, key=lambda k: k['id']))

                    self.annotated_images.append({'name': image_name})


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.annotated_images[index]['name']))
        old_width, old_height = img.size
        new_image = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        
        joint_heatmaps = []
        tensor_image = self.transform(img)
        joint_coords = []
        
        # Always get 16 joints 
        for i in range(16):
            if len(self.joint_arr[index]) < 16 and i >= len(self.joint_arr[index]):
                joint_coords.append([-1, -1])
                joint_heatmaps.append(np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE)))
                continue
            
            x = self.joint_arr[index][i]['x']
            y = self.joint_arr[index][i]['y']

            # Adjust joint locations
            x = round(x * self.IMAGE_SIZE / old_width)
            y = round(y * self.IMAGE_SIZE / old_height)

            joint_coords.append([x, y])
            joint_heatmaps.append(gaussian_heatmap((x, y), self.IMAGE_SIZE, 3))

        joint_heatmaps = np.array(joint_heatmaps, dtype=np.float32)
        joint_coords = np.array(joint_coords, dtype=int)
        return tensor_image, joint_heatmaps, joint_coords

    def __len__(self):
        """ Returns length of the dataset """
        return len(self.annotated_images)

def getTrainingValidationDataLoader(image_path, anno_path, split=0.7, batch_size=4, shuffle=True, num_workers=2, drop_last=True):
    fulldata = MPIIDataset(image_path, anno_path)
    split_len = int(len(fulldata) * split)

    train_split = list(range(0, split_len))
    val_split = list(range(split_len + 1, len(fulldata), 2))

    train_set = Subset(fulldata, train_split)
    val_set = Subset(fulldata, val_split)

    training = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    validation = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    return training, validation




# image_path = "D:\Downloads - SSD\mpii_human_pose_v1\images"
# anno_path = "D:\Downloads - SSD\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat"

# dataset = MPIIDataset(image_path, anno_path)
# print(len(dataset))

# annotations = loadmat("D:\Downloads - SSD\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat")
# annotations = annotations['RELEASE']
# annolist = annotations['img_train']
# print(sum(annolist))
