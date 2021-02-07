import torch.nn as nn
import torch.optim
from tqdm import tqdm
import pandas as pd

from torch.utils.data import DataLoader
from dataloader import getTrainingValidationDataLoader
from unipose import UniPose
from utils import evaluation_pckh

class Trainer():
    BATCH_SIZE = 16
    BEST_PCKH = -10000
    training_loss_data = []
    validation_loss_data = []
    mPCKh_data = []

    def __init__(self):
        image_path = "D:\Downloads - SSD\mpii_human_pose_v1\images"
        anno_path = "D:\Downloads - SSD\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat"

        self.training_dataloader, self.validation_dataloader = getTrainingValidationDataLoader(image_path=image_path, anno_path=anno_path, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=1, drop_last=True)

        self.model = UniPose()
        self.model.cuda()

        self.criterion = nn.MSELoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(100):
            self.epoch = epoch
            print("Epoch", epoch)
            self.train()
            self.validate()

                # Saves data into a csv file.
        df = pd.DataFrame(data={'training_loss': self.training_loss_data, 'validation_loss': self.validation_loss_data, 'mPCKh': self.mPCKh_data})
        df.to_csv('training_results.csv', index_label='Epoch')


    def train(self):
        tbar = tqdm(self.training_dataloader)
        self.model.train()
        train_loss = 0.0


        for i, (image, heatmaps, coords) in enumerate(tbar):
            image = image.cuda()
            heatmaps = heatmaps.cuda()
            self.optimizer.zero_grad() 

            out_heatmaps = self.model(image)

            loss = self.criterion(out_heatmaps, heatmaps)

            train_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()

            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1) * self.BATCH_SIZE)))

        self.training_loss_data.append(train_loss / len(self.training_dataloader))
        

    def validate(self):
        tbar = tqdm(self.validation_dataloader)
        self.model.eval()
        val_loss = 0.0
        mPCKH = 0

        for i, (image, heatmaps, coords) in enumerate(tbar):

            image = image.cuda()
            heatmaps = heatmaps.cuda()
            self.optimizer.zero_grad()

            out_heatmaps = self.model(image)
            loss = self.criterion(out_heatmaps, heatmaps)

            val_loss += loss.item()

            PCKH = evaluation_pckh(out_heatmaps.detach().cpu().numpy(), coords.numpy())
            mPCKH += PCKH

            tbar.set_description('Validation loss: %.6f, mPCKh: %.6f' % (val_loss * self.BATCH_SIZE / (i + 1), (mPCKH * self.BATCH_SIZE / (i+1))))
        
        self.validation_loss_data.append(val_loss / len(self.validation_dataloader))
        self.mPCKh_data.append(mPCKH)

        if self.BEST_PCKH < mPCKH:
            self.BEST_PCKH = mPCKH
            # print("Data", self.training_loss_data)

            torch.save({
                'epoch': i,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, 'models/model_%d.pth' % self.epoch)

        
if __name__ == '__main__':
    start = Trainer()
