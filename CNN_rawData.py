import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# Hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--nb_classes', dtype=int, default=4)
parser.add_argument('--batch_size', dtype=int, default=128)
parser.add_argument('--input_channels', dtype=int, default=1)
parser.add_argument('--learning_rate', dtype=float, default=0.01)
parser.add_argument('--num_epoch', dtype=int, default=5)
parser.add_argument('--root_dataset', dtype=str, default='./dataset')
parser.add_argument('--trainset_path', dtype=str, default='train.xlsx')
parser.add_argument('--testset_path', dtype=str, default='test.xlsx')
config = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = os.path.join(config.root_dataset, config.trainset_path)
test_path = os.path.join(config.root_dataset, config.trainset_path)

class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        xy = pd.read_excel(dataset_path, header=None, dtype=np.float32)
        self.len = xy.shape[0]
        xyMatrix = xy.values # DataFrame to ndarray

        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

train_dataset = CustomDataset(train_path)
test_dataset = CustomDataset(test_path)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=config.batch_size,
                         shuffle=True)
        


class Model(nn.Module):
    def __init__(self, input_channels, nb_classes):
        super(Model, self).__init__()
        # Input Size : 1x128
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            
        )
    
    def forward(self, x):
        out = self.conv(x)
        out.shape
        out = self.fc(out)
        
        return out


model = Model(config.input_channels, config.nb_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

total_step = len(train_loader)
for epoch in range(config.num_epoch):
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, config.num_epoch, idx+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Test Accuracy: {}".format(100*correct/total))