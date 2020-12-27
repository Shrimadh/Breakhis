from PIL import Image, ImageEnhance
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.image as img
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import PIL as Image
import os

class sharp(object):
    def __call__(self, pic):
        enhancer = ImageEnhance.Sharpness(pic)
        return enhancer.enhance(4)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CustomDataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

means = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize([28,28]),                              
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std),
                                    ])

test_data = CustomDataset(pd.read_csv("test.csv"), 'Cancer\Test', test_transform )
test_loader = DataLoader(dataset = test_data, batch_size = 25, shuffle=False, num_workers=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN()
model.load_state_dict(torch.load("modelwithoutpreprocessing.ckpt"))
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for i,(images, labels) in enumerate(test_loader):
        if(i>8000):
            break
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
          
    print('Test Accuracy of the model without preprocessing is: {} %'.format(100 * correct / total))


test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize([28,28]), 
                                     sharp(),                             
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std),
                                    ])

test_data = CustomDataset(pd.read_csv("test.csv"), 'Cancer\Test', test_transform )
test_loader = DataLoader(dataset = test_data, batch_size = 25, shuffle=False, num_workers=0)
model = CNN()
model.load_state_dict(torch.load("modelwithsharpening.ckpt"))
model.eval()  # it-disables-dropout

with torch.no_grad():
    correct = 0
    total = 0
    for i,(images, labels) in enumerate(test_loader):
        if(i>8000):
            break
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
          
    print('Test Accuracy of the model with SHARPENING is: {} %'.format(100 * correct / total))