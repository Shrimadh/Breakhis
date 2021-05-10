from PIL import Image, ImageEnhance
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import numpy as np
import matplotlib.image as img
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import PIL as Image

os.getcwd()
# place the files in your IDE working dicrectory .
labels = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_path = './Data/Cancer/Train'
test_path = './Data/Cancer/Test'

for col in labels.columns:
    print(col)
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
        img_path = os.path.join(img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class sharp(object):
    def __call__(self, pic):
        enhancer = ImageEnhance.Sharpness(pic)
        return enhancer.enhance(4)

    def __repr__(self):
        return self.__class__.__name__ + '()'

means = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize([28,28]),
                                      sharp()   ,                         
                                      transforms.ToTensor(),
                                      transforms.Normalize(means,std),
                                      ])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize([28,28]), 
                                     sharp(),                             
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std),
                                    ])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize([28,28]),
                                      sharp() ,                         
                                      transforms.ToTensor(),
                                     transforms.Normalize(means,std)])

train, valid_data = train_test_split(labels, stratify=labels.Benign, test_size=0.2)

train_data = CustomDataset(train, train_path, train_transform )
valid_data = CustomDataset(valid_data, train_path, valid_transform )
test_data = CustomDataset(test, test_path, test_transform )
print(len(train_data),len(test_data),len(valid_data))
num_epochs = 10
num_classes = 2
batch_size = 25
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)


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

# model = CNN()
# model = CNN().to(device)
model = models.resnet34(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

train_losses = []
valid_losses = []
n = 15
m_train = 100
m_valid = 100

def save(train_loss,valid_loss):
    global m_train
    global m_valid
    if(train_loss<m_train and valid_loss<m_valid):
        m_valid = valid_loss
        m_train = train_loss
        torch.save(model.state_dict(), 'resnetwithsharpening.ckpt')
        print("MODEL SAVED")

for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0
    
    # training-the-model
    model.train()
    for i,(data, target) in enumerate(train_loader):
        print(i)
        if(i>800):
            break
        # move-tensors-to-GPU 
        data = data.to(device)
        target = target.to(device)
        
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        
    # validate-the-model
    model.eval()
    for it,(data, target) in enumerate(valid_loader):
        if(it>200):
            break
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        
        loss = criterion(output, target)
        
        # update-average-validation-loss 
        valid_loss += loss.item() * data.size(0)
    
    # calculate-average-losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    save(train_loss,valid_loss)
    # print-training/validation-statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for i,(images, labels) in enumerate(test_loader):
        if(i>540):
            break
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
          
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

# Save 
torch.save(model.state_dict(), 'resnetnetwithsharpening.ckpt')