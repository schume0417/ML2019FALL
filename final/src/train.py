import sys
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch import device , cuda, optim
from torch.utils.data import Dataset , DataLoader
from torch import FloatTensor
import MCD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD , Adam
from time import time
import matplotlib.pyplot as plt

class dataset(Dataset):
    def __init__(self , data , label , return_label , transform , apply_transform):
        self.data = data
        self.label = label
        self.return_label = return_label
        self.transform = transform
        self.apply_transform = apply_transform
        return

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self , index):
        if (self.return_label):
            return (FloatTensor(self.data[index]) , self.label[index])
        else:
            return FloatTensor(self.data[index])

def discrepancy(output_1 , output_2):
    return torch.mean(torch.abs(F.softmax(output_1) - F.softmax(output_2)))

train_x = np.load('../data/trainX.npy')
train_x = np.transpose(train_x,(0,3,1,2))
train_x = train_x / 255

train_y = np.load('../data/trainY.npy').astype(np.int)

test_image = np.load('../data/testX.npy')

test_x = []
for i in range(test_image.shape[0]):
    image = cv2.cvtColor(test_image[i] , cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (32,32), cv2.INTER_LINEAR)
    test_x.append(image)
test_x = np.array(test_x).astype(np.float)
test_x = np.transpose(test_x,(0,3,1,2))
test_x = test_x / 255

use_gpu = torch.cuda.is_available()
transform = transforms.Compose([
    #transforms.RandomAffine(10 , translate = (0.1 , 0.1)) ,
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30,resample = Image.BILINEAR)
    ])

train_dataset = dataset(train_x , train_y , True , transform , True)
test_dataset = dataset(test_x , None , False , transform , False)
train_loader = DataLoader(train_dataset , batch_size = 64 , shuffle = True)
test_loader = DataLoader(test_dataset , batch_size = 64 , shuffle = True)
acc_loader = DataLoader(train_dataset, batch_size = 100, shuffle = False)
num_epoch = 80

generator = MCD.generator().cuda()
classifier_1 = MCD.classifier().cuda()
classifier_2 = MCD.classifier().cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer_generator = SGD(generator.parameters() , lr = 0.0025 , weight_decay = 0.0005 , momentum = 0.9)
optimizer_classifier_1 = SGD(classifier_1.parameters() , lr = 0.0025 , weight_decay = 0.0005 , momentum = 0.9)
optimizer_classifier_2 = SGD(classifier_2.parameters() , lr = 0.0025 , weight_decay = 0.0005 , momentum = 0.9)

for epoch in range(num_epoch):
    train_acc = 0
    start = time()
    generator.train()
    classifier_1.train()
    classifier_2.train()
    for ((data_source,label),data_target) in zip(train_loader, test_loader):
        #step1
        data_source = data_source.cuda()
        label = label.cuda()
        data_target = data_target.cuda()

        optimizer_generator.zero_grad()
        optimizer_classifier_1.zero_grad()
        optimizer_classifier_2.zero_grad()

        feature = generator(data_source)
        y1 = classifier_1(feature)
        y2 = classifier_2(feature)

        loss = criterion(y1, label) + criterion(y2, label)
        loss.backward()
        optimizer_generator.step()
        optimizer_classifier_1.step()
        optimizer_classifier_2.step()
        #step2
        optimizer_generator.zero_grad()
        optimizer_classifier_1.zero_grad()
        optimizer_classifier_2.zero_grad()

        feature = generator(data_source)
        y1 = classifier_1(feature)
        y2 = classifier_2(feature)

        loss1 = criterion(y1, label) + criterion(y2, label)

        feature = generator(data_target)
        y1 = classifier_1(feature)
        y2 = classifier_2(feature)

        loss2 = discrepancy(y1 , y2)

        loss = loss1-loss2
        loss.backward()
        optimizer_classifier_1.step()
        optimizer_classifier_2.step()
        #step3
        for j in range(4):
            feature = generator(data_target)
            y1 = classifier_1(feature)
            y2 = classifier_2(feature)
            loss = discrepancy(y1 , y2)
            loss.backward()
            optimizer_generator.step()
        end = time()

        feature = generator(data_source)
        y1 = classifier_1(feature)
        y2 = classifier_2(feature)
        y = (y1 + y2) / 2
        (temp , result) = torch.max(y , 1)
        train_acc += np.sum(result.cpu().numpy() == label.cpu().numpy())


    train_acc = train_acc/train_dataset.__len__()
    print('[%03d/%03d] %.2f sec(s) Train Acc: %.6f' % (epoch + 1, num_epoch, time()-start,train_acc))

torch.save(generator.state_dict() , 'generator.pth')
torch.save(classifier_1.state_dict() , 'classifier_1.pth')
torch.save(classifier_2.state_dict() , 'classifier_2.pth')

test_loader = DataLoader(test_dataset , batch_size = 100 , shuffle = False)

ans = open("final.csv",'w')
ans.write('id,label\n')
generator.eval()
classifier_1.eval()
classifier_2.eval()
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data.cuda()
        feature = generator(data)
        y1 = classifier_1(feature)
        y2 = classifier_2(feature)
        y = (y1 + y2) / 2
        (temp , pred_label) = torch.max(y , 1)
        for i in range(100):
            ans.write('%d,%d\n' %(batch_idx*100 + i, pred_label[i].item()))
        print(batch_idx*100)





