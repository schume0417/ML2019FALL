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

generator = MCD.generator()
classifier_1 = MCD.classifier()
classifier_2 = MCD.classifier()

generator.load_state_dict(torch.load('../model/generator.pth'))
classifier_1.load_state_dict(torch.load('../model/classifier_1.pth'))
classifier_2.load_state_dict(torch.load('../model/classifier_2.pth'))

generator = generator.cuda()
classifier_1 = classifier_1.cuda()
classifier_2 = classifier_2.cuda()

test_image = np.load('../data/testX.npy')

test_x = []
for i in range(test_image.shape[0]):
    image = cv2.cvtColor(test_image[i] , cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (32,32), cv2.INTER_LINEAR)
    test_x.append(image)
test_x = np.array(test_x).astype(np.float)
test_x = np.transpose(test_x,(0,3,1,2))
test_x = test_x / 255

transform = transforms.Compose([
    #transforms.RandomAffine(10 , translate = (0.1 , 0.1)) ,
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30,resample = Image.BILINEAR)
    ])

test_dataset = dataset(test_x , None , False , transform , False)
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