import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import cv2 as cv2 
import time
from model import Net, Classifier, Classifier2, Classifier3

testing_folder = sys.argv[1]
output_file = sys.argv[2]


class train_hw3(Dataset):
    def __init__(self,data_dir,label):
        self.data_dir = data_dir
        self.label = label
    def __getitem__(self,index):
        picfile = '{:0>5d}.jpg'.format(self.label[index][0])
        img = cv2.imread(os.path.join(self.data_dir,picfile), cv2.IMREAD_GRAYSCALE) /255.0
        img = np.expand_dims(img,0)
        return torch.FloatTensor(img), self.label[index,1]

    def __len__(self):
        return self.label.shape[0]

class test_hw3(Dataset):
    def __init__(self,data_dir,label):
        self.data_dir = data_dir
        self.label = label
    def __getitem__(self,index):
        picfile = '{:0>4d}.jpg'.format(self.label[index][0])
        img = cv2.imread(os.path.join(self.data_dir,picfile), cv2.IMREAD_GRAYSCALE) /255.0
        img = np.expand_dims(img,0)
        return torch.FloatTensor(img), self.label[index,1]

    def __len__(self):
        return self.label.shape[0]


if __name__ == '__main__':
    test_label = pd.read_csv('sample_submission.csv',header = None,skiprows =1)
    test_label = test_label.values
    #y_train = pd.read_csv('train.csv',header = None,skiprows =1)
    #y_train = y_train.values

    #train_dataset = train_hw3('train_img',y_train)
    #train_loader = DataLoader(train_dataset, batch_size=1,num_workers=8,shuffle=False)
    test_dataset = test_hw3(testing_folder,test_label)
    test_loader = DataLoader(test_dataset, batch_size=1,num_workers=8,shuffle=False)

    device = torch.device('cuda')
    #model = Classifier()
    #model.load_state_dict(torch.load('model.pth'))

    model2 = torch.load('model2.pth')
    model2 = model2.to(device)
    model2.eval()

    model3 = torch.load('model3.pth')
    model3 = model3.to(device)
    model3.eval()

    model4 = torch.load('model4.pth')
    model4 = model4.to(device)
    model4.eval()

    model5 = torch.load('model_final.pth')
    model5 = model5.to(device)
    model5.eval()

    ans = open(output_file,'w')
    ans.write('id,label\n')

    prediction = []
    for batch_idx, (img, index)in enumerate(test_loader):
        img = img.to(device)
        out2 = model2(img)
        out3 = model3(img)
        out4 = model4(img)
        out5 = model5(img)
        out = out2 + out3 + out4 + out5
        _, pred_label = torch.max(out, 1)
        prediction.append((batch_idx,pred_label.item()))
        ans.write('%d,%d\n' %(batch_idx,pred_label.item()) )


    # acc = 0
    # for batch_idx, (img, index)in enumerate(train_loader):
    #     img = img.to(device)
    #     out1 = model1(img)
    #     out2 = model2(img)
    #     out3 = model3(img)
    #     out4 = model4(img)
    #     out5 = model5(img)
    #     out = out1 + out2 + out3 + out4 + out5
    #     _, pred_label = torch.max(out, 1)
    #     if pred_label.item() == y_train[batch_idx][1]:
    #         acc += 1
    # print(acc/y_train.shape[0])
        

    
