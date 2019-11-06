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
import itertools

training_folder = sys.argv[1]
train_x = sys.argv[2]


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # [16, 24, 24]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [32, 12, 12]
            nn.Dropout(p=0.1),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]
            nn.Dropout(p=0.3),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(p=0.3),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [256, 3, 3]
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [256, 3, 3]
            nn.Dropout(p=0.5),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.7),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.7),
            nn.Linear(128, 7)
        )


        self.cnn.apply(gaussian_weights_init)

        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        #print(out.shape)
        return self.fc(out)

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


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    y_train = pd.read_csv(train_x,header = None,skiprows =1)
    y_train = y_train.values
    y_train_org = y_train

    # x_train = pd.read_csv('train_img')
    # print(x_train)
    print(type(y_train))
    print(y_train[:,1].shape)

    valid_index = np.arange(y_train.shape[0])
    np.random.shuffle(valid_index)


    y_valid = y_train[valid_index[25000:]]
    y_train = y_train[valid_index[:25000]]

    print(y_valid.shape)
    print(y_train.shape)

    train_dataset = train_hw3(training_folder,y_train)
    train_loader = DataLoader(train_dataset, batch_size=256,num_workers=8,shuffle=True)
    valid_dataset = train_hw3(training_folder,y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size = 256, num_workers=8,shuffle=True)

    num_epoch = 100

    
    device = torch.device('cuda')
    model = Classifier2().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    best_acc = 0.0
    Train_Acc = []
    Train_Loss = []
    Valid_Acc = []
    Valid_Loss = []

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        model.train()
        train_acc = 0
        train_loss = 0
        valid_acc = 0
        valid_loss = 0
        for batch_index, (data,label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            #print(data.shape)
            optimizer.zero_grad()
            output = model(data)


            loss = F.cross_entropy(output,label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())

            progress = ('#' * int(float(batch_index)/len(train_loader)*40)).ljust(40)
            print ('[%03d/%03d] %.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)

        train_acc = train_acc/train_dataset.__len__()
        # print("\nEPOCH:", epoch+1)
        # print("accuracy = ", train_acc)
        

        model.eval()
        for valid_index, (v_data,v_label) in enumerate(valid_loader):
            v_data, v_label = v_data.to(device), v_label.to(device)
            v_output = model(v_data)
            v_loss = F.cross_entropy(v_output,v_label)

            valid_acc += np.sum(np.argmax(v_output.cpu().data.numpy(), axis=1) == v_label.cpu().numpy())
            valid_loss += v_loss.item()

            progress = ('#' * int(float(valid_index)/len(valid_loader)*40)).ljust(40)
            print ('[%03d/%03d] %.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                    (time.time() - epoch_start_time), progress), end='\r', flush=True)

        valid_acc = valid_acc/valid_dataset.__len__()

        print('[%03d/%03d] %.2f sec(s) Train Acc: %.6f Loss: %.6f | Val Acc: %.6f loss: %.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                    train_acc, train_loss, valid_acc, valid_loss))
        Train_Acc.append(train_acc)
        Train_Loss.append(train_loss)
        Valid_Acc.append(valid_acc)
        Valid_Loss.append(valid_loss)

        if (valid_acc > best_acc):
            with open('acc.txt','w') as f:
                f.write(str(epoch)+'\t'+str(valid_acc)+'\n')
            torch.save(model, 'model.pth')
            best_acc = valid_acc
            print ('Model Saved!')

    #torch.save(model, 'model_final.pth')
    

    ### Plot Loss and accuracy
    # plt.plot(Train_Acc)
    # plt.plot(Valid_Acc)
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left') 
    # plt.show()

    ### Plot Confusion Matrix
    # y_pred = np.ones(28888)

    # pred_dataset = train_hw3('train_img',y_train_org)
    # pred_loader = DataLoader(pred_dataset, batch_size=1,num_workers=8,shuffle=False)

    # ans = open("train_pred.csv",'w')
    # ans.write('id,label\n')
    # prediction = []

    # for batch_idx, (img, index)in enumerate(pred_loader):
    #     img = img.to(device)
    #     out = model(img)
    #     _, pred_label = torch.max(out, 1)
    #     y_pred[batch_idx] = pred_label.item()
    #     prediction.append((batch_idx,pred_label.item()))
    #     ans.write('%d,%d\n' %(batch_idx,pred_label.item()) )


    # cm = confusion_matrix(y_train_org[:,1],y_pred)
    # print(cm)
    # print(cm.shape)

    # plt.figure(figsize=(8,6))
    # classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    # cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    # plt.imshow(cm,interpolation='nearest',cmap = plt.get_cmap('Blues'))
    # plt.title('Confusion matrix')
    # plt.colorbar()

    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks,classes,rotation = 45)
    # plt.yticks(tick_marks,classes)

    # for i,j in itertools.product(range(7),range(7)):
    #     plt.text(j,i, "{:0.2f}".format(cm[i,j]), horizontalalignment="center" ,  color="white" if cm[i,j] > (cm.max() / 1.5) else "black"           )
    
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predict label')
    # plt.show()


