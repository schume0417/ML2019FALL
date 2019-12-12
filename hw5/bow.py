import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
import sys
import time
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import spacy
from gensim.models import word2vec

class train_hw5(Dataset):
    def __init__(self,x,label):
        self.x = x
        self.label = label
    def __getitem__(self,index):        
        return self.x[index], self.label[index]
    def __len__(self):
        return self.label.shape[0]

class test_hw5(Dataset):
    def __init__(self,x):
        self.x = x
    def __getitem__(self,index):        
        return self.x[index]
    def __len__(self):
        return self.x.shape[0]

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(7618,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,2),
            nn.Softmax(dim = 1)
        )
    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], -1)
        out = self.dnn(inputs)
        return out

if __name__ == "__main__":

    x_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    x_test_path = sys.argv[3]

    x_train = pd.read_csv(x_train_path)
    x_train = x_train['comment'].values

    x_test = pd.read_csv(x_test_path)
    x_test = x_test['comment'].values

    y_train = pd.read_csv(y_train_path)
    y_train = y_train['label'].values

    nlp = spacy.load("en_core_web_sm")
    sentence = []
    for i in range(x_train.shape[0]):
        sentence.append(nlp(x_train[i]))
    sentence = np.array(sentence)

    fileSegWordDonePath ='seg.txt'
    with open(fileSegWordDonePath,'w') as fW:
        for i in range(len(sentence)):
            for j in range(len(sentence[i])):
                fW.write(str(sentence[i][j]) + ' ')
            fW.write('\n')

    sentences = word2vec.LineSentence("seg.txt")
    word_model = word2vec.Word2Vec(sentences, size=200, min_count=3)
    word_model.save("word2vec.model")

    embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size)).astype(np.float)
    word_index = {}

    vocab_list = [(word, word_model.wv[word]) for word, _ in word_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vector = vocab
        embedding_matrix[i + 1] = vector
        word_index[word] = i + 1

    bag_of_word = np.zeros((sentence.shape[0], len(word_model.wv.vocab.items()) + 1 ))
    for i,doc in enumerate(sentence):
        for j,word in enumerate(doc):
            try:
                k = word_index[str(word)]
                bag_of_word[i,k] += 1
            except:
                bag_of_word[i,j] = 0
    model = DNN()
    model = model.cuda()

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0
    batch_size = 64
    num_epoch = 100

    bag_of_word = torch.Tensor(bag_of_word)
    y_train = torch.Tensor(y_train)

    train_dataset = train_hw5(bag_of_word,y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = train_hw5(bag_of_word,y_train) # remove vn
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    Train_Acc = []
    Train_Loss = []

    for epoch in range(num_epoch):
        train_acc = 0
        train_loss = 0
        valid_acc = 0
        valid_loss = 0
        for i, (data, label) in enumerate(train_loader):
            epoch_start_time = time.time()
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(data)
            
            
            #output = output.view(label.size()[0], -1)
            label = label.long()
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())
            train_loss += loss.item()
            progress = ('#' * int(float(i)/len(train_loader)*30)).ljust(30)
            print ('[%03d/%03d] %.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                    (time.time() - epoch_start_time), progress), end='\r', flush=True)
            
        train_acc = train_acc/train_dataset.__len__()
        model.eval()

        for i, (data, label) in enumerate( valid_loader):
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            
            label = label.long()
            loss = criterion(output, label)
            
            valid_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())
            valid_loss += loss.item()
            
        valid_acc = valid_acc/valid_dataset.__len__()
            
        print('[%03d/%03d] %.2f sec(s) Train Acc: %.6f Loss: %.6f | Val Acc: %.6f loss: %.6f' % \
                    (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                        train_acc, train_loss, valid_acc, valid_loss))

        Train_Acc.append(train_acc)
        Train_Loss.append(train_loss)

        if (train_acc > best_acc):
            torch.save(model, 'model_bow.pth')
            best_acc = valid_acc
            print ('Model Saved!')



    nlp = spacy.load("en_core_web_sm")
    test_sentence = []
    for i in range(x_test.shape[0]):
        test_sentence.append(nlp(x_test[i]))
    test_sentence = np.array(test_sentence)

    bag_of_word_test = np.zeros((test_sentence.shape[0], len(word_model.wv.vocab.items()) + 1 ))
    print(bag_of_word_test.shape)
    for i,doc in enumerate(test_sentence):
        for j,word in enumerate(doc):
            try:
                k = word_index[str(word)]
                bag_of_word_test[i,k] += 1
            except:
                bag_of_word_test[i,j] = 0
    print(bag_of_word_test[0])

    model.eval()
    ans = open("bow.csv",'w')
    ans.write('id,label\n')
    bag_of_word_test = torch.Tensor(bag_of_word_test)
    test_dataset = test_hw5(bag_of_word_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prediction = []
    for i, (data) in enumerate(test_loader):
        data = data.cuda()
        out = model(data)
        _, pred_label = torch.max(out, 1)
        prediction.append((i,pred_label.item()))
        ans.write('%d,%d\n' %(i,pred_label.item()) )










