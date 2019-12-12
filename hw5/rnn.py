import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
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

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dimension, hidden_dimension, layer_number, dropout = 0.3):
        super(LSTM_Net, self).__init__()
        
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        
#         self.embedding.weight.requires_grad = False
#         self.embedding_dim = embedding.size(1)
#         self.hidden_dim = hidden_dimension
#         self.num_layers = layer_number
#         self.dropout = dropout
        
        
        self.lstm = nn.GRU(input_size=embedding.size(1),hidden_size=hidden_dimension,num_layers=layer_number,
                            dropout=dropout,batch_first=True)
        self.lstm2 = nn.GRU(input_size=100 ,hidden_size=100,num_layers=layer_number,
                            dropout=dropout,batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(100 * 116 ,2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
            nn.Softmax(dim = 1)
        )
    def forward(self,inputs):
        inputs = self.embedding(inputs)
        
        x3, (_,_) = self.lstm(inputs)
        x2, (_,_) = self.lstm2(x3)
        x, (_,_) = self.lstm2(x2)
    
        x = x.reshape((-1,100 * 116))

        x = self.classifier(x)
        return x

if __name__ == "__main__":
    

    x_train = pd.read_csv('train_x.csv')
    x_train = x_train['comment'].values

    x_test = pd.read_csv('test_x.csv')
    x_test = x_test['comment'].values

    y_train = pd.read_csv('train_y.csv')
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
    word_model = word2vec.Word2Vec(sentences, size=100, min_count=1)
    word_model.save("word2vec_rnn.model")

    embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
    word_index = {}
    vocab_list = [(word, word_model.wv[word]) for word, _ in word_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vector = vocab
        embedding_matrix[i + 1] = vector
        word_index[word] = i + 1

    new_corpus = np.zeros((sentence.shape[0], 116))
    for i,doc in enumerate(sentence):
        for j,word in enumerate(doc):
            try:
                new_corpus[i,j] = word_index[str(word)]
            except:
                new_corpus[i,j] = 0

    embedding_matrix = torch.FloatTensor(embedding_matrix)

    model = LSTM_Net(embedding_matrix, 100, 100, 2) 
    model = model.cuda()

    model.train()
    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0
    batch_size = 64
    num_epoch = 100

    new_corpus = torch.LongTensor(new_corpus)
    y_train = torch.LongTensor(y_train)

    train_dataset = train_hw5(new_corpus,y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    Train_Acc = []
    Train_Loss = []

    for epoch in range(num_epoch):
        train_acc = 0
        train_loss = 0
        epoch_start_time = time.time()
        for i, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(data)
            
            #if (i == 0):
            #    print(output)
            
            label = label.long()
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())
            train_loss += loss.item()
            progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
            print ('[%03d/%03d] %.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                    (time.time() - epoch_start_time), progress), end='\r', flush=True)
        train_acc = train_acc/train_dataset.__len__()
        
        print('[%03d/%03d] %.2f sec(s) Train Acc: %.6f Loss: %.6f' % \
                    (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                        train_acc, train_loss))
        Train_Acc.append(train_acc)
        Train_Loss.append(train_loss)

        if (train_acc > best_acc):
            torch.save(model, 'model_rnn.pth')
            best_acc = train_acc
            print ('Model Saved!')










