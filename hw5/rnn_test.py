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
import sys

class test_hw5(Dataset):
    def __init__(self,x):
        self.x = x
    def __getitem__(self,index):        
        return self.x[index]
    def __len__(self):
        return self.x.shape[0]


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
    x_test_path = sys.argv[1]
    output_path = sys.argv[2]
    x_test = pd.read_csv(x_test_path)
    x_test = x_test['comment'].values

    model = torch.load('model_rnn.pth')

    nlp = spacy.load("en_core_web_sm")
    test_sentence = []
    for i in range(x_test.shape[0]):
        test_sentence.append(nlp(x_test[i]))
    test_sentence = np.array(test_sentence)

    word_model = word2vec.Word2Vec.load("word2vec_rnn.model")

    embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size)).astype(np.float)
    word_index = {}

    vocab_list = [(word, word_model.wv[word]) for word, _ in word_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vector = vocab
        embedding_matrix[i + 1] = vector
        word_index[word] = i + 1

    test_corpus = np.zeros((test_sentence.shape[0], 116))
    for i,doc in enumerate(test_sentence):
        for j,word in enumerate(doc):
            try:
                test_corpus[i,j] = word_index[str(word)]
            except:
                test_corpus[i,j] = 0

    model.eval()
    ans = open(output_path,'w')
    ans.write('id,label\n')
    test_corpus = torch.LongTensor(test_corpus)
    test_dataset = test_hw5(test_corpus)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    prediction = []
    for i, (data) in enumerate(test_loader):
        data = data.cuda()
        out = model(data)
        _, pred_label = torch.max(out, 1)
        prediction.append((i,pred_label.item()))
        ans.write('%d,%d\n' %(i,pred_label.item()) )
