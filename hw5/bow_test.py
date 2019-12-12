import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
import sys
import time
from torchvision import transforms
import torchvision.models as models
import spacy
from gensim.models import word2vec
import pickle

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
    x_test_path = sys.argv[1]
    output_path = sys.argv[2]

    x_test = pd.read_csv(x_test_path)
    x_test = x_test['comment'].values

    model = torch.load('model_bow1.pth')
    #model2 = torch.load('model_bow2.pth')

    nlp = spacy.load("en_core_web_sm")
    test_sentence = []
    for i in range(x_test.shape[0]):
        test_sentence.append(nlp(x_test[i]))
    test_sentence = np.array(test_sentence)

    word_model = word2vec.Word2Vec.load("word2vec.model")

    embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size)).astype(np.float)
    word_index = {}

    vocab_list = [(word, word_model.wv[word]) for word, _ in word_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vector = vocab
        embedding_matrix[i + 1] = vector
        word_index[word] = i + 1

    bag_of_word_test = np.zeros((test_sentence.shape[0], len(word_model.wv.vocab.items()) + 1 ))
    print(bag_of_word_test.shape)
    for i,doc in enumerate(test_sentence):
        for j,word in enumerate(doc):
            try:
                k = word_index[str(word)]
                bag_of_word_test[i,k] += 1
            except:
                bag_of_word_test[i,j] = 0

    model.eval()
    ans = open(output_path,'w')
    ans.write('id,label\n')
    bag_of_word_test = torch.Tensor(bag_of_word_test)
    test_dataset = test_hw5(bag_of_word_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prediction = []
    for i, (data) in enumerate(test_loader):
        data = data.cuda()
        out = model(data)
        _, pred_label = torch.max(out, 1)
        prediction.append(pred_label.item())
        ans.write('%d,%d\n' %(i,pred_label.item()) )

    #model2.eval()
    #prediction2 = []
    #for i,(data) in enumerate(test_loader):
    #    data = data.cuda()
    #    out = model(data)
    #    _, pred_label = torch.max(out,1)
    #    prediction2.append(pred_label.item())

    #prediction3 = pickle.load(open("fuck.pkl","rb"))
    #prediction3 = prediction3.values[:,1]

    #print(type(prediction))

    #result = []
    #for i in range(len(prediction)):
    #    if (prediction[i] + prediction2[i] + prediction3[i] <= 1):
    #        result.append(0)
    #    else:
    #        result.append(1)
    #result = np.array(result)
    
    #df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    #df.to_csv(output_path,index=False)



