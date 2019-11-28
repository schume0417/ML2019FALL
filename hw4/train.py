import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn import manifold
import torch.nn.functional as F
import time
import sys

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # define: encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,8,3,2,2),
            nn.SELU(0.3),

            nn.Conv2d(8,16,3,2,2),
            nn.SELU(0.3),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,3,2,2),
            nn.SELU(0.3),

            nn.Conv2d(32,64,3,2,2),
            nn.SELU(0.3),

            #nn.Conv2d(64,128,3,2,2),
            #nn.SELU(0.3),
            # nn.MaxPool2d(2,2),
        )

        self.encoderLi = nn.Sequential(
            nn.Linear(576, 256),
            nn.SELU(0.3),
            # nn.Linear(256,128),
            # nn.SELU(0.3)
        )

        self.decoderLi = nn.Sequential(
            # nn.Linear(128, 256),
            # nn.SELU(0.3),
            nn.Linear(256,576),
            nn.SELU(0.3),
        )
 
        # define: decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(128,64,2,2),
            # nn.SELU(0.3),
            # nn.MaxUnpool2d(2,2),

            nn.ConvTranspose2d(64,32,2,1),
            nn.SELU(0.3),
            
            nn.ConvTranspose2d(32,16,2,2),
            nn.SELU(0.3),

            #nn.MaxUnpool2d(2,2),
            nn.ConvTranspose2d(16,8,2,2),
            nn.SELU(0.3),

            nn.ConvTranspose2d(8,3,2,2),
            nn.SELU(0.3),
            nn.Tanh(),
        )
 
 
    def forward(self, x):
 
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size()[0], -1)
        encodedl = self.encoderLi(encoded)

        decodedl = self.decoderLi(encodedl)
        decodedl = decodedl.view(decodedl.size()[0],64,3,3)

        decoded = self.decoder(decodedl)

        # Total AE: return latent & reconstruct
        return encodedl, decoded
 
class ModelOne(nn.Module):
    def __init__(self):
        super(ModelOne, self).__init__()
 
        # define: encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2), 

            nn.Conv2d(16,32,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2),
        )

        self.encoderLi = nn.Sequential(
            nn.Linear(512, 256),
            nn.SELU(0.2),
        )

        self.decoderLi = nn.Sequential(
            nn.Linear(256,512),
            nn.SELU(0.2),
        )
 
        # define: decoder
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(128,64,2,2),
            nn.SELU(0.2),
            
            nn.ConvTranspose2d(64,32,2,2),
            nn.SELU(0.2),

            #nn.MaxUnpool2d(2,2),
            nn.ConvTranspose2d(32,16,2,2),
            nn.SELU(0.2),

            nn.ConvTranspose2d(16,3,2,2),
            nn.SELU(0.2),
            nn.Tanh(),
        )
 
 
    def forward(self, x):
 
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size()[0], -1)
        encodedl = self.encoderLi(encoded)

        decodedl = self.decoderLi(encodedl)
        decodedl = decodedl.view(decodedl.size()[0],128,2,2)

        decoded = self.decoder(decodedl)

        # Total AE: return latent & reconstruct
        return encodedl, decoded

class ModelTwo(nn.Module):
    def __init__(self):
        super(ModelTwo, self).__init__()
 
        # define: encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,8,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2), 

            nn.Conv2d(8,16,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,1,1),
            nn.SELU(0.2),
            nn.MaxPool2d(2,2),

            #nn.Conv2d(64,128,3,1,1),
            #nn.SELU(0.2),
            #nn.MaxPool2d(2,2),
        )

        self.encoderLi = nn.Sequential(
            nn.Linear(256, 128),
            nn.SELU(0.2),
        )

        self.decoderLi = nn.Sequential(
            nn.Linear(128,256),
            nn.SELU(0.2),
        )
 
        # define: decoder
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(128,64,2,2),
            #nn.SELU(0.2),
            # nn.MaxUnpool2d(2,2),

            nn.ConvTranspose2d(64,32,2,2),
            nn.SELU(0.2),
            
            nn.ConvTranspose2d(32,16,2,2),
            nn.SELU(0.2),

            #nn.MaxUnpool2d(2,2),
            nn.ConvTranspose2d(16,8,2,2),
            nn.SELU(0.2),

            nn.ConvTranspose2d(8,3,2,2),
            nn.SELU(0.2),
            nn.Tanh(),
        )
 
 
    def forward(self, x):
 
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size()[0], -1)
        encodedl = self.encoderLi(encoded)

        decodedl = self.decoderLi(encodedl)
        decodedl = decodedl.view(decodedl.size()[0], 64,2,2)

        decoded = self.decoder(decodedl)

        # Total AE: return latent & reconstruct
        return encodedl, decoded

if __name__ == '__main__':
 
    # detect is gpu available.
    use_gpu = torch.cuda.is_available()
 
    autoencoder = Autoencoder()
    
    # load data and normalize to [-1, 1]
    trainXPath = sys.argv[1]
    outputPath = sys.argv[2]

    trainX = np.load(trainXPath)
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)
 
    # if use_gpu, send model / data to GPU.
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()
 
    # Dataloader: train shuffle = True
    train_dataloader = DataLoader(trainX, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(trainX, batch_size=1, shuffle=False)
 
 
    # We set criterion : L1 loss (or Mean Absolute Error, MAE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

 
    # Now, we train 100 epochs.
    for epoch in range(100):
 
        cumulate_loss = 0
        for x in train_dataloader:
            
            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cumulate_loss += loss.item() * x.shape[0]
 
        print(f'Epoch { "%03d" % epoch }: Loss : { "%.5f" % (cumulate_loss / trainX.shape[0])}')
 
    torch.save(autoencoder.state_dict(),'model.pth')


    # Collect the latents and stdardize it.
    latents = []
    reconstructs = []
    # model1 = torch.load('1.pth') # or any other model
    # model1 = model1.cuda()
    for x in test_dataloader:
        ## You can change the model from autoencoder to model1
        latent, reconstruct = autoencoder(x)
        latent = latent.cpu().detach().numpy()

        loss = criterion(reconstruct, x)
        loss = np.array(loss.item()).reshape((1,1))
        latent = np.concatenate((latent,loss),axis=1)

        latents.append(latent)
        reconstructs.append(reconstruct.cpu().detach().numpy())
 
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
 
    print("success")
    # Use PCA to lower dim of latents and use K-means to clustering.
    #latents = PCA(n_components=32).fit_transform(latents)
    latents = manifold.TSNE(n_components = 2, init='pca',learning_rate=100,n_iter=2000).fit_transform(latents)
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(latents)
    result = kmeans.labels_

    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
 
 
    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv('hw4.csv',index=False)


    y1 = pd.read_csv('1.csv',header = None,skiprows =1)
    y1 = y1.values[:,1]

    y2 = pd.read_csv('2.csv',header = None,skiprows =1)
    y2 = y2.values[:,1]

    ensemble = []

    for i in range(len(y1)):
        if (y1[i] + y2[i] + result[i] <= 1):
            ensemble.append(0)
        else:
            ensemble.append(1)

    ensemble = np.array(ensemble)

    df = pd.DataFrame({'id': np.arange(0,len(ensemble)), 'label': ensemble})
    df.to_csv(outputPath,index=False)
