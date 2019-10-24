import numpy as np
import pandas as pd
import sys
from numpy.linalg import inv

input_test_x = sys.argv[1]
output_file = sys.argv[2]

def Tune_data(x):

    ## Normalization
    col = [0,1,3,4,5]
    mean = np.mean(x[:,col], axis=0).astype(np.float32)
    std = np.std(x[:,col], axis=0)
    x[:,col] = (x[:,col]-mean) / std
    
    # Normalization to [0,1]
    col = [0,1,3,4,5]
    maxi = np.max(x[:,col],axis=0)
    mini = np.min(x[:,col],axis=0)
    #x[:,col] = (x[:,col] - mini) / (maxi - mini)  * 10

    return x

def Add_feature(x_train):
    col = [0,1,3,4,5]
    for i in range(len(col)):
        a = x_train[:,col[i]]

        b = np.tan(x_train[:,col[i]])
        b = b.reshape((-1,1))
        x_train = np.concatenate((x_train,b),axis = 1)

        b = np.cos(x_train[:,col[i]])
        b = b.reshape((-1,1))
        x_train = np.concatenate((x_train,b),axis = 1)

        b = np.sin(x_train[:,col[i]])
        b = b.reshape((-1,1))
        x_train = np.concatenate((x_train,b),axis = 1)


        for j in range(2,5):
            b = np.power(a,j)
            b = b.reshape((-1,1))
            x_train = np.concatenate((x_train,b),axis = 1)

            e = np.tan(b)
            x_train = np.concatenate((x_train,e),axis = 1)

            e = np.cos(b)
            x_train = np.concatenate((x_train,e),axis = 1)

            e = np.sin(b)
            x_train = np.concatenate((x_train,e),axis = 1)

    return x_train


w = np.load("gen.npy")

## Read Testing Data
x_test = np.genfromtxt(input_test_x, delimiter=',', skip_header=1)
x_test = np.array(x_test).astype(np.float32)

## Adjust data
x_test = Tune_data(x_test)
x_test = Add_feature(x_test)

## Add bias
bias = np.ones(shape=(x_test.shape[0], 1), dtype=np.float32)
x_test = np.concatenate((bias,x_test), axis=1)


## Predict On Testing Data
y_pred = np.sign(np.dot(x_test, w))
y_pred[y_pred == -1] = 0
print(np.sum(y_pred))

## Output Testing Results To gen.csv, Output Weight to gen.npy
with open(output_file, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_pred):
            f.write('%d,%d\n' %(i+1, v))

