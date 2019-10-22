import numpy as np
import pandas as pd
import sys
from numpy.linalg import inv

input_train_x = sys.argv[1]
input_train_y = sys.argv[2]
input_test_x = sys.argv[3]
output_file = sys.argv[4]

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


## Read Training Data
x = np.genfromtxt(input_train_x, delimiter=',', skip_header=1)
y = np.genfromtxt(input_train_y, delimiter=',')
x = np.array(x).astype(np.float32)
y = np.array(y)

## Read Testing Data
x_test = np.genfromtxt(input_test_x, delimiter=',', skip_header=1)
x_test = np.array(x_test).astype(np.float32)

## Adjust data
x = Tune_data(x)
x_test = Tune_data(x_test)
x = Add_feature(x)
x_test = Add_feature(x_test)

class_0 = []
class_1 = []
for i in range(len(y)):
    if y[i] == 1:
        class_0.append(i)
    else:
        class_1.append(i)

x_0 = x[class_0]
x_1 = x[class_1]

mean_0 = np.mean(x_0, axis=0).astype(np.float32)
mean_1 = np.mean(x_1, axis=0).astype(np.float32)

feature_num = x_0.shape[1]
cov_0 = np.zeros((feature_num,feature_num))
cov_1 = np.zeros((feature_num,feature_num))

cov_0 += (np.dot( (x_0 - mean_0).transpose()  , (x_0 - mean_0) )   / x_0.shape[0])
cov_1 += (np.dot( (x_1 - mean_1).transpose()  , (x_1 - mean_1) )   / x_1.shape[0])

cov = (x_0.shape[0] * cov_0 + x_1.shape[0] * cov_1) / (x_0.shape[0] + x_1.shape[0])

w = np.dot((mean_0 - mean_1),inv(cov)).transpose()
print("w_shape", w.shape)

b = (- 0.5) * np.dot(np.dot(mean_0,inv(cov)), mean_0) + (0.5) * np.dot(np.dot(mean_1,inv(cov)), mean_1) + np.log(x_0.shape[0]/x_1.shape[0]) 
print("b_shape", b.shape)

## Predict On Training Data
y_pred = np.sign(np.dot(x, w) + b)
y_pred[y_pred == -1] = 0
print("Acc:",np.sum(y_pred == y)/len(y_pred) ) 

## Predict On Testing Data
y_pred = np.sign(np.dot(x_test, w) + b)
y_pred[y_pred == -1] = 0
print(np.sum(y_pred))

## Output Testing Results To gen.csv, Output Weight to gen.npy
with open(output_file, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_pred):
            f.write('%d,%d\n' %(i+1, v))

b = np.array(b)
b = b.reshape((-1,))
w = np.concatenate((b,w))
np.save("gen.npy", w)
