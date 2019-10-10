import sys
import numpy as np
import pandas as pd 

input_file = sys.argv[1]
output_file = sys.argv[2]

w = np.load("hw1.npy")

def readdata(data):
    
    # 把有些數字後面的奇怪符號刪除
    for col in list(data.columns[2:]):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
    data = data.values
    
    # 刪除欄位名稱及日期
    data = np.delete(data, [0,1], 1)
    
    # 特殊值補0
    data[ data == 'NR'] = 0
    data[ data == ''] = 0
    data[ data == 'nan'] = 0
    data = data.astype(np.float)

    return data


def extract(data):
    N = data.shape[0] // 18

    x = []

    
    # Shape 會變成 (x, 18) x = 取多少hours
    for i in range(N):
        tmp = []
        tmp = np.hstack(data[i*18: i*18+18, :])
        x.append(tmp)

    x = np.array(x)
    return x




testing_data_pd = pd.read_csv(input_file)

year1 = readdata(testing_data_pd)

test_data = extract(year1)


y = []

for i in test_data:
    #162feature
    y.append(np.dot(i, w[1:]) + w[0])
    #9feature
    #y.append(np.dot(i[81:90], w[1:]) + w[0])

    #18feature
    #y.append(np.dot(i[72:90], w[1:]) + w[0])
    #y.append(np.dot(i, w))


print("id,value\n{}".format("\n".join("id_{},{}".format(i,txt[0]) for i,txt in enumerate(y))), file = open(output_file,"w"))