import numpy as np
import pandas as pd
import sys

def Add_feature(x_train,x_test):
	col = [0,1,3,4,5]
	for i in range(len(col)):
		a = x_train[:,col[i]]
		c = x_test[:,col[i]]

		b = np.tan(x_train[:,col[i]])
		b = b.reshape((-1,1))
		d = np.tan(x_test[:,col[i]])
		d = d.reshape((-1,1))
		x_train = np.concatenate((x_train,b),axis = 1)
		x_test = np.concatenate((x_test,d),axis = 1)

		b = np.cos(x_train[:,col[i]])
		b = b.reshape((-1,1))
		d = np.cos(x_test[:,col[i]])
		d = d.reshape((-1,1))
		x_train = np.concatenate((x_train,b),axis = 1)
		x_test = np.concatenate((x_test,d),axis = 1)

		b = np.sin(x_train[:,col[i]])
		b = b.reshape((-1,1))
		d = np.sin(x_test[:,col[i]])
		d = d.reshape((-1,1))
		x_train = np.concatenate((x_train,b),axis = 1)
		x_test = np.concatenate((x_test,d),axis = 1)

		for j in range(2,5):
			b = np.power(a,j)
			d = np.power(c,j)
			b = b.reshape((-1,1))
			d = d.reshape((-1,1))

			x_train = np.concatenate((x_train,b),axis = 1)
			x_test = np.concatenate((x_test,d),axis = 1)

			e = np.tan(b)
			f = np.tan(d)
			x_train = np.concatenate((x_train,e),axis = 1)
			x_test = np.concatenate((x_test,f),axis = 1)

			e = np.cos(b)
			f = np.cos(d)
			x_train = np.concatenate((x_train,e),axis = 1)
			x_test = np.concatenate((x_test,f),axis = 1)

			e = np.sin(b)
			f = np.sin(d)
			x_train = np.concatenate((x_train,e),axis = 1)
			x_test = np.concatenate((x_test,f),axis = 1)


	return x_train,x_test

def load_data(input_train_x,input_train_y,input_test_x):
    #讀檔如果像這樣把路徑寫死交到github上去會馬上死去喔
    #還不知道怎寫請參考上面的連結
    x_train = pd.read_csv(input_train_x)
    x_test = pd.read_csv(input_test_x)

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv(input_train_y, header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	
	return np.clip(res, 1e-6, 1-1e-6)

def normalize(x_train, x_test, index):

	x_all = np.concatenate((x_train, x_test), axis = 0)
	mean = np.mean(x_all, axis = 0)
	std = np.std(x_all, axis = 0)

	mean_vec = np.zeros(x_all.shape[1])
	std_vec = np.ones(x_all.shape[1])
	# minn_vec = np.zeros(x_all.shape[1])
	# maxx_vec = np.ones(x_all.shape[1])

	mean_vec[index] = mean[index]
	std_vec[index] = std[index]
	x_all_nor = (x_all - mean_vec) / std_vec

	# col = [0,1,3,4,5]
	# minn = np.min(x_all_nor,axis = 0)
	# maxx = np.max(x_all_nor,axis = 0)
	# minn_vec[col] = minn[col]
	# maxx_vec[col] = maxx[col]
	# x_all_nor = (x_all_nor - minn_vec) / (maxx_vec - minn_vec)

	x_train_nor = x_all_nor[0:x_train.shape[0]]
	x_test_nor = x_all_nor[x_train.shape[0]:]

	return x_train_nor, x_test_nor

if __name__ == '__main__':
    input_train_x = sys.argv[1]
    input_train_y = sys.argv[2]
    input_test_x = sys.argv[3]
    output_file = sys.argv[4]
    x_train, y_train, x_test = load_data(input_train_x,input_train_y,input_test_x)

    index = [0,1,3,4,5]
    x_train, x_test = normalize(x_train, x_test,index)
    x_train, x_test = Add_feature(x_train, x_test)

    w = np.load("ta2.npy")

    ## Add bias
    bias = np.ones(shape=(x_test.shape[0], 1), dtype=np.float32)
    x_test = np.concatenate((bias,x_test), axis=1)

    print(w.shape)

    y_pred = np.sign(np.dot(x_test, w))
    print(y_pred.shape)
    y_pred[y_pred == -1] = 0
    print(np.sum(y_pred))

    with open(output_file, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_pred):
            f.write('%d,%d\n' %(i+1, v))



