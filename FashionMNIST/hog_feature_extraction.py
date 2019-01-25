# @author: Titas De
# Fashion MNIST HOG feature extraction

import numpy as np
import pdb
from skimage.feature import hog
from os.path import expanduser
import pandas as pd

# Load Data
home_dir = expanduser("~")
train_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_train.csv')
Y_train = train_data['label'].values.astype(np.uint8)
train_data.drop(['label'],inplace=True, axis=1)
X_train = train_data.values.astype(np.float32)
max_train_val, min_train_val = np.amax(X_train), np.amin(X_train)
X_train = (X_train - min_train_val) / (max_train_val - min_train_val + 1e-4)  # 0-1 scaling
N_train = X_train.shape[0]
X_train = X_train.reshape(N_train,28,28)

test_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_test.csv')
Y_test = test_data['label'].values.astype(np.uint8)
test_data.drop(['label'],inplace=True, axis=1)
X_test = test_data.values.astype(np.float32)
X_test = (X_test - max_train_val) / (max_train_val - min_train_val + 1e-4)  # 0-1 scaling
N_test = X_test.shape[0]
X_test = X_test.reshape(N_test,28,28)

# pdb.set_trace()

pixels_per_cell = 2
cells_per_block = 2
N_hist_orient = 9
block_norm = 'L2-Hys'
# img_zp = np.zeros((30,30))
# img_zp[1:29,1:29] = X_train[0,:,:]
feat = hog(X_train[0,:,:], orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(cells_per_block, cells_per_block), block_norm=block_norm, feature_vector=True)

#pdb.set_trace()
feat_train = np.zeros((N_train,len(feat))).astype(np.float16)
feat_test = np.zeros((N_test,len(feat))).astype(np.float16)

for i in range(N_train):
		# img_zp[1:29,1:29] = X_train[i,:,:]
		# pdb.set_trace()
		feat_train[i,:] = hog(X_train[i,:,:], orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(cells_per_block, cells_per_block), block_norm=block_norm, feature_vector=True).astype(np.float16)
		
		if i<N_test:
			# img_zp[1:29,1:29] = X_test[i,:,:]
			feat_test[i,:] = hog(X_test[i,:,:], orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(cells_per_block, cells_per_block), block_norm=block_norm, feature_vector=True).astype(np.float16)
		

print('Feature extraction done')

# pdb.set_trace()

np.savez_compressed(home_dir+'/Downloads/Datasets/fashionmnist/'+'HOGfeatures-2x2pixels-2x2cells-L2Hys', X_train=feat_train, Y_train=Y_train, X_test=feat_test, Y_test=Y_test)


