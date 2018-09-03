'''
@author: Titas De
'''

import numpy as np
from os import path
import pdb

data = np.load('all_images_3x60x60.npy')
ids_scores = np.load('ids_scores.npy')

# Make a 80-20 split into training and test data

X_train = []
X_test = []
Y_train = []
Y_test = []

for i in range(1,data.shape[0]+1):
	if i%5==0:
		X_test.append(data[i,:,:,:])
		Y_test.append(ids_scores[i,1:])
	else:
		X_train.append(data[i,:,:,:])
		Y_train.append(ids_scores[i,1:])

X_test = np.array(X_test).astype(np.uint8)
Y_test = np.array(Y_test).astype(np.uint8)
X_train = np.array(X_train).astype(np.uint8)
Y_train= np.array(Y_train).astype(np.uint8)
np.save('all_data_3x60x60.npy',{'x_train':X_train,'x_test':X_test,'y_train':Y_train,'y_test':Y_test})

