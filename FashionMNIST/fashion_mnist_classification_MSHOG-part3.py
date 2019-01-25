# @author: Titas De
# Fashion MNIST classification

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics, multiclass
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.svm import SVC, LinearSVC
import pandas as pd 
import pdb
from skimage.feature import hog, canny
from os.path import expanduser
from skimage.measure import block_reduce
from xgboost import XGBClassifier as xgbclf
# from sklearn.ensemble import RandomForestClassifier

# Load Data
home_dir = expanduser("~")
train_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_train.csv')
Y_train = train_data['label'].values.astype(np.uint8)
train_data.drop(['label'],inplace=True, axis=1)
X_train = train_data.values.astype(np.float32)
X_train = (X_train - np.amin(X_train)) / (np.amax(X_train) - np.amin(X_train) + 0.0001)  # 0-1 scaling
N_train = X_train.shape[0]
X_train = X_train.reshape(N_train,28,28)

test_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_test.csv')
Y_test = test_data['label'].values.astype(np.uint8)
test_data.drop(['label'],inplace=True, axis=1)
X_test = test_data.values.astype(np.float32)
X_test = (X_test - np.amin(X_train)) / (np.amax(X_train) - np.amin(X_train) + 0.0001)  # 0-1 scaling
N_test = X_test.shape[0]
X_test = X_test.reshape(N_test,28,28)

pixels_per_cell = 4
N_hist_orient = 9
image_size = 28

img_zp = np.zeros((30,30))

last_idx1 = 0+(30/3-1)**2 * N_hist_orient*4
last_idx2 = last_idx1+(30/5-1)**2 * N_hist_orient*4
last_idx3 = last_idx2+(image_size/7-1)**2 * N_hist_orient*4
last_idx4 = last_idx3+(30/10-1)**2 * N_hist_orient*4
last_idx5 = last_idx4+(image_size/14-1)**2 * N_hist_orient*4

feat_train = np.zeros((N_train,last_idx5)).astype(np.float32)
feat_test = np.zeros((N_test,last_idx5)).astype(np.float32)
block_norm = 'L2'
print(feat_train.shape)
print(feat_test.shape)
for i in range(N_train):
	# pdb.set_trace()
	# feat_train[i, 0:last_idx1] = hog(X_train[i,:,:], orientations=N_hist_orient, pixels_per_cell=(2, 2), 
	# 					  cells_per_block=(2, 2), block_norm='L1', feature_vector=True)
	if np.mean(X_train[i,:,:14][:])>np.mean(X_train[i,:,14:][:]):
		X_train[i,:,:] = np.fliplr(X_train[i,:,:])

	X_train[i,:,:] = canny(X_train[i,:,:])
	img_zp[1:29,1:29] = X_train[i,:,:]
	feat_train[i, 0:last_idx1] = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(3, 3), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
	feat_train[i, last_idx1:last_idx2] = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(5, 5), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
	feat_train[i, last_idx2:last_idx3] = hog(X_train[i,:,:], orientations=N_hist_orient, pixels_per_cell=(7, 7), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
	feat_train[i, last_idx3:last_idx4] = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(10, 10), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
	feat_train[i, last_idx4:last_idx5] = hog(X_train[i,:,:], orientations=N_hist_orient, pixels_per_cell=(14, 14), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)

	if i<N_test:
		if np.mean(X_test[i,:,:14][:])>np.mean(X_test[i,:,14:][:]):
			X_test[i,:,:] = np.fliplr(X_test[i,:,:])
		
		X_test[i,:,:] = canny(X_test[i,:,:])
		img_zp[1:29,1:29] = X_test[i,:,:]
		feat_test[i, 0:last_idx1] = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(3, 3), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
		feat_test[i, last_idx1:last_idx2] = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(5, 5), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
		feat_test[i, last_idx2:last_idx3] = hog(X_test[i,:,:], orientations=N_hist_orient, pixels_per_cell=(7, 7), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
		feat_test[i, last_idx3:last_idx4] = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(10, 10), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)
		feat_test[i, last_idx4:last_idx5] = hog(X_test[i,:,:], orientations=N_hist_orient, pixels_per_cell=(14, 14), 
						  cells_per_block=(2, 2), block_norm=block_norm, feature_vector=True)

	
print('Feature extraction done')
#pdb.set_trace()

# rbm = BernoulliRBM(random_state=0, n_iter=100, n_components=200, batch_size=100, learning_rate=0.1)
# rbm.fit(feat_train)

# feat_train_reduce = np.dot(feat_train, rbm.components_.T)
# feat_test_reduce = np.dot(feat_test, rbm.components_.T)

# Models we will use
clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial')

# clf = xgbclf()

# clf = multiclass.OneVsOneClassifier(logistic, n_jobs=-1)
#clf = multiclass.OneVsRestClassifier(LinearSVC(random_state=0, C=100,kernel='poly'))
# clf = SVC(gamma='scale', kernel='poly', C=10.0, decision_function_shape='ovo', max_iter=1000, random_state=0)
# clf = LinearSVC(C=0.1, random_state=0, max_iter=100)

# clf = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=0, max_features=400)

# Training the Logistic regression classifier directly on the pixel
# raw_pixel_classifier = clf
#raw_pixel_classifier.C = 1.0
# raw_pixel_classifier.fit(feat_train, Y_train)

# feat_train_poly = np.hstack((feat_train,feat_train**2))
# feat_train_poly = np.hstack((feat_train_poly,feat_train**3))

# feat_test_poly = np.hstack((feat_test,feat_test**2))
# feat_test_poly = np.hstack((feat_test_poly,feat_test**3))

clf.fit(feat_train, Y_train)

# #############################################################################
# Evaluation
Y_pred = np.uint8(np.round(clf.predict(feat_test)))
print("XGboost classification using HOG features:\n%s\n" % (metrics.classification_report(Y_test, Y_pred)))

acc = np.sum(Y_pred==Y_test)*1.0/Y_test.shape[0]
print(acc)
# pdb.set_trace()
