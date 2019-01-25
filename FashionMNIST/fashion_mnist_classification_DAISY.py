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
from skimage.feature import hog
from os.path import expanduser

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
N_hist_orient = 16
feat_train = np.zeros((N_train,(28/pixels_per_cell)**2*N_hist_orient)).astype(np.float32)
feat_test = np.zeros((N_test,(28/pixels_per_cell)**2*N_hist_orient)).astype(np.float32)
#feat_train = []
for i in range(N_train):
	#print(i)
	feat_train[i,:] = hog(X_train[i,:,:], orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(1, 1), feature_vector=True)
	if i<N_test:
		feat_test[i,:] = hog(X_test[i,:,:], orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(1, 1), feature_vector=True)

#pdb.set_trace()


# Models we will use
clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000,
                                           multi_class='multinomial')

#clf = multiclass.OneVsOneClassifier(logistic, n_jobs=-1)
#clf = multiclass.OneVsRestClassifier(LinearSVC(random_state=0, C=100,kernel='poly'))
clf = SVC(gamma='scale', kernel='poly', C=10.0, decision_function_shape='ovo', max_iter=1000, random_state=0)
#clf = LinearSVC(C=0.1, random_state=0)


# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clf
#raw_pixel_classifier.C = 1.0
raw_pixel_classifier.fit(feat_train, Y_train)

# #############################################################################
# Evaluation
Y_pred = raw_pixel_classifier.predict(feat_test)
print("Logistic regression using raw pixel features:\n%s\n" % (metrics.classification_report(Y_test, Y_pred)))


