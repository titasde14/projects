# @author: Titas De
# Fashion MNIST classification

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.ndimage.filters import uniform_filter
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
from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier

# Load Data
home_dir = expanduser("~")
train_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_train.csv')
Y_train = train_data['label'].values.astype(np.uint8)
train_data.drop(['label'],inplace=True, axis=1)
X_train = train_data.values.astype(np.float32)
X_train = (X_train - np.amin(X_train)) / (np.amax(X_train) - np.amin(X_train) + 0.0001)  # 0-1 scaling
# X_train[X_train<0.0] = 0.0
# X_train[X_train>1.0] = 1.0
N_train = X_train.shape[0]
X_train = X_train.reshape(N_train,28,28)

test_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_test.csv')
Y_test = test_data['label'].values.astype(np.uint8)
test_data.drop(['label'],inplace=True, axis=1)
X_test = test_data.values.astype(np.float32)
X_test = (X_test - np.amin(X_test)) / (np.amax(X_test) - np.amin(X_test) + 0.0001)  # 0-1 scaling
# X_test[X_test<0.0] = 0.0
# X_test[X_test>1.0] = 1.0
N_test = X_test.shape[0]
X_test = X_test.reshape(N_test,28,28)


pixels_per_cell = 3
cells_per_block = 2
N_hist_orient = 9
image_size = 28
img_zp = np.zeros((30,30))
img_zp[1:29,1:29] = X_train[0,:,:]
feat = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(cells_per_block, cells_per_block), block_norm='L1', feature_vector=True)
# pdb.set_trace()
num_iterations = 5
feat_train = np.zeros((N_train*num_iterations, len(feat))).astype(np.float32)
feat_test = np.zeros((N_test, len(feat))).astype(np.float32)
# feat_train = np.zeros((N_train,784))
# feat_test = np.zeros((N_test,784))
#feat_train = []
print(feat_train.shape)
print(feat_test.shape)

clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial', random_state=0)

for iter_num in range(num_iterations):
	print(iter_num)
	for i in range(N_train):
		img_zp[1:29,1:29] = X_train[i,:,:]
		img_zp_rot = rotate(img_zp,np.random.randint(-10,11,size=1),reshape=False)
		feat_train[iter_num*N_train+i,:] = hog(img_zp_rot, orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(cells_per_block, cells_per_block), block_norm='L1', feature_vector=True)
		
		if iter_num*N_train+i<N_test:
			img_zp[1:29,1:29] = X_test[i,:,:]
			feat_test[i,:] = hog(img_zp, orientations=N_hist_orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell), 
						  cells_per_block=(cells_per_block, cells_per_block), block_norm='L1', feature_vector=True)
		
	# clf.fit(feat_train, Y_train)


print('Feature extraction done')

# percent_nz = np.zeros((len(feat),1))
# for i in range(len(feat)):
# 	percent_nz[i,0] = np.sum(feat_train[i,:]>0.0)*1.0/N_train

# idx_to_del = np.where(percent_nz<0.010)[0]
# feat_train = np.delete(feat_train,idx_to_del,1)
# feat_test = np.delete(feat_test,idx_to_del,1)

# print(feat_train.shape)
# print(feat_test.shape)

# Feature normalization
# std_feat = np.std(feat_train,0)
# mean_feat = np.mean(feat_train,0)

# feat_train = (feat_train - mean_feat)/(std_feat+1e-6)
# feat_test = (feat_test - mean_feat)/(std_feat+1e-6)
Y_train = np.repeat(Y_train,num_iterations,axis=0)
clf.fit(feat_train, Y_train)

# pdb.set_trace()

# rbm = BernoulliRBM(random_state=0, n_iter=100, n_components=200, batch_size=100, learning_rate=0.1)
# rbm.fit(feat_train)

# feat_train_reduce = np.dot(feat_train, rbm.components_.T)
# feat_test_reduce = np.dot(feat_test, rbm.components_.T)

# Models we will use


# clf = MultinomialNB(alpha=1e-4)
# clf = xgbclf()

# clf = multiclass.OneVsOneClassifier(clf, n_jobs=-1)
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



# #############################################################################
# Evaluation
Y_pred = np.uint8(np.round(clf.predict(feat_test)))
acc = np.sum(Y_pred==Y_test)*1.0/Y_test.shape[0]
print(acc)
print("XGboost classification using HOG features:\n%s\n" % (metrics.classification_report(Y_test, Y_pred)))


