# @author: Titas De
# RBM for Fashion MNIST classification

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import pandas as pd 
import pdb
from os.path import expanduser


home_dir = expanduser("~")

# Load Data
train_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_train.csv')
Y_train = train_data['label'].values.astype(np.uint8)
train_data.drop(['label'],inplace=True, axis=1)
X_train = train_data.values.astype(np.float32)
X_train = (X_train - np.amin(X_train)) / (np.amax(X_train) - np.amin(X_train) + 0.0001)  # 0-1 scaling

test_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_test.csv')
Y_test = test_data['label'].values.astype(np.uint8)
test_data.drop(['label'],inplace=True, axis=1)
X_test = test_data.values.astype(np.float32)
X_test = (X_test - np.amin(X_train)) / (np.amax(X_train) - np.amin(X_train) + 0.0001)  # 0-1 scaling

rbm = BernoulliRBM(random_state=0, n_iter=100, n_components=200, batch_size=100, learning_rate=0.1)
rbm.fit(X_train)

# X_train_reduce = np.zeros((X_train.shape[0],100))
# X_test_reduce = np.zeros((X_test.shape[0],100))
# for i in range(X_train.shape[0]):
# 	#pdb.set_trace()
# 	# X_train_reduce[i,:] = np.dot((X_train[i,:] - np.expand_dims(rbm.intercept_visible_, axis=0)),rbm.components_.T)+np.expand_dims(rbm.intercept_hidden_,axis=0)
# 	# if i<X_test.shape[0]:
# 	# 	X_test_reduce[i,:] = np.dot((X_test[i,:] - np.expand_dims(rbm.intercept_visible_, axis=0)),rbm.components_.T)+np.expand_dims(rbm.intercept_hidden_,axis=0)

# 	X_train_reduce[i,:] = np.dot(X_train[i,:], rbm.components_.T)
# 	if i<X_test.shape[0]:
# 		X_test_reduce[i,:] = np.dot(X_test[i,:], rbm.components_.T)
	

X_train_reduce = np.dot(X_train, rbm.components_.T)
X_test_reduce = np.dot(X_test, rbm.components_.T)

#X_train_reduce = rbm.transform(X_train)
#X_test_reduce = rbm.transform(X_test)

# X_train_reduce = X_train
# X_test_reduce = X_test

# Models we will use
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
logistic.fit(X_train_reduce, Y_train)

# #############################################################################
# Evaluation

Y_pred = logistic.predict(X_test_reduce)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred)))


# # #############################################################################
# # Plotting

# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(rbm.components_):
#     plt.subplot(n_comp_sqrt, n_comp_sqrt, i + 1)
#     plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle(str(rbm.n_components)+' components extracted by RBM', fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# plt.show()
