# @author: Titas De
# Fashion MNIST classification

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



# Load Data
train_data = pd.read_csv('/Users/titas/Downloads/fashionmnist/fashion-mnist_train.csv')
Y_train = train_data['label'].values.astype(np.uint8)
train_data.drop(['label'],inplace=True, axis=1)
X_train = train_data.values.astype(np.float32)
X_train = (X_train - np.amin(X_train)) / (np.amax(X_train) - np.amin(X_train) + 0.0001)  # 0-1 scaling

test_data = pd.read_csv('/Users/titas/Downloads/fashionmnist/fashion-mnist_test.csv')
Y_test = test_data['label'].values.astype(np.uint8)
test_data.drop(['label'],inplace=True, axis=1)
X_test = test_data.values.astype(np.float32)
X_test = (X_test - np.amin(X_test)) / (np.amax(X_test) - np.amin(X_test) + 0.0001)  # 0-1 scaling



# Models we will use
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000,
                                           multi_class='multinomial')


# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = logistic
raw_pixel_classifier.C = 100.0
# raw_pixel_classifier.fit(X_train, Y_train)

# #############################################################################
# Evaluation

# Y_pred = raw_pixel_classifier.predict(X_test)
# print("Logistic regression using raw pixel features:\n%s\n" % (
#     metrics.classification_report(Y_test, Y_pred)))


N_train = X_train.shape[0]
N_test = X_test.shape[0]
X_train2 = np.swapaxes(X_train.reshape(N_train,28,28),1,2).reshape(N_train,784)
X_test2 = np.swapaxes(X_test.reshape(N_test,28,28),1,2).reshape(N_test,784)
# raw_pixel_classifier.fit(X_train2, Y_train)
# Y_pred2 = raw_pixel_classifier.predict(X_test2)
# print("Logistic regression using raw pixel features transposed:\n%s\n" % (
#     metrics.classification_report(Y_test, Y_pred2)))

#pdb.set_trace()

X_train3 = np.hstack((X_train,X_train2))
X_test3 = np.hstack((X_test,X_test2))
#pdb.set_trace()
raw_pixel_classifier.fit(X_train3, Y_train)
Y_pred3 = raw_pixel_classifier.predict(X_test3)
print("Logistic regression using raw pixel features and its transpose:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred3)))

