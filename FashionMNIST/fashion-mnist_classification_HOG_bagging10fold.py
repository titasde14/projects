# @author: Titas De
# Fashion MNIST classification 10-fold bagging

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, multiclass
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.svm import SVC, LinearSVC
import pandas as pd 
import pdb
from os.path import expanduser
from xgboost import XGBClassifier as xgbclf
from sklearn.naive_bayes import MultinomialNB

# Load Data
home_dir = expanduser("~")
features_labels = np.load(home_dir+'/Downloads/Datasets/fashionmnist/HOGfeatures-3x3pixels-2x2cells-L2.npz')
X_train = features_labels['X_train'].astype(np.float64)
X_test = features_labels['X_test'].astype(np.float64)
Y_train = features_labels['Y_train']
Y_test = features_labels['Y_test']

N_test = Y_test.shape[0]
Y_pred_votes = np.zeros((N_test,10))

del features_labels

indices_by_folds = np.load('indices_by_tenfolds.npy')


for fold_ctr in range(10):
	indices = []
	for j in range(10):
		if j!=fold_ctr:
			indices.extend(indices_by_folds[j])

	print(len(indices))
	X_train_fold = X_train[indices,:]
	Y_train_fold = Y_train[indices]

	clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=20, multi_class='multinomial', random_state=0)
	clf.fit(X_train_fold, Y_train_fold)

	y_pred_fold = clf.predict(X_test)
	Y_pred_votes[np.arange(N_test),y_pred_fold] += 1


# pdb.set_trace()
Y_pred = np.argmax(Y_pred_votes,axis=1)
acc = np.sum(Y_pred==Y_test)*1.0/N_test
print(acc)
print("Multinomial classification using HOG features + 10-fold bagging:\n%s\n" % (metrics.classification_report(Y_test, Y_pred)))


