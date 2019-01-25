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
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import precision_recall_curve
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier

# Load Data
home_dir = expanduser("~")
features_labels = np.load(home_dir+'/Downloads/Datasets/fashionmnist/HOGfeatures-2x2pixels-2x2cells-L2.npz')
X_train = features_labels['X_train'].astype(np.float64)
X_test = features_labels['X_test'].astype(np.float64)
Y_train = features_labels['Y_train']
Y_test = features_labels['Y_test']

Y_train_2class = np.ones(Y_train.shape)
pos = (Y_train==6)
Y_train_2class[pos] = 0

Y_test_2class = np.ones(Y_test.shape)
pos = (Y_test==6)
Y_test_2class[pos] = 0


# tic = time.time()
# clf = RandomForestClassifier(n_estimators=100, random_state=0)
# clf.fit(X_train, Y_train_2class)
# print(time.time()-tic)
# Y_pred = clf.predict(X_test)
# print(metrics.classification_report(Y_test_2class, Y_pred))

# feat_imp = np.zeros((X_train.shape[1],2))
# feat_imp[:,0] = np.arange(X_train.shape[1])
# feat_imp[:,1] = clf.feature_importances_
# feat_imp = feat_imp[feat_imp[:,1].argsort()][::-1] 

# cum_imp = np.cumsum(feat_imp[:,1])
# idx = np.where(cum_imp>0.80)[0]

# X_train_reduced = np.delete(X_train,idx,axis=1)
# X_test_reduced = np.delete(X_test,idx,axis=1)

clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, multi_class='ovr', random_state=0)
# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
# clf = xgbclf(n_estimators=200,learning_rate=0.1,max_depth=2, random_state=0)
clf.fit(X_train, Y_train_2class)

Y_pred_2class = clf.predict(X_test)

acc = np.sum(Y_pred_2class==Y_test_2class)*1.0/Y_test.shape[0]
print(acc)
print(metrics.classification_report(Y_test_2class, Y_pred_2class))

Y_score = clf.decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(Y_test_2class, Y_score)
F1_scores = 2*precision*recall/(precision+recall)
idx = np.argmax(F1_scores)

#pdb.set_trace()





