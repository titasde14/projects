import numpy as np
import pdb
from skimage.feature import hog
from os.path import expanduser
import pandas as pd

# Load Data
home_dir = expanduser("~")
train_data = pd.read_csv(home_dir+'/Downloads/Datasets/fashionmnist/fashion-mnist_train.csv')
Y_train = train_data['label'].values.astype(np.uint8)

idx_by_folds = [[]]*5

for label in range(5):

	indices = np.where(Y_train==label)[0]
	np.random.shuffle(indices)
	for i in range(5):
		if len(idx_by_folds[i])==0:
			idx_by_folds[i] = list(indices[i*1200:(i+1)*1200])
		else:
			idx_by_folds[i].extend(indices[i*1200:(i+1)*1200])

np.save('indices_by_fivefolds',idx_by_folds)