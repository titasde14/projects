'''
@author: Titas De
'''

import numpy as np
import scipy as sp
import cv2 as cv
from joblib import Parallel, delayed
import multiprocessing
import time
from os import path
import pdb
import gc

base_path ='/home/titasde08/AVA-db/AVA_dataset/images/images/'
existing_ids = np.load('existing_ids.npy')

min_X = 1e6
min_Y = 1e6
max_val = -1
ctr = -1
for idx in existing_ids:
	ctr += 1
	#print(ctr,idx)
	file_name = base_path+str(idx)+'.jpg'
	img = cv.imread(file_name)
	min_X = min(min_X,img.shape[0])
	min_Y = min(min_Y,img.shape[1])
	max_val = max(max_val,np.amax(img[:]))
	print(min_X,min_Y,max_val)
	del img
	gc.collect()
pdb.set_trace()




