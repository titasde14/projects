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

N_size = 60

def get_image(idx):
	print(idx)
	img = cv.imread(base_path+'images/images/'+str(idx)+'.jpg')
	img_new = np.zeros((3,N_size,N_size))
	img_new[0,:,:] = cv.resize(img[:,:,0],(N_size,N_size))
	img_new[1,:,:] = cv.resize(img[:,:,1],(N_size,N_size))
	img_new[2,:,:] = cv.resize(img[:,:,2],(N_size,N_size))
	del img
	gc.collect()
	return img_new

def check_file_exist(idx):
	if path.exists(base_path+'images/images/'+str(idx)+'.jpg')==True:
		return idx
	else:
		return -1

base_path ='/home/titasde08/AVA-db/AVA_dataset/'
num_cores = multiprocessing.cpu_count()
data = np.load(base_path+'data/idsWscores.npy')
id_list = data.item(0)['id_list']
all_ids = [id_list[i] for i in range(id_list.shape[0])]

tic = time.time()
existing_ids = Parallel(n_jobs=num_cores)(delayed(check_file_exist)(i) for i in all_ids)
print(time.time()-tic)
print(len(existing_ids))
existing_ids = sorted([idx for idx in existing_ids if idx>-1])
print(len(existing_ids))

del all_ids
gc.collect()
np.save('existing_ids.npy',existing_ids)

tic = time.time()
all_imgs = np.array(Parallel(n_jobs=num_cores)(delayed(get_image)(i) for i in existing_ids))
print(time.time()-tic)

np.save('all_images_3x'+str(N_size)+'x'+str(N_size)+'.npy',all_imgs)
pdb.set_trace()




