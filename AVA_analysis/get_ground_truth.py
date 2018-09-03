'''
@author: Titas De
'''
import numpy as np
from os import path
import pdb

base_path = '/home/titasde08/AVA-db/AVA_dataset/'
existing_ids = np.load(base_path+'code_base/existing_ids.npy')
ids_dict = dict()
for i in range(len(existing_ids)):
	ids_dict[existing_ids[i]] = i

lines = np.array([line.rstrip('\r\n').split(' ')[1:12] for line in open(base_path+'AVA.txt')]).astype(np.int32)
ids_scores = np.array([lines[ids_dict[idx],:] for idx in existing_ids]).astype(np.int32)
np.save('ids_scores.npy',ids_scores)


