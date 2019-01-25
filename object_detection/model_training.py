import numpy as np 
import pdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = np.load('labelled_video_pixel_feat.npz')
x_feat, y_feat = data['x_feat'], data['y_feat']
#pdb.set_trace()

indices_0 = np.where(y_feat==0)[0]
indices_1 = np.where(y_feat==1)[0]
indices_2 = np.where(y_feat==2)[0]

np.random.shuffle(indices_0)
np.random.shuffle(indices_1)
np.random.shuffle(indices_2)

x_train = []
x_val = []
y_train = []
y_val = []

mid_idx_0 = int(len(indices_0)*0.8)
mid_idx_1 = int(len(indices_1)*0.8)
mid_idx_2 = int(len(indices_2)*0.8)

x_train.extend(x_feat[indices_0[:mid_idx_0],:])
y_train.extend(np.ones((mid_idx_0,))*0)
x_train.extend(x_feat[indices_1[:mid_idx_1],:])
y_train.extend(np.ones((mid_idx_1,))*1)
x_train.extend(x_feat[indices_2[:mid_idx_2],:])
y_train.extend(np.ones((mid_idx_2,))*2)

x_val.extend(x_feat[indices_0[mid_idx_0:],:])
y_val.extend(np.ones((len(indices_0)-mid_idx_0,))*0)
x_val.extend(x_feat[indices_1[mid_idx_1:],:])
y_val.extend(np.ones((len(indices_1)-mid_idx_1,))*1)
x_val.extend(x_feat[indices_2[mid_idx_2:],:])
y_val.extend(np.ones((len(indices_2)-mid_idx_2,))*2)

x_val = np.array(x_val)
y_val = np.array(y_val)
x_train = np.array(x_train)
y_train = np.array(y_train)


clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, class_weight='balanced', multi_class='multinomial').fit(x_train, y_train)
y_pred = clf.predict(x_val)

print(classification_report(y_val, y_pred, target_names=['class_0','class_1','class_2']))
#pdb.set_trace()
