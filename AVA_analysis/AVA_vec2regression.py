'''
@author: Titas De

The output of this script is X_final_score which stores the images ids in sorted order 
from the lowest aesthetic quality to the highest

'''

import numpy as np
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from skimage import transform


class AVA_Regressor(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Linear Layers
        self.layer1 = nn.Linear(40, 20)
        self.layer2 = nn.Linear(20, 10)
        
        
    def forward(self, vec):
        vec = F.relu(self.layer1(vec))
        vec = F.Softmax(self.layer2(vec))
        return vec
    

class AVAVecDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X.shape[0])

    def __getitem__(self, idx):
        return (self.X[idx,:],self.Y[idx])


# Hyperparameters
num_epochs = 1000
batch_size = 100

# Load features
data = np.load('train&testvec_vec.npy')
x_train = data.item(0)['train_vec'] 
x_test  = data.item(0)['test_vec']

del data
gc.collect()

# Load targets
data = np.load('all_data_3x60x60.npy')
y_train = np.float64(data.item(0)['y_train'][:,1:]) # ignroing the 1st column depicting image ID
y_test  = np.float64(data.item(0)['y_test'][:,1:])

del data
gc.collect()

# normalizing the scores across the rows
for i in range(y_train):
    y_train[i,:] = y_train[i,:]/np.sum(y_train[i,:]+1e-6)

for i in range(y_test):
    y_test[i,:] = y_test[i,:]/np.sum(y_test[i,:]+1e-6)


train_data = AVAVecDataset(x_train,y_train)
test_data = AVAVecDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)
test_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=16)

# Instantiate model
ava_model = AVA_Regressor().cuda()
loss_fn = nn.MSELoss().cuda()
optimizer = optim.Adam(ava_model.parameters(), lr=1e-3)

# Training loop
ava_model.train(True)
ava_model.zero_grad()
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
    
    for train_batch in train_loader:
        input_X = Variable(train_batch[0].float().cuda())
        input_Y = Variable(train_batch[1].float().cuda())   
        output_Y = ava_model(input_X)
        
        optimizer.zero_grad()
        loss = loss_fn(output_Y, input_Y)
        loss.backward()
        optimizer.step()
        
    print("Loss = %.3f" % loss.data[0])

ava_model.save_state_dict('AVA_vec_trained.pt')


X_distrib = np.zeros((x_train.shape[0],10))
ranks = np.array(range(1,11)).astype(np.float64) # rank array from 1 to 10
X_final_score = np.zeros((x_train.shape[0],3))

ava_model.train(False)
ava_model.zero_grad()
for i in range(len(x_train.shape[0])):
    X_distrib[i,:] = ava_model(Variable(x_train[i,:].float().cuda())).numpy()
    X_final_score[i,0] = i
    X_final_score[i,1] = y_train[i,0] # image ID
    X_final_score[i,2] = np.dot(X_distrib[i,:],ranks)/np.sum(X_distrib[i,:]+1e-6)

X_final_score = X_final_score[X_final_score[:,2].argsort()] # being sorted from the lowest aesthetic quality to the highest

np.save('sortedimages_lowtohigh_aesthetic_quality',X_final_score)
  



