'''
@author: Titas De
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


class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Encoder specification
        self.encode_cnn1 = nn.Conv2d(3, 6, kernel_size=5)
        self.encode_cnn2 = nn.Conv2d(6, 12, kernel_size=5)
        self.encode_cnn3 = nn.Conv2d(12, 24, kernel_size=5)
        self.encode_linear1 = nn.Linear(4 * 4 * 24, 120)
        self.encode_linear2 = nn.Linear(120, 40)
        
        # Decoder specification
        self.decode_linear1 = nn.Linear(40, 600)
        self.decode_linear2 = nn.Linear(600, 3600*3)
        
    def forward(self, images):
        vec = self.encode(images)
        out = self.decode(vec)
        return out, vec
    
    def encode(self, images):
        vec = self.encode_cnn1(images)
        vec = F.relu(F.max_pool2d(vec, 2))
        
        vec = self.encode_cnn2(vec)
        vec = F.relu(F.max_pool2d(vec, 2))
        
        vec = self.encode_cnn3(vec)
        vec = F.relu(F.max_pool2d(vec, 2))

        vec = vec.view([images.size(0), -1])
        vec = F.relu(self.encode_linear1(vec))
        vec = self.encode_linear2(vec)
        vec = vec.view([vec.size(0), -1])
        return vec
    
    def decode(self, vec):
        out = F.relu(self.decode_linear1(vec))
        out = F.sigmoid(self.decode_linear2(out))
        out = out.view([vec.size(0), 3, 60, 60])
        return out
    

class CAEDataset(Dataset):

    def __init__(self, img_array):
        self.img_array = img_array
        
    def __len__(self):
        return len(self.img_array.shape[0])

    def __getitem__(self, idx):
        return self.img_array[idx,:,:,:]


# Hyperparameters
num_epochs = 1000
batch_size = 100

# Load data
data = np.load('all_data_3x60x60.npy')
train_data = CAEDataset(np.float64(data.item(0)['x_train'])/255.0) # converting from uint8 to float64
test_data  = CAEDataset(np.float64(data.item(0)['x_test'])/255.0)

del data
gc.collect()

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=16)

# Instantiate model
autoencoder = AutoEncoder().cuda()
loss_fn = nn.BCELoss().cuda()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

# Training loop
autoencoder.train(True)
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
    
    for images in train_loader:   
        out, vec = autoencoder(Variable(images.float().cuda()))
        
        optimizer.zero_grad()
        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()
        
    print("Loss = %.3f" % loss.data[0])

autoencoder.save_state_dict('AVA_imgs_trained.pt')

# saving the encoder vectors
train_vec = np.zeros((train_data.size()[0],40))
test_vec = np.zeros((test_data.size()[0],40))

autoencoder.train(False)
autoencoder.zero_grad()
for i in range(len(train_data.size()[0])):
    _,train_vec[i,:] = autoencoder(Variable(train_data[i,:,:,:].float().cuda())).numpy()

for i in range(len(test_data.size()[0])):
    _,test_vec[i,:] = autoencoder(Variable(test_data[i,:,:,:].float().cuda())).numpy()

np.save('train&test_vec.npy',{'train_vec':train_vec,'test_vec':test_vec})
