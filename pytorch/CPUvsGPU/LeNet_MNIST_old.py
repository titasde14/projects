# created by Titas De

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms,datasets
import torch.optim as optim
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import torchvision
#import mkl

from ctypes import *
# Load the share library
mkl = cdll.LoadLibrary("libmkl_rt.so")
# mkl = cdll.LoadLibrary("/opt/intel/intelpython2/lib/libmkldnn.so")

import copy
import time
import gc

# Define network architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)        
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)



# Load data:
apply_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
BatchSize = 100
num_workers = 1
print("data_loader num workers = ",num_workers)


trainset = datasets.MNIST(root='./MNIST', train=True, download=True, transform=apply_transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize,
                                      shuffle=False, num_workers=num_workers) # Creating dataloader

testset = datasets.MNIST(root='./MNIST', train=False, download=True, transform=apply_transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=BatchSize,
                                     shuffle=False, num_workers=num_workers) # Creating dataloader


# Size of train and test datasets
print('No. of samples in train set: '+str(len(trainLoader.dataset)))
print('No. of samples in test set: '+str(len(testLoader.dataset)))
print(LeNet())
print('\n\n')


num_threads = 1

while num_threads<=32:
    torch.set_num_threads(num_threads)
    
    print("torch num threads = ",torch.get_num_threads())

    mkl.mkl_set_num_threads(byref(c_int(num_threads))) # try to set number of mkl threads
    print("mkl get max thread = ", mkl.mkl_get_max_threads())

    net = LeNet()
    # print(net)


# Check availability of GPU
    #use_gpu = torch.cuda.is_available()
    use_gpu = False
    if use_gpu:
        print('GPU is available!')   
        net = net.cuda()

    # Define loss function and optimizer
    criterion = nn.NLLLoss() # Negative Log-likelihood
    optimizer = optim.Adam(net.parameters(), lr=1e-4) # Adam

# Train the network

    iterations = 10
    trainLoss = []
    testAcc = []
    start = time.time()
    for epoch in range(iterations):
        epochStart = time.time()
        runningLoss = 0    
        net.train(True) # For training
        for data in trainLoader:
            inputs,labels = data
            # Wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)  
       
            # Initialize gradients to zero
            optimizer.zero_grad()
            # Feed-forward input data through the network        
            outputs = net(inputs)        
            # Compute loss/error
            loss = criterion(outputs, labels)
            # Backpropagate loss and compute gradients
            loss.backward()
            # Update the network parameters
            optimizer.step()
            # Accumulate loss per batch
            runningLoss += loss.data[0]    

            # del inputs, labels, loss, outputs
            # gc.collect()

        avgTrainLoss = runningLoss/60000.0
        trainLoss.append(avgTrainLoss)
    
        # Evaluating performance on test set for each epoch
        net.train(False) # For testing [Affects batch-norm and dropout layers (if any)]
        running_correct = 0
        for data in testLoader:
            inputs,labels = data
            # Wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu()
            else:
                inputs = Variable(inputs)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum()
            # del inputs, labels, outputs, predicted
            # gc.collect()

        avgTestAcc = running_correct/10000.0
        testAcc.append(avgTestAcc)
        
        epochEnd = time.time()-epochStart
        print('Iteration: {:2.0f}/{:2.0f}  ;  Training Loss: {:.6f} ; Testing Acc: {:.3f} ; Time consumed: {:.0f}m {:.0f}s '\
              .format(epoch + 1,iterations,avgTrainLoss,avgTestAcc*100,epochEnd//60,epochEnd%60))
    end = time.time()-start
    print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
    print('Average Training Time = {:.2f}s'.format(end/10))
    print('\n')

    del net, optimizer, criterion, inputs, outputs, labels
    gc.collect()
    num_threads = num_threads * 2

