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

# from ctypes import *
# Load the share library
# mkl = cdll.LoadLibrary("libmkl_rt.so")
# mkl = cdll.LoadLibrary("/opt/intel/intelpython2/lib/libmkldnn.so")

import copy
import time
import gc
import pdb


def two_conv_pool(F0, F1, F2, p):
    model = nn.Sequential(nn.Conv2d(F0,F1,kernel_size=3, padding=p), \
                            nn.BatchNorm2d(F1), nn.ReLU(), \
                            nn.Conv2d(F1,F2,kernel_size=3, padding=p), \
                            nn.BatchNorm2d(F2), nn.ReLU(), \
                            nn.MaxPool2d(kernel_size=2,stride=2))
    # x = nn.Conv2D(out_features=F1, kernel_size=3)(x)
    # x = nn.BatchNorm2d()(x)
    # x = F.relu(x)
    # x = nn.Conv2D(out_features=F2, kernel_size=3)(x)
    # x = nn.BatchNorm2d()(x)
    # x = F.relu(x)
    # x = nn.MaxPool2d(kernel_size=2,stride=2)(x)
    # return x
    return model

def three_conv_pool(F0, F1, F2, F3):
    # x = nn.Conv2D(out_features=F1, kernel_size=3)(x)
    # x = nn.BatchNorm2d()(x)
    # x = F.relu(x)
    # x = nn.Conv2D(out_features=F2, kernel_size=3)(x)
    # x = nn.BatchNorm2d()(x)
    # x = F.relu(x)
    # x = nn.Conv2D(out_features=F3, kernel_size=3)(x)
    # x = nn.BatchNorm2d()(x)
    # x = F.relu(x)
    # x = nn.MaxPool2d(kernel_size=2,stride=2)(x)
    # return x
    model = nn.Sequential(nn.Conv2d(F0,F1,kernel_size=3,padding=1), \
                            nn.BatchNorm2d(F1), nn.ReLU(), \
                            nn.Conv2d(F1,F2,kernel_size=3,padding=1), \
                            nn.BatchNorm2d(F2), nn.ReLU(), \
                            nn.Conv2d(F2,F3,kernel_size=3,padding=1), \
                            nn.BatchNorm2d(F3), nn.ReLU(), \
                            nn.MaxPool2d(kernel_size=2,stride=2))
    return model


# Define network architecture
class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        #self.bn1 = nn.BatchNorm2d()
        
        #self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        #self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        #self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)        
        #self.fc1 = nn.Linear(400, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        self.two_conv1 = two_conv_pool(1,32,32,0)
        self.two_conv2 = two_conv_pool(32,64,64,1)
        self.three_conv1 = three_conv_pool(64,128,128,128)
        self.three_conv2 = three_conv_pool(128,256,256,256)
        
        self.fc1 = nn.Linear(256,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,10)


    

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = self.pool1(x)
        #x = F.relu(self.conv2(x))
        #x = self.pool2(x)
        #x = x.view(-1, 400)
        #x = F.relu(self.fc1(x)) 
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        # pdb.set_trace()
        
        x = self.two_conv1(x)
        #print(x)
        x = self.two_conv2(x)
        #print(x)
        x = self.three_conv1(x)
        #print(x)
        x = self.three_conv2(x)
        #print(x)
        x = x.view(-1,256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)


        return F.log_softmax(x)



# Load data:
apply_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
BatchSize = 60000
num_workers = 1
print("data_loader num workers = ",num_workers)

trainset = datasets.MNIST(root='./MNIST', train=True, download=True, transform=apply_transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize,
                                      shuffle=False, num_workers=num_workers) # Creating dataloader

for data in trainLoader:
	inputs_train,labels_train = data
	 


testset = datasets.MNIST(root='./MNIST', train=False, download=True, transform=apply_transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=BatchSize,
                                     shuffle=False, num_workers=num_workers) # Creating dataloader

for data in testLoader:
	inputs_test,labels_test = data


# Size of train and test datasets
print('No. of samples in train set: '+str(len(trainLoader.dataset)))
print('No. of samples in test set: '+str(len(testLoader.dataset)))
#print(LeNet())
print('\n\n')


# num_thread_array = [1,2,4,8,12,16,32]*5
num_thread_array = [1]

thread_ctr = 0

while thread_ctr<len(num_thread_array):
    num_threads = num_thread_array[thread_ctr]
    torch.set_num_threads(num_threads)
    
    print("torch num threads = ",torch.get_num_threads())

    # mkl.mkl_set_num_threads(byref(c_int(num_threads))) # try to set number of mkl threads
    # print("mkl get max thread = ", mkl.mkl_get_max_threads())
    #pdb.set_trace()
    net = VGG()
    print(net)

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

    iterations = 1000
    trainLoss = []
    testAcc = []
    start = time.time()
    for epoch in range(iterations):
        epochStart = time.time()
        runningLoss = 0    
        net.train(True) # For training
        for train_ctr in range(0,600):
            inputs = inputs_train[train_ctr*100:(train_ctr+1)*100,:,:,:]
            labels = labels_train[train_ctr*100:(train_ctr+1)*100]	
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
        for test_ctr in range(0,100):
            inputs = inputs_test[test_ctr*100:(test_ctr+1)*100,:,:,:]
            labels = labels_test[test_ctr*100:(test_ctr+1)*100]	
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
        # print(avgTestAcc)
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
    # num_threads = num_threads * 2
    thread_ctr += 1

