# Created by Titas De
# Two View Resnet and DenseNet

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F



path_densenet_view1 = ""
path_densenet_view2 = ""

path_resnet_view1 = ""
path_resnet_view2 = ""


def densenet(view, weights, notop=True):

	# path of saved model
	if view == 'view1':
		path = path_densenet_view1
	elif view == 'view2':
		path = path_resnet_view2

	# weight initialization
	if weights == 'imagenet':
		net = models.densenet121(pretrained=True)
		net.classifier = nn.Linear(1024, 3)

	elif weights == 'random':
		net = models.densenet121(pretrained=False)
		net.classifier = nn.Linear(1024, 3)

	elif weights == 'oneview':
		net = models.densenet121(pretrained=False)
		net.classifier = nn.Linear(1024, 3)

		saved = torch.load(path)
		net.load_state_dict(saved['state_dict'])


	if notop == False:
		return net

	return net.features


class DenseNet_Twoview(nn.Module):
	def __init__(self, weights='imagenet', notop=True, **kwargs):
		super(DenseNet_Twoview, self).__init__()

		self.view1 = densenet(view='view1', weights=weights, notop=notop)
		self.view2 = densenet(view='view2', weights=weights, notop=notop)

		self.fc_twoview = nn.Linear(1024 * 2, 3)

	def forward(self, x1, x2):

		x1 = self.view1(x1)
		x1 = F.relu(x1, inplace=True)
		x1 = F.avg_pool2d(x1, kernel_size=7, stride=1).view(x1.size(0), -1)


		x2 = self.view2(x2)
		x2 = F.relu(x2, inplace=True)
		x2 = F.avg_pool2d(x2, kernel_size=7, stride=1).view(x2.size(0), -1)		

		x = self.fc_twoview(torch.cat((x1, x2), dim=1))
		return x


def resnet(view, weights, notop=True):

	# path of saved model
	if view == 'view1':
		path = path_resnet_view1
	elif view == 'view2':
		path = path_resnet_view2

	# weight initialization
	if weights == 'imagenet':
		net = models.resnet18(pretrained=True)
		net.fc = nn.Linear(512, 3)

	elif weights == 'random':
		net = models.resnet18(pretrained=False)
		net.fc = nn.Linear(512, 3)

	elif weights == 'oneview':
		net = models.resnet18(pretrained=False)
		net.fc = nn.Linear(512, 3)

		saved = torch.load(path)
		net.load_state_dict(saved['state_dict'])


	if notop == False:
		return net

	return nn.Sequential(*list(net.children())[:-1])

class ResNet_Twoview(nn.Module):
	def __init__(self, weights='imagenet', notop=True, **kwargs):
		super(ResNet_Twoview, self).__init__()

		self.view1 = resnet(view='view1', weights=weights, notop=notop)
		self.view2 = resnet(view='view2', weights=weights, notop=notop)

		self.fc_twoview = nn.Linear(512 * 2, 3)

	def forward(self, x1, x2):

		x1 = self.view1(x1)
		x1 = x1.view(x1.size(0), -1)

		x2 = self.view2(x2)
		x2 = x2.view(x2.size(0), -1)

		x = self.fc_twoview(torch.cat((x1, x2), dim=1))
		return x






# test densenet

import pdb
model = DenseNet_Twoview(weights='imagenet')


from torch.autograd import Variable
import numpy as np

arr = np.zeros((3, 224, 224))
brr = np.zeros((3, 224, 224))

import torch
arr = Variable(torch.from_numpy(arr).unsqueeze(0)).float()
brr= Variable(torch.from_numpy(brr).unsqueeze(0)).float()

outs = model(arr, brr)
pdb.set_trace()





# test resnet

import pdb
model = ResNet_Twoview(weights='imagenet')


from torch.autograd import Variable
import numpy as np

arr = np.zeros((3, 224, 224))
brr = np.zeros((3, 224, 224))

import torch
arr = Variable(torch.from_numpy(arr).unsqueeze(0)).float()
brr= Variable(torch.from_numpy(brr).unsqueeze(0)).float()

outs = model(arr, brr)
pdb.set_trace()
