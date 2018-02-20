# created by Titas De

from __future__ import print_function
import numpy as np
import pdb
import torch.utils.data as data_utils
import argparse
import torch.nn.functional as F
from torchvision import models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn as nn
import torch
import torchvision
# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, DataLoader
import time
import gc
import shutil
from skimage import transform
import copy
# import inception_dropoutcontrol
import inception_v3_withdropoutcontrol

# import vgg_dropoutcontrol
import random

MANUAL_SEED = 37
torch.manual_seed(MANUAL_SEED)
# torch.manual_seed_all(MANUAL_SEED)
torch.cuda.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed_all(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
# np.random.seed_all(MANUAL_SEED)
random.seed(MANUAL_SEED)
# random.seed_all(MANUAL_SEED)



class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        c, h, w = image.shape
        h = float(h)
        w = float(w)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = np.zeros((c,new_h,new_w))
        for ctr in range(0,c):
            img[ctr,:,:] = transform.resize(image[ctr,:,:], (new_h, new_w),mode='constant')

        return {'image': img, 'label': label}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        c, h, w = image.shape
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = np.zeros((c,new_h,new_w))
        for ctr in range(0,c):
            img[ctr,:,:] = image[ctr,top: top + new_h,
                      left: left + new_w]
        return {'image': img, 'label': label}


class RandomRotate(object):

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        c, h, w = image.shape
        degree_rot = (2*np.random.rand(1)[0]-1)*self.degrees
        img = np.zeros((c,h,w))
        for ctr in range(0,c):
            img[ctr,:,:] = transform.rotate(image[ctr,:,:], angle=degree_rot, order=5, resize=False)
        return {'image': img, 'label': label}


class MyDataset(Dataset):

	def __init__(self, img_array, labels_array, idx_array, my_transforms=None):
		self.img_array = img_array
		self.labels_array = labels_array
		self.idx_array = idx_array
		self.required_transforms = my_transforms
	
	def __len__(self):
		return len(self.labels_array)

	def __getitem__(self, index):
		# print(idx)
		idx = self.idx_array[index]
		curr_img = self.img_array[idx]
		curr_label = self.labels_array[idx]
		sample = {'image': curr_img, 'label': curr_label}
		if self.required_transforms:
			sample = self.required_transforms(sample)
		return sample

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, choices=['v','i','r','d'], required=True, help="Which model architecture to use ?")
parser.add_argument('--pretrained', type=str, choices=['p', 'r'], required=True, help="Load pretrained model or random initialization ?")
parser.add_argument('--epochs', type=int, default=2, required=False, help='No: of epochs')
parser.add_argument('--batch_size', type=int, default=2, required=False, help='Batch size')
parser.add_argument('--batch_size_shuffle', type=str, choices=['T','F'], required=True, help='whether to shuffle the batch size or not')
parser.add_argument('--num_workers', type=int, default=2, required=False, help='Number of workers')
parser.add_argument('--val_fold', type=int, default=0, choices=[0,1,2,3,4], required=False, help='Validation fold number')
parser.add_argument('--classification_type', type=str, choices=['0vs1','1vs2','all'], required=False, help='which classes to choose')
parser.add_argument('--transform_input', type=bool, default=False, choices=[True,False], required=False, help='Whether to transform_input or not')
parser.add_argument('--dropout_prob', type=float, default=0.2, required=False, help='dropout probability')
parser.add_argument('--just_top_layer', type=str, choices=['T','F'], required=True, help='whether to fine-tune just the top layer or the entire architecture')
parser.add_argument('--entire_dataset', type=str, choices=['T','F'], required=True, help='whether to run the model on the entire dataset or a small subset')



args = parser.parse_args()
model_name = args.model_name
pretrain = args.pretrained
batch_size = args.batch_size
batch_size_shuffle = args.batch_size_shuffle
epochs = args.epochs
num_workers=args.num_workers
val_fold = args.val_fold
classification_type = args.classification_type
transform_input = args.transform_input
dropout_prob = args.dropout_prob
just_top_layer = args.just_top_layer
entire_dataset = args.entire_dataset

if 'T' in batch_size_shuffle:
	batch_size_shuffle = True
else:
	batch_size_shuffle = False


if 'p' in pretrain:
	pretrained = True
else:
	pretrained = False

if 'T' in just_top_layer:
	just_top_layer = True
else:
	just_top_layer = False

if 'T' in entire_dataset:
	entire_dataset = True
else:
	entire_dataset = False

base_path = '/home/Titas/'

k_folds_path = base_path + '5folds.npy'


all_features = np.load(base_path + 'all_features.npy')
all_labels = np.load(base_path + 'all_labels.npy')
all_bit_depths = np.load(base_path + 'all_bit_depths.npy')

if 'all' in classification_type:
	num_classes = 3
else:
	num_classes = 2

if '0vs1' in classification_type:
	all_labels[all_labels>0] = 1


folds = np.load(k_folds_path)
train_ids = []
val_ids = []
for i in range(0,5):
	if i != val_fold:
		for idx_4 in folds[i]:
			train_ids.append(idx_4[0])
			train_ids.append(idx_4[1])
			train_ids.append(idx_4[2])
			train_ids.append(idx_4[3])
	else:
		for idx_4 in folds[i]:
			val_ids.append(idx_4[0])
			val_ids.append(idx_4[1])
			val_ids.append(idx_4[2])
			val_ids.append(idx_4[3])


if '1vs2' in classification_type:
	pos = np.nonzero(all_labels>0)[0]
	all_labels -= 1 
	train_ids =  list(set(pos).intersection(set(train_ids)))
	val_ids =  list(set(pos).intersection(set(val_ids)))


all_ids = range(0,len(all_labels))

bd12_ids = np.nonzero(all_bit_depths==12)[0]
bd16_ids = np.nonzero(all_bit_depths==16)[0]
train_bd12_ids =  np.array(list(set(bd12_ids).intersection(set(train_ids))), dtype = np.uint16)
train_bd16_ids =  np.array(list(set(bd16_ids).intersection(set(train_ids))), dtype = np.uint16)

bd12_ids =  np.array(list(set(bd12_ids).intersection(set(all_ids))), dtype = np.uint16)
bd16_ids =  np.array(list(set(bd16_ids).intersection(set(all_ids))), dtype = np.uint16)

all_features = all_features.astype(np.float32)

train_bd12_ch0_minval = np.amin(all_features[train_bd12_ids,0,:,:])
train_bd12_ch1_minval = np.amin(all_features[train_bd12_ids,1,:,:])
train_bd12_ch2_minval = np.amin(all_features[train_bd12_ids,2,:,:])

train_bd12_ch0_maxval = np.amax(all_features[train_bd12_ids,0,:,:])
train_bd12_ch1_maxval = np.amax(all_features[train_bd12_ids,1,:,:])
train_bd12_ch2_maxval = np.amax(all_features[train_bd12_ids,2,:,:])

train_bd16_ch0_minval = np.amin(all_features[train_bd16_ids,0,:,:])
train_bd16_ch1_minval = np.amin(all_features[train_bd16_ids,1,:,:])
train_bd16_ch2_minval = np.amin(all_features[train_bd16_ids,2,:,:])

train_bd16_ch0_maxval = np.amax(all_features[train_bd16_ids,0,:,:])
train_bd16_ch1_maxval = np.amax(all_features[train_bd16_ids,1,:,:])
train_bd16_ch2_maxval = np.amax(all_features[train_bd16_ids,2,:,:])

del all_bit_depths

all_features[bd12_ids,0,:,:] = (all_features[bd12_ids,0,:,:] - train_bd12_ch0_minval)/(train_bd12_ch0_maxval - train_bd12_ch0_minval)
all_features[bd12_ids,1,:,:] = (all_features[bd12_ids,1,:,:] - train_bd12_ch1_minval)/(train_bd12_ch1_maxval - train_bd12_ch1_minval)
all_features[bd12_ids,2,:,:] = (all_features[bd12_ids,2,:,:] - train_bd12_ch2_minval)/(train_bd12_ch2_maxval - train_bd12_ch2_minval)

all_features[bd16_ids,0,:,:] = (all_features[bd16_ids,0,:,:] - train_bd16_ch0_minval)/(train_bd16_ch0_maxval - train_bd16_ch0_minval)
all_features[bd16_ids,1,:,:] = (all_features[bd16_ids,1,:,:] - train_bd16_ch1_minval)/(train_bd16_ch1_maxval - train_bd16_ch1_minval)
all_features[bd16_ids,2,:,:] = (all_features[bd16_ids,2,:,:] - train_bd16_ch2_minval)/(train_bd16_ch2_maxval - train_bd16_ch2_minval)

all_features[all_features>1] = 1
all_features[all_features<0] = 0

#all_features[bd12_ids,0,:,:] = 2*(all_features[bd12_ids,0,:,:] - 0.5*(train_bd12_ch0_minval + train_bd12_ch0_maxval))/(train_bd12_ch0_maxval - train_bd12_ch0_minval)
#all_features[bd12_ids,1,:,:] = 2*(all_features[bd12_ids,1,:,:] - 0.5*(train_bd12_ch1_minval + train_bd12_ch1_maxval))/(train_bd12_ch1_maxval - train_bd12_ch1_minval)
#all_features[bd12_ids,2,:,:] = 2*(all_features[bd12_ids,2,:,:] - 0.5*(train_bd12_ch2_minval + train_bd12_ch2_maxval))/(train_bd12_ch2_maxval - train_bd12_ch2_minval)

#all_features[bd16_ids,0,:,:] = 2*(all_features[bd16_ids,0,:,:] - 0.5*(train_bd16_ch0_minval + train_bd16_ch0_maxval))/(train_bd16_ch0_maxval - train_bd16_ch0_minval)
#all_features[bd16_ids,1,:,:] = 2*(all_features[bd16_ids,1,:,:] - 0.5*(train_bd16_ch1_minval + train_bd16_ch1_maxval))/(train_bd16_ch1_maxval - train_bd16_ch1_minval)
#all_features[bd16_ids,2,:,:] = 2*(all_features[bd16_ids,2,:,:] - 0.5*(train_bd16_ch2_minval + train_bd16_ch2_maxval))/(train_bd16_ch2_maxval - train_bd16_ch2_minval)


x_val = []
x_train = []

ctr = 0
for idx in train_ids:
	ctr += 1
	x_train.append(copy.deepcopy(all_features[idx,:,:,:]))
	if entire_dataset==False and ctr==64:
		break

ctr = 0
for idx in val_ids:
	ctr += 1
	x_val.append(copy.deepcopy(all_features[idx,:,:,:]))
	if entire_dataset==False and ctr==16:
		break

del all_features


print(len(x_train),len(x_val))
y_val = all_labels[val_ids] #.astype(np.float32)
y_val = y_val.reshape((y_val.shape[0]))
y_train = all_labels[train_ids] #.astype(np.float32)
y_train = y_train.reshape((y_train.shape[0]))
print(len(y_train),len(y_val))
del all_labels

y_train = torch.from_numpy(np.array(y_train))
y_val = torch.from_numpy(np.array(y_val))
eps = int(1e6)


num_samples_array = Variable(torch.zeros(num_classes,1),requires_grad=False)
for class_ctr in range(0,num_classes):
	num_samples_array[class_ctr] = eps*torch.sum(y_train==class_ctr)+1

k = 1/torch.sum(torch.pow(num_samples_array,-1))
class_weights_train = k*torch.pow(num_samples_array,-1)
criterion_train = nn.CrossEntropyLoss(weight=class_weights_train.float().cuda()).cuda()

criterion_val = nn.CrossEntropyLoss().cuda()

# pdb.set_trace()
del class_weights_train, num_samples_array, k

print('Data loaded. ')

# load model
print('Pretrained : ', pretrained)

if 'v' in model_name:
	model_ft = vgg_dropoutcontrol.vgg16_bn(pretrained=pretrained, num_classes_modified=num_classes, dropout_prob=dropout_prob)

elif 'i' in model_name:
	model_ft = inception_v3_withdropoutcontrol.inception_v3_withdropoutcontrol(pretrained=pretrained, num_classes_modified=num_classes, transform_input=transform_input)

elif 'd' in model_name:
	print('pass')
elif 'r' in model_name:
	print('pass')

gc.collect()

if just_top_layer==True:
	if 'v' in model_name:
		for p in model_ft.features.parameters():
			p.requires_grad=False
		ctr = 0
		for p in model_ft.classifier.parameters():
			ctr += 1
			if ctr==5:
				break
			p.requires_grad=False

	elif 'i' in model_name:
		ctr = 0
		for p in model_ft.parameters():
			ctr += 1
			if ctr == 291 or ctr == 292:
				p.requires_grad = True
			else:
				p.requires_grad = False

model_ft.cuda()

print('Model loaded.\n')

# pdb.set_trace()

# my_data_transforms = transforms.Compose([RandomRotate(10), Rescale(256), RandomCrop(224)])
my_data_transforms = transforms.Compose([Rescale(299)])
# train_data = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
# pdb.set_trace()
train_idx_array = np.array(range(0,len(x_train)))
val_idx_array = np.array(range(0,len(x_val)))
np.random.shuffle(train_idx_array)
np.random.shuffle(val_idx_array)


if 'i' in model_name:
	train_data = MyDataset(x_train,y_train,train_idx_array,Rescale(299))
	val_data = MyDataset(x_val,y_val,val_idx_array,Rescale(299))
elif 'v' in model_name:
	train_data = MyDataset(x_train,y_train,train_idx_array)
	val_data = MyDataset(x_val,y_val,val_idx_array)

train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=batch_size_shuffle, num_workers=num_workers)
val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=batch_size_shuffle, num_workers=num_workers)

if just_top_layer==True:
	if 'v' in model_name:
		optimizer_ft = optim.Adam(model_ft.classifier[6].parameters(), lr=1e-3)
	elif 'i' in model_name:
		optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=1e-3)
else:
	if 'v' in model_name:
		optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)
	elif 'i' in model_name:
		optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)


# optimizer_ft = optim.Adagrad(model_ft.parameters())

#optimizer_fc = optim.SGD(model_ft.fc.parameters(), lr=1e-3, momentum=0.2, weight_decay=1e-4, nesterov=False)
#optimizer_Aux = optim.SGD(model_ft.AuxLogits.parameters(), lr=1e-3, momentum=0.2, weight_decay=1e-4, nesterov=False)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-5, momentum=0.9, weight_decay=0, nesterov=True)



#scheduler_fc = StepLR(optimizer_fc, step_size=10, gamma=0.99)
#scheduler_Aux = StepLR(optimizer_Aux, step_size=10, gamma=0.99)
#scheduler_ft = StepLR(optimizer_ft, step_size=10, gamma=0.99)


train_loss = np.zeros((epochs, 1), dtype=np.float32)
train_acc = np.zeros((epochs, 1), dtype=np.float32)
val_acc = np.zeros(shape=(epochs,1), dtype=np.float32)
val_loss = np.zeros(shape=(epochs, 1), dtype=np.float32)

all_metrics_file = base_path + 'saved/all_metrics_inception_v3_Adam_finaldropout2e-1_L20_all_layers.npy'


def validate():
	N_val = 0.0
	model_ft.eval()
	total_val_loss = 0.0
	total_val_acc = 0.0

	# torch.set_grad_enabled(False)
	for data_val in val_loader:
		inputs = data_val['image']
		curr_batch_size = inputs.size(0)
		N_val += curr_batch_size
		inputs = Variable(inputs.float().cuda(), requires_grad=False, volatile=True)
		labels = Variable(data_val['label'].long().cuda(), requires_grad=False, volatile=True)
		optimizer_ft.zero_grad()
		model_ft.zero_grad()
		outputs = model_ft(inputs)
		del inputs, data_val

		loss = criterion_val(outputs, labels)
		total_val_loss += loss.data[0]*curr_batch_size

		if 'v' in model_name:
			_, preds = torch.max(outputs.data, 1)
			total_val_acc += torch.sum(preds == labels.data)
		elif 'i' in model_name:
			_, preds = torch.max(outputs, 1)
			total_val_acc += torch.sum(preds == labels).data[0]

		del outputs

		del labels, preds, loss, curr_batch_size 
		gc.collect()

	# torch.set_grad_enabled(True)
	total_val_loss = total_val_loss/N_val
	total_val_acc = total_val_acc/N_val

	return (total_val_loss,total_val_acc)

def save_checkpoint(state, filename= base_path + 'saved/inception_v3_Adam_finaldropout2e-1_L20_all_layers_checkpoint.pth.tar'):
	torch.save(state, filename)
	if state['is_best']==True:
		shutil.copyfile(filename, base_path + 'saved/inception_v3_Adam_finaldropout2e-1_L20_all_layers_model_best.pth.tar')

best_acc = 0.0
time_start_all_epochs = time.time()
model_ft.train(True)
for epoch_cnt in range(epochs):
	tic = time.time()
	N_train = 0.0
	total_train_loss = 0.0
	total_train_acc = 0.0
	
	ctr = 0
	optimizer_ft.zero_grad()
	model_ft.zero_grad()
	# torch.manual_seed(555)
	for data_train in train_loader:
		ctr+=1
		print(ctr,end="\r")
		inputs = data_train['image']
		curr_batch_size = inputs.size(0)
		N_train += curr_batch_size
		inputs = Variable(inputs.float().cuda(), requires_grad=False)
		labels = Variable(data_train['label'].long().cuda(), requires_grad=False)

		optimizer_ft.zero_grad()
		model_ft.zero_grad()
		if 'i' in model_name:
			outputs = model_ft(inputs,dropout_prob)
		else:
			outputs = model_ft(inputs)
		del inputs, data_train

		if 'v' in model_name:
			loss = criterion_train(outputs, labels)
			_, preds = torch.max(outputs.data, 1)
			total_train_acc += torch.sum(preds == labels.data)
		elif 'i' in model_name:
			loss_fc = criterion_train(outputs[1], labels)
			loss_Aux = criterion_train(outputs[0], labels)
			loss = loss_Aux + loss_fc
			_, preds = torch.max(outputs[1], 1)
			total_train_acc += torch.sum(preds == labels).data[0]

		total_train_loss += loss.data[0]*curr_batch_size
		loss.backward()
		optimizer_ft.step()

		del outputs,curr_batch_size,labels, preds, loss
		if 'i' in model_name:
			del loss_fc, loss_Aux
		gc.collect()

	# print('')
	optimizer_ft.step()
	train_loss[epoch_cnt] = total_train_loss/(2.0*N_train)
	train_acc[epoch_cnt] = total_train_acc/N_train
	time_spent = time.time() - tic
	del total_train_loss, total_train_acc
	val_loss[epoch_cnt], val_acc[epoch_cnt] = validate()

	model_ft.train(True)
	# print('\n', end="")
	print('Epoch [%d/%d]'%(epoch_cnt+1, epochs))
	print('-'*20)
	print('Total Time: %0.2f'%time_spent)
	print('Time / sample : %.3f'%(time_spent/N_train))
	print('Train loss: %.3f' %(train_loss[epoch_cnt]))
	print('Train Acc: %.3f' %(train_acc[epoch_cnt]))
	print('Val loss: %.3f' %val_loss[epoch_cnt])
	print('Val Acc: %.3f' %val_acc[epoch_cnt])
	all_metrics = {}
	all_metrics['training_loss'] = train_loss
	all_metrics['training_accuracy'] = train_acc
	all_metrics['validation_loss'] = val_loss
	all_metrics['validation_accuracy'] = val_acc
	np.save(all_metrics_file,all_metrics)
	del all_metrics
	print()
	curr_val_acc = val_acc[epoch_cnt]
	is_best = curr_val_acc > best_acc
	best_acc = max(best_acc,curr_val_acc)
	save_checkpoint({'epoch': epoch_cnt,'state_dict': model_ft.state_dict(),'best_acc': best_acc,'optimizer_ft' : optimizer_ft.state_dict(), 'is_best' : is_best, 'batch_size' : batch_size})

print('Finished training.')
print('#'*20)
elapsed_time_all_epochs = (time.time()-time_start_all_epochs)/3600
print('Total training time for %d epochs: %.2f hrs'%(epochs,elapsed_time_all_epochs))

pdb.set_trace()



