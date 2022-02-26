'''
CNN-based controller using image input designed by qyz
'''
# import open3d
import argparse
import os
import time
import json
# import h5py
import datetime
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from matplotlib import pyplot as plt
import data_utils.my_log as log
import matplotlib.pyplot as plt

from model.CNN import CNN
from torchvision import transforms

from model.utils import load_pointnet

# from pcd_utils import mkdir, select_avaliable
#from data_utils.SemKITTI_Loader import SemKITTI_Loader
#from data_utils.kitti_utils import getpcd
from data_utils.fhy4_Sem_Loader import SemKITTI_Loader, PointsAndVel_Loader, pcd_normalize, Img_Loader
from data_utils.fhy4_datautils_test import Semantic_KITTI_Utils

import ipdb


ROOT = os.path.dirname(os.path.abspath(__file__)) + "/img_velo_label_327-002"

batch_size = 32#32
workers = 4
learning_rate = 0.001
init_epoch = 0
total_epoch = 50
mode = 'train'
model_name = 'angular'
pretrain = None

train_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)
	])


def calc_decay(init_lr, epoch):
	return init_lr * 1 / (1 + 0.03 * epoch)#0.03
	#return init_lr

def test_CNN(model, loader, print_info = False, name = None, plot = False):

	total_loss = 0
	groud_truth = []
	predict_vel = []
	for img, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):

		img = img.float().cuda()

		target = target.float().cuda()
		with torch.no_grad():
			loss_func = torch.nn.MSELoss(reduction='mean')
			prediction = model(img)

			#loss = loss_func(prediction[:,(0,5)], target[:,(0,5)])
			if name == 'linear':
				loss = loss_func(prediction,target[:, 0].reshape(-1,1))
				total_loss += loss
				if print_info == True:
					print('linear_truth:%f linear_predict:%f' % (target[:, 0], prediction))

			if name == 'angular':
				loss = loss_func(prediction, target[:, 1].reshape(-1,1))
				total_loss += loss
				if print_info == True:
					print('angular_truth:%f angular_predict:%f\n' % (target[:, 1], prediction))
				groud_truth.append(target[:,1])


	#return linear_loss
	total_loss /= len(loader)
	return total_loss

def train(name=None):
	experiment_dir = mkdir('experiment/')
	checkpoints_dir_linear = mkdir('experiment/%s/'% 'CNN')

	dataset = Img_Loader(ROOT, train=True, subset='inview',transform = None)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
							num_workers=workers, pin_memory=True)

	test_dataset = Img_Loader(ROOT, train=False, subset='inview')

	testdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
								num_workers=workers, pin_memory=True)

	model = CNN()
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=learning_rate,
		betas=(0.9, 0.999),
		eps=1e-08,
		weight_decay=1e-4)#1e-4
	#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
	torch.backends.cudnn.benchmark = True
	model = torch.nn.DataParallel(model)
	# use more than 1 gpu
	model.cuda()
	log.info('Using gpu:', '0')
	if pretrain is not None:
		log.info('Use pretrain model...')
		model.load_state_dict(torch.load(pretrain))
		init_epoch = int(pretrain[:-4].split('-')[-1])
		# init_epoch = 0
		log.info('Restart training', epoch=init_epoch)
	else:
		log.msg('Training from scratch')
		init_epoch = 0
	loss_func = torch.nn.MSELoss()

	best_loss = None
	loss_list = []
	epoch_time = []
	lr_list = []
	eval_loss_list=[]

	for epoch in range(init_epoch, total_epoch):
		model.train()
		lr = calc_decay(learning_rate, epoch)
		log.info(model='CNN', gpu='0', epoch=epoch, lr=lr)

		for img, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
			#points = points.float().transpose(2,1).cuda()
			print(img.shape)
			ipdb.set_trace()
			img = img.float().cuda()
			# print(img.cuda())
			target = target.float().cuda()

			prediction = model(img)
			#print(target[:,(0,5)])
			#print(prediction.shape)
			#loss = loss_func(prediction[:,(0,5)], target[:,(0,5)])
			if name == 'linear':
				loss = loss_func(prediction, target[:,0].reshape(-1,1))
			if name == 'angular':
				loss = loss_func(prediction, target[:, 1].reshape(-1,1))
			#train_linear_loss = loss_func(prediction[:, 0],target[:, 0])
			#train_angular_loss = loss_func(prediction[:, 1], target[:, 1])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		torch.cuda.empty_cache()
		#eval_loss = test_mlp(model.eval(), testdataloader)
		eval_loss = test_CNN(model.eval(), testdataloader,print_info = False, name=name)

		save_model = False
		if best_loss == None:
			best_loss = eval_loss
			save_model = True
		if eval_loss < best_loss:
			best_loss = eval_loss
			save_model = True

		loss_list.append(loss)
		eval_loss_list.append(eval_loss)
		#train_linear_loss_list.append(train_linear_loss)
		#train_angular_loss_list.append(train_angular_loss)
		#eval_linear_loss_list.append(eval_linear_loss)
		#eval_angular_loss_list.append(eval_angular_loss)
		epoch_time.append(epoch)
		lr_list.append(lr)
		if save_model:
			fn_pth = '%s-%.8f-%04d.pth' % ('CNN', best_loss, epoch)
			log.info('Save model...', fn=fn_pth)
			torch.save(model.state_dict(), os.path.join(checkpoints_dir_linear, fn_pth))
			log.info(Best_eval_loss=best_loss, curr_train_loss=loss, curr_eval_loss=eval_loss)

		else:
			log.info('No need to save model')
			log.info(Best_eval_loss=best_loss,curr_train_loss=loss,curr_eval_loss=eval_loss)

	label_size = {"size": 10}
	fig = plt.figure(1)
	plt.plot(epoch_time, loss_list, label="train_loss")
	plt.plot(epoch_time, eval_loss_list, label="eval_loss")
	# plt.plot(epoch_time,lr_list,label = "learning rate")
	plt.xlabel("epoch time", fontsize=10)
	plt.ylabel("value", fontsize=10)
	plt.title("training trendency", fontsize=20)
	plt.tick_params(labelsize=10)
	plt.legend(prop=label_size)
	plt.show()


def evaluate(name=None):
	test_dataset = Img_Loader(ROOT, train=False, subset='inview')

	testdataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
								num_workers=workers, pin_memory=True)
	model = CNN()
	torch.backends.cudnn.benchmark = True
	model = torch.nn.DataParallel(model)
	checkpoint = torch.load("./checkpoint/CNN-0.00005945-0099.pth")
	model.load_state_dict(checkpoint)

	for img, target in tqdm(testdataloader, total=len(testdataloader), smoothing=0.9, dynamic_ncols=True):
			checkimg = img.squeeze(0).numpy()
			
			print(img.shape)
			print(target)
			
			img = img.float().cuda()
			target = target.float().cuda()

			prediction = model(img)
			print(prediction)





if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	if mode == 'train':
		train(model_name)
	if mode == 'eval':
		evaluate(model_name)

