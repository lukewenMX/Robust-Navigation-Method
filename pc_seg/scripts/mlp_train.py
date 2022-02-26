'''
MLP-based robot controller designed by qyz
'''

import open3d
import argparse
import os
import time
import json

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

from model.qyz_mlp import Net

from model.utils import load_pointnet

from pcd_utils import mkdir, select_avaliable
#from data_utils.SemKITTI_Loader import SemKITTI_Loader
#from data_utils.kitti_utils import getpcd
from data_utils.fhy4_Sem_Loader import SemKITTI_Loader, PointsAndVel_Loader, pcd_normalize
from data_utils.fhy4_datautils_test import Semantic_KITTI_Utils

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/img_velo_label_327-002"
train_npts = 2000
test_npts =2000
batch_size = 64#32
workers = 4
learning_rate = 0.0005
init_epoch = 0
total_epoch = 50
mode = 'train'
model_name = 'angular'
pretrain = None
#pretrain = '/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/experiment/MLP_linear/MLP_linear-0.17034362-0005.pth'
#pretrain = '/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/experiment/MLP_angular/MLP_angular-0.01193680-0004.pth'

def calc_decay(init_lr, epoch):
	return init_lr * 1 / (1 + 0.03 * epoch)#0.03
	#return init_lr

def test_mlp(model, loader, print_info = False, name = None):

	total_loss = 0
	for points, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
		# in tqdm, loader is an iterable variable. loader here includes dataset with label
		#batch_size, num_point, _ = points.size()  # points

		points = points.float().transpose(2,1).cuda()
		target = target.float().cuda()
		with torch.no_grad():
			loss_func = torch.nn.MSELoss(reduction='mean')
			prediction = model(points)
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
	#return linear_loss
	total_loss /= len(loader)
	return total_loss


def train(name=None):
	experiment_dir = mkdir('experiment/')
	checkpoints_dir_linear = mkdir('experiment/%s/'% 'MLP_linear')
	checkpoints_dir_angular = mkdir('experiment/%s/' % 'MLP_angular')
	dataset = PointsAndVel_Loader(ROOT, train_npts, train=True, subset='inview')
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
							num_workers=workers, pin_memory=True)

	test_dataset = PointsAndVel_Loader(ROOT, test_npts, train=False, subset='inview')

	testdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
								num_workers=workers, pin_memory=True)

	model = Net(5,128)#2000，64
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=learning_rate,
		betas=(0.9, 0.999),
		eps=1e-08,
		weight_decay=0.1)#1e-4
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
	train_linear_loss_list=[]
	train_angular_loss_list = []
	eval_linear_loss_list=[]
	eval_angular_loss_list = []
	epoch_time = []
	lr_list = []
	eval_loss_list=[]

	for epoch in range(init_epoch, total_epoch):
		model.train()
		lr = calc_decay(learning_rate, epoch)
		log.info(model='MLP', gpu='0', epoch=epoch, lr=lr)

		for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
			points = points.float().transpose(2,1).cuda()
			#print(points.shape)
			target = target.float().cuda()

			prediction = model(points)
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
		eval_loss = test_mlp(model.eval(), testdataloader,print_info = False, name=name)

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
			if name == 'linear':
				fn_pth = '%s-%.8f-%04d.pth' % ('MLP_linear', best_loss, epoch)
				log.info('Save model...', fn=fn_pth)
				torch.save(model.state_dict(), os.path.join(checkpoints_dir_linear, fn_pth))
				log.info(Best_eval_loss=best_loss, curr_train_loss=loss, curr_eval_loss=eval_loss)
			if name == 'angular':
				fn_pth = '%s-%.8f-%04d.pth' % ('MLP_angular', best_loss, epoch)
				log.info('Save model...', fn=fn_pth)
				torch.save(model.state_dict(), os.path.join(checkpoints_dir_angular, fn_pth))
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
'''
	label_size = {"size": 10}
	fig = plt.figure(2)
	plt.plot(epoch_time, train_linear_loss_list, label="train_linear_loss")
	plt.plot(epoch_time, train_angular_loss_list, label="train_angular_loss")
	plt.plot(epoch_time, eval_linear_loss_list, label="eval_linear_loss")
	plt.plot(epoch_time, eval_angular_loss_list, label="eval_angular_loss")
	# plt.plot(epoch_time,lr_list,label = "learning rate")
	plt.xlabel("epoch time", fontsize=10)
	plt.ylabel("value", fontsize=10)
	plt.title("training trendency", fontsize=20)
	plt.tick_params(labelsize=10)
	plt.legend(prop=label_size)
	plt.show()
'''

def evaluate(name):

	test_dataset = PointsAndVel_Loader(ROOT, test_npts, train=False, subset='inview')
	testdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
								num_workers=workers, pin_memory=True)

	model = Net(5,5)
	torch.backends.cudnn.benchmark = True
	model = torch.nn.DataParallel(model)
	# use more than 1 gpu
	log.info('Using gpu:', '0')
	model.load_state_dict(torch.load(pretrain))
	model.cuda()
	model.eval()
	loss = test_mlp(model.eval(), testdataloader,print_info = True, name=name)
	log.info('Curr', Loss = loss)





if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	if mode == 'train':
		train(model_name)
	if mode == 'eval':
		evaluate(model_name)