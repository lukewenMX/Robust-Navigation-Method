import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .fhy_pointnet1 import PointNetSeg, PointNetSegLight, feature_transform_reguliarzer
from .pointnet2 import PointNet2SemSeg
from .qyz_mlp import LSTM
from .CNN import CNN
from .cnn_lstm import CNN_LSTM
from .eval_cnn_lstm import EVAL_CNN_LSTM
def load_pointnet(model_name, num_classes, fn_pth):
    if model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
    elif model_name == 'pointnet_3d':
        model = PointNetSeg(num_classes, input_dims = 3, feature_transform=True)
    elif model_name == 'pointnet_light':
        model = PointNetSegLight(num_classes, input_dims = 3, feature_transform=True)
    else:
        #model = PointNet2SemSeg(num_classes, feature_dims = 1)
        model = PointNet2SemSeg(num_classes)
    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

    assert fn_pth is not None,'No pretrain model'
    checkpoint = torch.load(fn_pth)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    return model

def load_lstm(model_name, fn_pth):
    if model_name == "LSTM":
        model = LSTM(10000, 10, 2, 1)
    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

    assert fn_pth is not None,'No pretrain model'
    checkpoint = torch.load(fn_pth)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    return model

def load_CNN(model_name, fn_pth):
    if model_name == "CNN":
        model = CNN()
    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

    assert fn_pth is not None,'No pretrain model'
    checkpoint = torch.load(fn_pth)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    return model

def load_cnn_lstm(model_name, fn_pth):
    if model_name == "cnn_lstm":
        model = EVAL_CNN_LSTM()
    checkpont = torch.load(fn_pth)
    model.load_state_dict(checkpont)
    model.cuda()
    model.eval()
    return model