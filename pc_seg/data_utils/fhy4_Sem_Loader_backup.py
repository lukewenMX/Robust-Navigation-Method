import os
import cv2
import json
import yaml
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import threading
import multiprocessing
from PIL import Image

#from .fhy4_datautils_test import Semantic_KITTI_Utils
#from redis_utils import Mat_Redis_Utils
from data_utils.fhy4_datautils_test import Semantic_KITTI_Utils

def pcd_jitter(pcd, sigma=0.01, clip=0.05):#0.01,0.05
    N, C = pcd.shape 
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(pcd.dtype)
    jittered_data += pcd
    return jittered_data

def pcd_normalize(pcd):
    # pcd = pcd.copy()
    # pcd[:,0] = pcd[:,0] / 70
    # pcd[:,1] = pcd[:,1] / 70
    # pcd[:,2] = pcd[:,2] / 3
    # pcd[:,3] = (pcd[:,3] - 0.5)*2
    # pcd = np.clip(pcd,-1,1)
    pcd = pcd.copy()
    centroid = np.mean(pcd, axis=0)
    pcd = pcd - centroid
    m = np.sqrt(np.sum(pcd ** 2, axis=0)/pcd.shape[0])
    #print(m)
    pcd = pcd / m
    pcd = np.clip(pcd, -1, 1)
    return pcd

def pcd_unnormalize(pcd):
    pcd = pcd.copy()
    pcd[:,0] = pcd[:,0] * 70
    pcd[:,1] = pcd[:,1] * 70
    pcd[:,2] = pcd[:,2] * 3
    pcd[:,3] = pcd[:,3] / 2 + 0.5
    return pcd

def pcd_tensor_unnorm(pcd):
    pcd_unnorm = pcd.clone()
    pcd_unnorm[:,0] = pcd[:,0] * 70
    pcd_unnorm[:,1] = pcd[:,1] * 70
    pcd_unnorm[:,2] = pcd[:,2] * 3
    pcd_unnorm[:,3] = pcd[:,3] / 2 + 0.5
    return pcd_unnorm

class SemKITTI_Loader(Dataset):
    def __init__(self, root, npoints, train = True, subset = 'inview'):
        self.root = root
        self.train = train
        self.npoints =npoints
        self.load_image = not train
        self.utils = Semantic_KITTI_Utils(root, subset)
        # self.np_redis = Mat_Redis_Utils()
        part_length = {'00': 2190,'01':1978,'02':1955,'03':2530, '04':1691, '05':1606, '06':1489, '07':2716, '11':10}
        self.keys = []
        if self.train:
            for part in ['00','01','03','04','05','06','07']:
            #for part in ['04']:
                length = part_length[part]
                for index in range(0,length,1):#3
                    self.keys.append('%s/%06d'%(part, index))
        else:
            # for part in ['07']:
            for part in ['11']:
                length = part_length[part]
                for index in range(0,length,1):#3
                    self.keys.append('%s/%06d'%(part, index))        

    def __len__(self):
            return len(self.keys)

    def get_data(self, key):
        part, index = key.split('/')
        point_cloud, label = self.utils.get_pts_l(part, int(index))
        return point_cloud, label

    def __getitem__(self, index):
        point_cloud, label = self.get_data(self.keys[index])
        pcd = pcd_normalize(point_cloud)
        #pcd = point_cloud
        if self.train:
            pcd = pcd_jitter(pcd)
        # length = pcd.shape[0]
        # if length == self.npoints:
        #     pass
        # elif length > self.npoints:
        #     start_idx = np.random.randint(0, length - self.npoints)
        #     end_idx = start_idx + self.npoints
        #     pcd = pcd[start_idx:end_idx]
        #     label = label[start_idx:end_idx]
        # else:
        #     rows_short = self.npoints - length
        #     pcd = np.concatenate((pcd,pcd[0:rows_short]),axis=0)
        #     label = np.concatenate((label,label[0:rows_short]),axis=0)

        length = pcd.shape[0]
        
        choice = np.random.choice(length, self.npoints, replace=True)
        pcd = pcd[choice]
        label = label[choice]
        # print(label.shape)
        return pcd, label

class PointsAndVel_Loader(Dataset):
    def __init__(self, root, npoints, train = True, subset = 'inview'):
        self.root = root
        self.train = train
        self.npoints =npoints
        self.load_image = not train
        self.utils = Semantic_KITTI_Utils(root, subset)
        # self.np_redis = Mat_Redis_Utils()
        #part_length = {'04':1691, '05':1606, '06':1489, '07':2716}
        part_length = {'08': 3141, '09': 2462, '10': 795}
        self.keys = []
        if self.train:
            #for part in ['04','05','07']:
            for part in ['08', '09']:
            #for part in ['04']:
                length = part_length[part]
                for index in range(0,length,1):#3
                    self.keys.append('%s/%06d'%(part, index))
        else:
            for part in ['10']:
                length = part_length[part]
                for index in range(0,length,1):#3
                    self.keys.append('%s/%06d'%(part, index))

    def __len__(self):
            return len(self.keys)

    def get_data(self, key):
        part, index = key.split('/')
        point_cloud, vel = self.utils.get_pts_vel(part, int(index))
        return point_cloud, vel

    def __getitem__(self, index):
        point_cloud, vel = self.get_data(self.keys[index])
        pcd = pcd_normalize(point_cloud)
        #pcd = point_cloud
        if self.train:
            pcd = pcd_jitter(pcd)
        # length = pcd.shape[0]
        # if length == self.npoints:
        #     pass
        # elif length > self.npoints:
        #     start_idx = np.random.randint(0, length - self.npoints)
        #     end_idx = start_idx + self.npoints
        #     pcd = pcd[start_idx:end_idx]
        #     label = label[start_idx:end_idx]
        # else:
        #     rows_short = self.npoints - length
        #     pcd = np.concatenate((pcd,pcd[0:rows_short]),axis=0)
        #     label = np.concatenate((label,label[0:rows_short]),axis=0)

        length = pcd.shape[0]
        choice = np.random.choice(length, self.npoints, replace=True)
        pcd = pcd[choice]
        vel = vel.reshape(2)
        return pcd, vel

if __name__ == '__main__':
    data_path = "/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/img_velo_label_327-002"
    s = PointsAndVel_Loader(data_path, 2000, train = True, subset = 'inview')
    pcd, label = s[300]
    #np.set_printoptions(threshold=np.inf)
    print(type(pcd))

