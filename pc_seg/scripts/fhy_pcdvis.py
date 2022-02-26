'''
PCD visualization scripts by fhy
'''
import open3d
import argparse
import os
import time
import json
# import h5py
import datetime
import cv2
import yaml
import colorsys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt
import data_utils.my_log as log

from model.fhy_pointnet1 import PointNetSeg, feature_transform_reguliarzer
from model.utils import load_pointnet

from pointnet_train import parse_args
from data_utils.fhy4_Sem_Loader import pcd_normalize
from data_utils.fhy4_datautils_test import Semantic_KITTI_Utils

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/img_velo_label_327-002"

class Window_Manager():
    def __init__(self):
        self.param = open3d.io.read_pinhole_camera_parameters('config/ego_view.json')
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=800, height=800, left=100)
        self.vis.register_key_callback(32, lambda vis: exit())
        self.vis.get_render_option().load_from_json('config/render_option.json')
        self.pcd = open3d.geometry.PointCloud()
    
    def update(self, pts_3d, colors):
        self.pcd.points = open3d.utility.Vector3dVector(pts_3d)
        self.pcd.colors = open3d.utility.Vector3dVector(colors/255)
        self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.param)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_screen(self,fn):
        self.vis.capture_screen_image(fn, False)

def export_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = cv2.VideoWriter('experiment/pn_compare.avi',fourcc, 15.0, (int(1600*0.8),int(740*0.8)))

    # mkdir('experiment/imgs/%s/'%(args.model_name))
    # vis_handle.capture_screen('experiment/imgs/%s/%d_3d.png'%(args.model_name,i))
    # cv2.imwrite('experiment/imgs/%s/%d_sem.png'%(args.model_name, i), img_semantic)

    for index in range(100, 320):
        pn_3d = cv2.imread('experiment/imgs/pointnet/%d_3d.png' % (index))
        pn_sem = cv2.imread('experiment/imgs/pointnet/%d_sem.png' % (index))
        pn2_3d = cv2.imread('experiment/imgs/pointnet2/%d_3d.png' % (index))
        pn2_sem = cv2.imread('experiment/imgs/pointnet2/%d_sem.png' % (index))

        pn_3d = pn_3d[160:650]
        pn2_3d = pn2_3d[160:650]

        pn_sem = cv2.resize(pn_sem, (800, 250))
        pn2_sem = cv2.resize(pn2_sem, (800, 250))

        pn = np.vstack((pn_3d, pn_sem))
        pn2 = np.vstack((pn2_3d, pn2_sem))

        cv2.putText(pn, 'PointNet', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn, 'PointNet', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)

        merge = np.hstack((pn, pn2))
        class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
        colors = [[255, 255, 255],[245, 150, 100],[30, 30, 255],[255, 0, 255],[0, 200, 255],[0, 175, 0]]
        for i,(name,c) in enumerate(zip(class_names, colors)):
            cv2.putText(merge, name, (200 + i * 200, 50), font,1, [c[2],c[1],c[0]], 2, cv2.LINE_AA)

        cv2.line(merge,(0,70),(1600,70),(255,255,255),2)
        cv2.line(merge,(800,70),(800,1300),(255,255,255),2)

        merge = cv2.resize(merge,(0,0),fx=0.8,fy=0.8)
        # cv2.imshow('merge', merge)
        # if 32 == waitKey(1):
        #     break
        out.write(merge)

        print(index)
    out.release()

def vis(args):
    kitti_utils = Semantic_KITTI_Utils(ROOT,args.subset)

    # vis_handle = Window_Manager()
    args.pretrain = './checkpoint/pointnet-0.39590-0040.pth'
    #args.pretrain = 'experiment/pointnet2/pointnet2-0.39690-0016.pth'

    model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)
    part = '11'
    for index in range(0,11):
        points, labels = kitti_utils.get_pts_l(part, index, True)      
        # resample point cloud
        # length = point_cloud.shape[0]
        # npoints = 25000
        # choice = np.random.choice(length, npoints, replace=True)
        # point_cloud = point_cloud[choice]
        # label = label[choice]
        pts3d = points[:,:-1]
        pcd = pcd_normalize(points)

        with log.Tick():
            points_tensor = torch.from_numpy(pcd).unsqueeze(0).transpose(2, 1).float().cuda()
            
            with torch.no_grad():
                logits,_ = model(points_tensor)
                pred = logits[0].argmax(-1).cpu().numpy()

            print(index, pred.shape, end='')
            print(np.unique(pred))

            with log.Tock("cpu"):
                pts2d = kitti_utils.project_3d_to_2d(pts3d)

        pred_color = np.ndarray.tolist(kitti_utils.mini_color_BGR[pred])
        orig_color = np.ndarray.tolist(kitti_utils.mini_color_BGR[labels])
        img1 = kitti_utils.draw_2d_points(pts2d, orig_color)
        img2 = kitti_utils.draw_2d_points(pts2d, pred_color)
        img = np.hstack((img1, img2))
        cv2.imshow('semantic', img)
        if 32 == cv2.waitKey(0):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    vis(args)