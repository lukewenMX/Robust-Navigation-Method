'''
do prediction on one pcd file with PointNet pretrained model, and save the result as .npz file for the following visualization
'''

import sys
import os
from tkinter import N
import torch
import ipdb
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from model.utils import load_pointnet
from data_utils.fhy4_Sem_Loader import pcd_normalize

MODEL_PATH = "checkpoint/offline_train/pointnet/pointnet-0.80816-0174.pth"
# MINI_COLOR = [[128, 64, 128], [50, 205, 50], [250, 170, 30], [220, 20, 60], [0, 0, 255]] #RGB
MINI_COLOR = [[128, 64, 128], [153, 153, 153], [255, 0, 0], [30, 170, 250], [50, 205, 50]]
# DATA_ROOT = '/home/yxdai/repository/nav_ws/src/prepare_data/NanyangLinkNight2'
DATA_ROOT = '/media/yxdai/HP P500/dataset/NanyangLink'
PCD_DIR = 'pts_label'
example_file = '11/000045'

h_fov, v_fov, d_range = (-42.5,42.5), (-20,20), (0,25)

def points_basic_filter(points):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            2. x,y,z distance limit
            return a bool array
        """
        def hv_in_range(m, n, fov, fov_type='h'):
            """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
                horizontal limit = azimuth angle limit
                vertical limit = elevation angle limit
            """
            if fov_type == 'h':
                return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                        np.arctan2(n, m) < (-fov[0] * np.pi / 180))
            elif fov_type == 'v':
                return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                        np.arctan2(n, m) > (fov[0] * np.pi / 180))
            else:
                raise NameError("fov type must be set between 'h' and 'v' ")
        def box_in_range(x,y,z,d, x_range, y_range, z_range, d_range):
            """ extract filtered in-range velodyne coordinates based on x,y,z limit """
            return np.logical_and.reduce((
                    x > x_range[0], x < x_range[1],
                    y > y_range[0], y < y_range[1],
                    z > z_range[0], z < z_range[1],
                    d > d_range[0], d < d_range[1]))

        # assert points.shape[1] == 5, points.shape # [N,4]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))
        r = np.sqrt(x ** 2 + y ** 2)
        # extract in-range fov points
        h_points = hv_in_range(x, y, h_fov, fov_type='h')
        v_points = hv_in_range(r, z, v_fov, fov_type='v')
        combined = np.logical_and(h_points, v_points)
        
        # extract in-range x,y,z points
        in_range = box_in_range(x,y,z,d, (-10000, 10000), (-10000, 10000), (-10000, 10000), d_range)
        combined = np.logical_and(combined, in_range)
        
        return combined

if __name__ == "__main__":
    net = load_pointnet(model_name='pointnet', num_classes=5, fn_pth=MODEL_PATH)
    # ipdb.set_trace()
    subset, name = example_file.split('/')
    pcd_path = os.path.join(DATA_ROOT, PCD_DIR, subset, 'pts_l' + name + '.npz')
    pts_l = np.load(pcd_path)['pts_l']
    pts_xyzi = pts_l[:,0:4]
    basic_filter = points_basic_filter(pts_l)
    crop_pts = pts_xyzi[basic_filter]
    normed_pts = pcd_normalize(crop_pts)
    pts_tensor = torch.from_numpy(normed_pts).unsqueeze(0).transpose(2,1).float().cuda()
    with torch.no_grad():
        logits, _ = net(pts_tensor)
        pred = logits[0].argmax(-1).cpu().numpy()
        pts_pred = np.concatenate((crop_pts, pred.reshape(-1,1)), axis = 1)
    np.savez("/home/yxdai/repository/nav_ws/src/prepare_data/pred_vis_day.npz",pts_l=pts_pred)

    
    
    
    



