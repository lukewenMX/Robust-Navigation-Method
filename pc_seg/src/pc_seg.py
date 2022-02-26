#!/usr/bin/env python3
import os
import torch
import rospy
import sys
import struct
import sensor_msgs.point_cloud2
import rospkg
rospack = rospkg.RosPack()
pkg_prefix = rospack.get_path("pc_seg")
sys.path.append(pkg_prefix)

import numpy as np
import pandas as pd
import torch.nn.functional as F
import data_utils.my_log as log
import std_msgs.msg
from sensor_msgs.point_cloud2 import create_cloud, read_points, read_points_list
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

from model.fhy_pointnet1 import PointNetSeg, feature_transform_reguliarzer
from model.utils import load_pointnet
from data_utils.fhy4_Sem_Loader import pcd_normalize
from data_utils.fhy4_datautils_test import Semantic_KITTI_Utils

# ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(pkg_prefix, 'checkpoint/pointnet-0.39590-0040.pth')
# print(MODEL_PATH)
MINI_NAME = ['road', 'sidewalk', 'building', 'obstacle', 'human', 'others']
MINI_COLOR = [[128, 64, 128], [50, 205, 50], [70, 70, 70], [250, 170, 30], [220, 20, 60], [0, 0, 255]] #RGB

################ TODO ###################
# 1. read raw points with sensor_msgs
# 2. crop raw points with data_utils
# 3. send raw points into model
# 4. get labelled points and publish

class PCSeg():
    def __init__(self) -> None:
        self.seg_model = load_pointnet("pointnet", len(MINI_NAME),  MODEL_PATH)
        self.set_filter([-45,45],[-20,20])
        self.node = rospy.init_node("pointcloud_segmentation", anonymous=True)
        self.cloud_pub = rospy.Publisher("/segmented_cloud", PointCloud2, queue_size=1)
        self.cloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.pointcloudHandler)
        # self.utils = Semantic_KITTI_Utils(ROOT, subset="inview") 
        rospy.spin()

    def pointcloudHandler(self, cloud_msg):
        # read raw points
        raw_points = read_points_list(cloud_msg, skip_nans=True) # list of points with (x,y,z,intensity,ring)
        raw_points = np.array(raw_points)
        # print(raw_points[0])
        raw_points_4d = np.array(raw_points)[:,:-1]
        # print("Raw Points Size = ", raw_points.shape)
        filtered = self.points_basic_filter(raw_points_4d)
        crop_points = raw_points_4d[filtered]
        # print("Crop Points Size = ", crop_points.shape)
        norm_points = pcd_normalize(crop_points)
        pred = []
        softmax_out = []
        with log.Tick():
            points_tensor = torch.from_numpy(norm_points).unsqueeze(0).transpose(2,1).float().cuda()
            with torch.no_grad():
                logits,_ = self.seg_model(points_tensor)
                softmax_out = logits[0].cpu().numpy()
                pred = logits[0].argmax(-1).cpu().numpy()
            # print(pred.shape, end='')
            print(np.unique(pred))

            # with log.Tock("cpu"):
            #     pass
        
        new_pts = []
        for i in range(crop_points.shape[0]):
            point = crop_points[i,:]
            softmax_prob = np.exp(softmax_out[i])
            # print(softmax_prob.tolist())
            # print(np.exp(softmax_prob))
            out_point = point.tolist()
            # out_point.extend(MINI_COLOR[pred[i]])
            # r, g, b = MINI_COLOR[pred[i]]
            # a = 255
            # # rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            # out_point.append(rgb)
            out_point.extend(softmax_prob.tolist()) 
            new_pts.append(out_point)
        # print(len(new_pts))
        ros_cloud = sensor_msgs.point_cloud2.create_cloud(cloud_msg.header, self.create_pc_fields(), new_pts)
        self.cloud_pub.publish(ros_cloud)

    def create_pc_fields(self):
        fields = []
        fields.append( PointField( 'x', 0, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'y', 4, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'z', 8, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'intensity', 12, PointField.FLOAT32, 1 ) )
        # fields.append( PointField( 'r', 16, PointField.UINT8, 1 ))
        # fields.append( PointField( 'g', 17, PointField.UINT8, 1 ))
        # fields.append( PointField( 'b', 18, PointField.UINT8, 1 ))
        # fields.append( PointField( 'rgba', 16, PointField.UINT32, 1 ) )
        fields.append( PointField( 'road', 16, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'sidewalk', 20, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'building', 24, PointField.FLOAT32, 1 ) ) 
        fields.append( PointField( 'obstacle', 28, PointField.FLOAT32, 1 ) ) 
        fields.append( PointField( 'human', 32, PointField.FLOAT32, 1 ) ) 
        fields.append( PointField( 'others', 36, PointField.FLOAT32, 1 ) )
        return fields 

    def set_filter(self, h_fov, v_fov, x_range = None, y_range = None, z_range = None, d_range = None):
        self.h_fov = h_fov if h_fov is not None else (-180, 180)
        self.v_fov = v_fov if v_fov is not None else (-25, 20)
        self.x_range = x_range if x_range is not None else (-10000, 10000)
        self.y_range = y_range if y_range is not None else (-10000, 10000)
        self.z_range = z_range if z_range is not None else (-10000, 10000)
        self.d_range = d_range if d_range is not None else (-10000, 10000)

    def points_basic_filter(self, points):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            2. x,y,z distance limit
            return a bool array
        """
        assert points.shape[1] == 4, points.shape # [N,4]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))
        r = np.sqrt(x ** 2 + y ** 2)
        # extract in-range fov points
        h_points = self.hv_in_range(x, y, self.h_fov, fov_type='h')
        v_points = self.hv_in_range(r, z, self.v_fov, fov_type='v')
        combined = np.logical_and(h_points, v_points)
        
        # extract in-range x,y,z points
        in_range = self.box_in_range(x,y,z,d, self.x_range, self.y_range, self.z_range, self.d_range)
        combined = np.logical_and(combined, in_range)
        
        return combined

    def hv_in_range(self, m, n, fov, fov_type='h'):
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

    def box_in_range(self,x,y,z,d, x_range, y_range, z_range, d_range):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return np.logical_and.reduce((
                x > x_range[0], x < x_range[1],
                y > y_range[0], y < y_range[1],
                z > z_range[0], z < z_range[1],
                d > d_range[0], d < d_range[1]))

    def project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape
        pts_3d = pts_3d.copy()
        
        # Concat and change shape from [N,3] to [N,4] to [4,N]
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float32)
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1).T

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(self.RT, xyz_v[:, i])

        xyz_c = xyz_v[:3]

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(self.K, xyz_c[:, i])

        # normalize image(pixel) coordinates(x,y)
        xy_i = xyz_c / xyz_c[2]

        # get pixels location
        pts_2d = xy_i[:2].T
        return pts_2d


if __name__ == "__main__":
    # rospy.init_node("pointcloud_segmentation", anonymous=True)
    try:
        pcseg = PCSeg()
    except rospy.ROSInternalException:
        print("ERROR")
    rospy.spin()
