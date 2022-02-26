#!/usr/bin/env python3
from math import cos, sin
import os
from matplotlib.pyplot import axis
from numpy.core.arrayprint import dtype_short_repr

from numpy.core.shape_base import stack
from numpy.lib.function_base import append
from numpy.lib.npyio import load
import torch
from torch.functional import norm


import math
import cv2
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
import matplotlib.pyplot as plt
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2, PointField, Image
from sensor_msgs.point_cloud2 import read_points_list
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry

from image_geometry import PinholeCameraModel
from model.cnn_lstm import CNN_LSTM
from model.utils import load_CNN, load_cnn_lstm, load_pointnet, load_lstm
from data_utils.fhy4_Sem_Loader import pcd_normalize
from scipy.spatial.transform import Rotation as R
from data_utils.pt2range import LaserScan

CLOUD_SEG_MODEL_PATH = os.path.join(pkg_prefix, "checkpoint/offline_train/pointnet/pointnet-0.80816-0174.pth")
CLOUD_PRED_MODEL_PATH = os.path.join(pkg_prefix, "checkpoint/CNN_LSTM-0.00006027-0030.pth")
VIZ_DATA_PATH = os.path.join(pkg_prefix, "vis")
CALIB_FILE = os.path.join(pkg_prefix, "params/camera_params.yaml")
MINI_NAME = ['road', 'building', 'terrain', 'vegetation', 'others']
MINI_COLOR = [[128, 64, 128], [50, 205, 50], [250, 170, 30], [220, 20, 60], [0, 0, 255]] #RGB


VERBOSE=False

class CloudPredict():
    def __init__(self) -> None:
        
        self.cloud_seg_model = load_pointnet("pointnet", 5, CLOUD_SEG_MODEL_PATH)
        self.cloud_pred_model = load_cnn_lstm("cnn_lstm", CLOUD_PRED_MODEL_PATH)
        self.set_filter([-42.5,42.5],[-20,20],d_range=[0,25])

        self.time_step = 10
        self.stack_list = [np.zeros((1,5))]
        self.pt_converter = LaserScan(project=True, H=16, W=200, fov_up=15, fov_down=-15, hfov=85)
        self.feature_list = []
        self.h = torch.autograd.Variable(torch.zeros(2,1,128)).cuda()
        self.c = torch.autograd.Variable(torch.zeros(2,1,128)).cuda()
        self.last_cmd = 0
        self.node = rospy.init_node("cloud_controller", anonymous=True)
        self.image_pub = rospy.Publisher("/range_image",Image, queue_size = 1)
        self.cloud_pub = rospy.Publisher("/segmented_cloud", PointCloud2, queue_size = 1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.cloud_sub = rospy.Subscriber("/velodyne_points_throttle", PointCloud2, self.cloudCallback, queue_size = 1)
        self.header = Header()
        self.header.frame_id = 'base_link'
        
        rospy.spin()

    def cloudCallback(self, cloud_msg):
        # Obtain Pointcloud
        raw_points = read_points_list(cloud_msg, skip_nans=True) # list of points with (x,y,z,intensity,ring)
        raw_points_5d = np.array(raw_points)
        filtered = self.points_basic_filter(raw_points_5d)
        crop_points = raw_points_5d[filtered]
        # choice = np.random.choice(crop_points.shape[0], 2000, replace=True)
        # crop_points = crop_points[choice]
        norm_points = pcd_normalize(crop_points[:,:-1])
        
        # Run Pointcloud Prediction
        with log.Tick(name="cloud_segmentation"):
            points_tensor = torch.from_numpy(norm_points).unsqueeze(0).transpose(2,1).float().cuda()
            with torch.no_grad():
                logits,_ = self.cloud_seg_model(points_tensor)
                cloud_pred_label = logits[0].argmax(-1).cpu().numpy()

            # print(np.unique(pred))
        points_label_rings = np.concatenate((crop_points[:,:4], cloud_pred_label.reshape(-1,1), crop_points[:,4].reshape(-1,1)), axis=1)
        self.pt_converter.set_points(points_label_rings[:,:3], points_label_rings[:,3], points_label_rings[:,4],points_label_rings[:,5])
        
        depth_img = self.pt_converter.proj_range 
        intensity_img = self.pt_converter.proj_remission
        label_img = self.pt_converter.proj_label
        color_img = self.pt_converter.proj_color
        # blured = cv2.blur(depth_img,(1,1))
        feature = np.array([depth_img, intensity_img, label_img])
        feature = feature.reshape(3, 16, 200)

        # Publish Range Image
        self.image_pub.publish(self.cv2_to_imgmsg(color_img, depth=False))

        self.feature_list.append(feature)
        # if len(self.feature_list) < 10:
        #     return
        if len(self.feature_list) > 1:
            self.feature_list.pop(0)
        
        with torch.no_grad():
            data = torch.from_numpy(np.array(self.feature_list)).unsqueeze(0).cuda()
            prediction, self.h, self.c = self.cloud_pred_model(data, self.h, self.c)
            pred_vel = prediction.reshape(-1).cpu().numpy()[0]
            print(pred_vel)
        cloud_pred_vel = (self.last_cmd + pred_vel) / 2.0
        self.last_cmd = pred_vel
        # Add Prediction
        # self.cloud_predict_w_list.append(cloud_predict_vel)

        # Print Prediction
        # print("[TIMESTAMP: %.3f] vel@cloud = %.3f" 
        #         % (cloud_msg.header.stamp.to_sec(), pred_vel))

        ################# Save File #############
        # np.save(os.path.join(VIZ_DATA_PATH, "cloud_predict.npy"), np.array(self.cloud_predict_w_list))

        ################# Publish ###############
        
        
        # Publish Pointcloud
        self.header.stamp = rospy.Time.now()
        self.publish_points(crop_points, cloud_pred_label, self.header)

        # Publish command
        msg = Twist()
        if abs(pred_vel) > 0.08:
            linear_vel = 0.6
        else:
            linear_vel = 0.8
        msg.linear.x = float(linear_vel)
        msg.angular.z = float(cloud_pred_vel) # or img_pred_vel
        self.cmd_vel_pub.publish(msg)


    def set_filter(self, h_fov, v_fov, x_range = None, y_range = None, z_range = None, d_range = None):
        self.h_fov = h_fov if h_fov is not None else (-180, 180)
        self.v_fov = v_fov if v_fov is not None else (-25, 20)
        print(self.h_fov, self.v_fov)
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
        # assert points.shape[1] == 5, points.shape # [N,4]
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

    def publish_points(self, points, pred, header):
        new_pts = []
        for i in range(points.shape[0]):
            point = points[i,:]
            point = point.tolist()
            point[-1] = int(point[-1])
            # point.extend(MINI_COLOR[pred[i]])
            r, g, b = MINI_COLOR[pred[i]]
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            point.append(rgb)
            new_pts.append(point)
        # print(len(new_pts[0]))
        ros_cloud = sensor_msgs.point_cloud2.create_cloud(header, self.create_pc_fields(), new_pts)
        self.cloud_pub.publish(ros_cloud)

    def cv2_to_imgmsg(self, img, depth=False):
        # easy implementation converting 3-channels image to sensor_msgs::Image
        img_msg = Image()
        img_msg.header.frame_id = 'base_link'
        img_msg.header.stamp = rospy.Time.now()
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        
        if depth == True:
            img_msg.encoding = "32FC1"
        else:
            img_msg.encoding = "bgr8"
        
        img_msg.data = img.tostring()
        img_msg.step = int(len(img_msg.data) / img_msg.height)
        
        return img_msg

    def create_pc_fields(self):
        fields = []
        fields.append( PointField( 'x', 0, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'y', 4, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'z', 8, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'intensity', 12, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'rings', 16, PointField.UINT32, 1 ) )
        fields.append( PointField( 'rgba', 20, PointField.UINT32, 1 ) )
        return fields

if __name__ == "__main__":
    try:
        pcpredict = CloudPredict()
    except rospy.ROSInterruptException:
        print("ERROR")
    r = rospy.Rate(5.0)
    while not rospy.is_shutdown():
        r.sleep()
    # rospy.spin()