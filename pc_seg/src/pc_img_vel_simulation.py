#!/usr/bin/env python3
from math import cos, sin
import os

from numpy.core.shape_base import stack
from numpy.lib.function_base import append
from numpy.lib.npyio import load
import torch
from torch.functional import norm


import math
import cv2
import rospy
import sys
import queue
import struct
import sensor_msgs.point_cloud2
import rospkg
import message_filters
import yaml
rospack = rospkg.RosPack()
pkg_prefix = rospack.get_path("pc_seg")
sys.path.append(pkg_prefix)

import numpy as np
import data_utils.my_log as log
import matplotlib.pyplot as plt
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points_list
from geometry_msgs.msg import Twist, TwistStamped

from image_geometry import PinholeCameraModel
from model.utils import load_CNN, load_pointnet, load_lstm
from data_utils.fhy4_Sem_Loader import pcd_normalize
from data_utils.noise_generator import OrnsteinUhlenbeckNoise
from scipy.spatial.transform import Rotation as R


CLOUD_SEG_MODEL_PATH = os.path.join(pkg_prefix, "checkpoint/cloud_seg_model.pth")
# CLOUD_PRED_MODEL_PATH = os.path.join(pkg_prefix, "checkpoint/cloud_vel_model.pth")
CLOUD_PRED_MODEL_PATH = os.path.join(pkg_prefix, "checkpoint/cloud_vel_left.pth")
IMG_PRED_MODEL_PATH = os.path.join(pkg_prefix, "checkpoint/new_cnn_model.pth")
VIZ_DATA_PATH = os.path.join(pkg_prefix, "vis")
CALIB_FILE = os.path.join(pkg_prefix, "params/camera_params.yaml")
MINI_NAME = ['road', 'sidewalk', 'building', 'obstacle', 'human', 'others']
MINI_COLOR = [[128, 64, 128], [50, 205, 50], [70, 70, 70], [250, 170, 30], [220, 20, 60], [0, 0, 255]] #RGB


VERBOSE=False

class FusePredict():
    def __init__(self) -> None:
        
        self.cloud_seg_model = load_pointnet("pointnet", 6, CLOUD_SEG_MODEL_PATH)
        self.cloud_pred_model = load_lstm("LSTM", CLOUD_PRED_MODEL_PATH)
        self.img_pred_model = load_CNN("CNN", IMG_PRED_MODEL_PATH)
        self.camera_model = PinholeCameraModel()
        self.load_cam_info(CALIB_FILE)
        self.set_filter([-45,45],[-20,20])
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=0, sigma=0.01)
        self.time_step = 4
        self.stack_list = [np.zeros((1,5))]

        self.cloud_predict_w_list = []
        self.img_predict_w_list = []
        self.random_w_list = []
        self.gt_w_list = []
        self.timeStamp = -1
        self.node = rospy.init_node("image_to_velocity", anonymous=True)
        self.cloud_pub = rospy.Publisher("/segmented_cloud", PointCloud2, queue_size = 5)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.cloud_sub = message_filters.Subscriber("/velodyne_points", PointCloud2, queue_size = 5)
        self.img_sub = message_filters.Subscriber("/camera/left/image_raw/compressed", CompressedImage, queue_size = 5)
        self.cmd_vel_sub = message_filters.Subscriber("/cmd_vel_stamped", TwistStamped, queue_size = 5)
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.cloud_sub, self.img_sub, self.cmd_vel_sub], 5, 0.08, allow_headerless=True)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.cmd_vel_sub], 5, 0.08, allow_headerless=True)
        
        self.ts.registerCallback(self.fusionCallback)
        plt.show()
        rospy.spin()

    def fusionCallback(self, cloud_msg, img_msg, vel_msg):
        print("Callback!")
        # Update TimeStamp
        current_timeStamp = img_msg.header.stamp.to_sec()
        if (self.timeStamp == -1):
            # self.timeStamp = current_timeStamp
            dt = 0
        else:
            dt = current_timeStamp - self.timeStamp
        self.timeStamp = current_timeStamp
        if (dt > 0): 
            # Obtain Pointcloud
            raw_points = read_points_list(cloud_msg, skip_nans=True) # list of points with (x,y,z,intensity,ring)
            raw_points_4d = np.array(raw_points)[:,:-1]
            filtered = self.points_basic_filter(raw_points_4d)
            crop_points = raw_points_4d[filtered]
            choice = np.random.choice(crop_points.shape[0], 2000, replace=True)
            crop_points = crop_points[choice]
            norm_points = pcd_normalize(crop_points)

            # Obtain Image
            img_arr = np.fromstring(img_msg.data, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR) # [height, width, 3]
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.undistort(img, self.K, self.D)
            img = self.img_normalize(img)

            # Obtain GroundTruth Velocity
            gt_vel = [vel_msg.twist.linear.x, vel_msg.twist.linear.y, vel_msg.twist.linear.z, 
                        vel_msg.twist.angular.x, vel_msg.twist.angular.y, vel_msg.twist.angular.z]
            
            # Run Pointcloud Prediction
            with log.Tick(name="cloud_segmentation"):
                points_tensor = torch.from_numpy(norm_points).unsqueeze(0).transpose(2,1).float().cuda()
                with torch.no_grad():
                    logits,_ = self.cloud_seg_model(points_tensor)
                    cloud_pred_label = logits[0].argmax(-1).cpu().numpy()

            norm_points_with_label = np.concatenate((norm_points, cloud_pred_label.reshape(-1,1)),axis=1) # shape = (n, 5)

            stack_size = len(self.stack_list)
            if stack_size < self.time_step:
                # queue not full, only push
                pass
                self.stack_list.append(norm_points_with_label)
                
            else:
                # queue is full
                self.stack_list.pop(0) # pop first
                self.stack_list.append(norm_points_with_label)
            
                # Vel Prediction by PointCloud
                
                input_data = torch.Tensor(self.stack_list)
                input_data = input_data.reshape(1, 4, 10000) # param "10000" same as that in func load_lstm
                input_data = input_data.cuda()

                with log.Tick(name="cloud_prediction"):
                    with torch.no_grad():
                        pred = self.cloud_pred_model(input_data)
                        cloud_predict_vel = pred.cpu().numpy()[0][0] #predict_vel as [[w]]
                
                # Vel Prediction by Image
                
                img_tensor = torch.from_numpy(img).unsqueeze(0).float().cuda() # img size in CNN: [batch_size, 3, height, width], but transpose is done inside CNN forward function
                print(img_tensor.shape)
                with log.Tick(name="image_prediction"):
                    with torch.no_grad():
                        pred = self.img_pred_model(img_tensor)
                        img_pred_vel = pred.cpu().numpy()[0][0]


                # Add Prediction
                self.cloud_predict_w_list.append(cloud_predict_vel)
                self.img_predict_w_list.append(img_pred_vel)
                self.gt_w_list.append(gt_vel[-1])
                print(dt)
                rand_vel = gt_vel[-1] + self.ou_noise(dt=dt)
                self.random_w_list.append(rand_vel)

                # Print Prediction
                print("[TIMESTAMP: %.3f] gt vel = %.3f, vel@cloud = %.3f, vel@image = %.3f, random vel = %.3f" 
                        % (self.timeStamp, gt_vel[-1], cloud_predict_vel, img_pred_vel, rand_vel))

                ################# Save File #############
                np.save(os.path.join(VIZ_DATA_PATH, "cloud_predict.npy"), np.array(self.cloud_predict_w_list))
                np.save(os.path.join(VIZ_DATA_PATH, "img_predict.npy"), np.array(self.img_predict_w_list))
                np.save(os.path.join(VIZ_DATA_PATH, "random.npy"), np.array(self.random_w_list))
                np.save(os.path.join(VIZ_DATA_PATH, "gt.npy"), np.array(self.gt_w_list))

                ################# Publish ###############
                # Publish Pointcloud
                self.publish_points(crop_points, cloud_pred_label, cloud_msg.header)

                # Publish command
                msg = Twist()
                msg.linear.x = float(gt_vel[0])
                msg.angular.z = float(img_pred_vel) # or img_pred_vel
                self.cmd_vel_pub.publish(msg)
        else: pass
            
            

    def img_normalize(self, img):
        mean = img.mean(axis=(0, 1), keepdims=True)
        std = img.std(axis=(0,1), keepdims=True)
        img = (img - mean) / std
        return img

    
    def load_cam_info(self, file_name):
        self.cam_info = CameraInfo()
        with open(file_name,'r') as cam_calib_file :
            cam_calib = yaml.load(cam_calib_file, Loader=yaml.FullLoader)
            self.cam_info.height = cam_calib['image_height']
            self.cam_info.width = cam_calib['image_width']
            self.img_width, self.img_height = self.cam_info.width, self.cam_info.height
            self.cam_info.K = cam_calib['camera_matrix']['data']
            self.cam_info.D = cam_calib['distortion_coefficients']['data']
            self.cam_info.distortion_model = cam_calib['distortion_model']
            self.K = np.array(self.cam_info.K).reshape(3,3)
            self.D = np.array(self.cam_info.D)
        print("Params Load Success")

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
            # point.extend(MINI_COLOR[pred[i]])
            r, g, b = MINI_COLOR[pred[i]]
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            point.append(rgb)
            new_pts.append(point)
        # print(len(new_pts[0]))
        ros_cloud = sensor_msgs.point_cloud2.create_cloud(header, self.create_pc_fields(), new_pts)
        self.cloud_pub.publish(ros_cloud)

    def create_pc_fields(self):
        fields = []
        fields.append( PointField( 'x', 0, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'y', 4, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'z', 8, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'intensity', 12, PointField.FLOAT32, 1 ) )
        fields.append( PointField( 'rgba', 16, PointField.UINT32, 1 ) )
        return fields

if __name__ == "__main__":
    try:
        pcpredict = FusePredict()
    except rospy.ROSInterruptException:
        print("ERROR")
    rospy.spin()