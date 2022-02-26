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
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Twist

from image_geometry import PinholeCameraModel
from model.utils import load_CNN

IMG_PRED_MODEL_PATH = os.path.join(pkg_prefix, "checkpoint/new_cnn_model.pth")
VIZ_DATA_PATH = os.path.join(pkg_prefix, "vis")
CALIB_FILE = os.path.join(pkg_prefix, "params/camera_params.yaml")
MINI_NAME = ['road', 'sidewalk', 'building', 'obstacle', 'human', 'others']
MINI_COLOR = [[128, 64, 128], [50, 205, 50], [70, 70, 70], [250, 170, 30], [220, 20, 60], [0, 0, 255]] #RGB

VERBOSE=False


class ImagePredict():
    def __init__(self) -> None:
        
        self.img_pred_model = load_CNN("CNN", IMG_PRED_MODEL_PATH)
        self.camera_model = PinholeCameraModel()
        self.load_cam_info(CALIB_FILE)

        self.img_predict_w_list = []
        self.last_cmd = 0
        self.node = rospy.init_node("image_to_velocity", anonymous=True)

        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)

        self.img_sub = rospy.Subscriber("/camera/left/image_raw/compressed", CompressedImage, self.imageCallback)
        
        rospy.spin()

    def imageCallback(self, img_msg):

        # Obtain Image
        img_arr = np.fromstring(img_msg.data, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR) # [height, width, 3]
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.undistort(img, self.K, self.D)
        img = self.img_normalize(img)

        # Vel Prediction by Image
        # img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).float().cuda() # img size needed by CNN: [batch_size, 3, height, width]
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().cuda()
        with log.Tick(name="image_prediction"):
            with torch.no_grad():
                pred = self.img_pred_model(img_tensor)
                pred_vel = pred.cpu().numpy()[0][0]

        
        # Add Prediction
        img_pred_vel = (self.last_cmd + pred_vel) / 2.0
        self.last_cmd = pred_vel
        self.img_predict_w_list.append(img_pred_vel)

        # Print Prediction
        print("[TIMESTAMP: %.3f] vel@image = %.3f" % (img_msg.header.stamp.to_sec(), img_pred_vel))

        ################# Save File #############

        # np.save(os.path.join(VIZ_DATA_PATH, "img_predict.npy"), np.array(self.img_predict_w_list))

        ################# Publish ###############


        # Publish command
        msg = Twist()
        msg.linear.x = float(1)
        msg.angular.z = float(img_pred_vel) # or img_pred_vel
        self.cmd_vel_pub.publish(msg)

            
            

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


if __name__ == "__main__":
    try:
        pcpredict = ImagePredict()
    except rospy.ROSInterruptException:
        print("ERROR")
    rospy.spin()