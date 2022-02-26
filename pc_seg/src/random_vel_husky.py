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
import pandas as pd
import torch.nn.functional as F
import data_utils.my_log as log
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points_list
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry

from image_geometry import PinholeCameraModel
from model.utils import load_CNN, load_pointnet, load_lstm
from data_utils.fhy4_Sem_Loader import pcd_normalize
from data_utils.noise_generator import OrnsteinUhlenbeckNoise
from scipy.spatial.transform import Rotation as R


VIZ_DATA_PATH = os.path.join(pkg_prefix, "vis")



if __name__ == "__main__":

    node = rospy.init_node("random_velocity", anonymous=True)
    ou_noise = OrnsteinUhlenbeckNoise(mu=0, sigma=0.01)
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
    random_w_list = []
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rand_vel = ou_noise(0.1)
        print("rand_vel = ", rand_vel)
        msg = Twist()
        msg.linear.x = float(1)
        msg.angular.z = float(rand_vel)
        random_w_list.append(rand_vel)
        # np.save(os.path.join(VIZ_DATA_PATH, "ou_random.npy"), np.array(random_w_list))
        cmd_vel_pub.publish(msg)
        rate.sleep()
