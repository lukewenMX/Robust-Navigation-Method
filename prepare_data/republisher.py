import os
import math
import random
import numpy as np

import rospy

from nav_msgs.msg import Odometry

from geometry_msgs.msg import Twist


from sensor_msgs.point_cloud2 import read_points



class VelRepublisher():
    def __init__(self):
        self.node = rospy.init_node('republisher', anonymous=True)
        self.pub_odometry = rospy.Publisher("/odom_topic", Odometry, queue_size=1)
        # self.sub_cam1_info = rospy.Subscriber(rospy.resolve_name( '/camera/left/camera_info'), CameraInfo, callback=self.cam1_info_callack, queue_size=1)
        self.twist_subscriber = rospy.Subscriber('/husky_velocity_controller/cmd_vel', Twist, callback=self.twist_callback)
        self.odometry = Odometry()
    def twist_callback(self,data):
        self.odometry.header.stamp = rospy.Time.now()
        print(rospy.Time.now())
        self.odometry.twist.twist.linear.x = data.linear.x
        self.odometry.twist.twist.linear.y = data.linear.y
        self.odometry.twist.twist.linear.z = data.linear.z
        self.odometry.twist.twist.angular.x = data.angular.x
        self.odometry.twist.twist.angular.y = data.angular.y
        self.odometry.twist.twist.angular.z = data.angular.z
        self.pub_odometry.publish(self.odometry)


if __name__ == '__main__':
    try:
        cloud_generator = VelRepublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
