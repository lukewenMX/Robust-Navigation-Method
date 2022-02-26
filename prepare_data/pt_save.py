import os
import numpy as np
import rospy
import cv2
import message_filters  

from sensor_msgs.msg import PointField, PointCloud2, Image
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2
from sensor_msgs.point_cloud2 import read_points
import yaml
from pt2range import LaserScan

DATA_DIR = "./sidewalk_data"
DATA_PATCH = 2


class CloudGenerator():
    def __init__(self, img_res = (672, 376)):
        if (DATA_PATCH >= 0 and DATA_PATCH < 10):
            self.data_patch = '0' + str(DATA_PATCH)
        else:
            self.data_patch = str(DATA_PATCH)
        self.label_path = os.path.join(DATA_DIR, 'pts', str(self.data_patch))
        
        self.pt2range_cvt = LaserScan(project=True, H=16, W=1024, fov_up=15, fov_down=-15)
        
        os.makedirs(self.label_path, exist_ok=True)
        self.num = 0

        self.node = rospy.init_node('lidar_to_rgb', anonymous=True)
        # self.bridge = CvBridge()
        self.pub_cloud = rospy.Publisher("/label_cloud", PointCloud2, queue_size = 1 )
        self.pub_image = rospy.Publisher("/range_image",Image, queue_size = 1 )

        self.sub_cloud = rospy.Subscriber('/velodyne_points', PointCloud2, self.cloudCallback, queue_size=1)
    
        rospy.spin()

    def cv2_to_imgmsg(self, img):
        # easy implementation converting 3-channels image to sensor_msgs::Image
        img_msg = Image()
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]

        img_msg.encoding = "32FC1"
        img_msg.data = img.tostring()
        img_msg.step = int(len(img_msg.data) / img_msg.height)
        
        return img_msg

        
    def cloudCallback(self, msg_cloud):
        pts_r = []
        for point in (read_points(msg_cloud, skip_nans=True)):
                pts_r.append(point)
        
        
        ## This part is added for testing the range image conversion
        scan = np.array(pts_r)
        self.pt2range_cvt.set_points(points=scan[:,0:3], remissions=scan[:,3], label=None, rings=scan[:,4])
        range_img = self.pt2range_cvt.proj_range_no_completion
        np.savez(os.path.join(self.label_path, 'pts_l%06d' % (self.num)), pts_l=pts_r)
        self.num += 1

        self.pub_image.publish(self.cv2_to_imgmsg(range_img))

if __name__ == '__main__':
    try:
        cloud_generator = CloudGenerator()
    except rospy.ROSInterruptException:
        pass