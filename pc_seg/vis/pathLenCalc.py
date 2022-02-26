from sys import path
from numpy.linalg.linalg import get_linalg_error_extobj
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from rospy.client import init_node


initialPose = False
lastPose = np.array([])
pathLength = 0

def poseCallback(odo_msg):
    pose = np.array([odo_msg.pose.pose.position.x, odo_msg.pose.pose.position.y, odo_msg.pose.pose.position.z])
    print(pose)
    global initialPose
    global lastPose
    global pathLength
    if not initialPose:
        lastPose = pose
        initialPose = True
    else:
        pathLength += np.linalg.norm(pose - lastPose)
        lastPose = pose






if __name__ == "__main__":
    rospy.init_node("path_length_calculation", anonymous=True)
    path_pub = rospy.Subscriber("/husky2_robot_pose", Odometry, poseCallback, queue_size=5)
    
    rospy.spin()

    print("\n pathLength = ", pathLength)
