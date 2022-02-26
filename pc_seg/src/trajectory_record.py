#!/usr/bin/env python2.7

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from rospy.impl.tcpros import init_tcpros
import tf

traj = []
init_x = 0
init_y = 0
init_z = 0
flag = False
if __name__ == "__main__":
    rospy.init_node("tf_saver", anonymous=True)
    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)
    
    while not rospy.is_shutdown():
        try:
            (trans, rot) =  listener.lookupTransform("/map","/base_link",rospy.Time(0))
            (x,y,z) = trans
            if not flag:
                flag = True
                init_x = x
                init_y = y
                init_z = z
                traj.append([0,0,0])
            else:
                traj.append([x - init_x, y - init_y, z - init_z])
            # print traj
            np.save("test.npy", np.array(traj))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        
        
        rate.sleep()
    

    # print("\n pathLength = ", pathLength)
