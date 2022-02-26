运行指令放在cmd.sh里面可以参考  
1.roscore  
2.rosrun image_transport republish compressed in:=/camera/left/image_raw raw out:=/camera/left/image_raw   
3.rosbag play rosbag --clock 暂停  
4.rosparam set use_sim_time true  
5.python cam_lidar_fusion.py  
6.python republisher.py  
7.播放rosbag  

#标签简化的映射关系在map.py里  
#提取数据以及保存的位置可在cam_lidar_fusion.py里修改  
