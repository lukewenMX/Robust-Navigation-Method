
roscore

rosrun image_transport republish compressed in:=/camera/left/image_raw raw out:=/camera/left/image_raw
rosbag play /media/qing/DATA3/DeepLabV3Plus-Pytorch/husky1-ICRA-_2021-03-11-22-01-01.bag
rosbag play /media/qing/DATA3/DeepLabV3Plus-Pytorch/husky1-ICRA-_2021-03-11-22-34-10.bag
rosbag play /media/qing/DATA3/DeepLabV3Plus-Pytorch/husky1-ICRA-_2021-03-11-22-52-26.bag
rosbag play /media/qing/DATA3/DeepLabV3Plus-Pytorch/husky2-ICRA-_2021-03-15-23-19-21.bag
rosparam set use_sim_time true
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-44-19_0.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-45-40_1.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-46-56_2.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-48-17_3.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-49-33_4.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-50-49_5.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-52-13_6.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-53-46_7.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-54-59_8.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-56-06_9.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-57-13_10.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-58-17_11.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-05-59-24_12.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-06-00-37_13.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-06-01-49_14.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-06-02-57_15.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-06-04-12_16.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-06-05-29_17.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-06-06-45_18.bag --clock
rosbag play husky1-continental-nanyanglink-_2020-08-06-06-09-38_20.bag --clock
rosbag play husky1-continental-nanyanglink-day-anti-_2021-01-19-03-03-09_5.bag --clock
python cam_lidar_fusion.py
python republisher.py
