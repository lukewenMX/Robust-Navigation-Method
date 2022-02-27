# Code Package for end2end navigation
- ```prepare_data``` directory provides scripts to
  - extract different types of data (images, pointclouds etc.) from rosbag
  - manually split a whole dataset into train and test part and generate list file respectively
- ```pc_seg``` directory provides scripts to 
  - train different variants of PointNet for point segmentation
  - run the navigation test on robot with different types of sensor input (image, point cloud and random controller etc.) and pretrained model
- The dataset we used to train and test the proposed method can be downloaded through [Google Drive](https://drive.google.com/file/d/19jlJjrdp_5CfxelZEjouCHW9srt6EP9C/view?usp=sharing)
