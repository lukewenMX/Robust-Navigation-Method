import os
import numpy as np
from numpy.core.records import array
from torch.utils.data import Dataset
from pt2range import LaserScan
import random
import math

class RangeDataset(Dataset):
    def __init__(self, path: str, Train: bool, is_augmentation: bool, is_seq: bool, acc : float) -> None:
        super(RangeDataset).__init__()

        self.path = path
        self.is_augmentation = is_augmentation
        self.is_seq = is_seq
        self.acc = acc
        self.Train = Train

        if Train:
            self.pcl_index = {"00":395, "06":788, "07":444, "08":498, "09":444, "10":398, "11":426, "12":413, "13":498, "14":496}
        else:
            self.pcl_index = {"15":581, "16":474, "17":541, "18":671, "19":536}
            # self.pcl_index = {"16":474}

        self.len = sum(self.pcl_index.values())
        self.keys = []

        for key,value in self.pcl_index.items():
            for index in range(0, value):
                self.keys.append((key, index))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        part, index = self.keys[i]
        if self.is_seq:
            # if not self.Train:
            #     timesteps = self.len
            # else:
            #     timesteps = 10
            timesteps = 10
            feature, velocity = self.get_seq_data(self.path, self.keys[i:i+timesteps])
        else:
            feature, velocity = self.get_data(self.path, part, index, self.is_augmentation)

        return feature, velocity

    def get_seq_data(self, path: str, keys: list):
        last_part = 0
        vel_list = []
        feat_list = []
        for i, (part, index) in enumerate(keys):
            if i == 0:
                last_part = part
            if part != last_part:
                break
            # velocity
            fn_vel = os.path.join(path, 'velocity', str(part))
            vel_data =  np.load(fn_vel+'/velocity%06d.npz' % (index))
            velocity = vel_data['vel'][0][-1]
            vel_list.append(velocity)

            # range image
            fn_points = os.path.join(path, 'label', str(part))
            vel_data =  fn_points+'/pts_l%06d.npz' % (index)
            converter = LaserScan(project=True, H=16, W=200, fov_up=15, fov_down=-15)
            converter.open_scan(vel_data)
            depth_img = converter.proj_range 
            intensity_img = converter.proj_remission
            label_img = converter.proj_label
            feature = np.array([depth_img, intensity_img, label_img])
            feature = feature.reshape(3, 16, 200)
            feat_list.append(feature)

        while len(feat_list) < 10:
            feat_list.append(feat_list[-1])
            vel_list.append(vel_list[-1])

        feat_list = np.array(feat_list)
        vel_list = np.array(vel_list)

        return feat_list, vel_list


    def get_data(self, data_path : str, part : int, index : int, is_augmentation : bool):
        # velocity
        fn_vel = os.path.join(data_path, 'velocity', str(part))
        vel_data =  np.load(fn_vel+'/velocity%06d.npz' % (index))
        velocity = vel_data['vel'][0][-1]

        # range image
        fn_points = os.path.join(data_path, 'label', str(part))
        vel_data =  fn_points+'/pts_l%06d.npz' % (index)
        converter = LaserScan(project=True, H=16, W=200, fov_up=15, fov_down=-15)
        converter.open_scan(vel_data)
        depth_img = converter.proj_range 
        intensity_img = converter.proj_remission
        label_img = converter.proj_label
        color_img = converter.proj_color
        # print("-----------")
        # print(f"depth image shape: {depth_img.shape}")
        # print(f"intensity image shape: {intensity_img.shape}")
        # print(f"label image shape: {label_img.shape}")
        # print(f"color image shape: {color_img.shape}")

        if is_augmentation:
            random_rotation = random.uniform(-7, 7)
            random_velocity = math.sqrt(abs(2 * self.acc * random_rotation * 3.14 / 180.0))
            if random_rotation < 0:
                random_velocity = -random_velocity
            depth_img, intensity_img, label_img = converter.do_rotation(clip_angle=70, rotation_angle=random_rotation)
            velocity = random_velocity + velocity

        feature = np.array([depth_img, intensity_img, label_img])
        feature = feature.reshape(3, 16, 165)
        # print(feature.shape)
        return feature, velocity