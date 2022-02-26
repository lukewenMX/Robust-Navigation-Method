'''
scripts for converting pointcloud to range image
'''

#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
# import ipdb
import matplotlib.pyplot as plt
# from numpy.ma.extras import _fromnxfunction_single
# from torch._C import dtype 
# import glog
import cv2
import time
mini_color = [[128, 64, 128], [50, 205, 50], [250, 170, 30], [220, 20, 60], [0, 0, 255]]

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin','.npz']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, hfov = 85.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.hfov = hfov
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m, 1]: remission
    self.label = np.zeros((0,1),dtype=np.uint32)
    self.rings = np.zeros((0,1),dtype=np.uint32)
    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)
    # projected label - [H,W] colored image
    self.proj_label = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
    # projected label color - [H,W,3] colored image
    self.proj_color = np.full((self.proj_H, self.proj_W, 3), 0, dtype=np.uint8)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    # scan = np.fromfile(filename, dtype=np.float32)
    scanfile = np.load(filename)
    scan = scanfile["pts_l"]

    # scan = scan.reshape((-1, 4))
    scan = scan.reshape((-1,6))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    label = scan[:, 4]
    rings = scan[:, 5]
    self.set_points(points, remissions, label, rings)

  def set_points(self, points, remissions=None, label=None, rings=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz

    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    self.label = label.astype(np.int32) if label is not None else None
    self.rings = rings.astype(np.int32) if rings is not None else None

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]
    
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    
    pitch = np.arcsin(scan_z / depth)

    hfov_rad = self.hfov / 180 * np.pi ### Preset HFOV
    # get projections in image coords
    # proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_x = 0.5 * (yaw / (hfov_rad / 2.0) + 1.0)          # in [0.0, 1.0]
    
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order 
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    
    proj_y = proj_y[order] if self.rings is None else 15 - self.rings[order]
    proj_x = proj_x[order]


    # assing to images
    
    self.proj_range[proj_y, proj_x] = depth
    self.proj_range_no_completion = self.proj_range.copy() # just for comparison
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_xyz[proj_y, proj_x] = points
    if self.label is not None:
      label = self.label[order]
      self.proj_label[proj_y, proj_x] = label
      self.proj_color[proj_y, proj_x] = np.array(mini_color, dtype='uint8')[label,::-1]

    idx = np.argwhere(np.all(self.proj_range[...,:] == -1, axis=0)) # find columns with all zeros
    
    start = time.time()
  
    if idx.size:    
      idx_1d = idx[:,0].transpose()
      # print("idx_1d = ", idx_1d)
      # front, end = (np.min(idx_1d) - 1 + self.proj_W) % self.proj_W, (np.max(idx_1d) + 1) % self.proj_W 
      # self.proj_range[:,idx_1d] =np.expand_dims(np.apply_along_axis(lambda x: np.mean(x), 1, self.proj_range[:,[end,front]]), 1).repeat(idx.size, axis=1)
      # self.proj_remission[:,idx_1d] = np.expand_dims(np.apply_along_axis(lambda x: np.mean(x), 1, self.proj_remission[:,[end,front]]), 1).repeat(idx.size, axis=1)
      # self.proj_label[:,idx_1d] = np.expand_dims(np.apply_along_axis(lambda x: np.random.choice(x), 1, self.proj_label[:,[end,front]]), 1).repeat(idx.size, axis=1)
      # self.proj_color[:,idx_1d] = np.expand_dims(np.apply_along_axis(lambda x: np.random.choice(x), 1, self.proj_color[:,[end,front]]), 1).repeat(idx.size, axis=1)
      
      ## Under Modification
      interval_list = []
      i = 0
      j = 0
      while (i <= j and j < len(idx_1d)):
        if (idx_1d[j] == idx_1d[i] + (j - i)):
          pass
        else:
          interval_list.append([idx_1d[i] - 1, idx_1d[j-1] + 1])
          i = j
        j+=1
      interval_list.append([idx_1d[i] -1, idx_1d[j - 1] + 1 if idx_1d[j-1] != self.proj_W - 1 else -1])
      # print("interval_list = ", interval_list)

      for interval in interval_list:
        front, end = interval
        if front != -1 and end != -1:
          self.proj_range[:,front+1:end] = np.expand_dims(np.apply_along_axis(lambda x: np.mean(x), 1, self.proj_range[:,[front,end]]), 1).repeat(end-front-1, axis=1)
          self.proj_remission[:,front+1:end] = np.expand_dims(np.apply_along_axis(lambda x: np.mean(x), 1, self.proj_remission[:,[front,end]]), 1).repeat(end-front-1, axis=1)
          self.proj_label[:,front+1:end] = np.expand_dims(np.apply_along_axis(lambda x: np.random.choice(x), 1, self.proj_label[:,[front,end]]), 1).repeat(end-front-1, axis=1)
          self.proj_color[:,front+1:end] = np.expand_dims(np.apply_along_axis(lambda x: np.random.choice(x), 1, self.proj_color[:,[front,end]]), 1).repeat(end-front-1, axis=1)
        elif front == -1:
          self.proj_range[:,front+1:end] = np.expand_dims(self.proj_range[:,end], 1).repeat(end-front-1, axis=1)
          self.proj_remission[:,front+1:end] = np.expand_dims(self.proj_remission[:,end], 1).repeat(end-front-1, axis=1)
          self.proj_label[:,front+1:end] = np.expand_dims(self.proj_label[:,end], 1).repeat(end-front-1, axis=1)
          self.proj_color[:,front+1:end] = np.expand_dims(self.proj_color[:,end], 1).repeat(end-front-1, axis=1)
        else:
          self.proj_range[:,front+1:] = np.expand_dims(self.proj_range[:,front], 1).repeat(self.proj_W-front-1, axis=1)
          self.proj_remission[:,front+1:] = np.expand_dims(self.proj_remission[:,front], 1).repeat(self.proj_W-front-1, axis=1)
          self.proj_label[:,front+1:] = np.expand_dims(self.proj_label[:,front], 1).repeat(self.proj_W-front-1, axis=1)
          self.proj_color[:,front+1:] = np.expand_dims(self.proj_color[:,front], 1).repeat(self.proj_W-front-1, axis=1)


    finish = time.time()
    # print(finish - start)

    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    
  def do_rotation(self, clip_angle = 0, rotation_angle = 0):
    # glog.check_lt(clip_angle + 2 * np.abs(rotation_angle), self.hfov)
    k = self.proj_W / self.hfov
    print(k)
    clip = (self.hfov - clip_angle) / 2.0
    print(clip)
    clip_start = int(max(0.0, (clip - rotation_angle) * k))
    print(clip_start)
    clip_end = int(min(float(self.proj_W), self.proj_W - (clip + rotation_angle) * k))
    print(clip_end)
    return self.proj_range[:,clip_start : clip_end]
    

class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self,  sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()

    # make semantic colors
    if sem_color_dict:
      # if I have a dict, make it
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      # otherwise make random
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      # force zero to a gray-ish color
      self.sem_color_lut[0] = np.full((3), 0.1)

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # # instance labels
    # self.inst_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    # self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=np.float)              # [H,W,3] color

    # # projection color with instance labels
    # self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
    #                                 dtype=np.int32)              # [H,W]  label
    # self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
    #                                 dtype=np.float)              # [H,W,3] color

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))

    # set it
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half
      self.inst_label = label >> 16    # instance id in upper half
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

    # # instances
    # self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    # self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]



if __name__ == "__main__":
  file_name = "/media/yxdai/HP P500/dataset/NanyangLink/pts_label/07/pts_l000200.npz"

  converter = LaserScan(project=True, H=16, W=200, fov_up=15, fov_down=-15)
  #### Usage 1 ####
  converter.open_scan(file_name)
  #### Usage 2 ####
  data = np.load(file_name)
  scan = data['pts_l']
  # converter.set_points(points=scan[:,0:3],remissions=scan[:,3],label=scan[:,4],rings=scan[:,5])
  #### Get Projection Result ####
  raw_depth_img = converter.proj_range_no_completion
  depth_img = converter.proj_range 
  intensity_img = converter.proj_remission
  label_img = converter.proj_label
  color_img = converter.proj_color

  # clip_img = converter.do_rotation(clip_angle=70, rotation_angle=7)
  ## Display ####
  plt.figure()
  plt.imshow(cv2.cvtColor(cv2.imread("/media/yxdai/HP P500/dataset/NanyangLink/img/07/000200.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
  # plt.imshow(raw_depth_img,cmap="gray")
  # plt.imshow(cv2.bilateralFilter(depth_img, 3, 75, 75), cmap="gray")
  plt.figure()
  plt.subplot(2,1,1)
  plt.imshow(depth_img, cmap="gray")
  plt.subplot(2,1,2)
  plt.imshow(color_img)
  plt.show()
