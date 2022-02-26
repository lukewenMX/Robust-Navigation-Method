import os
import cv2
import json
import yaml
import numpy as np
import random
from tqdm import tqdm
import torch
from PIL import Image

data_path = "/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/img_velo_label_327-002"

mini_name = ['road', 'sidewalk', 'building', 'obstacle', 'human', 'others']
mini_color = [[128, 64, 128], [50, 205, 50], [70, 70, 70], [250, 170, 30], [220, 20, 60], [0, 0, 255]] #RGB
mini_color_BGR = [[128, 64, 128], [50, 205, 50], [70, 70, 70], [30, 170, 250], [60, 20, 220], [255, 0, 0]]
kitti_colors = [[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],[153, 153, 153],
        [250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],[0, 130, 180],
        [220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70], [0, 60, 100],[0, 80, 100],[0, 0, 230],[119, 11, 32]]
map_kitti2mini = {
    'road':     'road',
     'sidewalk': 'sidewalk',
     'building': 'building',
     'wall':          'obstacle',
     'fence':         'obstacle',
     'pole':          'obstacle',
     'traffic_light': 'others',
     'traffic_sign': 'others',
     'vegetation':   'obstacle',
     'terrain':      'sidewalk',
     'sky':          'others',
     'person':    'human',
     'rider':     'obstacle',
     'car':       'obstacle',
     'truck':     'obstacle',
     'bus':       'obstacle',
     'train':      'obstacle',
     'motorcycle': 'obstacle',
     'bicycle':     'obstacle'
}

def do_map(x):
    x = x.copy()
    for src_id,src in enumerate(map_kitti2mini):
        dst_id = mini_name.index(map_kitti2mini[src])
        x[x==src_id] = dst_id
    return x

class Semantic_KITTI_Utils():
    def __init__(self, data_path, subset = 'all'):
        self.mini_name = mini_name
        self.mini_color = mini_color
        self.mini_color_BGR = mini_color_BGR
        self.data_path = data_path
        self.class_names = mini_name
        self.num_classes = len(self.class_names)
        self.R, self.T, self.K, self.D = self.calib_velo2cam(self.data_path+"/calib_range_to_cam_00.txt")
        self.RT = np.concatenate((self.R,self.T),axis=1)
        self.mini_color = np.array(mini_color)
        self.mini_color_BGR = np.array(mini_color_BGR)
        assert subset in ['all', 'inview'], subset
        self.subset = subset

        
    def get(self, index, load_image = False):

        if load_image:
            fn_frame = os.path.join(self.data_path, 'img/%06d.png' % (index))
            assert os.path.exists(fn_frame), 'Broken dataset %s' % (fn_frame)
            self.frame_BGR = cv2.imread(fn_frame)
            self.frame = cv2.undistort(self.frame_BGR, self.K, self.D)
            # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # self.frame_HSV = cv2.cvtColor(self.frame_BGR, cv2.COLOR_BGR2HSV)

        fn_velo = os.path.join(self.data_path, 'velody')
        # fn_label = os.path.join(data_path, 'semantic/%06d.npz' %(index))
        assert os.path.exists(fn_velo), 'Broken dataset %s' % (fn_velo)
        # assert os.path.exists(fn_label), 'Broken dataset %s' % (fn_label)
            
        data = np.load(fn_velo+'/%06d.npz' % (index))
        points = data['pointcloud']
        # raw_label = np.fromfile(fn_label, dtype=np.uint32)['pre_mini']

        # if raw_label.shape[0] == points.shape[0]:
        #     label = raw_label & 0xFFFF  # semantic label in lower half
        #     inst_label = raw_label >> 16  # instance id in upper half
        #     assert((label + (inst_label << 16) == raw_label).all()) # sanity check
        # else:
        #     print("Points shape: ", points.shape)
        #     print("Label shape: ", label.shape)
        #     raise ValueError("Scan and Label don't contain same number of points")
        
        # Map to learning 20 classes
        # label = np.array([self.learning_map[x] for x in label], dtype=np.int32)

        # Drop class -> 0
        # drop_class_0 = np.where(label != 0)
        # points = points[drop_class_0]
        # label = label[drop_class_0] - 1
        # assert (label >=0).all and (label<self.num_classes).all(), np.unique(label)

        if self.subset == 'inview':
            self.set_filter([-45, 45], [-20, 20]) # 水平[-45, 45]， 垂直[-20, 20]
            combined = self.points_basic_filter(points)
            points = points[combined]
            #label = label[combined]

        return points

    def get_classes(self, index, data_path, load_image = False):
        if load_image:
            fn_frame = os.path.join(data_path, 'img/%06d.png' % (index))
            assert os.path.exists(fn_frame), 'Broken dataset %s' % (fn_frame)
            self.frame_BGR = cv2.imread(fn_frame)
            self.frame = cv2.undistort(self.frame_BGR, self.K, self.D)
            #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            #self.frame_HSV = cv2.cvtColor(self.frame_BGR, cv2.COLOR_BGR2HSV)
        fn_velo = os.path.join(data_path, 'velody')
        fn_pixel = os.path.join(data_path, 'semantic')
        data = np.load(fn_velo+'/%06d.npz' % (index))
        points = data['pointcloud']
        data2 = np.load(fn_pixel+'/%06d.npz' % (index))
        classes = data2['pred_mini']

        if self.subset == 'inview':
            self.set_filter([-45, 45], [-20, 20])
            combined = self.points_basic_filter(points)
            points = points[combined]
        return points, classes

    def get_pts_l(self, part, index, load_image = False):
        if load_image:
            fn_frame = os.path.join(self.data_path, 'img', str(part),'%06d.png' % (index))
            assert os.path.exists(fn_frame), 'Broken dataset %s' % (fn_frame)
            self.frame_BGR = cv2.imread(fn_frame)
            self.frame = cv2.undistort(self.frame_BGR, self.K, self.D)
            # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # self.frame_HSV = cv2.cvtColor(self.frame_BGR, cv2.COLOR_BGR2HSV)
        fn_velo = os.path.join(self.data_path, 'label', str(part))
        data = np.load(fn_velo+'/pts_l%06d.npz' % (index))
        points_label = data['pts_l']
        points = points_label[:,:-1]
        labels = points_label[:,-1].astype(np.int32)
        return points, labels

    def get_pts_vel(self, part, index, load_image = False):
        if load_image:
            fn_frame = os.path.join(self.data_path, 'img', str(part),'%06d.png' % (index))
            assert os.path.exists(fn_frame), 'Broken dataset %s' % (fn_frame)
            self.frame_BGR = cv2.imread(fn_frame)
            self.frame = cv2.undistort(self.frame_BGR, self.K, self.D)
            # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # self.frame_HSV = cv2.cvtColor(self.frame_BGR, cv2.COLOR_BGR2HSV)
        fn_points = os.path.join(self.data_path, 'label', str(part))
        fn_vel = os.path.join(self.data_path, 'velocity', str(part))
        data = np.load(fn_points+'/pts_l%06d.npz' % (index))
        vel_data =  np.load(fn_vel+'/velocity%06d.npz' % (index))
        points_label = data['pts_l']
        points = points_label[:,:]
        velocity = vel_data['vel'][:,(0,5)]
        #velocity = vel_data['vel'][:,:]
        #velocity = np.array([vel_data['vel'][:,0],vel_data['vel'][:,5]*10])
        velocity = velocity.reshape(2)
        return points, velocity

    def set_filter(self, h_fov, v_fov, x_range = None, y_range = None, z_range = None, d_range = None):
        self.h_fov = h_fov if h_fov is not None else (-180, 180)
        self.v_fov = v_fov if v_fov is not None else (-25, 20)
        self.x_range = x_range if x_range is not None else (-10000, 10000)
        self.y_range = y_range if y_range is not None else (-10000, 10000)
        self.z_range = z_range if z_range is not None else (-10000, 10000)
        self.d_range = d_range if d_range is not None else (-10000, 10000)

    def hv_in_range(self, m, n, fov, fov_type='h'):
        """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
            horizontal limit = azimuth angle limit
            vertical limit = elevation angle limit
        """
        if fov_type == 'h':
            return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def box_in_range(self,x,y,z,d, x_range, y_range, z_range, d_range):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return np.logical_and.reduce((
                x > x_range[0], x < x_range[1],
                y > y_range[0], y < y_range[1],
                z > z_range[0], z < z_range[1],
                d > d_range[0], d < d_range[1]))

    def points_basic_filter(self, points):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            2. x,y,z distance limit
            return a bool array
        """
        assert points.shape[1] == 4, points.shape # [N,4]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))
        r = np.sqrt(x ** 2 + y ** 2)
        # extract in-range fov points
        h_points = self.hv_in_range(x, y, self.h_fov, fov_type='h')
        v_points = self.hv_in_range(z, r, self.v_fov, fov_type='v')
        combined = np.logical_and(h_points, v_points)

        # extract in-range x,y,z points
        in_range = self.box_in_range(x,y,z,d, self.x_range, self.y_range, self.z_range, self.d_range)
        combined = np.logical_and(combined, in_range)

        return combined

    def calib_velo2cam(self, fn_v2c):
        """
        get Rotation(R : 3x3), Translation(T : 3x1) matrix info
        using R,T matrix, we can convert velodyne coordinates to camera coordinates
        """
        for line in open(fn_v2c, "r"):
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)

            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
            
            if key == 'K':
                K = np.fromstring(val, sep=' ')
                K = K.reshape(3, 3)

            if key == 'D':
                D = np.fromstring(val, sep=' ')
                D = D.reshape(5, 1)

        # RT = np.concatenate((R, T), axis=1)
        # RT = np.concatenate((RT,[[0,0,0,1]]),axis=0)
        # iden = np.array([0,-1,0,0,0,0,-1,0,1,0,0,0,0,0,0,1]).reshape(4,4)
        # RT = np.dot(iden,RT)
        # RT = RT[:3]
        return R, T, K, D

    def project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape
        pts_3d = pts_3d.copy()
        
        # Concat and change shape from [N,3] to [N,4] to [4,N]
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float32)
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1).T

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(self.RT, xyz_v[:, i])

        xyz_c = xyz_v[:3]

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(self.K, xyz_c[:, i])

        # normalize image(pixel) coordinates(x,y)
        xy_i = xyz_c / xyz_c[2]

        # get pixels location
        pts_2d = xy_i[:2].T
        return pts_2d

    def torch_project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape
        pts_3d = pts_3d.copy()
        
        # Create a [N,1] array
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float32)
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1)

        RT = torch.from_numpy(self.RT).float().cuda()
        K = torch.from_numpy(self.K).float().cuda()
        xyz_v = torch.from_numpy(xyz_v).float().cuda()

        assert xyz_v.size(1) == 4, xyz_v.size()
    
        xyz_v = xyz_v.unsqueeze(2)
        RT_rep = RT.expand(xyz_v.size(0),3,4)
        K_rep = K.expand(xyz_v.size(0),3,3)

        xyz_c = torch.bmm(RT_rep, xyz_v)
        #log.info(xyz_c.shape, RT_rep.shape, xyz_v.shape)

        xy_v = torch.bmm(K_rep, xyz_c)
        #log.msg(xy_v.shape, P_rep.shape, xyz_c.shape)

        xy_i = xy_v.squeeze(2).transpose(1,0)
        xy_n = xy_i / xy_i[2]
        pts_2d = (xy_n[:2]).transpose(1,0)

        return pts_2d.detach().cpu().numpy()

    def draw_2d_points(self, pts_2d, colors, image = None):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape

        if image is None:
            image = self.frame.copy()
        pts = pts_2d.astype(np.int32).tolist()

        for (x,y),c in zip(pts, colors):
            cv2.circle(image, (x, y), 2, c, -1)

        return image

    def draw_2d_top_view(self, pcd_3d, colors):
        """ draw 2d points in camera image """
        assert pcd_3d.shape[1] == 3, pcd_3d.shape

        image = np.zeros((600,800,3),dtype=np.uint8)

        for (x,y,z),c in zip(pcd_3d.tolist(), colors.tolist()):
            X = int(-x*800+600)
            Y = int(-y*800+400)
            cv2.circle(image, (Y,X), 3, c, -1)

        return image

    def get_labels(self, pts3d, classes, pts2d):
        label = []
        color = []
        for i in range(0, pts2d.shape[0]):
            (x,y) = pts2d[i]
            if 0<=x and x<672 and 0<=y and y<376:
                label.append(classes[y,x])
            else:
                label.append(mini_name.index('others'))
        # label = np.array(label,dtype = np.uint8)
        return label
        

def export_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = cv2.VideoWriter('experiment/pn_compare.avi',fourcc, 15.0, (int(1600*0.8),int(740*0.8)))

    # mkdir('experiment/imgs/%s/'%(args.model_name))
    # vis_handle.capture_screen('experiment/imgs/%s/%d_3d.png'%(args.model_name,i))
    # cv2.imwrite('experiment/imgs/%s/%d_sem.png'%(args.model_name, i), img_semantic)

    for index in range(100, 320):
        pn_3d = cv2.imread('experiment/imgs/pointnet/%d_3d.png' % (index))
        pn_sem = cv2.imread('experiment/imgs/pointnet/%d_sem.png' % (index))
        pn2_3d = cv2.imread('experiment/imgs/pointnet2/%d_3d.png' % (index))
        pn2_sem = cv2.imread('experiment/imgs/pointnet2/%d_sem.png' % (index))

        pn_3d = pn_3d[160:650]
        pn2_3d = pn2_3d[160:650]

        pn_sem = cv2.resize(pn_sem, (800, 250))
        pn2_sem = cv2.resize(pn2_sem, (800, 250))

        pn = np.vstack((pn_3d, pn_sem))
        pn2 = np.vstack((pn2_3d, pn2_sem))

        cv2.putText(pn, 'PointNet', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn, 'PointNet', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)

        merge = np.hstack((pn, pn2))
        class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
        colors = [[255, 255, 255],[245, 150, 100],[30, 30, 255],[255, 0, 255],[0, 200, 255],[0, 175, 0]]
        for i,(name,c) in enumerate(zip(class_names, colors)):
            cv2.putText(merge, name, (200 + i * 200, 50), font,1, [c[2],c[1],c[0]], 2, cv2.LINE_AA)

        cv2.line(merge,(0,70),(1600,70),(255,255,255),2)
        cv2.line(merge,(800,70),(800,1300),(255,255,255),2)

        merge = cv2.resize(merge,(0,0),fx=0.8,fy=0.8)
        # cv2.imshow('merge', merge)
        # if 32 == waitKey(1):
        #     break
        out.write(merge)

        print(index)
    out.release()  

# if __name__ == '__main__':
#     p = Semantic_KITTI_Utils(data_path, 'inview')
#     # myrange = list(range(320,700))+list(range(940,4200))
#     for index in range(3000,7000):
#         pts3d, classes = p.get_classes(index, data_path, True)
#         pts2d = p.project_3d_to_2d(pts3d[:,:3]).astype(np.int32)
#         label = p.get_labels(pts3d, classes, pts2d)
#         label = np.array(label)
#         # print("label:",label.shape,"3d",pts3d.shape)
#         # pts3d_l = pts3d.copy()
#         # pts3d_l = np.concatenate((pts3d_l, label.reshape(-1,1)), axis = 1)
#         # np.savez(data_path + '/label/pts_l%06d'%index, pts_l = pts3d_l)
#         # print('finish saveing:%06d'%index)
#         color = p.mini_color[label]
#         color = np.ndarray.tolist(color)
#         print(pts3d.shape)
#         # label = []
#         # color = []
#         # for i in range(0,pts2d.shape[0]):
#         #     (x,y) = pts2d[i]
#         #     if 0<=x and x<672 and 0<=y and y<376:
#         #         label.append(classes[y,x])
#         #     else:
#         #         label.append(mini_name.index('others'))
#         #     color.append(p.mini_color[label[-1]])

#         image = p.draw_2d_points(pts2d,color)
#         cv2.imshow('img',image)
#         print(index)
#         if 27 == cv2.waitKey(1):
#             break

# if __name__ == '__main__':
#     p = Semantic_KITTI_Utils(data_path, 'inview')
#     for i in range(1,):
#         pts3d, classes = p.get_classes(i, data_path, load_image)
#         pts2d = p.project_3d_to_2d(pts3d[:,:3]).astype(np.int32)
#         label = p.get_labels(pts3d, classes, pts2d)
#         pts3d = p.get(800,True)
#         pts3d = pts3d[:,:3]
#         pts2d = p.project_3d_to_2d(pts3d)
#         print(pts2d)
#         color = np.ones((pts2d.shape[0],3), dtype=np.uint8)
#         color = np.ndarray.tolist(color*255)
#         img = p.draw_2d_points(pts2d, color)
#         cv2.imshow('img',img)
#         cv2.waitKey(0)

if __name__ == '__main__':
    p = Semantic_KITTI_Utils(data_path, 'inview')
    for index in range(1, 2000):
        points, labels = p.get_pts_l('07',index, True)
        #labels = do_map(labels)
        np.set_printoptions(threshold=np.inf)
        print(labels)
        #points = p.get(index,False)
        print(points.shape,labels.shape)
        pts3d = points[:,:-1]
        color = p.mini_color[labels]
        #kitti_colors = np.array(kitti_colors)
        #color = kitti_colors[labels]
        color = np.ndarray.tolist(color)
        pts2d = p.project_3d_to_2d(pts3d)
        image = p.draw_2d_points(pts2d,color)
        cv2.imshow('img',image)
        #print(pts2d.shape)
        #print(do_map(labels))
        print(index)
        cv2.waitKey(-1)
        # if 27 == cv2.waitKey(1):
        #     break


    