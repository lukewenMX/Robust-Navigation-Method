import open3d
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from map import mini_color, do_map, kitti_colors
from t import  DeepLabNode
# DATA_ROOT = '/home/yxdai/repository/nav_ws/src/prepare_data/NanyangLinkNight2'
DATA_ROOT = '/media/yxdai/HP P500/dataset/NanyangLink'
PCD_DIR = 'pts_label'
IMG_DIR = 'img'
MINIGT_DIR = 'miniclass_viz'

data_item = '11/000045'
calibration_file = './calib_data/velo-to-cam.txt'

img_width, img_height = 672, 376
intrinsic_matrix =     [
      3.507833068601e+02,
      0.000000000000e+00,
      3.454384750906e+02,
      0.000000000000e+00,
      3.508783430712e+02,
      1.668714311752e+02,
      0.000000000000e+00,
      0.000000000000e+00,
      1.000000000000e+00,
    ]

seg_nn = DeepLabNode()

def do_projection(pts_l: np.ndarray, img_to_draw: np.ndarray):
    pts_xyz_homo = np.concatenate((pts_l[:,0:3], np.ones_like(pts_l[:,0]).reshape(-1,1)), axis=1)
    pts_cam_xyz = extrinsic_matrix.dot(pts_xyz_homo.T).T
    pts_cam_xyz = pts_cam_xyz[:,0:3]
    pts_cam_xyz = (np.array(intrinsic_matrix).reshape(-3,3)).dot(pts_cam_xyz.T).T
    pts_uv = pts_cam_xyz[:,0:2] / pts_cam_xyz[:,2].reshape(-1,1)
    pts_uv = pts_uv.astype(np.int)
    filter_idx = np.where((pts_uv[:,0] >= 0) & (pts_uv[:,0] < img_width) & (pts_uv[:,1] >= 0) & (pts_uv[:,1] < img_height))[0]

    for idx in filter_idx:
        pt_label = int(pts_l[idx, 4])
        cv2.circle(img_to_draw, (int(pts_uv[idx, 0]), int(pts_uv[idx, 1])), 2, mini_color[pt_label], thickness=-1)
    return img_to_draw

def do_img_prediction(model, img: np.ndarray):
    label_mat = model.run_prediction(img)
    # mini_label_mat = do_map(label_mat)
    mini_color_img = np.array(kitti_colors, dtype='uint8')[label_mat,:]
    return mini_color_img

def visualize(subset: str, name: str):
    
    img_path = os.path.join(DATA_ROOT, IMG_DIR, subset, name + '.png')
    minigt_path = os.path.join(DATA_ROOT, MINIGT_DIR, subset, name + '.png')
    raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv_gt = raw_img.copy()
    cv_pred = raw_img.copy()
    pcd_path = os.path.join(DATA_ROOT, PCD_DIR, subset, 'pts_l' + name + '.npz')
    pred_pcd_path = './pred_vis_day.npz'
    pts_l = np.load(pcd_path)['pts_l'] # [x,y,z,intensity, mini_class, rings]
    pts_pred = np.load(pred_pcd_path)['pts_l'] # [x,y,z,intensity, mini_class]

    cv_gt = do_projection(pts_l, cv_gt)
    cv_pred = do_projection(pts_pred, cv_pred)

    cv_seg = do_img_prediction(seg_nn, raw_img)

    
    cv2.imshow("gt", cv_gt)
    cv2.imshow("pred", cv_pred)
    cv2.imshow("seg_new_colormap", cv_seg)
    cv2.waitKey(0)


    

    



def load_extrinsics_ntu(fn='calib_data/velo-to-cam.txt', param= 'cam_to_velo'):
        def get_inverse(Mat):
            try:
                return np.linalg.inv(Mat)
            except np.linalg.LinAlgError:
                print("Failed to Inverse")
                pass

        def read_file(file_name, name, shape=(3,3), suffix='', sep=' '):
            content_vec = []

            for line in open(file_name, "r"):
                (key, val) = line.split(':', 1)
                if key == (name + suffix):
                    content = np.fromstring(val, sep=sep).reshape(shape)
                    content_vec.append(content)
            return np.array(content_vec)

        return get_inverse(read_file(fn, param, shape=(4,4), sep=','))




if __name__ == '__main__':
    subset_name, file_name = data_item.split('/')
    extrinsic_matrix = load_extrinsics_ntu(calibration_file)[0]
    visualize(subset_name, file_name)



