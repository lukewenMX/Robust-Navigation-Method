'''
scripts for extracting data from rosbag  
'''
import os
import numpy as np
import rospy
import cv2
import message_filters  
from sensor_msgs.msg import Image, CompressedImage, PointField, CameraInfo
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2
from sensor_msgs.point_cloud2 import read_points
from t import DeepLabNode

from image_geometry import PinholeCameraModel
import yaml
from map import do_map, mini_name, map_kitti2mini, mini_color, kitti_colors
from pt2range import LaserScan

DATA_DIR = "./NanyangLinkNight2"
DATA_PATCH = 3

def create_pc_fields():
    fields = []
    fields.append( PointField( 'x', 0, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'y', 4, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'z', 8, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'intensity', 12, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'label', 16, PointField.UINT32, 1 ) )
    fields.append( PointField( 'rings', 20, PointField.UINT32, 1 ) )
    
    return fields

class CloudGenerator():
    def __init__(self, img_res = (672, 376)):
        if (DATA_PATCH >= 0 and DATA_PATCH < 10):
            self.data_patch = '0' + str(DATA_PATCH)
        else:
            self.data_patch = str(DATA_PATCH)

        self.colormap = np.array(kitti_colors, dtype='uint8')
        self.mini_colormap = np.array(mini_color, dtype='uint8')

        '''
        data types in the dataset
        pts_label: pointcloud with label in the format of [x,y,z,intensity,label,rings]
        image: RGB image
        velocity: robot velocity [v,w]
        class_viz: image colorized with full colormap for visualization
        miniclass_viz: image colorized with mini-colormap for visualization
        '''

        self.label_path = os.path.join(DATA_DIR, 'pts_label', str(self.data_patch))
        self.img_path = os.path.join(DATA_DIR, 'img', str(self.data_patch))
        self.vel_path = os.path.join(DATA_DIR, 'velocity', str(self.data_patch))
        self.raw_painted_path = os.path.join(DATA_DIR, 'class_viz', str(self.data_patch))
        self.mini_painted_path = os.path.join(DATA_DIR, 'miniclass_viz', str(self.data_patch))
        
        self.pt2range_cvt = LaserScan(project=True, H=16, W=200, fov_up=15, fov_down=-15)
        
        os.makedirs(self.label_path, exist_ok=True)
        os.makedirs(self.img_path, exist_ok=True)
        os.makedirs(self.vel_path, exist_ok=True)
        os.makedirs(self.raw_painted_path, exist_ok=True)
        os.makedirs(self.mini_painted_path, exist_ok=True)

        self.camera_model = PinholeCameraModel()
        self.img_width, self.img_height = img_res
        self.isTransformSet = False
        self.isCamInfoSet = False
        self.seg_nn =DeepLabNode()
        self.num = 0
        self.load_extrinsics_ntu(fn='calib_data/velo-to-cam.txt', param= 'cam_to_velo')
        self.load_cam_info('calib_data/cam_param.yaml')
        self.build_cams_model()


        self.node = rospy.init_node('lidar_to_rgb', anonymous=True)
        # self.bridge = CvBridge()
        self.pub_cloud = rospy.Publisher("/label_cloud", PointCloud2, queue_size = 1 )
        self.pub_image = rospy.Publisher("/projected_image",Image, queue_size = 1 )
        self.pub_image_raw = rospy.Publisher("range_image", Image, queue_size = 1)
        self.pub_velocity = rospy.Publisher("/odom", Odometry, queue_size=1)
        # self.sub_cam1_info = rospy.Subscriber(rospy.resolve_name( '/camera/left/camera_info'), CameraInfo, callback=self.cam1_info_callack, queue_size=1)
        self.sub_cam1_image = message_filters.Subscriber('/camera/left/image_raw/compressed', CompressedImage, queue_size=1)
        self.sub_cloud = message_filters.Subscriber('/velodyne_points', PointCloud2, queue_size=1)
        self.sub_velocity = message_filters.Subscriber('/odom_topic', Odometry, queue_size=1) # /odom_topic is the republished encoder measurment from republisher.py
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_cam1_image, self.sub_cloud, self.sub_velocity], 1, 0.08)
        self.ts.registerCallback(self.fusion_callback)
        


        rospy.spin()
        
    def read_file(self, file_name, name, shape=(3,3), suffix='', sep=' '):
        content_vec = []

        for line in open(file_name, "r"):
            (key, val) = line.split(':', 1)
            if key == (name + suffix):
                content = np.fromstring(val, sep=sep).reshape(shape)
                content_vec.append(content)
        return np.array(content_vec)
    
    def load_cam_info(self, file_name):
        self.cam_info = CameraInfo()
        with open(file_name,'r') as cam_calib_file :
            cam_calib = yaml.load(cam_calib_file)
            self.cam_info.height = cam_calib['image_height']
            self.cam_info.width = cam_calib['image_width']
            self.img_width, self.img_height = self.cam_info.width, self.cam_info.height
            self.cam_info.K = cam_calib['camera_matrix']['data']
            self.cam_info.D = cam_calib['distortion_coefficients']['data']
            # self.cam_info.P = np.concatenate((self.cam_info.K, [0,0,1]), axis=1)
            # cam_info.P = cam_calib['projection_matrix']['data']
            self.cam_info.distortion_model = cam_calib['distortion_model']
            self.K = np.array(self.cam_info.K).reshape(3,3)
            self.D = np.array(self.cam_info.D)
                
    def load_extrinsics_ntu(self, fn, param):
        def get_inverse(Mat):
            try:
                return np.linalg.inv(Mat)
            except np.linalg.LinAlgError:
                print("Failed to Inverse")
            #     # Not invertible. Skip this one.
                pass
        self.lidar_to_cams_RT_vect = get_inverse(self.read_file(fn, param, shape=(4,4), sep=',')) 
        self.isTransformSet = True
        
    def build_cams_model(self):
        self.cam_model=PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)
            

        self.isCamInfoSet = True
                   
    def load_extrinsics(self, fn_c2v_vec):
        
        self.cams_to_lidar_R_vec = [self.read_file(fn_c2v_vec, 'R')]
        self.cams_to_lidar_T_vec = [self.read_file(fn_c2v_vec, 'T', shape=(3,1))]
        to_homo = np.array([0, 0, 0, 1]).reshape(1, 4)
        self.cams_to_lidar_RT_vect = np.array([np.concatenate((np.concatenate((np.squeeze(R, axis=0), np.squeeze(T, axis=0)), axis=1), to_homo),axis=0) for R, T in zip(self.cams_to_lidar_R_vec, self.cams_to_lidar_T_vec)])
        # [print(np.squeeze(R, axis=0).shape, np.squeeze(T, axis=0).shape) for R, T in zip(self.cams_to_lidar_R_vec, self.cams_to_lidar_T_vec)]
        
        def get_inverse(Mat):
            try:
                return np.linalg.inv(Mat)
            except np.linalg.LinAlgError:
            #     # Not invertible. Skip this one.
                pass
            
        self.lidar_to_cams_RT_vect = [get_inverse(cam_tolidar_RT) for cam_tolidar_RT in self.cams_to_lidar_RT_vect]
        self.isTransformSet = True
        # print(self.lidar_to_cams_RT_vect[0] )
    def cv2_to_imgmsg(self, img, depth=False):
        # easy implementation converting 3-channels image to sensor_msgs::Image
        img_msg = Image()
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        
        if depth == True:
            img_msg.encoding = "32FC1"
        else:
            img_msg.encoding = "bgr8"
        
        img_msg.data = img.tostring()
        img_msg.step = int(len(img_msg.data) / img_msg.height)
        
        return img_msg

        
    def fusion_callback(self, msg_img, msg_cloud, msg_velocity):
        # print("fusion callback!")
        
        ## Get Image
        img_arr = np.fromstring(msg_img.data, np.uint8)
        cv_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))
        cv_image = cv2.undistort(cv_image, self.K, self.D)
        
        ## Get Velocity
        try:
            velocity = []
            velocity.append([msg_velocity.twist.twist.linear.x, msg_velocity.twist.twist.linear.y, msg_velocity.twist.twist.linear.z, msg_velocity.twist.twist.angular.x, msg_velocity.twist.twist.angular.y, msg_velocity.twist.twist.angular.z])
            
        except:
            print('velocity error')
        
        ## Image Segmentation
        label_mat = self.seg_nn.run_prediction(cv_image)
        ## Paint Image
        color_img = self.colormap[label_mat,:]
        mini_label_mat = do_map(label_mat)
        mini_color_img = self.mini_colormap[mini_label_mat,:]
        
        ## Point Projection
        if self.isTransformSet and self.isCamInfoSet:
            cv_temp = cv_image.copy()
            new_pts = []

            for point in (read_points(msg_cloud, skip_nans=True)):
                    
                    # pts_r.append(point)
                    cam_index = 0
                    pts_xyz_homo = [point[0],point[1],point[2], 1.0]
                    intensity = point[3]
                    intensityInt = int(intensity*255) # why * 255

                    pts_xyz_cam = self.lidar_to_cams_RT_vect[cam_index].dot(pts_xyz_homo)
                    if pts_xyz_cam[2]<0 or pts_xyz_cam[2]>25:#0<depth<25
                        continue
                    pts_uvz_pix = self.K.dot((pts_xyz_cam[0], pts_xyz_cam[1], pts_xyz_cam[2]))
                    # pts_uv_pix = self.camera_model.project3dToPixel((pts_xyz_cam[0], pts_xyz_cam[1], pts_xyz_cam[2]))
                            # xy_n = xy_i / xy_i[2]
                    pts_uvz_pix = pts_uvz_pix/pts_uvz_pix[2]
                    pts_uv_pix = (pts_uvz_pix[0], pts_uvz_pix[1])
                    #projection
                    if 0<=pts_uv_pix[0]<=self.img_width and 0<=pts_uv_pix[1]<=self.img_height:
                        cv2.circle(cv_temp, (int(pts_uv_pix[0]), int(pts_uv_pix[1])), 5, (intensityInt, intensityInt, intensityInt), thickness=-1 )
                        b,g,r = cv_image[int(pts_uv_pix[1]),int(pts_uv_pix[0])]
                        label = mini_label_mat[int(pts_uv_pix[1]),int(pts_uv_pix[0])]
                        # pts_r.append([point[0],point[1],point[2],point[3]])
                        new_pts.append([point[0],point[1],point[2],point[3],label,point[4]])
            
            if len(new_pts) == 0:
                print("no data!")
                return
            ## Range Image Conversion
            scan = np.array(new_pts)
            self.pt2range_cvt.set_points(points=scan[:,0:3],remissions=scan[:,3],label=scan[:,4],rings=scan[:,5])
            range_img = self.pt2range_cvt.proj_range
            label_img = self.pt2range_cvt.proj_label
            color_range_img = self.pt2range_cvt.proj_color
            intensity_img = self.pt2range_cvt.proj_remission
            ## Data Saving
            cv2.imwrite(os.path.join(self.img_path, '%06d.png' % (self.num)), cv_image)
            cv2.imwrite(os.path.join(self.raw_painted_path, '%06d.png' % (self.num)), color_img)
            cv2.imwrite(os.path.join(self.mini_painted_path, '%06d.png' % (self.num)), mini_color_img)
            np.savez(os.path.join(self.label_path, 'pts_l%06d' % (self.num)), pts_l=new_pts)
            np.savez(os.path.join(self.vel_path, 'velocity%06d' % (self.num)), vel=velocity)

            self.num += 1
            
            # self.pub_image.publish(self.cv2_to_imgmsg(cv_temp))
            self.pub_image.publish(self.cv2_to_imgmsg(color_range_img, depth=False))
            self.pub_image_raw.publish(self.cv2_to_imgmsg(range_img, depth=True))
            ros_cloud = sensor_msgs.point_cloud2.create_cloud(msg_cloud.header, create_pc_fields(), new_pts)
            self.pub_cloud.publish(ros_cloud)
            self.pub_velocity.publish(msg_velocity)

        else:
            print( 'Waiting for intrisincs and extrinsics')

if __name__ == '__main__':
    try:
        cloud_generator = CloudGenerator()
    except rospy.ROSInterruptException:
        pass