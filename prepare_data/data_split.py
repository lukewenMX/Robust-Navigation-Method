'''
script to manually divide the whole dataset into train & test subset
'''
import os
import numpy as np
from matplotlib import image, pyplot as plt
from PIL import Image
import cv2
from sklearn import datasets

DATASET_ROOT = "./NanyangLink"


class ImageViewer(object):
    def __init__(self, img_dir, viz_dir):
        
        self.train_file = open(os.path.join(DATASET_ROOT, 'train.txt'),'w')
        self.test_file = open(os.path.join(DATASET_ROOT, 'test.txt'),'w')
        self.img_dir = img_dir
        self.viz_dir = viz_dir
        self.img_list = []
        for split in os.listdir(img_dir):
            self.img_list.extend([os.path.join(split, name) for name in os.listdir(os.path.join(img_dir, split))]) 
        self.img_list.sort()

    def __del__(self):
        self.train_file.close()
        self.test_file.close()

    def path2array(self, img_path):
        im = cv2.imread(os.path.join(self.img_dir, img_path))
        viz = cv2.imread(os.path.join(self.viz_dir, img_path))
        
        return np.concatenate((im,viz),axis=0)

    def show(self):
        if self.img_list:
            cv2.namedWindow("viz")
            for img_path in self.img_list:
                viz = self.path2array(img_path)
                folder, file = os.path.split(img_path)
                name, ext = os.path.splitext(file)
                cv2.imshow("viz", viz)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('a'):
                    self.train_file.write("%s\n" % os.path.join(folder,name))
                    print("written into train file: ", os.path.join(folder,name))
                elif key == ord('d'):
                    self.test_file.write("%s\n" % os.path.join(folder,name))
                    print("written into test file: ", os.path.join(folder,name))
                elif key == 27:
                    break
            cv2.destroyAllWindows()
            


if __name__ == '__main__':
    img_dir = os.path.join(DATASET_ROOT, "img")
    viz_dir = os.path.join(DATASET_ROOT, 'miniclass_viz')
    viewer = ImageViewer(img_dir, viz_dir)
    for split in os.listdir(img_dir):
        img_list = [os.path.join(img_dir, split, name) for name in os.listdir(os.path.join(img_dir, split))]
        sorted(img_list)

    viewer.show()
