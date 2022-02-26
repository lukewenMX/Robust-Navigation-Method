import os
from tqdm import tqdm
import network
import utils
import sys
import time
import random
import argparse
import numpy as np
import cv2
from map import do_map, mini_name, map_kitti2mini
#from skimage.transform import resize
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import torchvision.transforms as transforms

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0])
train_id_to_color = np.array(train_id_to_color)
    
def decode_target(target):
    target[target == 255] = 19
    return train_id_to_color[target]
    

def get_transform():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    return transform


class DeepLabNode():
    def __init__(self, Image_Res=(672, 376), Input_Res=(672,376)):
        self.img_width, self.img_height = Image_Res
        self.INPUT_WIDTH, self.INPUT_HEIGHT = Input_Res
        self.load_env_variables()
        self.load_model()
        self.load_weights()      
        self.prepare_inference() 
        print('Ready.')

    def load_env_variables(self):
        self.opts = get_argparser().parse_args()
        self.opts.num_classes = 19
        self.opts.model = 'deeplabv3plus_mobilenet'
        self.opts.output_stride = 16
        self.opts.ckpt = "ckpt/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opts.gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % self.device)

        
    def load_model(self):
        model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet}
        self.Model = model_map[self.opts.model](num_classes=self.opts.num_classes, output_stride=self.opts.output_stride)

    def load_weights(self):
        if self.opts.ckpt is not None and os.path.isfile(self.opts.ckpt):
            checkpoint = torch.load(self.opts.ckpt, map_location=torch.device('cpu'))
            self.Model.load_state_dict(checkpoint["model_state"])
            self.Model = nn.DataParallel(self.Model)
            self.Model.to(self.device)    
            print("Model restored from %s" % self.opts.ckpt)
            del checkpoint  # free memory
            print("Finish Loaded Pretrained Model")
        else:
            print("Model not loaded")
    
    def prepare_inference(self):       
        self.val_transform = get_transform()
        self.Model.eval() # Set in evaluation mode
        
    def image_processing(self, image):
        img = image[...,::-1]
        img = cv2.resize(img, (self.INPUT_WIDTH, self.INPUT_HEIGHT))    
        img_tensor = self.val_transform(img).unsqueeze(0)    
        return img_tensor
    
    def run_prediction(self, image):
        class_probs = self.predict(image)      
        pred_confidences, pred_labels = torch.topk(input = class_probs, k = 1, dim = 1, largest = True, sorted = True)
        pred_labels = pred_labels.squeeze(0).cpu().numpy()
        # pred_confidences = pred_confidences.squeeze(0).cpu().numpy()
        # print(pred_labels.astype(np.int).shape)
        # semantic_color= decode_target(pred_labels.astype(np.int))[...,::-1]
        # print(semantic_color.shape)
        # print(semantic_color)
        # print(semantic_colors.shape)
        # semantic_color = semantic_colors[0]
        # confidence = pred_confidences[0]
        # print(confidence.shape)
        # cv2.imshow('Semantic segmantation', semantic_color[0].astype(np.uint8))
        # cv2.waitKey(0)    
        # a = pred_labels.astype(np.int)[0]
        # print(a[70:80, 70:80])
        # print(pred_labels.astype(np.int)[0])
        return pred_labels.astype(np.int)[0]
    
    def predict(self, image):
        with torch.no_grad():
            input_tensor = self.image_processing(image)          
            input_tensor = input_tensor.to(self.device, dtype=torch.float32)
            outputs = self.Model(input_tensor)  
            outputs = torch.nn.functional.softmax(outputs, 1)     
            return outputs 

def main(args):
    seg_nn = DeepLabNode()
    img = cv2.imread("000000.png")
    label = seg_nn.run_prediction(img)
    #label = do_map([19])
    np.set_printoptions(threshold=np.inf)
    print(label.shape)

if __name__ == '__main__':
    main(sys.argv)

