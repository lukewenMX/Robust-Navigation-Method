'''
PointNet training scripts modified by YUNXIANG, including original pointnet, pointnet with intensity information (pointnet4d), and pointnet-light
'''
import argparse
import os
from random import random, shuffle
import sys

from numpy.lib.function_base import percentile
from torch.utils import data
sys.path.append("../")
import time
import json
# import h5py
import datetime
import cv2
from matplotlib.colors import to_rgba_array
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, dataset, random_split
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

from model.fhy_pointnet1 import PointNetSeg, PointNetSegLight, feature_transform_reguliarzer
from model.pointnet2 import PointNet2SemSeg
from model.qyz_mlp import Net
from model.utils import load_pointnet

from data_utils.pcd_utils import mkdir, select_avaliable
#from data_utils.SemKITTI_Loader import SemKITTI_Loader
#from data_utils.kitti_utils import getpcd
from data_utils.fhy4_Sem_Loader import Nanyang_Loader, SemKITTI_Loader, pcd_normalize
from data_utils.fhy4_datautils_test import Semantic_KITTI_Utils
import data_utils.my_log as log

import visdom


# ROOT = os.path.dirname(os.path.abspath(__file__)) + "/img_velo_label_327-002"

Relative_PATH = "../../prepare_data/NanyangLink"
ROOT = os.path.abspath(Relative_PATH)

def parse_args(notebook = False):
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--mode', default='train', choices=('train', 'eval'))
    parser.add_argument('--model_name', type=str, default='pointnet', choices=('pointnet', 'pointnet_3d', 'pointnet_light', 'pointnet2'))
    parser.add_argument('--pn2', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')#16
    parser.add_argument('--subset', type=str, default='inview', choices=('inview', 'all'))
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs for training')#80
    # parser.add_argument('--pretrain', type=str, default='./checkpoint/pointnet-0.39590-0040.pth', help='whether use pretrain model')
    # parser.add_argument('--linear_pretrain', type=str,default='/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/experiment/MLP_linear/MLP_linear-0.08614054-0001.pth',help='whether use pretrain model')
    # parser.add_argument('--angular_pretrain', type=str,default='/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/experiment/MLP_angular/MLP_angular-0.02084015-0001.pth',help='whether use pretrain model')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')#0.001
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')#0.001
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--augment', default=False, action='store_true', help="Enable data augmentation")
    if notebook:
        #if using in jupyter notebook, you should change ' ' to '[]'
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    # if args.pn2 == False:
    #     args.model_name = 'pointnet'
    # else:
    #     args.model_name = 'pointnet2'
    return args

def calc_decay(init_lr, epoch):
    return init_lr * 1 / (1 + 0.03 * epoch)#0.03
    #return init_lr

def test_kitti_semseg(model, loader, model_name, num_classes, class_names):

    ious = np.zeros((num_classes,), dtype = np.float32)
    count = np.zeros((num_classes,), dtype = np.uint32)
    precision = np.zeros((num_classes,), dtype = np.float32)
    recall = np.zeros((num_classes,), dtype = np.float32)
    count[0] = 1
    accuracy = []
    
    
    for points, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
    # in tqdm, loader is an iterable variable. loader here includes dataset with label
        batch_size, num_point, _ = points.size() #points
        points = points.float().transpose(2, 1).cuda()
        target = target.long().cuda()

        with torch.no_grad():

            pred, _ = model(points)
            # if model_name == 'pointnet':
            #     pred, _ = model(points)
            # if model_name == 'pointnet2':
            #     pred,_ = model(points)

            pred_choice = pred.argmax(-1)
            target = target.squeeze(-1)

            # for class_id in range(num_classes):
            #     I = torch.sum((pred_choice == class_id) & (target == class_id)).cpu().item()
            #     U = torch.sum((pred_choice == class_id) | (target == class_id)).cpu().item()
            #     iou = 1 if U == 0 else I/U
            #     ious[class_id] += iou
            #     count[class_id] += 1
            for class_id in range(num_classes):
                TP = torch.sum((pred_choice == class_id) & (target == class_id)).cpu().item()
                FP = torch.sum((pred_choice == class_id) & (target != class_id)).cpu().item()
                FN = torch.sum((pred_choice != class_id) & (target == class_id)).cpu().item()
                TN = torch.sum((pred_choice != class_id) & (target != class_id)).cpu().item()

                I = torch.sum((pred_choice == class_id) & (target == class_id)).cpu().item()
                U = torch.sum((pred_choice == class_id) | (target == class_id)).cpu().item()
                
                iou = 1 if U == 0 else I/U
                ious[class_id] += iou
                count[class_id] += 1
                precision[class_id] += TP / (TP+FP) if (TP+FP) != 0 else 1
                recall[class_id] += TP / (TP+FN) if (TP+FN) != 0 else 1
            
            correct = (pred_choice == target).sum().cpu().item()
            accuracy.append(correct/ (batch_size * num_point)) 

    categorical_iou, precision, recall = ious / count, precision / count, recall / count
    
    df = pd.DataFrame(categorical_iou, columns=['mIOU'], index=class_names)
    df = df.sort_values(by='mIOU', ascending=False)
    log.info('categorical mIOU')
    log.msg(df)

    acc = np.mean(accuracy)
    # miou = np.mean(categorical_iou[1:]) # why start from 1
    miou = np.mean(categorical_iou[:-1])
    return acc, miou, categorical_iou, precision, recall

def train(args):
    # experiment_dir = mkdir('experiment/')
    checkpoints_dir = mkdir('../checkpoint/offline_train/%s'%(args.model_name))

    kitti_utils = Semantic_KITTI_Utils(ROOT, subset = args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes

    if args.subset == 'inview':
        train_npts = 2000#2000
        test_npts = 2500#2500
    
    if args.subset == 'all':
        train_npts = 10000#50000
        test_npts = 12500#100000

    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform = True)
        channels_num = 4
    if args.model_name == "pointnet_3d":
        model = PointNetSeg(num_classes, input_dims=3, feature_transform=True)
        channels_num = 3
    if args.model_name == "pointnet_light":
        model = PointNetSegLight(num_classes, input_dims=3, feature_transform=True)
        channels_num = 3
    if args.model_name == 'pointnet2':
        #model = PointNet2SemSeg(num_classes, feature_dims = 1)
        model = PointNet2SemSeg(num_classes)

    log.info(subset=args.subset, train_npts=train_npts, test_npts=test_npts)
    validation_split = 0.2
    
    # dataset = SemKITTI_Loader(ROOT, train_npts, train = True, subset = args.subset, channels = channels_num)
    dataset = Nanyang_Loader(ROOT, train_npts, train=True, subset=args.subset, channels=channels_num)
    train_size = int((1-validation_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.workers, pin_memory=True)
    # test_dataset = Nanyang_Loader(ROOT, test_npts, train=False, subset=args.subset, channels=channels_num)
    # test_dataset = SemKITTI_Loader(ROOT, test_npts, train = False, subset = args.subset, channels = channels_num)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=False, 
                            num_workers=args.workers, pin_memory=True)


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    #use more than 1 gpu
    model.cuda()
    log.info('Using gpu:',args.gpu)
    
    if args.pretrain is not None:
        log.info('Use pretrain model...')
        model.load_state_dict(torch.load(args.pretrain))
        init_epoch = int(args.pretrain[:-4].split('-')[-1])
        #init_epoch = 0
        log.info('Restart training', epoch=init_epoch)
    else:
        log.msg('Training from scratch')
        init_epoch = 0

    best_acc = 0
    best_miou = 0

    for epoch in range(init_epoch,args.epoch):
        model.train()
        lr = calc_decay(args.learning_rate, epoch) # 为什么不直接用lr_scheduler
        log.info(model=args.model_name, gpu=args.gpu, epoch=epoch, lr=lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            points = points.float().transpose(2, 1).cuda()
            target = target.long().cuda()

            if args.model_name in ['pointnet',"pointnet_3d","pointnet_light"]:
                logits, trans_feat = model(points)
            if args.model_name == 'pointnet2':
                logits, _ = model(points)

            #logits = logits.contiguous().view(-1, num_classes)
            #target = target.view(-1, 1)[:, 0]
            #loss = F.nll_loss(logits, target)

            logits = logits.transpose(2, 1)
            loss = nn.CrossEntropyLoss()(logits, target) #这部分只有pointnet的训练，mlp的训练呢

            if args.model_name in ['pointnet',"pointnet_3d","pointnet_light"]:
                loss += feature_transform_reguliarzer(trans_feat) * 0.001

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.cuda.empty_cache()
        '''train acc'''
        # every epoch do one test
        # acc, miou, cate_iou, percision, recall = test_kitti_semseg(model.eval(), dataloader,
        #                             args.model_name,num_classes,class_names)

        acc, miou, cate_iou, precision, recall = test_kitti_semseg(model.eval(), testdataloader, 
                                                                    args.model_name, num_classes, class_names)

        log.info("mIOU / others = ", miou)
        # miou_list.append(np.asscalar(miou))
        # acc_list.append(np.asscalar(acc))
        save_model = False
        if acc > best_acc:
            best_acc = acc
        
        if miou > best_miou:
            best_miou = miou
            save_model = True

        if save_model:
            fn_pth = '%s-%.5f-%04d.pth' % (args.model_name, best_miou, epoch)
            log.info('Save model...',fn = fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
        else:
            log.info('No need to save model')
        
        # 09.06 added by dyx
        # viz.line([loss.item()],[epoch], win="Train_Loss", update="append")
        # viz.line([acc],[epoch], win="Accuracy", update="append")
        # viz.line([cate_iou[0]],[epoch],win="mIOU@Road",update="append")
        # viz.line([cate_iou[1]],[epoch],win="mIOU@Building",update="append")
        # viz.line([cate_iou[2]],[epoch],win="mIOU@Terrain",update="append")
        # viz.line([cate_iou[3]],[epoch],win="mIOU@Vegetation",update="append")
        # viz.line([cate_iou[4]],[epoch],win="mIOU@Others",update="append")



# def show(args):
#     kitti_utils = Semantic_KITTI_Utils(ROOT, subset = args.subset)
#     pth_path = ROOT + "/experiment/pointnet"
#     pths = os.listdir(pth_path)
#     pths.sort()
#     pth_new = os.path.join(pth_path, pths[-1])
#     print(pth_new)
#     model = load_pointnet(args.model_name, kitti_utils.num_classes, pth_new)
#     part = '03'
#     index = 607
#     points, labels = kitti_utils.get_pts_l(part, index, True)
#     pts3d = points[:,:-1]
#     pcd = pcd_normalize(points)
#     points_tensor = torch.from_numpy(pcd).unsqueeze(0).transpose(2, 1).float().cuda()
#     with torch.no_grad():
#         logits,_ = model(points_tensor)
#         pred = logits[0].argmax(-1).cpu().numpy()
#     pts2d = kitti_utils.project_3d_to_2d(pts3d)
#     pred_color = np.ndarray.tolist(kitti_utils.mini_color_BGR[pred])
#     orig_color = np.ndarray.tolist(kitti_utils.mini_color_BGR[predlabels])
#     img1 = kitti_utils.draw_2d_points(pts2d, orig_color)
#     img2 = kitti_utils.draw_2d_points(pts2d, pred_color)
#     img = np.hstack((img1, img2))
#     cv2.imshow('img',img)
#     cv2.waitKey(0)

def evaluate(args):
    kitti_utils = Semantic_KITTI_Utils(ROOT, subset = args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes
    
    if args.subset == 'inview':
        test_npts = 2000
    if args.subset == 'all':
        test_npts = 100000

    log.info(args.model_name)
    
    if args.model_name == "pointnet":
        # model_ckpt_path = "../checkpoint/offline_train/pointnet_4d/pointnet-0.68132-0084.pth"
        model_ckpt_path = "../checkpoint/offline_train/pointnet/pointnet-0.80816-0174.pth"
        channels = 4
    if args.model_name == "pointnet_3d":
        model_ckpt_path = "../checkpoint/offline_train/pointnet_3d/pointnet_3d-0.66064-0171.pth"
        channels = 3
    if args.model_name == "pointnet_light":
        model_ckpt_path = "../checkpoint/offline_train/pointnet_light/pointnet_light-0.66373-0146.pth"
        channels = 3
            

    test_dataset = Nanyang_Loader(ROOT, test_npts, train=False, subset=args.subset, channels=channels)
    # test_dataset = SemKITTI_Loader(ROOT, test_npts, train=False, subset=args.subset, channels = channels)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.workers)

    # model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)
    model = load_pointnet(args.model_name, kitti_utils.num_classes, model_ckpt_path)
    acc, miou, cate_iou, precision, recall = test_kitti_semseg(model.eval(), testdataloader,args.model_name,num_classes,class_names)

    log.info('Curr', accuracy=acc, mIOU=miou, iou=cate_iou, precision=precision, recall=recall)

def produce_data(model, loader, model_name, num_classes, class_names):
    data = torch.tensor([])
    data = data.cuda()
    for points, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
    # in tqdm, loader is an iterable variable. loader here includes dataset with label
        batch_size, num_point, _ = points.size() #points
        points = points.float().transpose(2, 1).cuda()
        target = target.long().cuda()

        with torch.no_grad():
            if model_name == 'pointnet':
                pred, _ = model(points)
            if model_name == 'pointnet2':
                pred,_ = model(points)

            pred_choice = pred.argmax(-1)
            target = target.squeeze(-1)
            pred_choice = pred_choice.unsqueeze(1)
            new_points = torch.cat((points,pred_choice),1)
            #print(pred_choice.shape)
            #print(new_points.shape)
            data = torch.cat((data,new_points),0)


    return pred_choice,target,data

def produce_vel(model, loader):
    for points in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
        points = points.float().transpose(2, 1).cuda()
        with torch.no_grad():
            prediction = model(points)
            print(prediction)



def diffusion(args):
    kitti_utils = Semantic_KITTI_Utils(ROOT, subset=args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes

    if args.subset == 'inview':
        test_npts = 2000

    if args.subset == 'all':
        test_npts = 100000

    test_dataset = SemKITTI_Loader(ROOT, test_npts, train=False, subset=args.subset)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False,
                                num_workers=args.workers)

    seg_model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)
    label,target,data= produce_data(seg_model.eval(), testdataloader,args.model_name,num_classes,class_names)
    target = target.squeeze(-1)
    torch.set_printoptions(profile='full')
    print(data.shape)

    #new_testdataloader = DataLoader(data,batch_size=1,shuffle=False,num_workers=args.workers)

    LinearVel_model = Net(5, 64)
    AngularVel_model = Net(5, 64)
    LinearVel_model = torch.nn.DataParallel(LinearVel_model)
    AngularVel_model = torch.nn.DataParallel(AngularVel_model)
    torch.backends.cudnn.benchmark = True

    # use more than 1 gpu
    log.info('Using gpu:', '0')
    LinearVel_model.load_state_dict(torch.load(args.linear_pretrain))
    AngularVel_model.load_state_dict(torch.load(args.angular_pretrain))

    LinearVel_model.cuda()
    LinearVel_model.eval()
    AngularVel_model.cuda()
    AngularVel_model.eval()

    with torch.no_grad():
        for points in data:
            points = points.unsqueeze(0)
            #print(points.shape)
            prediction_linear = LinearVel_model(points)
            prediction_angular = AngularVel_model(points)
            prediction = torch.cat((prediction_linear,prediction_angular))
            prediction.resize(2,1)
            print(prediction)








if __name__ == '__main__':
    args = parse_args()
    # viz = visdom.Visdom(server="http://localhost", port=8097, env=args.model_name)
    # viz.line( Y=[0.], X=[0], win="Train_Loss", opts=dict(title='Train_Loss'))
    # viz.line( Y=[0.], X=[0], win='Accuracy', opts=dict(title='Accuracy'))
    # viz.line( Y=[0.], X=[0], win='mIOU@Road', opts=dict(title='mIOU@Road'))
    # viz.line( Y=[0.], X=[0], win='mIOU@Building', opts=dict(title='mIOU@Building'))
    # viz.line( Y=[0.], X=[0], win='mIOU@Terrain', opts=dict(title='mIOU@Terrain'))
    # viz.line( Y=[0.], X=[0], win='mIOU@Vegetation', opts=dict(title='mIOU@Vegetation'))
    # viz.line( Y=[0.], X=[0], win='mIOU@Others', opts=dict(title='mIOU@Others'))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "train":
        train(args)
    if args.mode == "eval":
        evaluate(args)
    if args.mode == "diff":
        diffusion(args)


