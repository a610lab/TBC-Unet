import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from Unet_Demo.UNet2D_BraTs_Three_Branch.dataset import Dataset
from Unet_Demo.UNet2D_BraTs_Three_Branch.metrics import dice_coef, batch_iou, mean_iou, iou_score
import Unet_Demo.UNet2D_BraTs_Three_Branch.losses
from Unet_Demo.UNet2D_BraTs_Three_Branch.utils import str2bool, count_params
import pandas as pd
import Unet_Demo.UNet2D_BraTs_Three_Branch.unet
arch_names = list(Unet_Demo.UNet2D_BraTs_Three_Branch.unet.__dict__.keys())
loss_names = list(Unet_Demo.UNet2D_BraTs_Three_Branch.losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='model_file name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='demo',
                        choices=arch_names,
                        help='model_file architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="Yongpy",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    args = parser.parse_args()
    return args
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    model.train()
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])
    return log
def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])
    return log
def main():
    args = parse_args()
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)
    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = Yongpy.Unet_Demo.UNet2D_BraTs_Three_Branch.losses.__dict__[args.loss]().cuda()
    cudnn.benchmark = True
    img_paths = glob(r'*')
    mask_paths = glob(r'*')
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("%s"%str(len(train_img_paths)))
    print("%s"%str(len(val_img_paths)))
    print(" %s" %args.arch)
    model = Unet_Demo.UNet2D_BraTs_Three_Branch.unet.Unet(n_channels=4, n_classes=3, bilinear=True)
    model = model.cuda()
    print(count_params(model))
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])
    best_iou = 0
    trigger = 0
    for epoch in range(args.epochs):
        print(' [%d/%d]' %(epoch, args.epochs))
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        val_log = validate(args, val_loader, model, criterion)
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
            %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])
        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)
        trigger += 1
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model_file.pth' %args.name)
            best_iou = val_log['iou']
            trigger = 0
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
