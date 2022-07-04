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
from skimage.io import imread, imsave
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
from Yongpy.Unet_Demo.UNet2D_BraTs_Three_Branch.dataset import Dataset
import Yongpy.Unet_Demo.UNet2D_BraTs_Three_Branch.unet
from Yongpy.Unet_Demo.UNet2D_BraTs_Three_Branch.metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import Yongpy.Unet_Demo.UNet2D_BraTs_Three_Branch.losses
from Yongpy.Unet_Demo.UNet2D_BraTs_Three_Branch.utils import str2bool, count_params
import joblib
from hausdorff import hausdorff_distance
import imageio
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Yongpy_Unet',
                        help='model_file name')
    parser.add_argument('--mode', default="Calculate",
                        help='Calculate or GetPicture')
    args = parser.parse_args()
    return args
def main():
    val_args = parse_args()
    args = joblib.load('models/%s/args.pkl' %val_args.name)
    print("args:", args)
    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    print("=> creating model_file %s" %args.arch)
    model = Yongpy.Unet_Demo.UNet2D_BraTs_Three_Branch.unet.Unet(n_channels=4, n_classes=3, bilinear=True)
    model = model.cuda()
    img_paths = glob(r'*')
    mask_paths = glob(r'*')
    val_img_paths = img_paths
    val_mask_paths = mask_paths
    model.load_state_dict(torch.load('models/%s/model_file.pth' %args.name))
    model.eval()
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    if val_args.mode == "GetPicture":
        print("val_args.mode == GetPicture")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input = input.cuda()

                    if args.deepsupervision:
                        output = model(input)[-1]
                    else:
                        output = model(input)
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                    for i in range(output.shape[0]):
                        npName = os.path.basename(img_paths[i])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName  + ".png"
                        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i,0,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,1,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,2,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                        imsave('output/%s/'%args.name + rgbName,rgbPic)
            torch.cuda.empty_cache()
        val_gt_path = 'output/%s/'%args.name + "GT/"
        if not os.path.exists(val_gt_path):
            os.mkdir(val_gt_path)
        for idx in tqdm(range(len(val_mask_paths))):
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            rgbName = name + ".png"
            npmask = np.load(mask_path)
            GtColor = np.zeros([npmask.shape[0],npmask.shape[1],3], dtype=np.uint8)
            for idx in range(npmask.shape[0]):
                for idy in range(npmask.shape[1]):
                    if npmask[idx, idy] == 1:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 0
                    elif npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 128
                        GtColor[idx, idy, 2] = 0
                    elif npmask[idx, idy] == 4:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0
            imageio.imwrite(val_gt_path + rgbName, GtColor)
    if val_args.mode == "Calculate":
        print("val_args.mode == Calculate")
        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []
        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        wtPbList = []
        tcPbList = []
        etPbList = []
        maskPath = glob("output/%s/" % args.name + "GT\*.png")
        pbPath = glob("output/%s/" % args.name + "*.png")
        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])
            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    if mask[idx, idy, 1] == 128:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 128:
                        etpbregion[idx, idy] = 1
            dice = dice_coef(wtpbregion,wtmaskregion)
            wt_dices.append(dice)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sensitivity_n)
            dice = dice_coef(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sensitivity_n)
            dice = dice_coef(etpbregion, etmaskregion)
            et_dices.append(dice)
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(etpbregion, etmaskregion)
            et_sensitivities.append(sensitivity_n)
        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")
if __name__ == '__main__':
    main( )
