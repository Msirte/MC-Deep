  
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from unet import UNet
from misc_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="UNet_Test")
parser.add_argument("--logdir", type=str, default="logs_unet_l1_1024", help='path of log files')
parser.add_argument("--test_data_noisy", type=str, default='./noisy', help='noisy data')
parser.add_argument("--test_data_clean", type=str, default='./clean', help='clean data')
parser.add_argument("--test_data_output", type=str, default='./output', help='output images')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Create output dirs
    if not os.path.exists(opt.test_data_output):
        os.makedirs(opt.test_data_output)
    # Build model
    print('Loading model ...\n')
    net = UNet(n_class=1)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    imgs_noisy = glob.glob(os.path.join(opt.test_data_noisy, '*.png'))
    imgs_clean = glob.glob(os.path.join(opt.test_data_clean, '*.png'))
    imgs_noisy.sort()
    imgs_clean.sort()
    # process data
    psnr_test = 0
    region=[0,0,-1,-1]
    blocks=(1,1)
    resize_=(1024,768)
    for index in range(len(imgs_noisy)):
        img_noisy_filename = imgs_noisy[index]
        img_clean_filename = imgs_clean[index]
        img_noisy = cv2.imread(img_noisy_filename)
        img_clean = cv2.imread(img_clean_filename)
        img_noisy = img_noisy[region[1]:region[3],region[0]:region[2],0]
        img_clean = img_clean[region[1]:region[3],region[0]:region[2],0]
        img_noisy = cv2.resize(img_noisy,resize_,interpolation=cv2.INTER_CUBIC)
        img_clean = cv2.resize(img_clean,resize_,interpolation=cv2.INTER_CUBIC)
        if len(img_noisy.shape) == 2:
            img_noisy = np.expand_dims(img_noisy,-1)
            img_clean = np.expand_dims(img_clean,-1)
        h, w, c = img_noisy.shape
        out = np.zeros(img_noisy.shape)
        for j in range(blocks[0]):
            for k in range(blocks[1]):
                img_noisy_ = img_noisy[h//blocks[0]*j:h//blocks[0]*(j+1),w//blocks[1]*k:w//blocks[1]*(k+1),:]
                img_noisy_ = normalize(img_noisy_)
                img_noisy_ = np.transpose(img_noisy_, (2, 0, 1))
                img_noisy_ = np.expand_dims(img_noisy_,0)
                img_noisy_ = torch.Tensor(img_noisy_)
                img_noisy_ = Variable(img_noisy_.cuda())
                with torch.no_grad():
                    out_ = torch.clamp(model(img_noisy_),0.,1.)
                # print(out_.shape)
                out_ = tensor2cv2(out_)
                # print("out shape: ",out.shape)
                # print("out_ shape: ",out_.shape)
                out[h//blocks[0]*j:h//blocks[0]*(j+1),w//blocks[1]*k:w//blocks[1]*(k+1),:] = out_.copy()
        psnr = peak_signal_noise_ratio(img_clean,out,data_range=255)
        cv2.imwrite(os.path.join(opt.test_data_output,"{}".format(epoch+1)+os.path.basename(img_noisy_filename)),out)
        psnr_test += psnr
        print("%s PSNR %f" % (img_noisy_filename, psnr))
    psnr_test /= len(imgs_noisy)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
