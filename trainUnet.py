import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from unet import UNet
from dataset import prepare_data, Dataset
from misc_utils import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="UNet")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--valBatchSize", type=int, default=8, help="Validation batch size")
parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs_unet_l1_1024", help='path of log files')
opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, pin_memory=True, batch_size=opt.batchSize, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=8, batch_size=opt.valBatchSize)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of validation samples: %d\n" % int(len(dataset_val)))
    # Build model
    net = UNet(n_class=1)
    # net = ResUnet(channel=3)
    net.apply(weights_init_kaiming)
    criterion = nn.L1Loss(reduction='sum')
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    psnr_list = [0]
    for epoch in range(opt.epochs):
        time_start=time.time()
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            imgn_train, imgc_train = data
            imgn_train, imgc_train = Variable(imgn_train.cuda()), Variable(imgc_train.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, imgc_train)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, imgc_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 100 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()

        # validate
        with torch.no_grad():
            psnr_val = 0
            for i, data in enumerate(loader_val, 0):
                imgn_val, imgc_val = data
                imgn_val, imgc_val = Variable(imgn_val.cuda()), Variable(imgc_val.cuda())
                out_val = torch.clamp(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, imgc_val, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
            psnr_list.append(psnr_val)
            # log the images
            Imgc = utils.make_grid(imgc_train.data, nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
            writer.add_image('clean image', Imgc, epoch)
            writer.add_image('noisy image', Imgn, epoch)
            writer.add_image('reconstructed image', Irecon, epoch)
            time_end = time.time()
            print("epoch {0} running time: {1}s.".format(epoch+1,time_end-time_start))

if __name__ == "__main__":
    if opt.preprocess:
        prepare_data('/home/data/mcdeep', region=[0,0,1024,768], size=(1024,768), blocks=1, interpolation=cv2.INTER_NEAREST)
    time_start=time.time()
    main()
    time_end = time.time()
    print("total running time: {}s".format(time_end-time_start))

