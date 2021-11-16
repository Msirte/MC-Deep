import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from tqdm import tqdm

def normalize(data):
    return data/255.

def generate_h5file(img_list, region, size, blocks, channels, interpolation, out_filename):
    files = img_list
    files.sort()
    h5f =  h5py.File(out_filename, 'w')
    img_num = 0
    for i in tqdm(range(len(files))):
        img = cv2.imread(files[i])
        if channels == 3:
            if len(img.shape) == 2:
                img = np.expand_dims(img,axis=-1).repeat(3,axis=-1)
            Img = img[region[1]:region[3],region[0]:region[2],:]
            Img = cv2.resize(Img, size, interpolation=interpolation)
            # Img = np.expand_dims(Img.copy(), 0)
            h, w, c = Img.shape
            for j in range(blocks[0]):
                for k in range(blocks[1]):
                    img = Img[h//blocks[0]*j:h//blocks[0]*(j+1),w//blocks[1]*k:w//blocks[1]*(k+1),:]
                    img = np.float32(normalize(img))
                    img = img * 2.0
                    img = np.clip(img, 0., 1.)
                    h5f.create_dataset(str(img_num), data=img)
                    img_num += 1
        if channels == 1:
            if len(img.shape) == 3:
                img = img[:,:,0]
            Img = img[region[1]:region[3],region[0]:region[2]]
            Img = cv2.resize(Img, size, interpolation=interpolation)
            # Img = np.expand_dims(Img.copy(), 0)
            h, w = Img.shape
            for j in range(blocks[0]):
                for k in range(blocks[1]):
                    img = Img[h//blocks[0]*j:h//blocks[0]*(j+1),w//blocks[1]*k:w//blocks[1]*(k+1)]
                    img = np.float32(normalize(img))
                    img = img * 2.0
                    img = np.clip(img, 0., 1.)
                    h5f.create_dataset(str(img_num), data=img)
                    img_num += 1
    h5f.close()
    return img_num

def prepare_data(data_path, region=[0,0,1024,768], size=256, blocks=1, channels=1, interpolation=cv2.INTER_NEAREST):
    if type(size) == int:
        size = (size,size)
    if type(blocks) == int:
        blocks = (blocks,blocks)
    # train
    print('process training data')
    print('\nprocess noisy data')
    files = glob.glob(os.path.join(data_path, 'train', 'noisy', '*.png'))
    train_noisy_num = generate_h5file(files,region,size,blocks,channels,interpolation,'train_noisy.h5')
    print('\nprocess clean data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'train', 'clean', '*.png'))
    train_clean_num = generate_h5file(files,region,size,blocks,channels,interpolation,'train_clean.h5')

    # val
    print('\nprocess validation data')
    print('\nprocess noisy data')
    # files.clear()
    files = glob.glob(os.path.join(data_path, 'val', 'noisy', '*.png'))
    val_noisy_num = generate_h5file(files,region,size,blocks,channels,interpolation,'val_noisy.h5')
    print('\nprocess clean data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'val', 'clean', '*.png'))
    val_clean_num = generate_h5file(files,region,size,blocks,channels,interpolation,'val_clean.h5')

    print('training set, # noisy samples {0}, # clean samples {1}.\n'.format(train_noisy_num,train_clean_num))
    print('val set, # noisy samples {0}, # clean samples {1}.\n'.format(val_noisy_num,val_clean_num))

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            # h5f = h5py.File('train.h5', 'r')
            h5fn = h5py.File('train_noisy.h5', 'r')
            h5fc = h5py.File('train_clean.h5', 'r')
        else:
            h5fn = h5py.File('val_noisy.h5', 'r')
            h5fc = h5py.File('val_clean.h5', 'r')
        assert len(list(h5fn.keys())) == len(list(h5fc.keys()))
        self.keys = list(h5fn.keys())
        random.shuffle(self.keys)
        h5fn.close()
        h5fc.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5fn = h5py.File('train_noisy.h5', 'r')
            h5fc = h5py.File('train_clean.h5', 'r')
        else:
            h5fn = h5py.File('val_noisy.h5', 'r')
            h5fc = h5py.File('val_clean.h5', 'r')
        key = self.keys[index]
        noisy = np.array(h5fn[key])
        clean = np.array(h5fc[key])
        if len(noisy.shape) == 2:
            noisy = np.expand_dims(np.array(h5fn[key]),-1)
            clean = np.expand_dims(np.array(h5fc[key]),-1)
        noisy = torch.from_numpy(np.transpose(noisy, (2, 0, 1)))
        clean = torch.from_numpy(np.transpose(clean, (2, 0, 1)))
        h5fn.close()
        h5fc.close()
        return noisy, clean

