import torch
import torch.utils.data as data

import glob
import random
import h5py
import os
import numpy as np

from PIL import Image
from natsort import natsorted


class Dataset_test(data.Dataset):
    def __init__(self, src_path, sigma, transform=None):
        self.src_path =     (glob.glob(os.path.join(src_path, f'original_png', '*.[jp][pn]g')))
        self.src_path_noise = natsorted(glob.glob(os.path.join(src_path, f'noisy{sigma}', '*.[jp][pn]g')))
        self.transform = transform

    def __getitem__(self, index):
        clean = Image.open(self.src_path[index])
        noisy = Image.open(self.src_path_noise[index])

        clean = self.transform(clean)
        noisy = self.transform(noisy)

        return (clean, noisy)
    
    def __len__(self):
        return len(self.src_path)


class real_Dataset_test(data.Dataset):
    def __init__(self, src_path, transform=None):
        self.src_path = natsorted(glob.glob(os.path.join(src_path, 'clean','*.[jp][pn]g')))
        self.src_path_noise = natsorted(glob.glob(os.path.join(src_path, 'noisy', '*.[jp][pn]g')))
        if not self.src_path:
            self.src_path = self.src_path_noise
        elif len(self.src_path) != len(self.src_path_noise):
            print(f"The number of Clean dataset {len(self.src_path)} and Noisy images {len(self.src_path_noise)} is not equal!")
            assert len(self.src_path) == len(self.src_path_noise)
        self.transform = transform

    def __getitem__(self, index):
        clean = Image.open(self.src_path[index])
        noisy = Image.open(self.src_path_noise[index])

        clean = self.transform(clean)
        noisy = self.transform(noisy)
        label = os.path.split(self.src_path_noise[index])[-1][:-4]

        return (clean, noisy, label)
    
    def __len__(self):
        return len(self.src_path_noise)


class Dataset_from_h5(data.Dataset):

    def __init__(self, src_path_N, src_path_C, sigma, transform=None):
        self.path_N = src_path_N
        self.path_C = src_path_C

        # load generator dataset
        h5f_N = h5py.File(self.path_N, 'r')
        self.keys_G = list(h5f_N.keys())
        random.shuffle(self.keys_G)
        h5f_N.close()
        ####################################
        self.sigma = sigma

        # load clean dataset
        h5f_C = h5py.File(self.path_C, 'r')
        self.keys_D = list(h5f_C.keys())
        random.shuffle(self.keys_D)
        h5f_C.close()
        ####################################

        self.transform = transform
        
    def __getitem__(self, index):
        h5f_N = h5py.File(self.path_N, 'r')
        key_N = self.keys_G[index]
        data = np.array(h5f_N[key_N]).reshape(h5f_N[key_N].shape)
        gt = Image.fromarray(np.uint8(data*255))
        
        h5f_N.close()

        if self.transform:
            gt = self.transform(gt)

        noise = torch.normal(torch.zeros(gt.size()), self.sigma/255.0)        
        noisy = gt + noise 
        noisy = torch.clamp(noisy, 0.0, 1.0)

        # for clean data
        h5f_C = h5py.File(self.path_C, 'r')
        key_C = self.keys_D[index]
        data = np.array(h5f_C[key_C]).reshape(h5f_C[key_C].shape)
        clean_input = Image.fromarray(np.uint8(data*255))

        h5f_C.close()

        if self.transform:
            clean_input = self.transform(clean_input)

        return (noisy, clean_input)

    def __len__(self):
        return min(len(self.keys_G), len(self.keys_D))


class realDataset_from_h5(data.Dataset):
    def __init__(self, src_path_N, src_path_C, transform=None):
        self.path_N = src_path_N
        self.path_C = src_path_C

        # load generator dataset
        h5f_N = h5py.File(self.path_N, 'r')
        self.keys_G = list(h5f_N.keys())
        random.shuffle(self.keys_G)
        h5f_N.close()
        ####################################


        # load clean dataset
        h5f_C = h5py.File(self.path_C, 'r')
        self.keys_D = list(h5f_C.keys())
        random.shuffle(self.keys_D)
        h5f_C.close()
        ####################################
        assert sorted(self.keys_G) == sorted(self.keys_D)
        self.transform = transform
        
    def __getitem__(self, index):
        h5f_N = h5py.File(self.path_N, 'r')
        key_N = self.keys_G[index]
        data = np.array(h5f_N[key_N]).reshape(h5f_N[key_N].shape)
        noisy = Image.fromarray(data)  # to PIL
        
        h5f_N.close()

        if self.transform:
            noisy = self.transform(noisy)
        
        noisy = torch.clamp(noisy, 0.0, 1.0)

        # for clean data
        h5f_C = h5py.File(self.path_C, 'r')
        # key_C = self.keys_D[index]
        key_C = key_N
        data = np.array(h5f_C[key_C]).reshape(h5f_C[key_C].shape)
        
        clean_input = Image.fromarray(np.uint8(data*255))
        h5f_C.close()

        if self.transform:
            clean_input = self.transform(clean_input)

        return (noisy, clean_input) 

    def __len__(self):
        return min(len(self.keys_G), len(self.keys_D))