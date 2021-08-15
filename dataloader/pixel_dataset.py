import torch
import numpy as np
from tqdm import tqdm
import sys
import pickle5 as pickle
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
from glob import glob
import torch.nn.functional as F
import cv2

sys.path.append('.')


class PixelDataset(Dataset):
    def __init__(self, data_folder, ann_file, type):
        super().__init__()
        self.data_folder = data_folder
        self.__parse_annotation(ann_file, type)

    def __parse_annotation(self, ann_file, type):
        ann_file = open(ann_file, "rb")
        ann = pickle.load(ann_file)
        if type == 'train':
            self.indexes = ann['train']
        elif type == 'val':
            self.indexes = ann['val']
        else:
            raise Exception("invalid dataset type")

    def __getitem__(self, index):
        # load
        image_name = self.indexes[index]

        label = cv2.imread(f'{self.data_folder}/img_train2/{image_name}')
        label = (label[:,:,0])

        image = cv2.imread(f'{self.data_folder}/img_train_shape/{image_name}')
        image = (image[:,:,0])
        # image = np.expand_dims(image, axis=0)
        
        return image, label

    def __len__(self):
        return len(self.indexes)

def build_dataset(cfg):
    train_ds = PixelDataset(cfg.data_folder, cfg.ann_file, 'train')
    val_ds = PixelDataset(cfg.data_folder, cfg.ann_file, 'val')

    return train_ds, val_ds
