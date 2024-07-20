import os
import time
import copy
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import tf.transformations as tft


def transform_pose(pose):
    px, py, pz = pose[0], pose[1], pose[2]
    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    
    transformed_px = -pz
    transformed_py = -px
    transformed_pz = py
    
    # 쿼터니언 변환
    quat = [qx, qy, qz, qw]
    rot_quat_x = tft.quaternion_from_euler(np.pi / 2, 0, 0)
    
    # z축을 기준으로 -90도 회전하는 쿼터니언
    rot_quat_z = tft.quaternion_from_euler(0, 0, -np.pi / 2)
    
    # 기존 쿼터니언과 두 회전 쿼터니언을 곱하여 새로운 쿼터니언 생성
    quat_after_x = tft.quaternion_multiply(rot_quat_x, quat)
    transformed_quat = tft.quaternion_multiply(rot_quat_z, quat_after_x)
    
    return transformed_px, transformed_py, transformed_pz, transformed_quat[0], transformed_quat[1], transformed_quat[2], transformed_quat[3]

class CustomDataset(Dataset):
    def __init__(self, image_path, metadata_path, mode, transform, num_val=3000):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        raw_lines = open(self.metadata_path, 'r').readlines()
        self.lines = raw_lines[0:]

        print(self.lines.__len__())
        print(self.lines[0])

        self.test_filenames = []
        self.test_poses = []
        self.train_filenames = []
        self.train_poses = []

        for i, line in enumerate(self.lines):
            splits = line.split()
            filename = splits[0]
            values = splits[1:]
            values = list(map(lambda x: float(x.replace(",", "")), values))

             # 변환된 pose 정보 사용
            transformed_values = transform_pose(values)

            filename = os.path.join(self.image_path, filename)

            if self.mode == 'train':
                self.train_filenames.append(filename)
                self.train_poses.append(values)
            elif self.mode in ['val', 'test']:
                self.test_filenames.append(filename)
                self.test_poses.append(values)
                if self.mode == 'val' and i > num_val:
                    break
            else:
                assert 'Unavailable mode'

        self.num_train = len(self.train_filenames)
        self.num_test = len(self.test_filenames)
        print("Number of Train", self.num_train)
        print("Number of Test", self.num_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(self.train_filenames[index])
            pose = self.train_poses[index]
        elif self.mode in ['val', 'test']:
            image = Image.open(self.test_filenames[index])
            pose = self.test_poses[index]

        return self.transform(image), torch.Tensor(pose)

    def __len__(self):
        if self.mode == 'train':
            return self.num_train
        elif self.mode in ['val', 'test']:
            return self.num_test


def get_loader(model, image_path, metadata_path, mode, batch_size, is_shuffle=False, num_val=3000):
    # Predefine image size
    if model == 'Googlenet':
        img_size = 300
        img_crop = 299
    elif model in ['Resnet34', 'Resnet50', 'Resnet101', 'Resnet152']:
        img_size = 256
        img_crop = 224
    else:
        raise ValueError(f"Invalid model name: {model}")

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_crop),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        datasets = {
            'train': CustomDataset(image_path, metadata_path, 'train', transform, num_val),
            'val': CustomDataset(image_path, metadata_path, 'val', transform, num_val)
        }
        data_loaders = {
            'train': DataLoader(datasets['train'], batch_size, shuffle=is_shuffle, num_workers=4),
            'val': DataLoader(datasets['val'], batch_size, shuffle=is_shuffle, num_workers=4)
        }
    elif mode == 'test':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        batch_size = 1
        is_shuffle = False
        dataset = CustomDataset(image_path, metadata_path, 'test', transform)
        data_loaders = DataLoader(dataset, batch_size, shuffle=is_shuffle, num_workers=4)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return data_loaders


