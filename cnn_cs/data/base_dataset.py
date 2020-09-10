import os
import scipy.io as sio
import copy
import json
import torch
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from misc.util import get_image_paths, VectorNormalize

class CS80kDataset(data.Dataset):
    def __init__(self, opt, genre):
        self.opt = opt
        self.genre = genre

        dataset_dir = os.path.join(opt.data_dir, opt.dataset)
        if genre == 'test':
            dataset_dir = os.path.join(dataset_dir, 'test')
            self.imgs = get_image_paths(dataset_dir)
        else:
            imgs = torch.from_numpy(
                sio.loadmat(
                    os.path.join(dataset_dir, 'Training_Data_Img91.mat'))['labels'])
            if genre == 'train':
                self.imgs = imgs[:85000]
            elif genre == 'val':
                self.imgs = imgs[85000:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.genre == 'test':
            path = self.imgs[idx]
            img = Image.open(path)
            transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
            img = transform(img)

            return {'path': path,
                    'image': img}
        else:
            img = self.imgs[idx]
            transform = VectorNormalize()
            img = transform(img).view(-1, 1, 33, 33)
            return {'image': img}

class CS20kDataset(data.Dataset):
    def __init__(self, opt, genre):
        """
        genre(str): train, val, test
        """
        self.opt = opt
        dataset_dir = os.path.join(opt.data_dir, opt.dataset, genre)
        self.image_paths = get_image_paths(dataset_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path)
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        img = transform(img)

        return {'path': path,
                'image': img}


class FilterDataset(data.Dataset):
    def __init__(self, dataset, f):
        self.items = [item for item in dataset if f(item)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def split_dataset(dataset, ratio):
    demacation = int(len(dataset) * ratio)
    return dataset[:demacation], dataset[demacation:]
