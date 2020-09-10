import torch.utils.data as data
from os.path import join
import cv2
import json
import numpy as np
import random
from torchvision import transforms
from PIL import Image
from util import *


class VID(data.Dataset):
    def __init__(self, file='dataset/dataset.json', root='dataset/crop_125_2.0', range=10, train=True):
        self.imdb = json.load(open(file, 'r'))
        self.root = root
        self.range = range
        self.train = train
        self.mean = [109,120,119]

    def crop_transform(self, img):
        afterCrop = transforms.Compose([
            transforms.RandomCrop(50),
            transforms.Resize((63, 63), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            transforms.Lambda(RGB2BGR),
            transforms.Lambda(sub_mean(self.mean))
        ])
        trans = transforms.Compose([
            transforms.Lambda(jigsawCrop(self.order)),
            transforms.Lambda(trans_map(afterCrop)),
            transforms.Lambda(lambda imgs: torch.stack(imgs))
        ])
        return trans(img)

    def img_transform(self, img):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(RGB2BGR),
            transforms.Lambda(sub_mean(self.mean))
        ])
        return trans(img)

    def __getitem__(self, item):
        if self.train:
            target_id = self.imdb['train_set'][item]
        else:
            target_id = self.imdb['val_set'][item]

        # range_down = self.imdb['down_index'][target_id]
        range_up = self.imdb['up_index'][target_id]
        # search_id = np.random.randint(-min(range_down, self.range), min(range_up, self.range)) + target_id
        search_id1 = np.random.randint(1, min(range_up, self.range+1)) + target_id
        search_id2 = np.random.randint(1, min(range_up, self.range+1)) + target_id

        target = Image.open(join(self.root, '{:08d}.jpg'.format(target_id)))
        search1 = Image.open(join(self.root, '{:08d}.jpg'.format(search_id1)))
        search2 = Image.open(join(self.root, '{:08d}.jpg'.format(search_id2)))

        self.order = random.randint(0, 23)
        t = self.img_transform(target)
        s1 = self.img_transform(search1)
        s2 = self.img_transform(search2)
        tj = self.crop_transform(target)
        s1j = self.crop_transform(search1)
        s2j = self.crop_transform(search2)
        return t, tj, s1, s1j, s2, s2j, self.order

    def __len__(self):
        if self.train:
            return len(self.imdb['train_set'])
        else:
            return len(self.imdb['val_set'])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    data = VID(train=True)
    n = len(data)
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])

    for i in range(n):
        z, x = data[i]
        z, x = np.transpose(z, (1, 2, 0)).astype(np.uint8), np.transpose(x, (1, 2, 0)).astype(np.uint8)
        zx = np.concatenate((z, x), axis=1)

        ax.imshow(cv2.cvtColor(zx, cv2.COLOR_BGR2RGB))
        p = patches.Rectangle(
            (125/3, 125/3), 125/3, 125/3, fill=False, clip_on=False, linewidth=2, edgecolor='g')
        ax.add_patch(p)
        p = patches.Rectangle(
            (125 / 3+125, 125 / 3), 125 / 3, 125 / 3, fill=False, clip_on=False, linewidth=2, edgecolor='r')
        ax.add_patch(p)
        plt.pause(0.5)
