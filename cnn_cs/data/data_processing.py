import os
import re
from random import shuffle
from PIL import Image
from functools import partial
import shutil
from collections import defaultdict

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    """Return True if the file is an image.

    >>> is_image_file('front_1.jpg')
    True
    >>> is_image_file('bs')
    False
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(image_dir):
    return [os.path.join(root, fname)
            for root, _, fnames in sorted(os.walk(image_dir))
            for fname in fnames
            if is_image_file(fname)]

def crop_save(image_path, width, height, stride, save_dir):
    img = Image.open(image_path)
    w, h = img.size

    x_points = range(0, w - width, stride)
    y_points = range(0, h - height, stride)

    for x in x_points:
        for y in y_points:
            region = img.crop((x, y, x + width, y + height))
            fname = os.path.join(save_dir,
                                 '{}_{}_{}'.format(x, y, os.path.basename(image_path)))
            region.save(fname)

    print("The size of {}:  {}".format(image_path, img.size))
    print("crop {} image blocks from {}".format(len(x_points) * len(y_points),
                                                image_path))

def calu_output_num(image_lst, width, height, stride):
    def calu_from_single(path):
        img = Image.open(path)
        w, h = img.size

        x_points = range(0, w - width, stride)
        y_points = range(0, h - height, stride)
        return len(x_points) * len(y_points)

    return sum([calu_from_single(path) for path in image_lst])

# calu_num = partial(calu_output_num,
#                    image_lst=image_paths,
#                    width=33,
#                    height=33)

def split_dataset(image_dir):
    cate = ['train', 'val']
    dst_dirs = [os.path.join(image_dir, c) for c in cate]
    for dst_dir in dst_dirs:
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
    image_lst = get_image_paths(image_dir)
    shuffle(image_lst)
    train_set = image_lst[:20000]
    val_set = image_lst[20000:]

    def get_dst(path, is_train):
        dst_dir = 'train' if is_train else 'val'
        return os.path.join(os.path.dirname(path),
                            dst_dir,
                            os.path.basename(path))

    for path in train_set:
        shutil.move(path, get_dst(path, is_train=True))

    for path in val_set:
        shutil.move(path, get_dst(path, is_train=False))

def get_id(image_path):
    return re.search('[0-9]+_[0-9]+_(.+\.bmp)',
                     image_path).groups()[0]

def id_dict(id_lst):
    d = defaultdict(int)
    for i in id_lst:
        d[i] += 1

    total = sum(d.values())
    for k in d.keys():
        d[k] = d[k] / total
    return d

def compare_dict(dict1, dict2):
    assert sorted(dict1.keys()) == sorted(dict2.keys())
    d = defaultdict(int)
    for k in dict1.keys():
        d[k] = dict1[k] - dict2[k]
    return d


if __name__ == '__main__':
    image_dir = 'Train'
    for path in get_image_paths(image_dir):
        crop_save(path, 33, 33, 14, 'crop')

    split_dataset('crop')

    val_id_status = id_dict([get_id(path) for path in get_image_paths('crop/val')])
    train_id_status = id_dict([get_id(path) for path in get_image_paths('crop/train')])
    difference = sum(abs(i) for i in compare_dict(val_id_status, train_id_status).values())

