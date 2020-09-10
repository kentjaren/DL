import os
import logging
from torchvision import datasets
from torchvision import transforms

from .base_dataset import FilterDataset, split_dataset, CS20kDataset, CS80kDataset

cifar_class = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def CreateDataset(opt, genre):
    """
    genre(str): train, val, test
    """
    dataset = None
    flag = False if genre == 'test' else True
    if opt.dataset == 'CS20k':
        dataset = CS20kDataset(opt, genre)
    elif opt.dataset == 'CS80k':
        dataset = CS80kDataset(opt, genre)
    elif opt.dataset == 'cifar10':
        dataset = FilterDataset(
            datasets.CIFAR10(
                os.path.join(opt.data_dir, 'cifar10'),
                train = flag,
                transform=transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                ]),
                download=True),
            lambda t: t)
        if genre == 'train':
            dataset = split_dataset(dataset, 0.9)[0]
        elif genre == 'val':
            dataset = split_dataset(dataset, 0.9)[1]

    elif opt.dataset in cifar_class:
        dataset = FilterDataset(
            datasets.CIFAR10(
                os.path.join(opt.data_dir, 'cifar10'),
                train=flag,
                transform=transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                ]),
                download=True),
            lambda t: t[1] == cifar_class.index(opt.dataset))

        if genre == 'train':
            dataset = split_dataset(dataset, 0.9)[0]
        elif genre == 'val':
            dataset = split_dataset(dataset, 0.9)[1]
    else:
        raise ValueError("Dataset {} was not recongnized.".format(opt.dataset))

    logging.info("{} dataset {} was created.".format(genre, opt.dataset))
    return dataset
