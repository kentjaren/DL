import os
import yaml
import random
import logging
import logging.config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import datetime
from math import sqrt
from math import log10
from functools import reduce
import operator
import numbers
from functools import partial
from torchvision import transforms

class VectorNormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        assert tensor.dim() < 3
        new = (tensor - self.mean) / self.std
        return new

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
    '.tif',
    '.TIF'
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

def creat_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def is_same_size(tensors):
    assert isinstance(tensors, list)
    assert not len(tensors) == 0

    size = tensors[0].size()
    return all([t.size() == size for t in tensors])

def arrange_by_index(tensors):
    """
    tensors(list of tensors)
    """
    assert is_same_size(tensors)

    dim0 = tensors[0].size(0)

    def collect_ith(i):
        return torch.stack([t[i] for t in tensors], dim=0)

    return [collect_ith(i) for i in range(dim0)]

def generate_measure_matrix(side, cr, file_name):
    n = side * side
    m = int(n * cr)
    mat = torch.randn(n, m) / sqrt(m)
    torch.save(mat, file_name)

def norm(tensor, range=None, scale_each=False):
    assert tensor.dim() == 4
    if range is not None:
        assert isinstance(range, tuple)

    def norm_ip(img, minimum, maximum):
        img.clamp_(min=minimum, max=maximum)
        img.add_(-minimum).div_(maximum - minimum + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    tensor = tensor.clone()

    if scale_each is True:
        for t in tensor:
            norm_range(t, range)
    else:
        norm_range(tensor, range)

    return tensor

def l2_norm(tensor):
    tmp = torch.sum(tensor * tensor)
    return sqrt(tmp.item())

def calu_diff_ratio(img, target, normalize=True):
    assert img.dim() == 4
    assert img.size() == target.size()

    if normalize:
        img = norm(img)
        target = norm(target)

    return l2_norm(img - target) / l2_norm(target)

def calu_diff_ratio_tensor(imgs, target, normalize=True):
    return torch.tensor([calu_diff_ratio(img, target, normalize)
                         for img in imgs])

def calu_psnr(img, target, normalize=True):
    """
    img(tensor)
    target(tensor)
    the value of tensors should be in [0, 1]
    """
    assert img.dim() == 4
    assert img.size() == target.size()

    if normalize:
        img = norm(img)
        target = norm(target)

    mse = nn.MSELoss()(img, target)
    psnr = 10 * log10(1 / mse.item())
    return psnr

def calu_psnr_tensor(imgs, target, normalize=True):
    """
    imgs(list of tensor)
    target(tensor)
    """
    return torch.tensor([calu_psnr(img, target, normalize)
                         for img in imgs])

def calu_pixel(img, target, ratio=0.1, normalize=True):
    assert img.dim() == 4
    assert img.size() == target.size()

    if normalize:
        img = norm(img)
        target = norm(target)

    mask = (torch.abs(img - target) / target < ratio)
    sum_pixels = reduce(operator.mul, img.size())
    return torch.sum(mask == 1).float() / sum_pixels * 100

def calu_pixel_tensor(imgs, target, ratio=0.1, normalize=True):
    return torch.tensor([calu_pixel(img, target, ratio, normalize)
                         for img in imgs])

def calu_ensemble_status(syns, target, ratio=0.1, normalize=True):
    assert target.dim() == 4
    assert isinstance(syns, list)
    assert is_same_size(syns + [target])

    if normalize:
        syns = [norm(s) for s in syns]
        target = norm(target)

    mask = sum([(torch.abs(s - target) / target < ratio) for s in syns])
    sum_pixels = reduce(operator.mul, target.size())
    calu_prop = lambda i: torch.sum(mask == i).float() / sum_pixels * 100
    return torch.Tensor([calu_prop(i) for i in range(len(syns) + 1)])

def setup_logging(config_path='logging_config.yaml',
                  default_level=logging.INFO,
                  logging_path = 'output.log'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
            config['handlers']['file']['filename'] = logging_path
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def logging_psnr(tensor):
    assert len(tensor) > 1
    syn_fmt = "syn{}:\t{:.2f}dB"
    ret_fmt = "ret:\t{:.2f}dB"
    logging.info("----------------PSNR---------------")
    for i, t in enumerate(tensor[:-1]):
        logging.info(syn_fmt.format(i, t.item()))
    logging.info(ret_fmt.format(tensor[-1].item()))

def logging_diff_ratio(tensor):
    assert len(tensor) > 1
    syn_fmt = "syn{}:\t{:.2f}"
    ret_fmt = "ret:\t{:.2f}"

    logging.info("------------diff_ratio--------------")
    for i, t in enumerate(tensor[:-1]):
        logging.info(syn_fmt.format(i, t.item()))
    logging.info(ret_fmt.format(tensor[-1].item()))

def logging_pixel(tensor):
    assert len(tensor) > 1
    syn_fmt = "syn{}:\t{:.2f}%"
    ret_fmt = "ret:\t{:.2f}%"
    logging.info("------------pixel_status-----------")
    for i, t in enumerate(tensor[:-1]):
        logging.info(syn_fmt.format(i, t.item()))
    logging.info(ret_fmt.format(tensor[-1].item()))

def logging_ensemble(tensor):
    assert len(tensor) > 1
    fmt = "{} close:\t{:.2f}%"
    logging.info("----------ensemble-status----------")
    for i, t in enumerate(tensor):
        logging.info(fmt.format(i, t.item()))

def logging_train(net_index, epoch, duration, loss, same_time):
    if same_time:
        logging.info("Training first {} net.".format(net_index+1))
    else:
        logging.info("Training {}th net.".format(net_index))
    logging.info("End of epoch {}\tTime Taken: {:.2f} sec\tLoss : {:.2f}\n".format(
        epoch,
        duration,
        loss))

def logging_status(net_index, duration, psnr, diff_ratio_status,
                   pixel_status, ensemble_status, epoch, genre):
    if genre == 'Val':
        logging.info("{} {}th net of epoch {}".format(genre, net_index, epoch))
    else:
        logging.info("{} {}th net".format(genre, net_index))

    logging.info("{} Time Taken:  {:.2f} sec".format(genre, duration))
    logging_psnr(psnr)
    logging_diff_ratio(diff_ratio_status)
    logging_pixel(pixel_status)
    logging_ensemble(ensemble_status)


logging_val = partial(logging_status, genre='Val')
logging_test = partial(logging_status, epoch=None, genre='Test')

class Criterion(nn.Module):
    def __init__(self, device, alpha, lam, loss_func_type):
        super(Criterion, self).__init__()
        self.device = device
        self.alpha = alpha
        self.lam = lam
        main_choice, diff_choice = loss_func_type // 10, loss_func_type % 10
        if main_choice == 1:
            self.main = nn.MSELoss()
        elif main_choice == 2:
            self.main = SparseMSELoss()
        elif main_choice == 3:
            self.main = MixLoss(alpha=self.alpha)
        else:
            raise ValueError("loss_func_type wrong type")

        if diff_choice == 1:
            self.diff_position = diff_position1
        elif diff_choice == 2:
            self.diff_position = diff_position2
        else:
            raise ValueError("loss_func_type wrong type")

    def forward(self, syn, syns, groundtruth):
        syn = syn.to(self.device)
        syns = [s.to(self.device) for s in syns]
        groundtruth = groundtruth.to(self.device)
        main_part = self.main(syn, groundtruth)
        norm_part = self.lam * self.diff_position(
            syn, syns, groundtruth, self.device)
        # print("main part:\t{}".format(main_part))
        # print("norm part:\t{}".format(norm_part))

        return main_part + norm_part

def diff_position1(syn, syns, groundtruth, device):
    def diff(syn1, syn2, groundtruth):
        res1 = torch.abs(syn1 - groundtruth)
        res2 = torch.abs(syn2 - groundtruth)
        return nn.L1Loss()(torch.mul(res1, res2),
                           torch.zeros(groundtruth.size()).to(
                               device))

    # detach()
    return sum([
        diff(syn, s.detach(), groundtruth) for s in syns
        if syn is not s
    ])


def diff_position2(syn, syns, groundtruth, device):
    def diff(syn1, syn2, groundtruth):
        res1 = syn1 - groundtruth
        res2 = syn2 - groundtruth
        tmp = torch.zeros(groundtruth.size()).to(device)

        batchsize = tmp.size(0)
        subtraction = nn.L1Loss()((res1 + res2), tmp)
        subtracted = torch.abs(torch.sum(torch.mul(res1, res2)) / batchsize)
        return subtracted - subtraction

    return sum([diff(syn, s.detach(), groundtruth) for s in syns
                if syn is not s
    ])

def _sparse_mse_loss(input, target):
    t = torch.abs(input - target)
    return torch.where(t < 0.1, 0.2 * t - 0.01 , t ** 2)

class SparseMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SparseMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        ret = _sparse_mse_loss(input, target)
        if self.reduction != 'None':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret

class MixLoss(nn.Module):
    def __init__(self, alpha=0.1, reduction='mean'):
        super(MixLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        part1 = nn.MSELoss(reduction=self.reduction)
        part2 = nn.L1Loss(reduction=self.reduction)
        return part1(input, target) + self.alpha * part2(input, target)

def img2blocks(img, size, padding=True):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2

    block_h, block_w = size

    _, _, height, width = img.size()

    if padding:
        h_points = range(0, height, block_h)
        w_points = range(0, width, block_w)
        h_pad = len(h_points) * block_h - height
        w_pad = len(w_points) * block_w - width
        img = F.pad(input=img, pad=(0, w_pad, 0, h_pad),
                    mode='constant', value=0)
    else:
        h_points = range(0, height-block_h, block_h)
        w_points = range(0, width-block_w, block_w)

    block_lst = [img[:, :, h:h+block_h, w:w+block_w]
                 for h in h_points
                 for w in w_points]

    return torch.cat(block_lst, 0)

class Blocks2Img(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, blocks):
        b, c, block_h, block_w = blocks.size()
        h_points = range(0, self.height, block_h)
        w_points = range(0, self.width, block_w)
        device = blocks.device
        img = torch.zeros(1, c, len(h_points)*block_h, len(w_points)*block_w).to(device)

        count = 0
        for h in h_points:
            for w in w_points:
                img[0, :, h:h+block_h, w:w+block_w] = blocks[count]
                count = count + 1

        return img[:, :, :self.height, :self.width]

def set_manualSeed(opt, rand=False):
    if rand:
        manualSeed = random.randint(1, 10000)
    else:
        manualSeed = 100
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if not opt.device == 'cpu':
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

def weights_init_default(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
