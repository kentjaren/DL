# import cv2
import torch
import itertools
import numpy as np
import torch.nn as nn


def nth(iterable, n, default=None):
    return next(itertools.islice(iterable, n, None), default)


def reseq(xs, idxs):
    return tuple(xs[idx] for idx in idxs)


def trans_map(t):
    def helper(imgs):
        return tuple(t(img) for img in imgs)

    return helper


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


def RGB2BGR(tensor):
    return tensor[[2, 1, 0], :, :]


def sub_mean(mean):
    def helper(tensor):
        dtype = tensor.dtype
        m = torch.tensor(mean, dtype=dtype, device=tensor.device)
        tensor.mul_(255).sub_(m[:, None, None])
        return tensor

    return helper


def jigsawCrop(order):
    assert 0 <= order <= 23, "order must be in [0, 23]."
    order = nth(itertools.permutations(range(4)), order)

    def helper(img):
        image_width, image_height = img.size
        crop_height, crop_width = image_height // 2, image_width // 2
        tl = img.crop((0, 0, crop_width, crop_height))
        tr = img.crop((crop_width, 0, image_width, crop_height))
        bl = img.crop((0, crop_height, crop_width, image_height))
        br = img.crop((crop_width, crop_height, image_width, image_height))
        imgs = tuple([tl, tr, bl, br])
        return reseq(imgs, order)

    return helper


class ResponseLossL2(nn.Module):
    def __init__(self, size_average=False, only_drop=False):
        super(ResponseLossL2, self).__init__()
        self.size_average = size_average
        self.only_drop = only_drop

    def forward(self, r1, r2, output, label):
        batchsize = label.size(0)
        ret = (output - label)**2
        perSample_loss = torch.sum(ret.view(batchsize, -1), 1)

        _, index = torch.sort(perSample_loss)
        sample_drop = torch.zeros(batchsize).to(ret.device)
        max_num = int(0.9 * batchsize)
        sample_drop[index[:max_num]] = 1

        if self.only_drop:
            weight = sample_drop.view(batchsize, 1, 1, 1)
        else:
            motion = (r1 - label)**2 + (r2 - r1)**2
            motion_sum = torch.sum(motion.view(batchsize, -1), 1)
            delta = sample_drop * motion_sum
            delta_norm = delta / torch.sum(delta) * batchsize

            # import ipdb; ipdb.set_trace()
            weight = torch.exp(label) * delta_norm.view(batchsize, 1, 1, 1)

        loss = ret * weight
        if self.size_average:
            return torch.sum(loss) / batchsize
        return torch.sum(loss)

def guassian_label_transform(response, init_y):
    b, _, h, w = response.shape
    _, index = torch.max(response.view(b, -1), 1)
    r_max, c_max = np.unravel_index(index, [h, w])
    fake_y = np.zeros((b, 1, h, w))
    for j in range(b):
        shift_y = np.roll(init_y, r_max[j])
        fake_y[j,...] = np.roll(shift_y, c_max[j])
    return fake_y
