import torch.nn as nn
import torch  # pytorch 0.4.0! fft
import numpy as np
import cv2
from torchvision.models import vgg19


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFFeature(nn.Module):
    def __init__(self):
        super(DCFFeature, self).__init__()
        self.feature = nn.Sequential(*(self.make_unit(3, 32, pooling=True) +
                                       self.make_unit(32, 64, pooling=True) +
                                       self.make_unit(64, 128, pooling=True) +
                                       self.make_unit(128, 256, pooling=True) +
                                       self.make_unit(256, 256, pooling=True)))

    def make_unit(self, in_ch, out_ch, pooling=True):
        unit = [nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                # nn.ReLU(inplace=True)]
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)]
        if pooling:
            return unit + [nn.MaxPool2d(2,2)]
        return unit

    def forward(self, x, idxs=None):
        results = []
        for i, f in enumerate(self.feature):
            x = f(x)
            if i in idxs:
                results.append(x)
            if len(results) == len(idxs):
                break
        _, _, h, w = results[0].shape
        upsampler = nn.Upsample(size=(h, w), mode='bilinear')
        rest = [upsampler(x) for x in results[1:]]
        return results[:1] + rest

class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFFeature()
        self.idxs = [3, 13, 23]
        self.ws = [1.0, 0.2, 0.05]
        # self.idxs = [3]
        # self.ws = [1.0]
        self.model_alphaf = []
        self.model_zf = []
        self.config = config

    def forward(self, x):
        feats = self.feature(x, self.idxs)
        def helper(x, zf, alphaf):
            x = x * self.config.cos_window
            xf = torch.rfft(x, signal_ndim=2)
            kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
            response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
            return response
        return sum(helper(x, zf, alphaf) * w
                   for (x, zf, alphaf, w) in zip(feats, self.model_zf, self.model_alphaf, self.ws))

    def update(self, z, lr=1.):
        feats = self.feature(z, self.idxs)
        def helper(z):
            z = z * self.config.cos_window
            zf = torch.rfft(z, signal_ndim=2)
            kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
            alphaf = self.config.yf / (kzzf + self.config.lambda0)
            return alphaf, zf
        def adjust(lr, before, after):
            return (1 - lr) * before + lr * after
        if lr > 0.99:
            self.model_alphaf, self.model_zf = zip(*(helper(feat) for feat in feats))
        else:
            alphafs, zfs = zip(*(helper(feat) for feat in feats))
            self.model_alphaf = [adjust(lr, x.data, y.data) for (x,y) in zip(self.model_alphaf, alphafs)]
            self.model_zf = [adjust(lr, x.data, y.data) for (x,y) in zip(self.model_zf, zfs)]

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict'] 
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'classifier' not in k:
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
            else:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'classifier' not in k:
                        new_state_dict[k] = v
            self.load_state_dict(new_state_dict)
        else:
            self.feature.load_state_dict(checkpoint)


if __name__ == '__main__':

    # network test
    net = DCFNetFeature()
    net.eval()
    for idx, m in enumerate(net.modules()):
        print(idx, '->', m)
    for name, param in net.named_parameters():
        if 'bias' in name or 'weight' in name:
            print(param.size())
    from scipy import io
    import numpy as np
    p = io.loadmat('net_param.mat')
    x = p['res'][0][0][:,:,::-1].copy()
    x_out = p['res'][0][-1]
    from collections import OrderedDict
    pth_state_dict = OrderedDict()

    match_dict = dict()
    match_dict['feature.0.weight'] = 'conv1_w'
    match_dict['feature.0.bias'] = 'conv1_b'
    match_dict['feature.2.weight'] = 'conv2_w'
    match_dict['feature.2.bias'] = 'conv2_b'

    for var_name in net.state_dict().keys():
        print var_name
        key_in_model = match_dict[var_name]
        param_in_model = var_name.rsplit('.', 1)[1]
        if 'weight' in var_name:
            pth_state_dict[var_name] = torch.Tensor(np.transpose(p[key_in_model],(3,2,0,1)))
        elif 'bias' in var_name:
            pth_state_dict[var_name] = torch.Tensor(np.squeeze(p[key_in_model]))
        if var_name == 'feature.0.weight':
            weight = pth_state_dict[var_name].data.numpy()
            weight = weight[:, ::-1, :, :].copy()  # cv2 bgr input
            pth_state_dict[var_name] = torch.Tensor(weight)


    torch.save(pth_state_dict, 'param.pth')
    net.load_state_dict(torch.load('param.pth'))
    x_t = torch.Tensor(np.expand_dims(np.transpose(x,(2,0,1)), axis=0))
    x_pred = net(x_t).data.numpy()
    pred_error = np.sum(np.abs(np.transpose(x_pred,(0,2,3,1)).reshape(-1) - x_out.reshape(-1)))

    x_fft = torch.rfft(x_t, signal_ndim=2, onesided=False)


    print('model_transfer_error:{:.5f}'.format(pred_error))


import torch.nn as nn
