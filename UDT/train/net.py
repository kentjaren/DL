import torch  # pytorch 0.4.0! fft
import torch.nn as nn
import numpy as np
from util import *
from torchvision.models import vgg19
import pdb


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)

class DCFLayer(nn.Module):
    def __init__(self, lambda0):
        super(DCFLayer, self).__init__()
        self.lambda0 = lambda0

    def forward(self, z, x, label):
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)

        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)

        alphaf = label.to(device=z.device) / (kzzf + self.lambda0)
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)

        return response

class DCFFeature(nn.Module):
    def __init__(self):
        super(DCFFeature, self).__init__()
        self.feature = nn.Sequential(*(self.make_unit(3, 32, pooling=True) +
                                       self.make_unit(32, 64, pooling=True) +
                                       self.make_unit(64, 128, pooling=True) +
                                       self.make_unit(128, 256, pooling=True) +
                                       self.make_unit(256, 256, pooling=True)))
        self.classifier = nn.Sequential(nn.Linear(4 * 256, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, 24))

    def make_unit(self, in_ch, out_ch, pooling=True):
        unit = [nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                # nn.ReLU(inplace=True)]
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)]
        if pooling:
            return unit + [nn.MaxPool2d(2,2)]
        return unit

    def get_outputs(self, x, idxs=None):
        if idxs is None:
            return self.feature(x)
        results = []
        for i, f in enumerate(self.feature):
            x = f(x)
            if i in idxs:
                results.append(x)
            if len(results) == len(idxs):
                break
        _, _, h, w = results[0].shape
        upsampler = nn.Upsample(size=(h, w), mode='bilinear')
        rest = [upsampler(result) for result in results[1:]]
        return results[:1] + rest

    def forward(self, xs):
        B, T, C, H, W = xs.size()
        xs = xs.transpose(0, 1)
        outs = [self.get_outputs(x) for x in xs]
        out = torch.cat([out.view(B, -1) for out in outs], 1)
        out = self.classifier(out)
        return out

class DCF(nn.Module):
    def __init__(self, config=None):
        super(DCF, self).__init__()
        #  self.idxs = [3, 13]
        #  self.ws = [1.0, 0.5]
        self.idxs = [3]
        self.ws = [1.0]
        self.feature = DCFFeature()
        self.dcflayer = DCFLayer(config.lambda0)
        self.init_y = config.y.copy()
        self.yf = config.yf.clone()

    def compose(self, ts, ss, yf):
        assert len(ts) == len(ss) == len(self.ws)
        response = sum(self.dcflayer(t, s, yf) * w for (t, s, w) in zip(ts, ss, self.ws)) / sum(self.ws)
        return response

    def get_order(self, jigsaw):
        return self.feature(jigsaw)

    def get_response(self, template, search1, search2):
        ts = self.feature.get_outputs(template, self.idxs)
        s1s = self.feature.get_outputs(search1, self.idxs)
        s2s = self.feature.get_outputs(search2, self.idxs)

        B = template.shape[0]
        label = self.yf.repeat(B,1,1,1,1).cuda(non_blocking=True)

        with torch.no_grad():
            s1_response = self.compose(ts, s1s, label)
        fake_y = guassian_label_transform(s1_response, self.init_y)
        fake_yf = torch.rfft(torch.Tensor(fake_y).to(template.device), signal_ndim=2).cuda(non_blocking=True)

        with torch.no_grad():
            s2_response = self.compose(s1s, s2s, fake_yf)
        fake_y = guassian_label_transform(s2_response, self.init_y)
        fake_yf = torch.rfft(torch.Tensor(fake_y).to(template.device), signal_ndim=2).cuda(non_blocking=True)

        t_response = self.compose(s2s, ts, fake_yf)

        return t_response, s1_response, s2_response

    def load_param(self, path='./pretrained.pth.tar'):
        checkpoint = torch.load(path)
        new_state_dict = self.state_dict()
        #  import ipdb; ipdb.set_trace()
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict'] 
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
            else:
                for k, v in state_dict.items():
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict)
        else:
            self.feature.load_state_dict(checkpoint, strict=False)

        for idx, layer in enumerate(self.parameters()):
            if idx < 4:
                layer.requires_grad = False
            else:
                break


# class DCFNet(nn.Module):
#     def __init__(self, config=None):
#         super(DCFNet, self).__init__()
#         self.feature = DCFNetFeature()
#         self.dcflayer = DCFLayer(config.lambda0)
#         self.init_y = config.y.copy()
#         self.yf = config.yf.clone()

#     def forward(self, template, search1, search2):
#         t = self.feature(template)
#         s1 = self.feature(search1)
#         s2 = self.feature(search2)

#         B = template.shape[0]
#         label = self.yf.repeat(B,1,1,1,1).cuda(non_blocking=True)

#         with torch.no_grad():
#             s1_response = self.dcflayer(t, s1, label)
#         fake_y = guassian_label_transform(s1_response, self.init_y)
#         fake_yf = torch.rfft(torch.Tensor(fake_y).to(template.device), signal_ndim=2).cuda(non_blocking=True)

#         with torch.no_grad():
#             s2_response = self.dcflayer(s1, s2, fake_yf)
#         fake_y = guassian_label_transform(s2_response, self.init_y)
#         fake_yf = torch.rfft(torch.Tensor(fake_y).to(template.device), signal_ndim=2).cuda(non_blocking=True)

#         t_response = self.dcflayer(s2, t, fake_yf)

#         return t_response, s1_response, s2_response

# class DCFNet(nn.Module):
#     def __init__(self, config=None):
#         super(DCFNet, self).__init__()
#         self.feature = DCFNetFeature()
#         self.yf = config.yf.clone()
#         self.lambda0 = config.lambda0

#     def forward(self, z, x, label):
#         # z = self.feature(z)
#         # x = self.feature(x)
#         zf = torch.rfft(z, signal_ndim=2)
#         xf = torch.rfft(x, signal_ndim=2)

#         kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
#         kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
                  
#         alphaf = label.to(device=z.device) / (kzzf + self.lambda0)  
#         #alphaf = self.yf.to(device=z.device) / (kzzf + self.lambda0) # very Ugly
#         response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
           
#         return response


# if __name__ == '__main__':

#     # network test
#     net = DCFNet()
#     net.eval()



