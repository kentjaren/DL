import torch
import torch.nn as nn

class ReconNet(nn.Module):
    def __init__(self, in_chs):
        super(ReconNet, self).__init__()

        self.first = nn.Sequential(
            nn.Linear(in_chs, 33*33),
            nn.ReLU())

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 11, 1, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 7, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 64, 11, 1, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 7, 1, 3),
            nn.BatchNorm2d(1),
            nn.Tanh()
            )

    def forward(self, x):
        out = self.first(x)
        out = self.main(out.view(-1, 1, 33, 33))
        return out

class DeepInverse(nn.Module):
    pass

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False, is_last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.is_last = is_last
        if downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = False

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        if self.is_last:
            out = self.tanh(out)
        else:
            out = self.relu(out)

        return out

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 11, 1, 5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.block2 = BasicBlock(64, 64)
        self.block3 = BasicBlock(64, 1, downsample=True, is_last=True)

    def forward(self, x):
        out = self.pre(x)
        out = self.block2(out)
        out = self.block3(out)
        return out

