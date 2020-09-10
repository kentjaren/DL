import os
import time
import torch
import logging
import scipy.io as sio
import torch.optim as optim
import torchvision.utils as vutils
from collections import namedtuple
from functools import reduce

from .structure import ReconNet, DeepInverse
from misc.util import weights_init_default
from misc.util import img2blocks, Blocks2Img
from misc.util import Criterion
from misc.util import arrange_by_index, creat_dir
from misc.util import calu_psnr_tensor, calu_pixel_tensor
from misc.util import calu_ensemble_status, calu_diff_ratio_tensor

class EnsembleNet(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.in_chs = int(33*33*self.opt.cr)

        phi_path = os.path.join(opt.ms_dir,
                                'phi_0_{}_1089.mat'.format(
                                    int(self.opt.cr * 100)))
        phi = sio.loadmat(phi_path)['phi'].transpose()
        self.phi = torch.from_numpy(phi).float().to(self.device)

        if self.opt.model == 'ReconNet':
            net_names = ['ReconNet{}'.format(i) for i in range(3)]
            net_structure = [ReconNet(self.in_chs).to(self.device),
                             ReconNet(self.in_chs).to(self.device),
                             ReconNet(self.in_chs).to(self.device)]
        elif self.opt.model == 'DeepInverse':
            net_names = ['DeepInverse{}'.format(i) for i in range(3)]
            net_structure = [DeepInverse(self.in_chs).to(self.device),
                             DeepInverse(self.in_chs).to(self.device),
                             DeepInverse(self.in_chs).to(self.device)]

        net_lrs = [opt.lr0, opt.lr1, opt.lr2]

        optims = [
            optim.Adam(net_s.parameters(), lr=net_lr)
            for (net_s, net_lr) in zip(net_structure, net_lrs)
        ]

        Net = namedtuple('Net', 'name structure optimizer')
        self.nets = [Net._make(m) for m in zip(net_names, net_structure, optims)]

        self.default_model_paths = [os.path.join(self.opt.save_dir,
                                                 "{}.pth".format(net.name))
                                    for net in self.nets]

        self.criterion = Criterion(self.device, opt.alpha, opt.lam, opt.loss_func_type)

    def init_weights(self):
        for net in self.nets:
            net.structure.apply(weights_init_default)

    def set_input(self, batch, genre):
        """
        genre(str): train, val, test
        """
        if self.opt.dataset == 'cifar10':
            pass
        elif self.opt.dataset == 'CS20k':
            self.targets = batch['image'].to(self.device)
            tmp = img2blocks(self.targets, 33) if genre=='test' else self.targets
            self.inputs = torch.matmul(tmp.view(tmp.size(0), -1), self.phi)
        elif self.opt.dataset == 'CS80k':
            self.targets = batch['image'].to(self.device)
            tmp = img2blocks(self.targets, 33) if genre=='test' else self.targets
            self.inputs = torch.matmul(tmp.view(tmp.size(0), -1), self.phi)

    def save_model(self, net_index, model_path=None):
        if model_path is None:
            model_path = self.default_model_paths[net_index]
        net = self.nets[net_index]
        torch.save(net.structure.state_dict(), model_path)

    def load_model(self, net_index, model_path=None):
        if model_path is None:
            model_path = self.default_model_paths[net_index]
        net = self.nets[net_index]
        net.structure.load_state_dict(
            torch.load(model_path, map_location=self.device))

    def train_same_time(self, net_index, dataloader):
        nets = self.nets[:net_index+1]
        for net in nets:
            net.structure.train()

        start_time = time.time()
        epoch_loss = 0
        for batch in dataloader:
            self.set_input(batch, genre='train')
            self.syns = [net.structure(self.inputs) for net in nets]
            losses = [self.criterion(s, self.syns, self.targets)
                      for s in self.syns]
            epoch_loss += sum(losses)

            for net, loss in zip(nets, losses):
                net.structure.zero_grad()
                loss.backward()
                net.optimizer.step()

        self.epoch_loss = epoch_loss
        self.epoch_time = time.time() - start_time

    def train_ith_net(self, net_index, dataloader):
        """
        Train the ith net. net_index start from 0.
        """
        fixed_nets = self.nets[:net_index]
        current_net = self.nets[net_index]
        for net in fixed_nets:
            net.structure.eval()

        current_net.structure.train()

        start_time = time.time()
        epoch_loss = 0
        for batch in dataloader:
            self.set_input(batch, genre='train')
            with torch.no_grad():
                fixed_syns = [net.structure(self.inputs) for net in fixed_nets]
            current_syn = current_net.structure(self.inputs)
            self.syns = fixed_syns + [current_syn]
            loss = self.criterion(current_syn, self.syns, self.targets)
            epoch_loss += loss

            current_net.structure.zero_grad()
            loss.backward()
            current_net.optimizer.step()

        self.epoch_loss = epoch_loss.item()
        self.epoch_time = time.time() - start_time

    def val_ith_net(self, net_index, dataloader, epoch,
                    save_image=True, show_status=True):
        self.infer_ith_net(net_index, dataloader, 'val', epoch,
                           save_image, show_status)

    def test_ith_net(self, net_index, dataloader,
                     save_image=True, show_status=True):
        self.infer_ith_net(net_index, dataloader, 'test', None,
                           save_image, show_status)

    def infer_ith_net(self, net_index,
                      dataloader,
                      genre,
                      epoch,
                      save_image=True,
                      show_status=True):
        fixed_nets = self.nets[:(net_index+1)]
        for net in fixed_nets:
            net.structure.eval()

        start_time = time.time()
        avg_psnr = torch.zeros(net_index + 2)
        avg_pixel_status = torch.zeros(net_index + 2)
        avg_ensemble_status = torch.zeros(net_index + 2)
        avg_diff_ratio_status = torch.zeros(net_index + 2)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                self.set_input(batch, genre=genre)
                self.syns = [net.structure(self.inputs) for net in fixed_nets]
                self.ret = self.assemble(self.syns)

                if not self.ret.size() == self.targets.size():
                    _, _, height, width = self.targets.size()
                    blocks2img = Blocks2Img(height, width)
                    self.syns = [blocks2img(s) for s in self.syns]
                    self.ret = blocks2img(self.ret)

                if show_status:
                    tmp = self.syns + [self.ret]
                    avg_psnr += calu_psnr_tensor(tmp, self.targets)
                    avg_pixel_status += calu_pixel_tensor(tmp, self.targets)
                    avg_ensemble_status += calu_ensemble_status(self.syns, self.targets)
                    avg_diff_ratio_status += calu_diff_ratio_tensor(tmp, self.targets)

                if save_image:
                    self.save_image(genre, net_index, epoch, i)

        num_batch = len(dataloader)
        self.infer_time = time.time() - start_time
        self.psnr = avg_psnr / num_batch
        self.pixel_status = avg_pixel_status / num_batch
        self.ensemble_status = avg_ensemble_status / num_batch
        self.diff_ratio_status = avg_diff_ratio_status / num_batch
        self.ret_psnr = self.psnr[-1]

    def save_image(self, genre, net_index, epoch, batch):
        if genre == 'val':
            dirname = os.path.join(self.opt.save_dir, genre, 'net{}'.format(net_index),
                                   'epoch_{}'.format(epoch))
        else:
            dirname = os.path.join(self.opt.save_dir, genre, 'net{}'.format(net_index))
        creat_dir(dirname)

        save_content = [self.targets] + self.syns + [self.ret]
        save_content = arrange_by_index(save_content)
        for i, content in enumerate(save_content):
            img_name = 'batch{}_{}th.bmp'.format(batch, i)
            vutils.save_image(content, os.path.join(dirname, img_name), normalize=True)

    def assemble(self, syns):
        length = len(syns)
        if length < 3:
            return sum(syns) / length

        assert length == 3
        if self.opt.assemble == 'avg':
            return sum(syns) / length
        else:
            def min_position(x, items):
                pos = [(x <= item) for item in items if item is not x]
                return reduce(torch.mul, pos).float()

            minps = [min_position(s, syns) for s in syns]
            minum = sum([minps[i] * syns[i]
                         for i in range(length)]) / sum(minps)

            def mid_position(x, items):
                y, z = [item for item in items if item is not x]
                return ((y - x) * (z - x) <= 0).float()

            midps = [mid_position(s, syns) for s in syns]
            middle = sum([midps[i] * syns[i]
                          for i in range(length)]) / sum(midps)

            def max_position(x, items):
                pos = [(x >= item) for item in items if item is not x]
                return reduce(torch.mul, pos).float()

            maxps = [max_position(s, syns) for s in syns]
            maxium = sum([maxps[i] * syns[i]
                          for i in range(length)]) / sum(maxps)

            near = minum * ((maxium - middle) >=
                            (middle - minum)).float() + maxium * (
                                (maxium - middle) < (middle - minum)).float()
            if self.opt.assemble == 'near':
                return (near + middle) / 2
            elif self.opt.assemble == 'mix':
                delta = self.opt.delta * (middle - middle.min())
                mask = ((maxium - minum > delta) *
                        (torch.abs(near - middle) < delta)).float()
                excp = ((near + middle) / 2 * mask)
                usual = sum(syns) / length * (1 - mask)
                return excp + usual
