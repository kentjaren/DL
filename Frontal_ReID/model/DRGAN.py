import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np

from Component import Generator
from Component import Discriminator
from Component import one_hot
from Component import weights_init_normal
from Component import Tensor2Image
from base_model import BaseModel

class Single_DRGAN(BaseModel):
    """
    The model of Single_DRGAN according to the options.
    """

    def name(self):
        return 'Single_DRGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_Train = opt.is_Train

        self.G = Generator(N_p=opt.N_p, N_z=opt.N_z)
        self.D = Discriminator(N_p=opt.N_p, N_d=opt.N_d)
        if self.is_Train:
            self.optimizer_G = optim.Adam(self.G.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.D.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.criterion = nn.CrossEntropyLoss()
            self.L1_criterion = nn.L1Loss()
            self.w_L1 = opt.w_L1

        self.N_z = opt.N_z
        self.N_p = opt.N_p
        self.N_d = opt.N_d

    def init_weights(self):
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)

    def set_input(self, input, test_pose=None):
        """
        the structure of the input.
        {
        'image':Bx3x96x96 FloatTensor
        'pose':Bx1 FloatTensor
        'identity':Bx1 FloatTensor
        }

        test_pose (B): used for the test='initial learning rate
        """
        self.image = input['image']
        self.batchsize = len(self.image)
        self.pose = input['pose'].long()            #convert to LongTensor

        if self.is_Train:
            self.input_pose = one_hot(self.pose, self.N_p)
        else:
            self.input_pose = one_hot(test_pose.long(), self.N_p)

        self.identity = input['identity'].long()    #convert to LongTensor
        self.name = input['name']
        self.fake_identity = torch.zeros(self.batchsize).long() # 0 indicates fake
        self.noise = torch.FloatTensor(np.random.normal(loc=0.0, scale=0.3, size=(self.batchsize, self.N_z)))

        #cuda
        if self.opt.gpu_ids:
            self.image = self.image.cuda()
            self.pose = self.pose.cuda()
            self.input_pose = self.input_pose.cuda()
            self.identity = self.identity.cuda()
            self.fake_identity = self.fake_identity.cuda()
            self.noise = self.noise.cuda()

        self.image = Variable(self.image)
        self.pose = Variable(self.pose)
        self.input_pose = Variable(self.input_pose)
        self.identity = Variable(self.identity)
        self.fake_identity = Variable(self.fake_identity)
        self.noise = Variable(self.noise)

    def forward(self, input, test_pose=None):
        self.set_input(input, test_pose)

        self.syn_image = self.G(self.image, self.input_pose, self.noise)
        self.syn = self.D(self.syn_image)
        self.syn_identity = self.syn[:, :self.N_d+1]
        self.syn_pose = self.syn[:, self.N_d+1:]

        self.real = self.D(self.image)
        self.real_identity = self.real[:, :self.N_d+1]
        self.real_pose = self.real[:, self.N_d+1:]

    def backward_G(self):
        self.Loss_G_syn_identity = self.criterion(self.syn_identity, self.identity)
        self.Loss_G_syn_pose = self.criterion(self.syn_pose, self.pose)
        self.L1_Loss = self.L1_criterion(self.syn_image, self.image)

        self.Loss_G = self.Loss_G_syn_identity + self.Loss_G_syn_pose + self.w_L1 * self.L1_Loss
        self.Loss_G.backward(retain_graph=True)

    def backward_D(self):
        self.Loss_D_real_identity = self.criterion(self.real_identity, self.identity)
        self.Loss_D_real_pose = self.criterion(self.real_pose, self.pose)

        self.Loss_D_syn = self.criterion(self.syn_identity, self.fake_identity)

        self.Loss_D = self.Loss_D_real_identity + self.Loss_D_real_pose + self.Loss_D_syn
        self.Loss_D.backward()

    def optimize_G_parameters(self):
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_D_parameters(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def print_current_errors(self):
        print('Loss_G: {0} \t Loss_D: {1}'.format(self.Loss_G.data[0], self.Loss_D.data[0]))

    def save(self, epoch):
        self.save_network(self.G, 'G', epoch, self.gpu_ids)
        self.save_network(self.D, 'D', epoch, self.gpu_ids)

    def save_result(self, epoch=None):
        for i, syn_img in enumerate(self.syn_image.data):
            img = self.image.data[i]
            filename = self.name[i]

            if epoch:
                filename = 'epoch{0}_{1}'.format(epoch, filename)

            path = os.path.join(self.result_dir, filename)
            img = Tensor2Image(img)
            syn_img = Tensor2Image(syn_img)

            width, height = img.size
            result_img = Image.new(img.mode, (width*2, height))
            result_img.paste(img)
            result_img.paste(syn_img, box=(width, 0))
            result_img.save(path)

class Multi_DRGAN(Single_DRGAN):
    """
    The model of Multi_DRGAN according to the options.
    """
    def name(self):
        return 'Multi_DRGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_Train = opt.is_Train

        self.G = Generator(N_p=opt.N_p, N_z=opt.N_z, single=False)
        self.D = Discriminator(N_p=opt.N_p, N_d=opt.N_d)
        if self.is_Train:
            self.optimizer_G = optim.Adam(self.G.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.D.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.criterion = nn.CrossEntropyLoss()
            self.L1_criterion = nn.L1Loss()
            self.w_L1 = opt.w_L1

        self.N_z = opt.N_z
        self.N_p = opt.N_p
        self.N_d = opt.N_d
