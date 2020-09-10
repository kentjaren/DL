import os
import nni
import yaml
import torch
import random
import logging
import argparse
import numpy as np
import datetime

from misc.util import creat_dir, setup_logging


class BaseOption(object):
    def __init__(self, is_Train=True):
        self.parser = argparse.ArgumentParser()
        self.is_Train = is_Train

        #######################################################################
        #                                 data                                #
        #######################################################################
        self.parser.add_argument(
            '--data_dir', default='./data', help='dir to store the data')
        self.parser.add_argument(
            '--dataset', default='CS20k', help='name of the dataset')
        self.parser.add_argument('--num_workers', type=int, default=0)

        self.parser.add_argument(
            '--cr',
            default=0.25,
            type=float,
            help='the compression ratio')

        self.parser.add_argument(
            '--ms_dir',
            default='PHI',
            type=str,
            help='the measure matrix dir.')

        #######################################################################
        #                                model                             #
        #######################################################################
        self.parser.add_argument(
            '--model',
            default='ReconNet',
            help='name of the model, e.g ReconNet DeepInverse')

        self.parser.add_argument(
            '--load_path',
            default=None,
            type=str,
            help='the load path of the model.')

        #######################################################################
        #                                train                                #
        #######################################################################
        self.parser.add_argument(
            '--epochs', default=1000, type=int, help='the num of epochs')
        self.parser.add_argument(
            '--batch_size',
            default=64,
            type=int,
            help='the num of files in the batch')
        self.parser.add_argument(
            '--val_freq',
            default=5,
            type=int,
            help='the frequence of saving model')

        self.parser.add_argument(
            '--lr0',
            default=0.001,
            type=float,
            help='the learning rate of the net0')
        self.parser.add_argument(
            '--lr1',
            default=0.001,
            type=float,
            help='the learning rate of the net1')
        self.parser.add_argument(
            '--lr2',
            default=0.001,
            type=float,
            help='the learning rate of the net2')
        self.parser.add_argument(
            '--lam',
            default=0.0001,
            type=float,
            help='the hyperparameter of the loss structure')
        self.parser.add_argument(
            '--alpha',
            default=0.1,
            type=float,
            help='the hyperparameter of the MixLoss.')
        self.parser.add_argument(
            '--early_stop',
            default=5,
            type=int,
            help='the max early stop times')
        self.parser.add_argument(
            '--assemble',
            default='avg',
            type=str,
            help='the method of assemble images, eg. avg, mix, near')
        self.parser.add_argument(
            '--delta',
            default=0.1,
            type=float,
            help='the threshold of mix assemble method compare to the middle.')
        self.parser.add_argument(
            '--loss_func_type',
            default=32,
            type=int,
            help='choose the loss function')

        #######################################################################
        #                                 misc                                #
        #######################################################################
        self.parser.add_argument(
            '--checkpoint_dir',
            default='./checkpoint',
            help='dir to store the checkpoints and option')
        self.parser.add_argument(
            '--gpu_ids',
            type=str,
            default='0',
            help='gpu ids: e.g 0 1. use -1 for cpu')
        self.parser.add_argument(
            '--close_rate',
            default=0.1,
            type=float,
            help='the close rate of a syn image to real image.')
        self.parser.add_argument(
            '--automl',
            default=False,
            type=bool,
            help='switch to automl')
        self.parser.add_argument(
            '--load_config_path',
            default=None,
            type=str,
            help='the config file, will cover the other setting.')
        self.parser.add_argument(
            '--train_single',
            default=False,
            type=bool,
            help='train single net or train 3 nets')
        self.parser.add_argument(
            '--train_same_time',
            default=False,
            type=bool,
            help='train multi nets at the same time')

    def parse(self):
        self.opt = self.parser.parse_args()

        if self.opt.load_config_path is not None:
            with open(self.opt.load_config_path) as f:
                config = yaml.load(f)
            self.update(config)

        if self.opt.automl:
            params = nni.get_next_parameter()
            self.update(params)

        if self.opt.load_path:
            self.opt.load_path = self.opt.load_path.split(',')

        timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        prefix = 'train' if self.is_Train else 'test'
        self.opt.save_dir = os.path.join(
            self.opt.checkpoint_dir, "{}_{}_{}_{}".format(prefix,
                                                          self.opt.dataset,
                                                          self.opt.model,
                                                          timestamp))

        creat_dir(self.opt.save_dir)
        self.set_gpu()
        setup_logging(logging_path=os.path.join(self.opt.save_dir, 'output.log'))

        args = vars(self.opt)
        logging.info('-----------------Options----------------')
        for k, v in sorted(args.items()):
            logging.info('{0}: {1}'.format(str(k), str(v)))
        logging.info('-------------------End-------------------\n')

        return self.opt

    def update(self, params):
        opt = vars(self.opt)
        for k in params.keys():
            opt[k] = params[k]
        self.opt = argparse.Namespace(**opt)

    def set_gpu(self):
        gpu_ids = self.opt.gpu_ids
        device = gpu_ids.split()[0]
        if torch.cuda.is_available() and not (gpu_ids == '-1'):
            self.opt.device = torch.device('cuda:{}'.format(device))
        else:
            self.opt.device = "cpu"
        os.environ['CUDA_VISIBLE_DEVICES'] = "" if gpu_ids == '-1' else gpu_ids

