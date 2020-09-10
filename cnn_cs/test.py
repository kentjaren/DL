import time
import nni
from torch.utils.data import DataLoader

from option import BaseOption
from data.data_loader import CreateDataset
from model.model_loader import CreateModel

from misc.util import logging_test

opt = BaseOption(is_Train=False).parse()

dataset = CreateDataset(opt, 'test')

dataloader = DataLoader(
    dataset,
    batch_size=1,
    drop_last=False,
    shuffle=False,
    num_workers=opt.num_workers)

model = CreateModel(opt)

model.test_ith_net(2, dataloader)
logging_test(2, model.infer_time, model.psnr,
             model.pixel_status, model.ensemble_status)

if opt.automl:
    nni.report_final_result(model.ret_psnr)
