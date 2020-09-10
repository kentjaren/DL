import os
import nni
import time
import logging
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from option import BaseOption
from data.data_loader import CreateDataset
from model.model_loader import CreateModel

from misc.util import set_manualSeed
from misc.util import logging_train, logging_val, logging_test

opt = BaseOption().parse()
set_manualSeed(opt)

writer = SummaryWriter(os.path.join(opt.save_dir, 'event'))

train_data= CreateDataset(opt, 'train')
val_data = CreateDataset(opt, 'val')
test_data = CreateDataset(opt, 'test')

train_loader = DataLoader(
    train_data,
    batch_size=opt.batch_size,
    drop_last=False,
    shuffle=True,
    num_workers=opt.num_workers)

val_loader = DataLoader(
    val_data,
    batch_size=1,
    drop_last=False,
    shuffle=False,
    num_workers=opt.num_workers)

test_loader = DataLoader(
    test_data,
    batch_size=1,
    drop_last=False,
    shuffle=False,
    num_workers=opt.num_workers)

model = CreateModel(opt)

def train_same_time(net_index):
    max_psnr = 0
    count = opt.early_stop
    for epoch in range(1, opt.epochs + 1):
        model.train_same_time(net_index, train_loader)
        writer.add_scalar("Train_Loss", model.epoch_loss, epoch)

        logging_train(net_index, )


def train_stage(net_index, same_time=False):
    max_psnr = 0
    count = opt.early_stop
    for epoch in range(1, opt.epochs + 1):
        if same_time:
            model.train_same_time(net_index, train_loader)
            writer.add_scalar("Train_Loss", model.epoch_loss, epoch)
        else:
            model.train_ith_net(net_index, train_loader)
            writer.add_scalar("Net{}_Train_Loss".format(net_index), model.epoch_loss, epoch)
        logging_train(net_index, epoch, model.epoch_time, model.epoch_loss, same_time)

        if epoch % opt.val_freq == 0:
            model.val_ith_net(net_index, val_loader, epoch)
            writer.add_scalar('Net{}_Val_PSNR'.format(net_index), model.ret_psnr, epoch)
            logging_val(net_index, model.infer_time, model.psnr,
                        model.diff_ratio_status, model.pixel_status,
                        model.ensemble_status, epoch)

            if opt.automl:
                nni.report_intermediate_result(model.ret_psnr)

            if (model.ret_psnr > max_psnr):
                model.save_model(net_index)
                max_psnr = model.ret_psnr
                count = opt.early_stop
                max_epoch = epoch
            else:
                count = count - 1

            if (not opt.automl) and (count == 0):
                logging.info("epoch {} should be the choice for Net{}\n".format(
                    max_epoch, net_index))
                model.load_model(net_index)
                model.test_ith_net(net_index, test_loader)
                logging_test(net_index, model.infer_time, model.psnr,
                             model.diff_ratio_status, model.pixel_status,
                             model.ensemble_status)
                break

if opt.train_single:
    train_stage(0)
elif opt.train_same_time:
    train_stage(2, same_time=True)
else:
    for i in range(3):
        train_stage(i)

if opt.automl:
    nni.report_final_result(model.ret_psnr)
