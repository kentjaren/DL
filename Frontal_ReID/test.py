import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from model.model_Loader import CreateModel

opt = TestOptions().parse()

data_loader = CreateDataLoader(opt)
model = CreateModel(opt)

total_steps = 0
for i, data in enumerate(data_loader):
    batchsize = data['pose'].size()[0]
    total_steps += batchsize

    test_pose = torch.ones(batchsize)
    model.forward(data, test_pose)
    print(i)
    model.save_result()
